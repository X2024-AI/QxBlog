# Internship Experience — Embodied AI / Robot Learning

> **Yang Qianxi (杨骞玺)** · [fill in role title, e.g. "Research Intern"]
> **Organization:** X2024-AI · [fill in team/lab name]
> **Period:** [fill in, e.g. "Aug 2026 — present"] · Remote (A100 cluster)
> **Project:** LingBot-VA — a video-action robot-manipulation model (Wan2.1-based world model)
> **Robot:** OpenArm (dual-arm, bimanual)
>
> *This doc distills my work from a set of Claude Code session transcripts. All numbers below
> are measured results from those sessions.*

---

## TL;DR

During this internship I worked on **LingBot-VA**, a video-action (world-model) robot
policy for bimanual manipulation, across the full sim→real loop: benchmarking, data
tooling, model serving, and performance engineering. The headline result is a **2.65×
end-to-end inference speedup** (10.3 s → 3.88 s per action chunk, **bit-identical**
output) by finding and fixing the real bottleneck — which turned out to be CPU/launch
overhead in the KV-cache pool and an accidental FSDP wrap, *not* GPU compute.

| Workstream | What I did | Result |
|---|---|---|
| RoboTwin-2.0 eval | Stood up the full eval stack (checkpoint, paths, configs, env fixes) | `adjust_bottle` 2/2 (100%) smoke |
| Inference optimization | Profiled + fixed server latency | 10,278 → 3,878 ms (2.65×), bit-identical |
| Real-time monitoring panel | Built a web panel for inference timing | Live phase + before/after; caught a real regression |
| Real-robot serving | Deployed + babysat the grasp_bottle server | Stable tmux serve, ~1.6–3.5 s/chunk |
| Data tooling | Inspection tool + ground-truth replay harness | 200 eps audited; client verified |
| Model comparison | Built EEF dataset + train config for openpi pi0.5 | Dataset validated, training config ready |

---

## Context

X2024-AI develops an embodied-AI stack around the **OpenArm** dual-arm robot. LingBot-VA
is the lab's video-action policy: it conditions on multi-camera observations and predicts
action chunks autoregressively, served to the robot over a websocket. My work sat where
the model meets the real system — making it run fast enough to control a robot in
real time, and making sure it can be deployed, evaluated, and monitored reliably on a
shared 8×A100 cluster.

**What I inherited:** a working but slow model, an official eval harness full of
placeholders, and no monitoring/ops tooling around the server.

---

## Contributions (in detail)

### 1. Inference optimization — 10.3 s → 3.88 s per chunk (2.65×)

The model was ~79× over the **130 ms** streaming budget. I was asked to try FP8 /
TensorRT-LLM / FlashInfer, but first **profiled before optimizing** — and the profile
changed the whole plan.

**Diagnosis** (torch.profiler + Nsight Systems, real server, synthetic client replaying
the exact control-loop protocol):
- `infer` (one chunk: 25 video + 50 action denoise steps): **10,278 ms** mean.
- `compute_kv_cache` (VAE encode + 2 fwd): **1,127 ms**.
- **GPU utilization only 23.4%** — ~3,000 idle gaps >5 ms; CPU 35.3 s vs GPU busy 6.0 s.
  The server is **CPU/launch-bound, not compute-bound**.

The time went to: (a) the KV-slot pool calling `torch.nonzero`/`any` 2–3× per layer per
step (~86 ms/step × 75 steps ≈ **6.5 s**); (b) cache gather/scatter; (c) fp64 RoPE + fp32
norms; (d) `empty_cache()` in the hot path; and — found while benchmarking — (e) **FSDP
(`fully_shard`) wrapping the transformer even at `world_size=1`** (~4.3 s/chunk of
allgather churn).

**Fixes shipped:**
- **RingKVPool** — pre-allocated contiguous ring buffer replacing the nonzero/argsort slot
  pool; zero device syncs. (1.40× alone; 2.7× in-process.)
- **Gate FSDP on `world_size>1`** — the single biggest win.
- **Hot-path cleanup** — removed `empty_cache()` (5 sites), per-episode caches for
  `cfg_text_emb`/grid-ids, kept fp64 RoPE (fp32 diverged the action trajectory by chunk 2,
  corr 0.96).

**Verification** (what made this trustworthy, not just fast): output confirmed
**bit-identical** to baseline across all **4,680 attention calls** of a full control
sequence (fixed-seed harness + `script/test_ring_kv_equivalence.py`).

**What I correctly rejected, and why:**
| Accelerator | Verdict | Reason |
|---|---|---|
| FP8 | ✗ | A100 is sm80 — no FP8 tensor cores |
| TensorRT-LLM | ✗ | wrong model class (not a standard transformer) |
| paged KV-cache | △ | the ring buffer already solved it |
| FlashInfer | △ | attention is only ~7.5% of GPU time — save for last |

**Remaining bottleneck documented:** VAE encode = 85% of the kv-cache call (~700 ms).

### 2. Real-time inference monitoring panel

Built a web panel (`script/inference_panel.html` + timing probes at 0.0003% overhead) that
shows, live and per phase: the stack of inference phases, trend vs the 130 ms budget, a
**phase-share advisory** (optimize / later / skip, with reasons), and a **before/after
exhibit** of the measures shipped. Added **multi-server support** (tab switchover) so both
the sim and real-robot servers can be watched at once.

This panel immediately paid for itself: it caught a real regression where a neighbor job
(~200 procs) was inflating every phase 3.8 s → 8–9 s — the launch-bound signature, which a
blind latency number alone wouldn't have explained.

### 3. RoboTwin-2.0 evaluation (sim benchmark)

Set up the official LingBot-VA eval from scratch: downloaded the 24.4 GB checkpoint
(retry wrapper, ~1.5 h), fixed placeholder paths and a missing config
(`policy/ACT/deploy_policy.yml`), pinned `curobo` to v0.7.8, made flash-attn optional in
favor of SDPA, and rewrote the launch scripts to pass absolute paths. Found the
`enable_offload=True` trap (>1 h/episode → ~1 min after turning it off). Smoke test
`adjust_bottle` **2/2 success**.

### 4. Real-robot serving (grasp_bottle)

Deployed the real-robot `grasp_bottle` model (checkpoint 1710) from the live repo: built
the deploy dir (symlinks + `attn_mode` flex→torch patch), added the serve config
(`num_inference_steps=8`, `action_num_inference_steps=20`, `attn_window=36`), and wrote a
smoke-test protocol. Kept it up through three server deaths with three distinct causes
(process-group reaping → `setsid` → `tmux`) — learning the ops reality that a serving
process's lifetime is as much about process-tree topology as about the code.

### 5. Data & client tooling

- **`script/inspect_data.py`** — inspection tool for the paper_cup_relay dataset
  (`.parquet`/`.pth`/`.pt`); audited 200 episodes / 600 segments / 1,800 latent files.
- **MuJoCo ground-truth replay harness** (`replay_gt.py`) — replays recorded expert actions
  through the unchanged client to test it without a live server. Key finding:
  `success=False` was often a harness artifact (event-driven grasp/release stages), not a
  physics failure — replaying joint targets alone never fires `try_attach`/`prepare_release`.

### 6. Model comparison (openpi pi0.5)

Prepared a comparison run training openpi's pi0.5 on the same paper_cup_relay data:
built the EEF dataset (`paper_cup_relay_eef`, 14-dim Cartesian deltas swapped in without
re-encoding video), and wrote the `pi05_paper_cup_relay_eef` train config
(`action_horizon=50`, `batch_size=64`).

---

## Skills demonstrated

- **Performance engineering:** profiling-driven optimization (Nsight / torch.profiler),
  identifying CPU/launch-bound bottlenecks, KV-cache design, bit-exact verification.
- **Robot learning:** video-action / world-model policies, diffusion action generation,
  VAE latent spaces, RoboTwin / MuJoCo / Isaac simulation, behavior cloning, imitation
  learning.
- **ML infra / ops:** serving deployment, tmux/process management on shared GPUs,
  websocket+msgpack protocols, SSH tunnels, monitoring dashboards.
- **Data engineering:** LeRobot v2.1 datasets, EEF action-space conversion, inspection
  tooling.
- **Stack:** PyTorch, torch.distributed / FSDP, CUDA, Linux, Git (incl. graft commits),
  Python.

## Impact

- **2.65×** end-to-end inference speedup, output **bit-identical** — a result a
  researcher can trust, not just a faster number.
- Made the model's latency **observable** (panel) and its deployment **reproducible**
  (deploy dir + configs + smoke protocol + docs).
- Benchmarked and served the model end-to-end on the real-robot stack.

## Reflection

The most valuable lesson: **profile before optimizing.** The four "obvious" accelerators
(FP8, TensorRT-LLM, …) were all wrong for this workload; the real win came from a
`nonzero` call in the KV-cache pool and an accidental FSDP wrap — things only a profile
reveals. And on the ops side: a serving process is only as reliable as the process-tree
topology you launch it under.

---

*[Fields in brackets at the top — role title, team name, exact dates — were not stated in
the source transcripts and should be filled in before sharing.]*

# LingBot-VA — Deployment & Operations Guide

> Internal how-to for serving, evaluating, and monitoring the LingBot-VA robot model.
> Covers the real-robot `grasp_bottle` server, RoboTwin-2.0 eval, the MuJoCo client
> harness, the data-inspection tool, the inference panel, and the host quirks.
> Author: Yang Qianxi (杨骞玺) · Last updated 2026-08-29.

---

## 1. What LingBot-VA is

LingBot-VA is a **video-action (VA) robot-manipulation policy** built on the Wan2.1
video-diffusion stack — a `WanTransformer3DModel` transformer over a VAE latent space,
conditioned by a T5 text encoder. It predicts action chunks autoregressively from
multi-view camera observations, served over a **server–client** architecture
(websocket + msgpack).

**Component layout** (all derived from one config field
`wan22_pretrained_model_name_or_path`, a directory containing `transformer/`, `vae/`,
`tokenizer/`, `text_encoder/`):

| Component | Purpose |
|---|---|
| `vae/` | `AutoencoderKLWan` — encodes camera frames to latent; a second instance runs for wrist cams on bimanual tasks |
| `tokenizer/` | `T5TokenizerFast` |
| `text_encoder/` | `UMT5EncoderModel` (~11 GB, 3 shards) |
| `transformer/` | `WanTransformer3DModel.from_pretrained(..., attn_mode="torch")` |

**Tasks / checkpoints:**

| Model | Task | Checkpoint |
|---|---|---|
| `grasp_bottle_serve` | real-robot grasp_bottle (OpenArm bimanual) | `/mnt/checkpoints_lingbot/checkpoints/checkpoint_step_1710` |
| `robotwin` | RoboTwin-2.0 benchmark | `pretrained/lingbot-va-posttrain-robotwin` (HF `robbyant/lingbot-va-posttrain-robotwin`) |
| `paper_cup_relay_serve` | MuJoCo sim, paper-cup relay | trains to `/mnt/qx_data/sim_lingbot_train_v2` |

---

## 2. Hardware & environments

- **Host:** 8× NVIDIA A100-SXM4-80GB node (sm80 — **no FP8 tensor cores**), shared with
  other users' training jobs. **Always `nvidia-smi` before launching** — GPU occupancy
  shifts constantly (e.g. a 4-GPU training run took GPUs 0–3 at 02:33 on 08-28).
- **Model env:** `/data/envs/lingbot-va` — torch 2.9.0+cu126, **no flash-attn** (import
  was made optional; `attn_mode="torch"`/SDPA works).
- **Sim env (RoboTwin):** conda `robotwin` (sapien 3.0.0b1, mplib 0.2.1, curobo v0.7.8).
- **openpi env:** `/data/qx_workspace/openpi` (lerobot 0.1.0), MuJoCo rendered with
  `MUJOCO_GL=egl`.

---

## 3. Serving a checkpoint (real robot)

### 3.1 Deploy directory

A checkpoint is not served directly. Build a deploy dir, e.g.
`pretrained/grasp_bottle_serve/`:

- `vae/`, `tokenizer/`, `text_encoder/` → symlinks to `pretrained/lingbot-va-base/`
- transformer weights → symlink to the checkpoint (`checkpoint_step_1710`)
- `config.json` → copied from the checkpoint with **`attn_mode` patched `flex → torch`**
  (the only diff vs the raw checkpoint; required for serving).

### 3.2 Config registration

- `wan_va/configs/va_grasp_bottle_cfg.py` — task geometry (30-dim actions,
  `action_per_frame=12`, `openarm_bimanual`, obs keys `head/wrist_left/wrist_right`).
- `wan_va/configs/va_grasp_bottle_serve_cfg.py` — serve knobs:
  `num_inference_steps=8`, `action_num_inference_steps=20`, `attn_window=36`, RTC enabled.
- Registered as `grasp_bottle_serve` in `wan_va/configs/__init__.py`.

> Deliberately **not** ported from the old clone: `video_exec_step=4` / `async_video_refine`
> (this repo's server lacks the async-refine path).

### 3.3 Launch

Launch **inside tmux** — never from a tool call. `nohup`/`setsid`-launched procs get
reaped when the launching process group is cleaned (observed 08-27 ~14:24: all
Claude-launched servers died together). tmux survives because its procs belong to the
tmux server's tree.

```bash
tmux new-session -d -s lingbot_grasp -c /data/qx_workspace/lingbot-va \
  'CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
   /data/envs/lingbot-va/bin/python -m torch.distributed.run --nproc_per_node 1 \
   --master_port 29062 wan_va/wan_va_server.py \
   --config-name grasp_bottle_serve --port 29537 --metrics-port 29538 \
   2>&1 | tee -a /tmp/lingbot_va_server_grasp.log'
```

Restart: `tmux kill-session -t lingbot_grasp` then re-create.
Log: `/tmp/lingbot_va_server_grasp.log`. Health: `GET :29537/healthz` → `"OK"`.

### 3.4 Port map (this box)

| Port | Owner |
|---|---|
| **29536** | `paper_cup_relay_serve` eval server (another session, GPU 3 — **do not kill**) |
| **29537** | `grasp_bottle_serve` websocket |
| **29538** | `grasp_bottle_serve` metrics (must differ from the websocket port — a collision was caught pre-launch) |
| 29062 | torchrun master port |
| 29056 / 29061 | robotwin eval server/client |

### 3.5 Smoke-test protocol

Order matters (the server survives client errors, but to *succeed*):

1. `infer(dict(reset=True, prompt=...))` first — initializes `frame_st_id` (else `AttributeError`).
2. First chunk: obs as a **single dict** (not a list) — a cold-cache VAE temporal conv
   crashes on a multi-frame list.
3. `infer(dict(obs=[8 keyframe dicts], compute_kv_cache=True, state=prev_action))` — needs `state`.
4. Step chunks with single-dict obs.

Obs keys: `observation.images.head` (256×512), `observation.images.wrist_left` /
`wrist_right` (256×256), `observation.state` (30,). Actions: `(30, 2, 12)`-style chunks.

---

## 4. RoboTwin-2.0 evaluation

```bash
# model server (single GPU)
bash evaluation/robotwin/launch_server.sh
# client
task_name="adjust_bottle"; save_root="results/"
bash evaluation/robotwin/launch_client.sh ${save_root} ${task_name}
```

Key gotchas (all hit and fixed on this box):

- **`enable_offload` must stay `False`** on 80 GB cards — with `True` the VAE encodes on
  CPU and one episode takes >1 h; off it, minutes.
- **Websocket client must connect to `127.0.0.1`**, not the `0.0.0.0` default — the
  `ws://0.0.0.0:<port>` URI fails the handshake through this host's proxy.
- **RoboTwin checkout** `c659e0d` uses `env_cfg/task_config/` (not top-level `task_config/`).
- **`curobo` pinned to `v0.7.8`** — `planner.py` needs the old API
  (`curobo.types.math`, `curobo.wrap.reacher.motion_gen`).
- **Vulkan/OIDN:** the client can hit `OIDN Error: illegal memory access` when its GPU is
  shared — pin server and client to the same idle GPU.
- Result: `adjust_bottle` smoke test **2/2 success**.

---

## 5. MuJoCo ground-truth replay harness

`evaluation/openarm_mujoco/replay_gt.py` + `launch_replay.sh` — replays recorded expert
actions through the unchanged `client.py` pipeline (`GroundtruthPolicy` stub with the exact
`WebsocketClientPolicy.infer` interface), for testing the client without a live server.

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 EPISODES=1 START_EPISODE=180 \
  bash evaluation/openarm_mujoco/launch_replay.sh
```

Critical insight: **`success=False` was often a harness artifact, not a physics failure** —
stage transitions (`right_attached`, …) are event-driven; the expert calls
`task.try_attach(side)` / `task.prepare_release(side)` at specific frames. Replaying joint
targets alone never fires them. Reset must also call
`apply_visual_randomization(mission, seed)` to mirror data collection.

---

## 6. Data-inspection tool

`script/inspect_data.py` — inspects `.parquet` / `.pth` / `.pt` files under
`paper_cup_relay/`. Used to audit 200 episodes / 600 segments / 1800 latent files.

Dataset shape (for reference): `paper_cup_relay` = 200 episodes × 643 frames (128,600
frames, 20 fps, LeRobot v2.1, 3 cameras). The `action` column is 30-dim joint-space; the
EEF variant (`paper_cup_relay_eef`) swaps in 14-dim Cartesian deltas via
`script/convert_paper_cup_relay/build_eef_dataset.py`.

---

## 7. Real-time inference monitoring panel

`script/inference_panel.html` + `script/launch_panel.sh <gpu> <port> <metrics_port>`,
probes via `wan_va/utils/timing_probe.py` (0.0003% overhead), JSON served at `/metrics`
by `script/metrics_http.py` (`/metrics`, `/healthz`, dashboard at `/`).

- Multi-server tab registry, 1 Hz polling, `localStorage` key `panel-servers-v2`.
- Shows live phase stacking, trend vs the **130 ms** streaming budget, a phase-share
  advisory (optimize / later / skip, with reasons), and the before/after of shipped
  optimizations.
- Caught a real regression: a neighbor job (`openpi_dxq`, ~200 procs) inflated chunks
  3.8 s → 8–9 s proportionally across all phases (the launch-bound signature).

**Remote access** (the panel runs on the A100 box, viewed from a laptop) — SSH tunnel:

```bash
ssh -f -N -o ServerAliveInterval=30 -L <local>:127.0.0.1:<metrics_port> A100
```

---

## 8. GitHub push recipe (host network quirks)

This host's default git proxy is dead. Fixes that work:

- Proxy: `git config http.https://github.com.proxy http://127.0.0.1:7890` (NOT `[::1]:7890`).
- Push auth: URL-embedded PAT — `https://x-access-token:<PAT>@github.com/X2024-AI/lingbot_va_repro.git`
  (`http.extraHeader="Authorization: Basic ..."` returns HTTP 400 through the proxy;
  add `http.version=HTTP/1.1` + `postBuffer=52428800`).
- SSH: `github.com:22` blocked; use `ssh.github.com:443` with `ProxyCommand nc -X connect -x 127.0.0.1:7890 %h %p`.
- Remote layout built via a **graft commit** (`2824e01`): `AnydexGrasp/` + `Unified_Aggregater_Server/` + `lingbot-va/`, ~44 M, no force-push.

---

## 9. Server-stability post-mortems (why "it's down again")

Three distinct deaths, three fixes:

1. **Process-group reaping** — `nohup ... &` from a tool call dies when the call's
   process group is cleaned. → use `setsid nohup`.
2. **setsid still reaped** when the session ended (08-27 ~14:24). → use **tmux**.
3. **tmux also died once** overnight 08-28→08-29 (session `lingbot_grasp` gone by 05:00
   despite a clean log) + a **drift warning**: another session relaunched the server
   *without* `--metrics-port` (orphan PPID-1 instance, `:29537/healthz` 200 but `:29538`
   dead → panel tab reads "offline" though serving works).

**Checklist on any restart:** tmux session exists **AND** `:29537/healthz` 200 **AND**
`:29538/metrics` alive **AND** `nvidia-smi` shows the model on the expected GPU.

---

## 10. Cross-cutting host facts

- `~/.git-credentials` holds a `zhengbi-yong` PAT with **pull-only** access; `gh` token expired.
- The plaintext GitHub PAT from the 08-26 session was pasted in-shell — **revoke/rotate it**
  if not already done.
- Benchmarks are only comparable when CPU load is quiet — neighbor jobs inflate latency
  proportionally across all phases (launch-bound signature).

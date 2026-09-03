# LingBot-VA Inference Speedup — SOP & Factor Reference

> Distilled from the "optimize the inference speed of lingbot server" session.
> One method: **profile first, optimize second.** Every factor mentioned is listed,
> grouped by the step of the procedure it belongs to.

**Headline result:** `infer` 10,278 ms → 3,878 ms (**2.65×**), `compute_kv_cache` 1,127 → 872 ms (**1.29×**), output **bit-identical** to baseline.

---

## Step 0 — Hardware & environment reality check

Factors that decide the whole optimization space (check these *before* proposing anything):

| Factor | Finding | Consequence |
|---|---|---|
| GPU micro-arch | A100-SXM4-80GB = **sm80 (Ampere)** | **No FP8 tensor cores** (need Hopper sm90 / Ada sm89) |
| Installed kernels | No TensorRT-LLM, FlashInfer, flash-attn, TransformerEngine | Only torch 2.9.0+cu126 + SDPA available |
| FlashAttention | SDPA **already dispatches to flash kernels** (`aten::_flash_attention_forward`) | "Enable FlashAttention" was already true; no action needed |
| Host | Shared 8-GPU box, other users' training jobs | Benchmarks only comparable when CPU load is quiet |

---

## Step 1 — Establish a real baseline (measure before touching code)

| Factor | Value |
|---|---|
| Streaming budget | **130 ms** per robot tick — the threshold that defines "fast enough" |
| One `infer` chunk | 25 video + 50 action denoise steps |
| Baseline `infer` | **10,278 ms** (p90 10,769) — **~79× over budget** |
| Baseline `compute_kv_cache` | **1,127 ms** (p90 1,204) |
| Baseline per-step | 70 ms video / 85 ms action denoise |
| Caveat | action chunking → robot executes 32 steps/chunk, so the *per-tick* wait is really the ~1.1 s kv-cache call |
| Methodology trap | The repeat-`infer` loop violated the streaming-VAE state machine (`_encode_obs` only runs on the first chunk per episode) — the timing loop must mirror `reset → infer → compute_kv_cache → …` |

---

## Step 2 — Profile to find *where* the time goes

Decisive metric: **GPU utilization 23.4%** (nsys: 23.9 s busy over 102 s, ~3,000 idle gaps >5 ms; CPU 35.3 s vs GPU busy 6.0 s) → the server is **CPU/launch-bound, not bandwidth/compute-bound**.

The five diagnosed hot spots:

1. **KV-slot pool syncs** — `allocate_slots`/`update_cache`/`_next_cache_id` call `torch.nonzero`/`any` 2–3× per layer per denoise step. Each `nonzero` = **~1.4 ms CPU stall** → ~86 ms/step × 75 steps ≈ **6.5 s** of the 10 s chunk.
2. **Cache gather/scatter** (`key_pool[:, valid]`, `cache['k'][:, slots] = key`) — ~1.2 s GPU + ~3.6 s CPU per 2 cycles.
3. **RoPE in float64 + `FP32LayerNorm` everywhere** — ~30% of kernels are casts/norms.
4. **Hot-path waste** — `torch.cuda.empty_cache()` (5 sites), per-step `prompt_embeds` clone (4 MB), grid-id rebuild, 32k H2D copies per 2 cycles.
5. **FSDP at world_size=1** (found *during* benchmarking, not profiling) — `_configure_model` ran `fully_shard` even single-rank → **~4.3 s/chunk** of allgather/rescatter (~6,750 collectives).

---

## Step 3 — Rank the proposed accelerators against measurements (reject what doesn't fit)

| Proposal | Verdict | Why (the factor that decided it) |
|---|---|---|
| **FP8** | ✗ | No sm80 tensor cores; AND the model is launch-bound, so bit-width isn't the binding constraint anyway |
| **TensorRT-LLM** | ✗ | Targets autoregressive LLM decoders; this is a dual-stream video+action *diffusion* transformer with custom 3-D RoPE and dual heads |
| **Paged KV (TileRT)** | △ | Right problem (KV-copy overhead), wrong fix — a pre-allocated ring buffer kills the syncs with no new dependency |
| **FlashInfer** | △ | Attention is only ~7.5% of GPU time; revisit after launch-bound is fixed |
| **`flash-attn` package** | ~No gain | Attention slice is small; FA2 vs SDPA-flash differs ~10–20% *of* 7.5% → <1.5% E2E, adds a build dep |

---

## Step 4 — Implement in ranked order

1. **RingKVPool** (`wan_va/modules/model.py`) — pre-allocated contiguous ring buffer replacing the nonzero/argsort slot pool; host-side pointers, zero device syncs, slice-view reads. → infer 10,278 → 7,342 ms (1.40×).
2. **Skip FSDP when world_size==1** (`wan_va/distributed/util.py`) — the biggest single win (~4.3 s/chunk).
3. **Hot-path cleanup** — remove `empty_cache()`, per-episode `cfg_text_emb` (was 150 clones/chunk), grid-id caching, log-spam removal.
4. **RoPE kept fp64** — fp32 tested and *reverted* (see Step 5).

---

## Step 5 — Verify **bit-identical** (fast ≠ correct)

Factors that make the result trustworthy, and the bugs they caught:

| Factor | Detail |
|---|---|
| Fixed-seed equivalence | final actions `np.array_equal` under fixed seeds |
| Digest-stream diff | all **4,680 attention calls** across a full control sequence compared to HEAD |
| Equivalence suite | `script/test_ring_kv_equivalence.py` — 6 cases: append / transient / pred-rollback / eviction / wrap |
| **My own bug** | transient path rebound `key` *before* `drop_last(key.shape[1])` → silently zeroed the pool each step. Unit tests missed it (they called `drop_last(n)` directly); the digest-stream diff caught it |
| **Legacy bug** | legacy frees transient slots by ascending insertion id → frees a real entry while keeping stale garbage (can't fire at production capacity 18,432 tokens) |
| Chaos amplification | a tiny FP reorder compounds through 75 flow-match ODE steps — chunk-2 corr can drop to 0.40; fp32 RoPE → corr 0.96 by chunk 2, so fp64 kept |
| FSDP class rename | `fully_shard` renames the class to `FSDPWanAttention` — post-wrap class-level monkeypatches silently no-op |

---

## Step 6 — Benchmark honestly (control the environment)

Factors that skewed numbers during benchmarking:

- GPU squatters (a 32–59 GB job could land mid-benchmark).
- Two servers bound to the **same port** (29057) → client round-robined, contended.
- `save_async` writing latents/actions to a single-worker pool inflated E2E.
- OOM from a forgotten `CUDA_VISIBLE_DEVICES`.
- The in-process harness vs websocket-E2E gap (3.66 s vs 8 s) — resolved by finding the FSDP path, then A/B under identical conditions (same wrapper, same GPU).

**Final (websocket E2E, idle A100):**

| Phase | Before | After | Speedup |
|---|---|---|---|
| `infer` chunk | 10,278 ms | **3,878 ms** | **2.65×** |
| `compute_kv_cache` | 1,127 ms | **872 ms** | 1.29× |

---

## Step 7 — Make it observable (the panel)

Why: the panel caught a real regression a latency number alone wouldn't explain.

- **Probes** (`wan_va/utils/timing_probe.py`) around 5 phases — `vae_encode`, `video_diffusion`, `action_diffusion`, `kv_transformer`, `postprocess`. Overhead **0.01 ms/chunk (0.0003%)**, zero numerical impact.
- **Endpoint** (`script/metrics_http.py`) — `/metrics` JSON + `/healthz` + dashboard at `/`, inside the server process (`--metrics-port`, must differ from the websocket port).
- **Panel** (`script/inference_panel.html`) — KPI row, stacked per-call phase columns, log trend vs 130 ms, phase-share advisory (▲ optimize / ◆ later / ✕ skip with reasons), before/after exhibit, multi-server tabs.
- **Launch-bound signature** — a neighbor job (`openpi_dxq`, ~200 CPU procs) inflates **all** phases proportionally (3.8 s → 8–9 s), confirming the CPU-bound diagnosis.
- **Real-robot config** (`num_inference_steps=8`, `action=20`): video 0.77 s + action 1.05 s; **VAE encode 0.7 s = 85% of every kv-cache tick** — the top remaining lever.

---

## Step 8 — Remaining levers (ranked by measured headroom)

| # | Lever | Target | Expected | Risk / cost |
|---|---|---|---|---|
| 1 | `torch.compile` the denoise step (`max-autotune-no-cudagraphs`) | both loops | 3.9 → **~2.5 s** | Low — verify drift like RoPE; `aten::copy_` = 18% GPU time |
| 2 | CUDA-graph the 77 fixed-shape forwards | both loops | +20–30% | Medium — ring pool is graph-safe (host pointers, no syncs) |
| 3 | Action steps 50 → 25 | action loop | **−1.27 s** | Needs RoboTwin success-rate eval |
| 4 | VAE encode optimization | kv call | 0.87 → ~0.5 s | `channels_last_3d` + `cudnn.benchmark`, batch the two sequential streaming-VAE passes, or fp16 |
| 5 | Drop CFG on video loop (guidance_scale 5 → batch ×2) | video loop | −0.6 s | Needs eval; video loop is amortized in streaming |
| 6 | Step/distillation retraining (few-step flow matching) | everything | 2–4× | The only path to <130 ms; training-time |
| 7 | Multi-GPU (pipeline parallel) | everything | ~linear | FSDP measured *harmful* at ws=1 — needs re-benchmark |

**Realistic serving-side budget without touching weights** (#1+#2+#3+#4): **infer ~1.3–1.8 s, kv ~0.5 s**. Getting under the 130 ms budget requires #6 (distillation) — the 77-step diffusion structure puts a hard floor under eager/compiled inference.

---

## One-line takeaways

- The four "obvious" accelerators (FP8 / TRT-LLM / FlashInfer) were **all wrong**; the wins came from a `nonzero` sync, an accidental FSDP wrap, and hot-path `empty_cache` — things only a profile reveals.
- Fast is not enough: **bit-identical verification** is what turns a speedup into a publishable/shippable result.
- On a shared box, **CPU contention is a first-class failure mode** for launch-bound workloads — that's why observability (the panel) is part of the SOP, not an afterthought.

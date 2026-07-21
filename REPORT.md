# LLM Inference on a Single RTX A5000: A Field Guide

> A hands-on, numbers-first walkthrough of what actually happens when you serve a small, fine-tuned language model from one 24 GB GPU. Everything below comes from four reconciled measurement studies on the *same* rig. Read the caveats — they are the difference between "I saw a number" and "I understand what the number means."

---

## 0. The Setup (read this first)

Everything in this guide runs on **one machine, one GPU**:

| Component | Choice | Notes |
|---|---|---|
| GPU | **1× NVIDIA RTX A5000, 24 GB** (~24,111–24,564 MiB usable) | Power cap ~230 W. Single card, no tensor/pipeline parallelism. |
| Base model | **Qwen3-4B-Instruct-2507** (rev `cdbee75f`, Apache-2.0) | 4B params, 2560 hidden, 151,936-token vocab, tied embeddings. |
| Fine-tune | **legal-ops LoRA**, `r=16`, `α=32` (⇒ α/r = 2.0), dropout 0.05 | Targets q/k/v/o/gate/up/down_proj. Merged into the base, not served as a live adapter. |
| Merge → deploy | LoRA **merged** (bit-exact) → converted to **bf16 GGUF** → quantized to **Q6_K** | ~3.3 GB, 6.56 bits-per-weight. |
| Engine | **llama.cpp** (CUDA, build `91d2fc38`, arch 86) | f16 KV cache, `--no-kv-unified`, `--parallel == concurrency`, 768-token context per slot (except the conversation test). |

The through-line of this whole guide: **a 4B model at Q6_K is tiny relative to 24 GB.** The interesting questions are therefore *not* "does it fit?" (it does, ~3.3 GB of weights) but "how many users can I serve, how fast, and *what runs out first*?" The answer to the last one turns out to be surprising — and it is not VRAM, and it is not even the GPU's compute.

A note on honesty before we start: two of the four studies contain a **planned comparison that did not happen** (the vLLM engine crashed) and **a probe that measures something subtler than its nickname suggests** (the "extract a LoRA with SVD" study). I flag both loudly and repeatedly, because a field guide that lets you quote a number you shouldn't quote has failed you.

---

## 1. Quantization & GGUF — shrinking the model without lying about it

**What quantization is.** The trained weights are 16-bit floats. Quantization stores them in fewer bits (8, 6, 5, 4…) using per-block scales, trading a small quality loss for a large size and bandwidth win. **GGUF** is llama.cpp's container format; `Q6_K` is a k-quant scheme at ~6.56 bits/weight.

**The pipeline here is provenance-clean and verified.** This matters more than people think — a silently broken merge or a bad narrowing cast will quietly degrade quality and you'll never see an error. The recorded verification chain:

- Merge loads base in **bf16**, forces the LoRA `B@A` product to **FP32**, computes it, then casts **once** to bf16 (`merge_and_unload(safe_merge=True)`), re-ties the head.
- **Bit-exact assertion:** merged `q_proj` == `base + (α/r)·BA`; tied embeddings preserved; **no `lm_head` synthesized**; no embedding resize.
- Convert with `convert_hf_to_gguf.py --outtype bf16` (feeds the native quantizer, avoiding a bf16→f16 narrowing), then `llama-quantize … Q6_K` with **no** `--pure`/`--allow-requantize`/`--leave-output-tensor`.
- GGUF metadata verified: `general.architecture == qwen3`, 151,936 vocab, tied output head (`output.weight` absent), chat-template SHA-256 match, filetype `MOSTLY_Q6_K`.

**Result:** ≈ **3.3 GB, 6.56 bpw.** Downstream sanity: **0 failures across 2,620 requests**, and `tokens_predicted_total` matched the client exactly (5,120 / 153,600 / 512,000). That exact-match is the tell that the served model is doing what the harness thinks it is doing.

> **Teacher's note.** One caveat is baked into the artifact: the manifest's `source_git_commit` is `null` — the *exact repo state* of the run wasn't captured. The model provenance is pinned; the code provenance is not. Small, but honest.

The *speed* consequences of the quantization choice belong to Section 4 (precision knobs), where we measure them directly. Preview: **lower-bit weights decode faster**, because decode is bandwidth-bound.

---

## 2. LoRA — and the "is a LoRA just matmul / SVD extraction?" study (labeled honestly)

**LoRA in one breath.** Instead of updating a weight matrix `W` (shape `d×k`), you learn two small matrices `B` (`d×r`) and `A` (`r×k`) with rank `r ≪ d,k`, and use `W + (α/r)·B A`. Here `r=16`, `α=32`. Merging just *does that addition* and bakes it into `W`. So yes — **a merged LoRA is, mechanically, "the base weights plus a low-rank matmul."** That part is not mysterious.

The tempting next question is the interesting one, and it's where you must be careful:

> *"If a merged LoRA is base + BA, can I take any two model checkpoints, subtract them, run SVD on the difference, and recover the LoRA that produced it?"*

The study probes exactly this, with a **control** (a delta we *know* is a rank-16 LoRA) and a **treatment** (the delta between two published checkpoints). Here is the whole thing, honestly labeled:

| Metric (252 target-linear matrices) | **CONTROL:** merged − Instruct (known `r=16` LoRA) | **Instruct-2507 − Base** (inter-checkpoint delta) | Ratio |
|---|---|---|---|
| mean eff_rank (energy-entropy, /2560) | 42.83 | 1507.3 | 35.2× |
| mean stable_rank | 2.27 | 247.87 | 109× |
| mean rel_delta (‖ΔW‖_F/‖W‖_F) | 0.00195 (~0.2%) | 0.16952 (~17%) | ~87× |
| **rank-16 SVD retained Frobenius energy** | **74.2%** (recon 0.5075) | **3.55%** (recon 0.98211) | — |
| rank-256 / rank-512 retained energy | — | 29.1% / 47.4% | — |
| attn eff_rank / stable_rank | 38.06 / 2.35 | 1110.67 / 184.08 | — |
| mlp eff_rank / stable_rank | 49.19 / 2.16 | 2036.2 / 332.93 | — |
| rank-16 retained: attn vs mlp | 74.2% / 74.3% | 5.45% / **2.84% (MLP harder)** | — |
| embed_tokens rel_delta | 0.0 | **0.2041 (~20.4%)** | — |
| model.norm.weight rel_delta | 0.0 | 0.00362 (~0.36%) | — |

**How to read this:**

1. **The control behaves like a low-rank thing should.** A known rank-16 adapter reads as eff-rank ~43, stable-rank ~2.27, touches embeddings/norms *exactly zero*, and a rank-16 SVD recovers ~74% of its energy. (It isn't a clean "16" because we read it back through bf16-merged weights — bf16 rounding noise, ~2⁻⁸·|W|, is non-trivial against a delta that's only ~0.2% of the base norm, so it smears energy across the spectrum. That's the intended calibration baseline, not a bug — but note it inflates the absolute rank/recon floors for *both* deltas.)

2. **The inter-checkpoint delta is high-rank.** Rank-16 SVD keeps only **3.55%** of the energy; even rank-512 (32× the adapter rank) keeps under half. It moves **embeddings ~20%** — something a target-linear LoRA (q/k/v/o/gate/up/down only) *literally cannot represent*. And **MLP is harder than attention** (down_proj is the extreme: eff_rank ~2088).

**Now the mandatory honest framing — do not skip this:**

- **It is an INTER-CHECKPOINT delta, not a proven FullFT delta.** Instruct-2507 declares no `base_model`; its actual production process is not established from these artifacts. Calling it "the full fine-tuning delta" would be fabricating a lineage.
- **Any rank-r SVD of a weight delta is "LoRA-REPRESENTABLE," never "a working extracted LoRA."** SVD gives the *best rank-r approximation of the observed weights*. **No task behavior was trained or validated** — the workflow measures spectra, not downstream quality. An "extracted adapter" that was never run against a task is not an adapter you can trust.
- **This high-rank result does NOT refute "LoRA can match FullFT."** (Cf. Thinking Machines, *"LoRA Without Regret."*) A *trained* low-rank adapter stores **task-relevant information**, not the full weight delta, and matches full fine-tuning when applied to all layers (especially MLP) with adequate capacity. Measuring that a *checkpoint difference* happens to be high-rank says nothing about the quality a trained LoRA can achieve — the two are orthogonal.

> **Takeaway.** "A merged LoRA is base + a low-rank matmul" — true and useful. "Therefore I can SVD any model diff back into the LoRA that made it" — **false in general**, and this study is the receipt: the real between-checkpoint delta is high-rank, embedding-shifting, and MLP-heavy, i.e. *not* low-rank-recoverable.

---

## 3. Concurrency Capacity — the core section

This is the heart of the guide. **Question:** on this one A5000, as we raise the number of concurrent streams `C`, what happens to aggregate throughput, per-user experience, VRAM, and — crucially — *which resource becomes the bottleneck*?

**Method (label it precisely).** llama.cpp, closed-loop / **barrier-synchronous** load, `ignore_eos`, fixed **256-token prompt / 256-token generation**, `--parallel == C`, 768-token context per slot. This is a **saturation-capacity probe**, not a real arrival-pattern latency test. Keep that in your pocket for Sections 6–7.

### 3.1 The master table

| C | Agg tok/s | tok/min | Speedup vs C1 | Fair-share tok/s (agg/C) | Lat p50 (s) | Lat p95 (s) | TTFT p50 (s) | TTFT p95 (s) | GPU util med % (p95) | Power med W | MemCtrl proxy % | Srv-proc CPU % | VRAM MiB (fit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 118.4 | 7,105 | 1.0× | 118.4 | 2.18 | 2.20 | 0.073 | 0.081 | 94 (95) | 229 | 71 | 2.55 | 3584 |
| 2 | 205.5 | 12,332 | 1.7× | 102.8 | 2.49 | 2.54 | 0.120 | 0.138 | 91 (92) | 229 | 65 | 2.60 | 3692 |
| 4 | 293.5 | 17,613 | 2.5× | 73.4 | 3.48 | 3.55 | 0.227 | 0.247 | 86 (87) | 227 | 50 | 2.67 | 3908 |
| 8 | 375.4 | 22,525 | 3.2× | 46.9 | 5.45 | 5.48 | 0.442 | 0.482 | 80 (94) | 228 | 36 | 2.80 | 4340 |
| 16 | 637.6 | 38,258 | 5.4× | 39.9 | 6.42 | 6.51 | 0.585 | 0.899 | 61 (99) | 223 | 39 | 3.87 | 5204 |
| 24 | 704.2 | 42,254 | 5.9× | 29.3 | 8.68 | 8.77 | 0.839 | 1.101 | 54 (99) | 203 | 34 | 5.27 | 6072 |
| 32 | 748.5 | 44,911 | 6.3× | 23.4 | 10.94 | 11.29 | 0.892 | 1.348 | 50 (92) | 193 | 32 | 7.22 | 6932 |
| 32b | 744.1 | 44,645 | 6.3× | 23.3 | 11.00 | 11.46 | 0.901 | 1.310 | 53 (99) | 196 | 33 | 7.17 | 6932 |
| 48 | 789.0 | 47,338 | 6.7× | 16.4 | 15.54 | 16.32 | 0.933 | 1.574 | 44 (99) | 178 | 28 | 10.44 | 8664 |
| **64** | **801.8** | **48,108** | **6.8×** | **12.5** | 20.41 | 21.54 | 0.966 | 2.195 | 46 (99) | 174 | 29 | 13.82 | 10388 |
| 96 | 772.7 | 46,360 | 6.5× | 8.0 | 31.79 | 33.65 | 0.902 | 3.193 | 38 (99) | 141 | 23 | 21.26 | 13854 |
| 128 | 749.3 | 44,958 | 6.3× | 5.9 | 43.64 | 47.21 | 0.659 | 4.047 | 41 (99) | 168 | 26 | 29.13 | 17308 |
| *WorkA 30* | 720.5 | 43,230 | 5.9× | 24.0 | 10.51 | 11.34 | 0.896 | 0.959 | 51 (99) | 194 | 32 | n/a | 6744 |
| *WorkA 100* | 765.4 | 45,927 | 6.3× | 7.7 | 33.26 | 34.54 | 0.647 | 1.136 | 41 (99) | 157 | 24 | n/a | 14550 |

*(C=32b is a repeat run; it agrees with C=32 within ~0.6%, so run-to-run noise is ~1%. WorkA is a second, independent run with 2× the sampled tokens per worker — it lands on the same curve, confirming the shape is real, not sampling noise. C=24 had 1 failed request out of 240; all other points were 0 failures.)*

### 3.2 Throughput peaks, then *declines* — there is no simple "knee at 30"

Aggregate output throughput **rises to a broad plateau over C=32–64, peaks at C=64 = 801.8 tok/s (48,108 tok/min = 6.77× the single-stream rate), then falls** to 772.7 at C=96 and 749.3 at C=128 — back down to roughly the C=32 level.

```
tok/s
 802 ┤                                   ● 64  (PEAK)
 789 ┤                              ● 48
 773 ┤                                        ● 96
 749 ┤              ● 32                            ● 128
 704 ┤          ● 24
 638 ┤       ● 16
 375 ┤   ● 8
 294 ┤ ● 4
 206 ┤● 2
 118 ┤● 1
     └┬──┬──┬──┬──┬───┬────┬──────┬──────┬──────┬──
      1  2  4  8 16  24   32     48     64     96  128   → C
                                    ↑ plateau ↑  ↓ decline ↓
```

**The marginal-gain view makes it unambiguous.** The single biggest batching win is **C8→C16 (+262 tok/s)**. After that, each doubling adds less: C16→24 +67, C24→32 +44, C32→48 +41, C48→64 **+13**, then it goes **negative**: C64→96 **−29**, C96→128 **−23**. Adding streams past 64 *loses* aggregate throughput.

> **Teacher's note.** Don't say "the knee is at 30." The efficient region rolls off well before the peak, and the *absolute* peak is at 64 — but the plateau (C=32–64, all ≥93% of peak) is broad and flat, and everything past it declines. "Peak at 64, plateau 32–64, decline beyond" is the honest one-liner.

### 3.3 Per-user experience collapses the whole time

Aggregate throughput is the *server's* view. The *user's* view — **fair-share throughput (aggregate ÷ C)** — falls monotonically from the very first step: 118.4 → 102.8 → 73.4 → … → 12.5 at the C=64 peak → **5.9 at C=128**. That's a **20.1× degradation**.

Because each request generates a fixed 256 tokens, **end-to-end latency is just 256 ÷ fair-share**, so it's the mirror image: p50 climbs **2.18 s → 43.64 s (20.1×)**. At the throughput *peak* (C=64) each user is already waiting **20.4 s** for their 256 tokens.

TTFT tells a two-part story: **p50 stays sub-second at every C** (the barrier keeps prompt processing cheap and orderly — even 0.66 s at C=128), but the **tail explodes**: TTFT p95 grows 0.081 s → 4.05 s (~50×) and p99 to 7.47 s as queueing bites.

> **The tension you must internalize:** the server is happiest (max tok/s) at exactly the concurrency where individual users are already miserable (20 s latency). "Throughput-optimal" and "latency-acceptable" are *different operating points*. Choose per your SLO, not per the peak.

### 3.4 VRAM is preallocated **per configured slot**, not per live request

This is a clean, beautiful result: VRAM at *server-ready idle* (before any traffic) is a **near-perfect straight line** in configured slots.

- Fit: **slope = 108.07 MiB/slot, intercept = 3475.6 MiB, R² = 1.00000** (the idle-capture fit is identical: 108.09, 3483.9, R²=1.00000).
- Measured `vram_ready`: 1→3584, 2→3692, 4→3908, 8→4340, 16→5204, 24→6072, 32→6932, 48→8664, 64→10388, 96→13854, 128→17308 MiB.
- `mem_used` is **flat across the entire run** (e.g. 17,318 MiB pinned for all 439 samples at C=128) — **nothing grows with active requests.**

The mechanism is confirmed: a 768-token f16 KV cache for Qwen3-4B is ~108 MiB, which *is* the slope. So each `--parallel` slot reserves its KV up front. The ~3476 MiB intercept is model + runtime + driver overhead for the whole board.

**Practical consequence:** you provision VRAM by **how many slots you configure**, not by load. Extrapolating the fit, ~195 slots would fill the 24,564 MiB board — **but throughput already peaks at C=64 and declines**, so *you will never reach the VRAM wall for a reason worth reaching it*. VRAM is not your binding constraint here.

### 3.5 The bottleneck moves OFF the GPU — the host-bound evidence

This is the most important and most counterintuitive finding. As `C` rises, **the GPU does *less*, and the CPU does *more*.**

| Signal | C=1 | → | C=128 | Direction |
|---|---|---|---|---|
| GPU util **median** % | 94 | ↓ | 41 | **falls** |
| GPU util **p95** % | 95 | — | 99 | pinned high |
| Power (median W) | 229 (≈230 cap) | ↓ | 168 (dips to 141 @ C=96) | **falls far below cap** |
| **llama-server process CPU %** | 2.55 | ↑ | **29.13** | **~11.4× rise** |
| System CPU % | 5.1 | ↑ | 31.2 | tracks server proc |
| MemCtrl-busy proxy % | 71 | ↓ | 26 | falls |

Read it carefully:

- At **C=1** the run is **GPU-bound**: util 94%, power at the 230 W cap, server CPU ~2.5%.
- As `C` climbs, **GPU util median falls to ~40% and power falls to 141–174 W** — the GPU is *not saturated overall*. But `p95 = 99%` at every C≥16.
- Meanwhile the **llama-server process CPU rises ~11×** and, at C=128, is ~93% of *all* host CPU.

**The GPU isn't lightly loaded — it's *bursty and starved*.** The raw 1 Hz telemetry alternates idle rows (0% util, 82 W) with busy rows (99% util, 232 W): the GPU **waits between CPU-scheduled batches**. The median-vs-p95 gap is the fingerprint of this. Per-batch CPU orchestration (scheduling, sampling, request bookkeeping) grows faster than added parallelism helps — so throughput saturates near ~800 tok/s and then **declines** as the CPU-side cost per batch overtakes the batching benefit.

> **This is why more slots eventually hurt.** The limiter isn't FLOPs and isn't VRAM — it's **host-side scheduling**. On this rig, past C=64 you're paying CPU orchestration overhead for parallelism the GPU can't cash in.

**One measurement-hygiene note you must respect:** the "MemCtrl proxy %" (nvidia-smi `utilization.memory`) is the **fraction of time the memory controller was busy** — it falls (71→26%) simply because the GPU idles more between bursts. It is **NOT achieved bandwidth in GB/s.** Never quote it as bandwidth. (More on this in Limitations.)

---

## 4. Precision & Hyperparameter Knobs — what actually moves the needle

These are **single-stream `llama-bench` micro-benchmarks** (raw kernel throughput: `pp512` = prefill, `tg128` = decode; no HTTP server, no sampling, no real prompts). Different quantity from the server numbers above — don't cross the streams.

### 4.1 Weight quantization: decode is bandwidth-bound, prefill is compute-bound

| Weight | GGUF size | Prefill pp512 (tok/s) | Decode tg128 (tok/s) |
|---|---|---|---|
| Q4_K_M | 2.32 GiB | 5901.5 ± 799 | **167.99** |
| Q5_K_M | 2.69 GiB | 5947.7 ± 383 | 155.15 |
| Q6_K | 3.07 GiB | 5534.2 ± 294 | 133.26 |
| Q8_0 | 3.98 GiB | 6497.4 ± 452 | 125.37 |
| BF16 | 7.49 GiB | 6222.6 ± 793 | 77.01 |

Two clean, opposite lessons:

- **Decode falls monotonically with weight bits** — Q4_K_M is **2.18× faster than BF16** (167.99 vs 77.01); Q8_0 is 1.63×. Decode reads every weight once per token, so fewer bytes = more tokens/sec. **Decode is memory-bandwidth-bound.**
- **Prefill is flat and *non-monotonic*** — range 5534–6497, and BF16 (biggest) prefills *faster* than Q6_K while Q8_0 is fastest of all. That's the *opposite* of a bandwidth story: prefill is a big batched GEMM, so it's **compute-bound**, and bit-width doesn't order it. (Caveat: only 3 samples/point, first is a cold outlier, stddev up to ±790 ≈ 13% — the ~17% prefill spread is *within/near noise*, which reinforces "flat," not any real ordering.)

### 4.2 Flash-attention: helps prefill more than decode

Prefill uplift: Q4 +15.6%, Q5 +18.9%, Q6 +16.7%, Q8 +21.0%, BF16 +16.7%. Decode uplift is smaller: Q4 +8.6%, Q6 +6.7%, BF16 +4.2%. **Turn flash-attention on** — it's a free win, larger where the work is compute-heavy (prefill).

### 4.3 KV-cache dtype: a memory knob, barely a speed knob

| KV dtype | Prefill pp512 | Decode tg128 | KV memory |
|---|---|---|---|
| f16 | 5385.7 | 128.62 | 16-bit (baseline) |
| bf16 | 5296.6 | 124.02 | 16-bit (same) |
| q8_0 | 5166.5 | 123.90 | ~8-bit (~½) |

`q8_0` KV **halves the KV footprint for only ~3.7% decode / ~4.1% prefill loss.** `bf16` KV is same size as f16 but slightly slower — **strictly worse here**, skip it. Important context: this is a **128-token single-stream** test, so the KV cache is tiny and the memory saving buys no speed — only a small penalty shows. **The memory win only pays off at long context / high batch** (e.g. the 100-slot server in Section 6, or the 4096-ctx conversation test). The ~2× saving is *inferred from dtype width*, not directly measured here.

> **Knob priority for this rig:** (1) pick weight quant for your decode-speed vs quality budget — Q4_K_M if speed rules, Q6_K/Q8_0 if quality rules; (2) flash-attention on, always; (3) use q8_0 KV only when KV memory is actually the constraint (long ctx / many slots).

---

## 5. Engine Choice — llama.cpp vs vLLM (and why this comparison is VOID)

**This is where you must be most careful, because the tempting headline is unearned.**

The intent (study "E5") was a head-to-head at C = 1 / 30 / 100 on a **fixed 100-slot llama.cpp server** vs vLLM, to test the common hypothesis *"vLLM wins big at high concurrency (PagedAttention + continuous batching)."*

**The llama.cpp side ran cleanly:**

| Engine | C | Throughput (tok/s) | TTFT p50 / p99 (s) | Latency p50 / p99 (s) | Reqs (ok/fail) | GPU util med/p95 | VRAM |
|---|---|---|---|---|---|---|---|
| llama.cpp Q6_K | 1 | 118.11 | 0.074 / 0.080 | 2.17 / 2.20 | 20/0 | 93% / 94% | 14,290 MiB |
| llama.cpp Q6_K | 30 | 370.45 | 0.904 / 1.70 | 22.1 / 32.2 | 600/0 | 76% / 95% | 14,300 MiB |
| llama.cpp Q6_K | 100 | 753.12 | 0.646 / 5.29 | 33.6 / 39.5 | 2000/0 | 41% / 99% | 14,300 MiB |
| **vLLM bf16** | 1/30/100 | **CRASHED — 0 requests served** | — | — | 0/all | — | — |

**The vLLM side never served a single request.** vLLM 0.25.1 loaded the merged Qwen3-4B (bf16, 7.64 GiB), sized its KV cache to **13.25 GiB = 96,448 tokens (47.09× max concurrency)** — and then **`EngineCore` died at `kernel_warmup`** with a Triton JIT parse bug (`AttributeError: 'NoneType' object has no attribute 'start'` in `triton/runtime/jit.py`) triggered by importing an *unrelated* `minimax_m3` sparse-attention Triton kernel. No `benchmark-vllm-*.json` exists; the server never listened.

> **Mandatory framing — do not violate this:**
> - **The engine comparison is VOID / one-sided.** There is **no vLLM measurement**. Do **not** report any llama.cpp-vs-vLLM ratio, latency delta, or "vLLM wins/loses" claim. The hypothesis is **untested, not refuted, not confirmed.**
> - **Even if vLLM had run, it would not have been iso-precision:** llama.cpp was Q6_K (6-bit, 3.07 GiB) while vLLM was configured bf16 (16-bit, 7.49 GiB). llama.cpp carries a ~2.4× smaller weight-read advantage for decode *plus* a small quality handicap — so a naïve tok/s comparison would have been apples-to-oranges anyway.
> - Use the **complete run `eng-…T002012Z`** (3 benchmark JSONs + both server logs). The earlier `eng-…T001714Z` is **incomplete** (telemetry for c001/c030 only, no c100, no benchmark JSONs) — don't source headline numbers from it.

What the llama.cpp side *does* independently confirm (consistent with Section 3): throughput scales **sublinearly** (6.4× for 100× clients), tail latency and TTFT degrade sharply, and **median GPU util/power *drop* as C rises** (93/94% → 41/99% util; 229 W → 156 W) even as p95 util pins at 99% — the same bursty, host-bound behavior. VRAM is constant ~14.3 GiB (fixed 100-slot preallocation), 0 failures across 2,620 requests.

**Deployment-shape caveat:** this E5 server is **one fixed config** (`n_slots=100`, 768 tok/slot, Q6_K, ~14.3 GiB) reused for C=1/30/100. It is **not re-provisioned per C** like Section 3. So C=1 here runs with 99 idle slots (over-provisioned), and it's not directly comparable to the per-C-tuned sweep.

---

## 6. Real-World Finite-User Load (Locust) — closed-loop, correctly labeled

Studies E2/E3 use **Locust with a finite pool of users and think-time between requests**. This is the closest thing here to "real traffic," but it is still **CLOSED-loop** — read the labeling caveat carefully before comparing anything.

**Both experiments are complete and clean: 0 failures across all three runs.** (Token windows are named `short-u20.json`, `short-u60.json`, `convo-u20.json`; steady-state comes from trailing windows — 120 s for E2, 180 s for E3 — so window req counts are smaller than totals.)

### E2 — ShortChatUser (server `-np 64`, 768 tok/slot; ~17-token prompts)

| User level | Steady tok/s | tok/min | comp tok/req | reqs (win) | TTFT p50/p95 (ms) | e2e p50/p95 (ms) | req/s | fails |
|---|---|---|---|---|---|---|---|---|
| u20 | 292.6 | 17,558 | 140.5 | 250 | 109 / 152 | 7,874 / 8,853 | 2.07 | 0 |
| u60 | 412.3 | 24,736 | 142.2 | 348 | 230 / 316 | 20,143 / 22,306 | 2.89 | 0 |

- More users → more aggregate tok/s (292.6 → 412.3), the batched-decode tradeoff.
- **TTFT stays low** (prefill is trivial for ~17-token prompts: server p50 = 17 tokens) but grows with concurrency — TTFT is queue/scheduling-dominated, not prefill.
- **e2e balloons 7.9 s → 20.1 s (2.6×)** from u20 to u60 — and critically, this is **not queueing** (60 users < 64 slots). It's **per-slot decode collapse under heavier batching**: effective per-request gen rate falls ~17.8 → ~7.1 tok/s (per-slot tg p50 8.24 t/s). Expected batched-inference physics, not a defect.

### E3 — ConversationUser (server `-np 20`, 4096 tok/slot; 8 growing turns)

| User level | Steady tok/s | comp tok/req | prompt tok/req | prefill:decode | TTFT p50/p95 (ms) | e2e p50/p95 (ms) | reqs (win) | fails |
|---|---|---|---|---|---|---|---|---|
| u20 | 205.9 | 133.3 | 578.8 | **4.34 : 1** | 247 / 465 | 10,231 / 11,791 | 278 | 0 |

**Per-turn growing-context effect (u20):**

| Turn | e2e median (ms) | server prefill tokens | server prefill time (ms) |
|---|---|---|---|
| 1 | 9,600 | ~13–21 | ~40–64 |
| 2 | 9,500 | ~150–200 | ~60–90 |
| 3–4 | 9,700 / 10,000 | ~350–700 | ~110–170 |
| 5–6 | 10,000 / 10,000 | ~700–1000 | ~170–250 |
| 7 | 11,000 | ~1050–1150 | ~260–290 |
| 8 | 9,800 | ~1200–1253 | ~290–330 |

**The lesson:** re-prefilling the *entire growing history every turn* (no cross-turn prefix-cache reuse) produces a **4.3:1 prefill:decode token ratio** — a real and large "repeated-prefill tax" *in token terms*. **But** prefill runs at ~4,200 tok/s while decode (the e2e bottleneck) runs ~13 tok/s effective, so the tax costs only **~40 ms → ~300 ms of TTFT (~6×, ~250 ms absolute)** and **barely moves the ~10 s decode-dominated e2e**. Throughput (205.9 tok/s) is *lower* than E2 precisely because a big share of GPU work goes to re-prefill instead of new tokens.

> **Why E3 is a great teaching case:** the scary-sounding metric (4.3:1 prefill ratio) turns out to be *cheap* because prefill is ~300× faster than decode per token. Always ask "expensive in what units, and is that unit the bottleneck?" Here: expensive in tokens, cheap in seconds.
>
> *(Honesty note: per-turn TTFT was not recorded client-side — only aggregate TTFT p50/p95 = 247/465 ms. The ~40→300 ms per-turn growth is derived from server prefill times as a proxy and excludes network/scheduling wait.)*

---

## 7. When to Use What — a decision guide

**These are engineering choices, read off the data above. Match the operating point to your goal, not to a peak number.**

| Your goal | Do this | Why (from the data) |
|---|---|---|
| **Lowest latency per user** | Keep concurrency **very low (C≤4)**; over-provision. | Fair-share is 118→73 tok/s at C≤4; p50 latency 2.2–3.5 s. Beyond that, per-user rate collapses. |
| **Max total throughput** | Run **C≈48–64**; expect ~789–802 tok/s. | Peak 801.8 tok/s at C=64; C=32–64 all ≥93% of peak. |
| **Balanced (good tok/s, tolerable latency)** | **C≈24–32**; ~700–750 tok/s, p50 ~9–11 s. | Efficient region ends around here; marginal gains shrink fast past C=32. |
| **Avoid wasting resources** | **Do not exceed C≈64.** | Past 64, aggregate throughput *declines* (−29, −23 tok/s) — you pay CPU orchestration for nothing. |
| **Fastest decode / most tok/s per byte** | Quantize weights **as low as quality allows** (Q4_K_M = 2.18× BF16 decode). | Decode is bandwidth-bound; smaller weights = faster. |
| **Best quality within budget** | **Q6_K or Q8_0**; Q6_K is the deployed default (3.3 GB, 6.56 bpw). | Prefill is compute-bound (flat), so higher-bit weights cost little on prefill; you pay only in decode. |
| **Long context / many slots, VRAM-tight** | Use **q8_0 KV cache** (~½ footprint, ~4% speed cost). | Only worth it when KV memory is the real constraint; useless at short-ctx single-stream. |
| **Any config** | **Flash-attention ON.** | +15–21% prefill, +4–9% decode, free. |
| **Sizing VRAM** | Budget **~3.5 GiB base + 108 MiB × slots** (768-ctx). | Linear per-slot preallocation (R²=1.00000); VRAM is never your binding limit before throughput peaks. |
| **Choosing llama.cpp vs vLLM** | **Undecided from this data — benchmark it yourself, iso-precision.** | vLLM crashed at warmup; there is no comparison. Don't assume either way. |
| **Reasoning about the ceiling** | Remember the limiter is **host-side CPU scheduling**, not FLOPs/VRAM. | GPU util median falls to ~40% while server-proc CPU rises ~11×; GPU is bursty/starved at high C. |

---

## 8. LIMITATIONS & CAVEATS (read this as carefully as the results)

**A number without its caveat is a liability. These are the ones that will bite you if you forget them.**

1. **One GPU, one build, one model.** Everything is on a **single RTX A5000 (24 GB)**, llama.cpp CUDA build `91d2fc38`, f16 KV, `--no-kv-unified`, 768-tok slots, Qwen3-4B-Instruct-2507 + legal-ops LoRA merged to Q6_K. **Results are hardware- and build-specific** and do **not** transfer to other GPUs, quantizations, batch policies, or prompt/gen lengths. The peak (C=64), the ~108 MiB/slot slope, the CPU crossover — all A5000-and-this-build specific.

2. **Closed-loop ≠ open-loop; never mix the numbers.** The Section 3 concurrency sweep is a **barrier-synchronous saturation probe** (fixed 256/256, `ignore_eos`, fully-backlogged server). Its p50/p95 latencies and TTFTs reflect a **saturated** server and would differ under real Poisson arrivals. The Locust runs (Section 6) are **closed-loop finite-user with think-time** — user count is an *upper bound* on in-flight concurrency, and offered load is throttled by the user pool, **not pushed to saturation** (u20 = 2.07 req/s, u60 = 2.89 req/s). **Do NOT compare Locust tok/s (E2: 292.6/412.3; E3: 205.9) head-to-head with the E1 saturation tok/s.** Even u60's 412 tok/s is think-time-throttled, not a ceiling. They measure different regimes.

3. **`utilization.memory` is a proxy, NOT bandwidth.** The "MemCtrl proxy %" is the *fraction of time the memory controller was busy*. It falls at high C because the GPU idles more between bursts. **It is not GB/s.** Any bandwidth claim must come from a real bandwidth measurement, which these studies do not have.

4. **GPU util median is misleading alone.** A low *median* (≈40% at high C) coexists with **p95 = 99%**. The GPU is **bursty/starved**, not lightly loaded. Reason from the **median-vs-p95 gap**, never the median by itself.

5. **CPU % is a psutil sample, not a core count.** `cpu_server_proc_pct` samples the llama-server process. The **load-bearing signal is the ~11× trend and its tracking of system CPU**, not an exact core count read off the percentage.

6. **VRAM figures are whole-board `memory.used`.** The ~3476 MiB intercept includes driver/other overhead; the 108.1 MiB/slot slope is specific to 768-tok f16 KV for this model. The "~195 slots fills the board" figure is a **linear extrapolation** — and throughput peaks (C=64) and declines *long before* VRAM binds, so it's academic.

7. **The engine comparison is VOID.** vLLM served **0 requests** (Triton JIT crash at `kernel_warmup`, v0.25.1). Every E5 number is **llama.cpp-only**. No llama.cpp-vs-vLLM ratio, delta, or winner can be stated. And even had it run, **the precision was mismatched** (llama.cpp Q6_K 6-bit vs vLLM bf16 16-bit) — it would not have been iso-precision.

8. **E4 is statistically thin.** Each `llama-bench` point is **3 samples**, the first a cold outlier (prefill stddev up to ±790 ≈ 13%). The ~17% prefill spread between quants is **within/near noise** — hence "prefill is flat," not any real ordering. `e4a_quant_fa.json` and `e4a_summary.md` are *separate* invocations differing a few percent (e.g. Q4 decode 167.99 vs 164.45) — treat as two runs.

9. **E4 and E5 measure different quantities.** E4 = synthetic single-stream raw kernel throughput (no HTTP, no sampling, no real prompts). E5/E1/Locust = closed-loop HTTP with real prompts and networking. E4's Q6_K decode 133 tok/s and E1's C=1 end-to-end 118 tok/s are **not the same number** — the latter includes TTFT, prompt eval, and networking.

10. **The SVD study measures an INTER-CHECKPOINT delta, not a FullFT delta.** Instruct-2507 declares no `base_model`; its production process is unproven from these artifacts. Any rank-r SVD of the delta is **"LoRA-REPRESENTABLE" (best rank-r weight approximation), never "a working extracted LoRA"** — no task behavior was trained or validated. And the high-rank result **does NOT refute "LoRA can match FullFT"**: a trained low-rank adapter stores *task* information (not the full weight delta) and matches FullFT with all-layer coverage and adequate capacity. A high-rank *checkpoint difference* is orthogonal to trained-LoRA quality.

11. **bf16 read-back inflates the SVD floors.** The control's rank-16 delta reads as eff-rank ~43 (not 16) and ~74% (not 100%) energy because it's read from bf16-merged weights: bf16 rounding noise (~2⁻⁸·|W|) is non-trivial against a delta only ~0.2% of the base norm. This is the intended calibration baseline, but it means absolute eff-rank/recon floors are **quant-inflated for both deltas**.

12. **Minor data hygiene.** C=24 had 1 failed request of 240 (all other sweep points 0 failures). C=32/32b agree within ~0.6% (run-to-run noise ~1%). Locust CSV percentiles are bucketed to 100 ms/1000 ms — the JSON token-window values are the precise ones and were quoted preferentially. The Q6_K pipeline's manifest `source_git_commit` is `null` (code provenance not pinned).

---

### The one-paragraph summary you can repeat back

On this single A5000, a 4B Q6_K model is bandwidth-bound on **decode** (lower-bit weights decode faster, up to 2.18× for Q4 vs BF16) and compute-bound on **prefill** (flat across quant). Serving many users, aggregate throughput climbs to a **broad plateau (C=32–64), peaks at ~802 tok/s at C=64, then declines** — because the bottleneck **leaves the GPU** (util median 94%→40%, power 229→~155 W) and **lands on host-side CPU scheduling** (server-process CPU up ~11×). VRAM is a non-issue (linear ~108 MiB/slot, board never fills before the peak). Per-user latency degrades ~20× the whole way up, so **"max throughput" and "good latency" are different operating points** — pick per SLO. The engine shootout **didn't happen** (vLLM crashed at warmup), and the "SVD-extract a LoRA" probe measures a **high-rank inter-checkpoint delta** that is at most *LoRA-representable* and refutes nothing about trained-LoRA quality. Never compare the closed-loop saturation numbers with the finite-user Locust numbers — they measure different worlds.
---

## Reproduction & provenance

All numbers in this guide come from committed result files under `results/`:

| Study | Result dir |
|---|---|
| Q6_K 1/30/100 benchmark (Work A) | `results/a5000-20260720T162318Z/` |
| E1 concurrency sweep (12 points) | `results/sweep-20260720T191546Z/` |
| E4 precision / hyperparameter | `results/precision-20260720T191131Z/` |
| E5 engine comparison (llama.cpp; vLLM crashed) | `results/engines/eng-20260721T002012Z/` |
| E2/E3 Locust finite-user + context-growth | `results/locust/loc-20260721T005851Z/` |
| SVD LoRA-extraction study | `results/extraction/` |

Pipeline: `./run.sh reproduce` (download → merge → build llama.cpp → convert →
quantize → verify → benchmark). Sweeps: `scripts/sweep_concurrency.sh` (E1),
`scripts/precision_sweep.sh` (E4), `scripts/engine_compare.sh` (E5),
`scripts/locust_run.sh` (E2/E3), `scripts/svd_extract.py` (SVD). Reference rig:
1× RTX A5000 24 GB, 2× Xeon Silver 4210R (40 threads), 125 GB RAM, PCIe Gen3 x16,
driver CUDA 13.3, llama.cpp `91d2fc38`. Two codex-council reviews and a 5-agent
analysis workflow informed this report; remaining overclaims were corrected per
their findings.

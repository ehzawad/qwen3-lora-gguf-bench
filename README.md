# qwen3-lora-gguf-bench

Reproducible pipeline that takes an official **Qwen3-4B** model plus a public
**LoRA adapter**, merges them, converts to **GGUF**, quantizes to **Q6_K**, serves
the result with **llama.cpp**, and **benchmarks concurrent inference** (1 / 30 /
100 concurrent short-chat requests) on a **single NVIDIA RTX A5000 (24 GB)** —
reporting output **tokens/minute**, TTFT, latency, and GPU/CPU/VRAM telemetry.

> **Results are hardware- and revision-specific.** They describe one RTX A5000
> at the recorded driver, build flags, server settings, clocks, thermal state,
> and host load. They are **not** intrinsic model or llama.cpp numbers and will
> not transfer unchanged to other GPUs, drivers, contexts, or schedules.

## 📖 The report

The full write-up — a mentor-style **field guide to LLM inference** covering all
six studies (concurrency capacity, precision, engines, LoRA/SVD, and finite-user
load), reconciled through three independent review passes — lives in:

- **[`REPORT.md`](REPORT.md)** — the complete report (renders on GitHub).
- **[`report/`](report/)** — a self-contained, theme-aware **HTML suite** with
  interactive charts: [field guide](report/index.html) ·
  [concurrency](report/concurrency.html) · [precision](report/precision.html) ·
  [engines &amp; chatbot](report/engines.html) · [LoRA &amp; SVD](report/lora.html) ·
  [Locust](report/locust.html). (Open locally or via GitHub Pages.)

## Scope and non-goals

- **In scope:** an end-to-end, pinned, verifiable path from `base + adapter` to a
  6-bit GGUF served under real concurrency, with honest throughput/bottleneck
  measurement on one 24 GB Ampere card.
- **Non-goals:** production serving advice, cross-GPU generalization, adapter
  quality claims, or a vLLM/TGI comparison. This is a single-card llama.cpp study.

## Inputs, revisions, licenses

| Component | Identity | Revision | License |
|---|---|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | Apache-2.0 (upstream) |
| LoRA adapter | `narcolepticchicken/qwen3-4b-legal-ops-contract-intake-lora` | `8e5c6a9d99c9079fb775f2df5957d57a619659f9` | Apache-2.0 (model-card metadata; no standalone LICENSE file) |
| Inference engine | `ggml-org/llama.cpp` | `91d2fc387529940230555abd297a8b5e99737d3f` | MIT |

Adapter weight SHA-256: `4848bae64fa74ba689c5300730aaed0f339d850589f08f59af4fe59808c15763`
(r=16, α=32, dropout=0.05, targets = q/k/v/o/gate/up/down_proj).

See [`NOTICE`](NOTICE) for full provenance. **No model weights, merged models,
GGUFs, or llama.cpp binaries are redistributed here** — everything large is
fetched or generated locally and git-ignored.

## Reference environment

| | |
|---|---|
| GPU | NVIDIA RTX A5000, 24564 MiB, Ampere GA102 (CC 8.6), ~768 GB/s, 230 W |
| Driver / CUDA UMD | 13.3 · nvcc 13.2 · `CMAKE_CUDA_ARCHITECTURES=86` |
| Host | 2× Intel Xeon Silver 4210R (40 threads), 125 GB RAM, PCIe Gen3 x16 |
| Python | 3.10.6 · torch 2.5.1+cu121 · transformers 4.56.2 · peft 0.19.1 |

**Prerequisites:** ~25 GB free disk (base 7.6 GB + merged 7.6 GB + GGUFs 11 GB),
a CUDA 8.6 GPU with ≥ ~12 GB free for the C=100 run, and a C++/CUDA toolchain.

## Quick start

```bash
./run.sh reproduce      # preflight -> corpus -> download -> merge -> build
                        # -> convert -> quantize -> verify -> benchmark
```

Or step by step:

```bash
./run.sh preflight      # GPU / disk / python sanity
./run.sh corpus         # generate prompts/short-chat.jsonl (seeded, ~256 tok)
./run.sh download       # pinned base + adapter -> models/ (verifies adapter SHA)
./run.sh merge          # PEFT merge (verified) -> models/merged  [A5000]
./run.sh build          # build pinned llama.cpp (CUDA, arch 86) -> vendor/
./run.sh convert        # merged HF -> bf16 GGUF  (--outtype bf16)
./run.sh quantize       # bf16 GGUF -> Q6_K
./run.sh verify         # GGUF metadata / tied head / vocab / template checks
./run.sh benchmark      # C=1,30,100 -> results/<run>/  + summary
./run.sh serve 30       # foreground server (manual poking), 30 slots
```

## Exact settings

**Merge** (`scripts/merge.py`): load base in bf16, apply adapter with LoRA
matrices forced to FP32 (so `B@A` is computed in FP32 then cast once to bf16),
`merge_and_unload(safe_merge=True)`, re-tie the head, save uniformly bf16. The
**official base tokenizer** is used and saved — *not* the adapter's, whose
`tokenizer_config.json` has `extra_special_tokens` as a list that crashes
transformers 4.56.2. Chat templates are byte-identical (verified SHA-256).

Verification asserts (all must hold): tied embeddings preserved, no `lm_head`
synthesized, no embedding resize, merged `q_proj` bit-exact `base + (α/r)·BA`,
merged model closer to adapter-**ON** than **OFF** and preserving ON's top-1.

**Convert:** `convert_hf_to_gguf.py --outtype bf16` (bf16 is a native quantizer
input; avoids an unnecessary bf16→f16 narrowing before Q6_K).

**Quantize:** `llama-quantize <bf16> <out> Q6_K` (no `--pure`,
`--allow-requantize`, or `--leave-output-tensor`). Result ≈ 3.3 GB, 6.56 bpw.

**Build:** `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=OFF
-DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=OFF`.

**Serve** (per concurrency `C`, 768 tokens/slot):
`llama-server -ngl all -fa on -np C --ctx-size C*768 --no-kv-unified -cb
-b 2048 -ub 512 -ctk f16 -ctv f16 --cache-ram 0 --no-cache-prompt
--no-context-shift --fit off -dev CUDA0 -sm none -rea off --jinja`, with
`CUDA_VISIBLE_DEVICES=0` and `GGML_CUDA_ENABLE_UNIFIED_MEMORY` unset (no silent
RAM spill). `--no-kv-unified` gives each slot a guaranteed 768-token context and
avoids the unified-KV throughput degradation at high occupancy.

**Benchmark** (`scripts/benchmark.py`): closed-loop, `C` workers released by a
barrier; each does 2 discarded warmup + 20 measured requests. Each request:
streaming `/v1/chat/completions`, `max_tokens=256`, `ignore_eos=true`,
`temperature=0`, `cache_prompt=false` → exactly 256 completion tokens.

## Metric definitions

- **Aggregate output throughput** = `60 × Σ(completion_tokens) / makespan`, where
  makespan is a single wall clock from the first measured request's start to the
  last measured request's finish. Per-request rates are **never** averaged or
  summed. Cross-checked against the server's `llamacpp:tokens_predicted_total`
  delta.
- **TTFT** = start → first SSE chunk with non-empty assistant content (role-only
  and usage-only chunks ignored). **Latency** = start → final usage/`[DONE]`.
  Queueing time is included in both.
- **Telemetry** sampled at 1 s via `nvidia-smi`. Note `utilization.memory` is a
  memory-**controller-busy proxy**, not measured GB/s — a "bandwidth-bound"
  claim would require a profiler (Nsight GA10x DRAM throughput), which this study
  does not run. See interpretation below.

## Results

Single RTX A5000, short-chat (~256-tok prompt, 256-tok generation, `ignore_eos`),
f16 KV, `--no-kv-unified`, `--parallel == concurrency`. Throughput =
`60·Σ(completion_tokens)/makespan` (single wall clock). Every run had **0 failed
requests**, and the server's `tokens_predicted_total` counter matched the client
token count **exactly** (5,120 / 153,600 / 512,000).

| Concurrency | Output tok/min | Output tok/s | Speedup vs C=1 | Per-stream tok/s | TTFT p50 (s) | TTFT p95 (s) | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak (MiB) | OK/Fail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7,346 | 122.4 | 1.0× | 122.4 | 0.07 | 0.08 | 2.09 | 2.12 | 94 | 229 | 3,590 | 20/0 |
| 30 | 43,230 | 720.5 | 5.9× | 24.0 | 0.90 | 0.96 | 10.51 | 11.34 | 51 | 194 | 6,744 | 600/0 |
| 100 | 45,927 | 765.4 | 6.3× | 7.7 | 0.65 | 1.14 | 33.26 | 34.54 | 41 | 157 | 14,550 | 2,000/0 |

Raw data: [`results/a5000-20260720T162318Z/`](results/a5000-20260720T162318Z)
(`benchmark-c0NN.json`, `telemetry-c0NN.csv`, `summary.md`).

### Interpretation & bottlenecks

A single stream decodes at **122.4 tok/s**, and batching lifts *aggregate*
throughput to **720.5 tok/s at C=30 (5.9×)** and **765.4 tok/s at C=100 (6.3×)**.
The knee is unambiguous: **C=30 captures essentially all available throughput**,
and pushing to C=100 — 3.3× more concurrent streams — buys only **+6.2%**
aggregate (720.5 → 765.4 tok/s). Per-stream decode collapses over the same range,
from 122.4 tok/s down to 24.0 (C=30) and 7.7 (C=100) tok/s, because the fixed
batched-decode budget is shared across more slots. Practically, **C=30 is the
balanced operating point**; C=100 only makes sense if its tail latency is
acceptable.

That tradeoff is the TTFT-vs-latency story. **TTFT p50 stays sub-second** at every
point (0.07 → 0.90 → 0.65 s; the dip at C=100 is a closed-loop/barrier-synced
scheduling artifact of spreading 2,000 requests over a ~669 s makespan — TTFT p95
still climbs monotonically to 1.14 s). **End-to-end latency grows roughly linearly
past the knee: p50 2.09 → 10.51 → 33.26 s.** Beyond C=30, extra streams do not add
throughput — they queue and time-share the batch, so per-request latency inflates
while first-token responsiveness stays cheap.

This is **not** a memory-capacity problem. Weights are ~3.3 GB (Q6_K) and the f16
KV cache grows with slots × 768-ctx, but VRAM peaks at just 3,590 / 6,744 /
**14,550 MiB** — even C=100 uses only ~59% of the A5000's 24 GB, with ~10 GB of
headroom. Zero failures across all 2,620 requests (20 / 600 / 2,000 OK) → no slot
exhaustion or OOM. The system is throughput-saturated and latency-bound well
before it is VRAM- or KV-bound. Temperatures stayed ≤ 78 °C (no thermal throttle);
host RAM had ~110 GB free. (This 1/30/100 run did not record CPU telemetry; the
**E1 sweep does** — and it is what substantiates the host-bound claim below.)

The most telling signal: **GPU utilization and power *fall* as concurrency rises**
— util median 94% → 51% → 41%, power 229 W → 194 W → 157 W — while throughput
plateaus. Only the single-stream case saturates the silicon (util 94%, power
pinned at the **230 W cap**, SM clock throttled to ~1,552 MHz while the memory
clock stays pinned at its 7,601 MHz max), which points to a **memory-bandwidth-
bound decode co-limited by the power cap**. This is an **inference, not a
measurement**: `nvidia-smi utilization.memory` is a memory-controller-busy proxy,
not GB/s, so the bandwidth claim rests on that proxy plus the clock signature, the
power wall, and roofline arithmetic (122.4 tok/s × 3.3 GB ≈ 404 GB/s ≈ 53% of the
768 GB/s spec peak) aligning — proving DRAM saturation would require a profiler
(Nsight Compute / DCGM DRAM-active counters). At C=30/100 the GPU is instead
visibly **underfed** (util and power *below* the C=1 levels). The fine-grained
**E1 sweep** (see the full report / `results/sweep-*`) confirms the cause with CPU
telemetry: as concurrency climbs to 128, the llama-server **process CPU rises from
~2.5% to ~29% of the 40-thread host** while GPU util falls to ~41% and power to
~168 W — evidence the ceiling is the **serving/host pipeline** (scheduling,
sampling, SSE handling across many streams), not GPU compute.
*(Correction: an earlier draft here claimed a batch-64 MMQ→cuBLAS kernel switch.
That is **wrong** for Q6_K on Ampere at this llama.cpp commit — the MMQ path is
used regardless of batch — so the claim is withdrawn.)*

**Answering the original questions directly:** a single A5000 running this Q6_K
4B via llama.cpp serves **~46,000 output tokens/minute** whether you offer it 30
or 100 concurrent short-chat requests — it *handles* 100 concurrent fine (no
failures, ~15 GB VRAM), but the extra 70 requests mostly wait: throughput barely
moves (+6%) while median latency triples to ~33 s. The **E1 sweep** locates the
actual throughput **peak at C≈64 (~800 tok/s), which then *declines* beyond** —
and ~C=30 already reaches ~93% of that peak. Under an interactive-latency
preference, **~30 concurrent is a reasonable operating point** (~43k tok/min at
~10 s latency); this is a tradeoff choice, not an objective optimum. VRAM headroom
means you could push context or model size further before memory becomes the
limit; the practical ceiling here is the serving pipeline feeding a
power-capped GPU, not the 24 GB.

## Artifact map (what stays local)

Committed: scripts, `run.sh`, `config/`, `prompts/`, `requirements/`, results
JSON/CSV/Markdown, `README`/`LICENSE`/`NOTICE`. Fetched/generated at runtime and
git-ignored: `models/` (base, adapter, merged, GGUFs), `vendor/llama.cpp`,
`logs/`. Defense-in-depth `.gitignore` blocks `*.safetensors`, `*.gguf`, and
compiled objects even if placed elsewhere.

## Verification checks

`./run.sh verify` reads both GGUFs (without loading tensors) and asserts:
`general.architecture == qwen3`, 151936-token vocab with correct special tokens,
tied output head (`output.weight` absent), chat-template SHA-256 match, and
file types `MOSTLY_BF16` / `MOSTLY_Q6_K`. The merge step prints logit
diagnostics proving the adapter is genuinely baked in.

## Licensing & credits

Repository code/docs: **Apache-2.0** ([`LICENSE`](LICENSE)). This does **not**
relicense Qwen, the adapter, or llama.cpp, and no third-party weights or binaries
are redistributed. Base model © Alibaba Cloud (Apache-2.0); LoRA adapter by
`narcolepticchicken` (Apache-2.0 per model card); llama.cpp under MIT. The
adapter targets legal contract-intake ops and **is not legal advice**. Adapter
self-reported eval claims are not reproduced or endorsed here.

## Troubleshooting

- **Port in use:** the default server port is `8199`; set `PORT=<free>` if taken.
- **OOM at C=100:** ensure `GGML_CUDA_ENABLE_UNIFIED_MEMORY` is unset and no other
  process holds VRAM; the run needs ~12–15 GB free.
- **`/props` 404 / wrong responses:** something else is on your port — you are
  talking to another server. Pick a free port.
- **Converter import errors:** run it with `PYTHONPATH=vendor/llama.cpp/gguf-py`
  (handled by `scripts/convert.sh`).

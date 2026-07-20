# Concurrency benchmark summary

Model: Qwen3-4B-Instruct-2507 + legal-ops LoRA, merged, **Q6_K** GGUF, llama.cpp CUDA on a single RTX A5000. Short-chat: ~256-tok prompt, 256-tok generation (`ignore_eos`), f16 KV, `--no-kv-unified`, `--parallel == concurrency`.

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7346 | 122.4 | 0.071 | 0.075 | 2.09 | 2.12 | 94 | 229 | 3590 | 20/0 |
| 30 | 43230 | 720.5 | 0.896 | 0.959 | 10.51 | 11.34 | 51 | 194 | 6744 | 600/0 |
| 100 | 45927 | 765.4 | 0.647 | 1.136 | 33.26 | 34.54 | 41 | 157 | 14550 | 2000/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 122.4 | 1.0x | 122.4 |
| 30 | 720.5 | 5.9x | 24.0 |
| 100 | 765.4 | 6.3x | 7.7 |

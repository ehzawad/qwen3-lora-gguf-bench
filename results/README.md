# results/

Each `a5000-<UTC-timestamp>/` directory is one benchmark run and contains:

| File | What |
|---|---|
| `manifest.json` | Sanitized environment/hardware capture (GPU, driver, versions, git SHAs) |
| `experiment.json` | Frozen experiment config (copy of `config/experiment.json`) |
| `benchmark-c0NN.json` | Per-concurrency raw result: throughput, TTFT/latency percentiles, telemetry summary, token cross-check, server startup line |
| `telemetry-c0NN.csv` | 1 Hz `nvidia-smi` samples during the measured phase (raw) |
| `server-c0NN.log` | llama-server stdout/stderr for that concurrency |
| `summary.md` / `summary.json` | Aggregated table + scaling across all concurrencies |

`llamacpp_commit.txt` records the pinned llama.cpp build commit.

Throughput is `60 × Σ(completion_tokens) / makespan` on a single wall clock;
see the repo README for full metric definitions and the mandatory
hardware-specificity caveat.

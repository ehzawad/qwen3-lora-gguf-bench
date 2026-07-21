#!/usr/bin/env python3
"""Aggregate benchmark-*.json in a run dir into summary.md + summary.json.

Usage: report.py <run_dir>
"""
import glob
import json
import os
import sys

run_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(run_dir, "benchmark-*.json")))
rows = []
for fp in files:
    with open(fp) as f:
        rows.append(json.load(f))
rows.sort(key=lambda r: r.get("concurrency", 0))


def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p, default)
    return d


def fnum(x, nd=0):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


lines = []
lines.append("# Concurrency benchmark summary\n")
lines.append("Model: Qwen3-4B-Instruct-2507 + legal-ops LoRA, merged, **Q6_K** GGUF, "
             "llama.cpp CUDA on a single RTX A5000. Short-chat: ~256-tok prompt, "
             "256-tok generation (`ignore_eos`), f16 KV, `--no-kv-unified`, "
             "`--parallel == concurrency`.\n")
lines.append("Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).\n")
hdr = ("| Concurrency | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) "
       "| Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | "
       "VRAM peak MiB | OK/Fail |")
sep = "|" + "---|" * 11
lines.append(hdr)
lines.append(sep)

summary = []
for r in rows:
    c = r.get("concurrency")
    tpm = r.get("output_tokens_per_min")
    tps = r.get("output_tokens_per_s")
    row = (
        f"| {c} | {fnum(tpm,0)} | {fnum(tps,1)} "
        f"| {fnum(g(r,'ttft_s','p50'),3)} | {fnum(g(r,'ttft_s','p95'),3)} "
        f"| {fnum(g(r,'latency_s','p50'),2)} | {fnum(g(r,'latency_s','p95'),2)} "
        f"| {fnum(g(r,'telemetry','gpu_util_pct','median'),0)} "
        f"| {fnum(g(r,'telemetry','power_w','median'),0)} "
        f"| {fnum(g(r,'telemetry','mem_used_mib','peak'),0)} "
        f"| {r.get('requests_ok')}/{r.get('requests_failed')} |"
    )
    lines.append(row)
    summary.append({
        "concurrency": c,
        "output_tokens_per_min": tpm,
        "output_tokens_per_s": tps,
        "ttft_p50_s": g(r, "ttft_s", "p50"),
        "ttft_p95_s": g(r, "ttft_s", "p95"),
        "latency_p50_s": g(r, "latency_s", "p50"),
        "latency_p95_s": g(r, "latency_s", "p95"),
        "gpu_util_median_pct": g(r, "telemetry", "gpu_util_pct", "median"),
        "mem_ctrl_util_median_pct_proxy": g(r, "telemetry", "mem_controller_util_pct_proxy", "median"),
        "power_median_w": g(r, "telemetry", "power_w", "median"),
        "sm_clock_median_mhz": g(r, "telemetry", "sm_clock_mhz", "median"),
        "vram_peak_mib": g(r, "telemetry", "mem_used_mib", "peak"),
        "requests_ok": r.get("requests_ok"),
        "requests_failed": r.get("requests_failed"),
        "server_predicted_tokens_delta": r.get("server_predicted_tokens_delta"),
    })

# Scaling note relative to concurrency 1
base = next((s for s in summary if s["concurrency"] == 1), None)
if base and base["output_tokens_per_s"]:
    lines.append("\n## Throughput scaling vs single stream\n")
    lines.append("| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |")
    lines.append("|---|---|---|---|")
    for s in summary:
        c = s["concurrency"]
        tps = s["output_tokens_per_s"] or 0
        speedup = tps / base["output_tokens_per_s"] if base["output_tokens_per_s"] else 0
        per_stream = tps / c if c else 0
        lines.append(f"| {c} | {tps:.1f} | {speedup:.1f}x | {per_stream:.1f} |")

with open(os.path.join(run_dir, "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n".join(lines))
print(f"\n[report] wrote {run_dir}/summary.md and summary.json")

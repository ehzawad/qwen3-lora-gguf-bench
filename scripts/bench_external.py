#!/usr/bin/env python3
"""Closed-loop benchmark against an ALREADY-RUNNING OpenAI-compatible server
(vLLM or llama-server). Reuses benchmark.py's validated request/telemetry logic
so the engine comparison uses the identical harness, payload, and metric.

The payload in benchmark.do_request pins model="qwen3-4b-legal-q6k",
max_tokens=256, ignore_eos=true, temperature=0 -> serve the engine under that
served-model name so 256 fixed output tokens make it apples-to-apples with E1.

Usage: bench_external.py --url http://127.0.0.1:8199 --engine vllm \
         --concurrency 30 --outdir results/engines/<run> --tag vllm-c030 \
         [--measured 20 --warmup 2]
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import threading  # noqa: E402
from urllib.parse import urljoin  # noqa: E402

import requests  # noqa: E402
from benchmark import Telemetry, get_metrics, gpu_mem_used  # noqa: E402


def do_request(base, prompt, max_tokens=256):
    """OpenAI-clean streaming request (no llama.cpp-only fields, so vLLM accepts
    it). ignore_eos is honored by both llama.cpp and vLLM -> exactly 256 tokens."""
    payload = {
        "model": "qwen3-4b-legal-q6k", "messages": [{"role": "user", "content": prompt}],
        "stream": True, "stream_options": {"include_usage": True},
        "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0.0, "seed": 12345,
    }
    url = urljoin(base, "/v1/chat/completions")
    t_start = time.perf_counter()
    ttft = None
    completion_tokens = prompt_tokens = finish_reason = status = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as r:
            status = r.status_code
            if r.status_code != 200:
                return {"ok": False, "status": status, "error": r.text[:200],
                        "t_start": t_start, "t_end": time.perf_counter()}
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                ch = obj.get("choices") or []
                if ch:
                    if (ch[0].get("delta") or {}).get("content") and ttft is None:
                        ttft = time.perf_counter() - t_start
                    if ch[0].get("finish_reason"):
                        finish_reason = ch[0]["finish_reason"]
                if obj.get("usage"):
                    completion_tokens = obj["usage"].get("completion_tokens")
                    prompt_tokens = obj["usage"].get("prompt_tokens")
    except requests.RequestException as e:
        return {"ok": False, "status": status, "error": str(e)[:200],
                "t_start": t_start, "t_end": time.perf_counter()}
    t_end = time.perf_counter()
    ok = status == 200 and completion_tokens is not None and finish_reason == "length"
    return {"ok": ok, "status": status, "t_start": t_start, "t_end": t_end,
            "ttft": ttft, "latency": t_end - t_start,
            "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens,
            "finish_reason": finish_reason}


def run_phase(base, prompts, concurrency, per_worker, measured, sink):
    counter = {"i": 0}
    clock = threading.Lock()
    barrier = threading.Barrier(concurrency)

    def nxt():
        with clock:
            i = counter["i"]; counter["i"] += 1
        return prompts[i % len(prompts)]

    def worker():
        barrier.wait()
        for _ in range(per_worker):
            res = do_request(base, nxt())
            if measured:
                sink.append(res)

    ts = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def wait_ready(base, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for ep in ("/health", "/v1/models"):
            try:
                if requests.get(base.rstrip("/") + ep, timeout=3).status_code < 400:
                    return True
            except requests.RequestException:
                pass
        time.sleep(1.0)
    raise RuntimeError("server never became ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine", default="unknown")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="prompts/short-chat.jsonl")
    ap.add_argument("--measured", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    base = args.url.rstrip("/") + "/"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(l)["prompt"] for l in open(args.prompts) if l.strip()]

    wait_ready(base)
    telem = Telemetry(str(outdir / f"telemetry-{args.tag}.csv"))
    vram_ready = gpu_mem_used()

    run_phase(base, prompts, args.concurrency, args.warmup, False, [])  # warmup
    time.sleep(2)
    vram_idle = gpu_mem_used()
    m_before = get_metrics(base)
    telem.start(); time.sleep(1)

    sink = []
    run_phase(base, prompts, args.concurrency, args.measured, True, sink)

    time.sleep(1); telem.stop()
    m_after = get_metrics(base)

    ok = [r for r in sink if r.get("ok")]
    bad = [r for r in sink if not r.get("ok")]
    comp = sum(r["completion_tokens"] for r in ok)
    makespan = (max(r["t_end"] for r in ok) - min(r["t_start"] for r in ok)) if ok else 0.0
    tps = comp / makespan if makespan else 0.0

    def pct(vals, p):
        s = sorted(vals)
        return s[min(len(s) - 1, int(p * len(s)))] if s else None

    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    lats = [r["latency"] for r in ok]
    # vLLM and llama.cpp both export prometheus generation counters (names differ)
    srv = None
    for k in ("vllm:generation_tokens_total", "llamacpp:tokens_predicted_total",
              "llamacpp:n_tokens_predicted_total"):
        if k in m_after and k in m_before:
            srv = m_after[k] - m_before[k]
            break

    result = {
        "engine": args.engine, "tag": args.tag, "concurrency": args.concurrency,
        "requests_ok": len(ok), "requests_failed": len(bad),
        "completion_tokens_total": comp, "makespan_s": makespan,
        "output_tokens_per_s": tps, "output_tokens_per_min": tps * 60,
        "server_generated_tokens_delta": srv,
        "prompt_tokens_example": ok[0]["prompt_tokens"] if ok else None,
        "ttft_s": {"p50": pct(ttfts, .5), "p90": pct(ttfts, .9),
                   "p95": pct(ttfts, .95), "p99": pct(ttfts, .99),
                   "max": max(ttfts) if ttfts else None},
        "latency_s": {"p50": pct(lats, .5), "p90": pct(lats, .9),
                      "p95": pct(lats, .95), "p99": pct(lats, .99),
                      "max": max(lats) if lats else None},
        "vram_ready_mib": vram_ready, "vram_idle_mib": vram_idle,
        "telemetry": telem.summarize(), "failures_sample": bad[:5],
    }
    with open(outdir / f"benchmark-{args.tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ext] {args.engine} C={args.concurrency}: {tps*60:.0f} tok/min "
          f"({tps:.1f} tok/s) ok={len(ok)} fail={len(bad)} "
          f"ttft_p50={result['ttft_s']['p50']}")
    return 0 if len(bad) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

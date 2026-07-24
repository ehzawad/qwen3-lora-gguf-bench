#!/usr/bin/env python3
"""Closed-loop benchmark against an ALREADY-RUNNING OpenAI-compatible server.

This harness is used for engine comparisons. It reuses benchmark.py's telemetry
and metric parsing, but sends only fields accepted by both llama.cpp and vLLM.
A run is successful only when every planned request produced a valid record and,
when available, the server's generation counter agrees with the client count.

Usage: bench_external.py --url http://127.0.0.1:8199 --engine vllm \
         --concurrency 30 --outdir results/engines/<run> --tag vllm-c030 \
         [--measured 20 --warmup 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import Telemetry, get_metrics, gpu_mem_used  # noqa: E402


BENCH_VERSION = "1.1"


def do_request(base: str, prompt: str, max_tokens: int = 256) -> dict[str, Any]:
    """Send one streaming request and return timings plus usage."""
    payload = {
        "model": "qwen3-4b-legal-q6k",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 12345,
    }
    url = urljoin(base, "/v1/chat/completions")
    t_start = time.perf_counter()
    ttft = None
    completion_tokens = prompt_tokens = finish_reason = status = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as response:
            status = response.status_code
            if response.status_code != 200:
                return {
                    "ok": False,
                    "status": status,
                    "error": response.text[:200],
                    "t_start": t_start,
                    "t_end": time.perf_counter(),
                }
            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    if (choices[0].get("delta") or {}).get("content") and ttft is None:
                        ttft = time.perf_counter() - t_start
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                if obj.get("usage"):
                    completion_tokens = obj["usage"].get("completion_tokens")
                    prompt_tokens = obj["usage"].get("prompt_tokens")
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status": status,
            "error": str(exc)[:200],
            "t_start": t_start,
            "t_end": time.perf_counter(),
        }

    t_end = time.perf_counter()
    ok = status == 200 and completion_tokens is not None and finish_reason == "length"
    return {
        "ok": ok,
        "status": status,
        "t_start": t_start,
        "t_end": t_end,
        "ttft": ttft,
        "latency": t_end - t_start,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
    }


def failed_request(exc: BaseException) -> dict[str, Any]:
    now = time.perf_counter()
    return {
        "ok": False,
        "status": None,
        "error": f"{type(exc).__name__}: {exc}"[:200],
        "t_start": now,
        "t_end": now,
        "ttft": None,
        "latency": 0.0,
        "completion_tokens": None,
        "prompt_tokens": None,
        "finish_reason": None,
    }


def run_phase(
    base: str,
    prompts: list[str],
    concurrency: int,
    per_worker: int,
) -> list[dict[str, Any]]:
    """Run a barrier-synchronized closed-loop phase.

    Worker exceptions are converted into failed request records instead of being
    lost on background threads. The returned list therefore contains exactly
    ``concurrency * per_worker`` records unless thread creation itself fails.
    """
    counter = {"i": 0}
    counter_lock = threading.Lock()
    sink: list[dict[str, Any]] = []
    barrier = threading.Barrier(concurrency)

    def next_prompt() -> str:
        with counter_lock:
            i = counter["i"]
            counter["i"] += 1
        return prompts[i % len(prompts)]

    def worker() -> None:
        try:
            barrier.wait()
        except threading.BrokenBarrierError as exc:
            sink.extend(failed_request(exc) for _ in range(per_worker))
            return
        for _ in range(per_worker):
            try:
                result = do_request(base, next_prompt())
            except Exception as exc:  # keep thread failures visible in the artifact
                result = failed_request(exc)
            sink.append(result)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sink


def wait_ready(base: str, timeout: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        for endpoint in ("/health", "/v1/models"):
            try:
                if requests.get(base.rstrip("/") + endpoint, timeout=3).status_code < 400:
                    return
            except requests.RequestException:
                pass
        time.sleep(1.0)
    raise RuntimeError("server never became ready")


def percentile(values: list[float], p: float) -> float | None:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))] if ordered else None


def counter_matches(client_tokens: int, server_tokens: float | None) -> bool | None:
    if server_tokens is None:
        return None
    return abs(float(client_tokens) - float(server_tokens)) <= 0.5


def successful_run(
    requests_ok: int,
    requests_failed: int,
    expected_requests: int,
    server_counter_matches: bool | None,
) -> bool:
    return (
        requests_ok == expected_requests
        and requests_failed == 0
        and server_counter_matches is not False
    )


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    prompt = item["prompt"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid prompt record: {exc}") from exc
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(f"{path}:{line_number}: prompt must be a non-empty string")
                prompts.append(prompt)
    except OSError as exc:
        raise ValueError(f"cannot read prompts from {path}: {exc}") from exc
    if not prompts:
        raise ValueError(f"prompt corpus is empty: {path}")
    return prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--engine", default="unknown")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prompts", default="prompts/short-chat.jsonl")
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)

    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.measured <= 0:
        parser.error("--measured must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    try:
        prompts = load_prompts(Path(args.prompts))
    except ValueError as exc:
        parser.error(str(exc))

    base = args.url.rstrip("/") + "/"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / f"benchmark-{args.tag}.json"
    expected_measured = args.concurrency * args.measured
    result: dict[str, Any] = {
        "bench_version": BENCH_VERSION,
        "engine": args.engine,
        "tag": args.tag,
        "concurrency": args.concurrency,
        "warmup_per_worker": args.warmup,
        "measured_per_worker": args.measured,
        "requests_expected": expected_measured,
        "ok": False,
    }

    telemetry = Telemetry(str(outdir / f"telemetry-{args.tag}.csv"))
    telemetry_started = False
    try:
        wait_ready(base)
        vram_ready = gpu_mem_used()

        warmup = run_phase(base, prompts, args.concurrency, args.warmup)
        expected_warmup = args.concurrency * args.warmup
        warmup_bad = [record for record in warmup if not record.get("ok")]
        if len(warmup) != expected_warmup or warmup_bad:
            raise RuntimeError(
                "warmup failed: "
                f"records={len(warmup)}/{expected_warmup}, failures={len(warmup_bad)}"
            )

        time.sleep(2)
        vram_idle = gpu_mem_used()
        metrics_before = get_metrics(base)
        try:
            telemetry.start()
            telemetry_started = True
        except Exception:
            telemetry.stop()
            raise
        try:
            time.sleep(1)
            sink = run_phase(base, prompts, args.concurrency, args.measured)
            time.sleep(1)
        finally:
            if telemetry_started:
                telemetry.stop()
                telemetry_started = False
        metrics_after = get_metrics(base)

        ok_records = [record for record in sink if record.get("ok")]
        bad_records = [record for record in sink if not record.get("ok")]
        completion_tokens = sum(record["completion_tokens"] for record in ok_records)
        makespan = (
            max(record["t_end"] for record in ok_records)
            - min(record["t_start"] for record in ok_records)
            if ok_records
            else 0.0
        )
        output_tps = completion_tokens / makespan if makespan else 0.0
        ttfts = [record["ttft"] for record in ok_records if record["ttft"] is not None]
        latencies = [record["latency"] for record in ok_records]

        server_generated = None
        for key in (
            "vllm:generation_tokens_total",
            "llamacpp:tokens_predicted_total",
            "llamacpp:n_tokens_predicted_total",
        ):
            if key in metrics_after and key in metrics_before:
                server_generated = metrics_after[key] - metrics_before[key]
                break
        counters_match = counter_matches(completion_tokens, server_generated)
        run_ok = successful_run(
            len(ok_records), len(bad_records), expected_measured, counters_match
        )

        result.update({
            "ok": run_ok,
            "requests_ok": len(ok_records),
            "requests_failed": len(bad_records),
            "completion_tokens_total": completion_tokens,
            "makespan_s": makespan,
            "output_tokens_per_s": output_tps,
            "output_tokens_per_min": output_tps * 60,
            "server_generated_tokens_delta": server_generated,
            "server_counter_matches_client": counters_match,
            "prompt_tokens_example": ok_records[0]["prompt_tokens"] if ok_records else None,
            "ttft_s": {
                "p50": percentile(ttfts, 0.5),
                "p90": percentile(ttfts, 0.9),
                "p95": percentile(ttfts, 0.95),
                "p99": percentile(ttfts, 0.99),
                "max": max(ttfts) if ttfts else None,
            },
            "latency_s": {
                "p50": percentile(latencies, 0.5),
                "p90": percentile(latencies, 0.9),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
                "max": max(latencies) if latencies else None,
            },
            "vram_ready_mib": vram_ready,
            "vram_idle_mib": vram_idle,
            "telemetry": telemetry.summarize(),
            "failures_sample": bad_records[:5],
        })
    except Exception as exc:
        if telemetry_started:
            telemetry.stop()
        result.update({
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        write_result(result_path, result)
        print(f"[ext] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[ext] wrote failure artifact {result_path}", file=sys.stderr)
        return 2

    write_result(result_path, result)
    print(
        f"[ext] {args.engine} C={args.concurrency}: "
        f"{result['output_tokens_per_min']:.0f} tok/min "
        f"({result['output_tokens_per_s']:.1f} tok/s) "
        f"ok={result['requests_ok']} fail={result['requests_failed']} "
        f"ttft_p50={result['ttft_s']['p50']}"
    )
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

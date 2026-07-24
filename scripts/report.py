#!/usr/bin/env python3
"""Aggregate benchmark-*.json into summary.md and summary.json.

The report is always diagnostic: failed points remain visible, but they are
marked as failed and excluded from throughput-scaling calculations. The command
exits non-zero when any point failed, a request count is incomplete, a server
counter disagrees with the client, or an expected concurrency point is missing.

Usage: report.py <run_dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any


class ReportError(RuntimeError):
    """The run directory cannot be summarized safely."""


def g(d: Any, *path: str, default: Any = None) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p, default)
    return d


def fnum(x: Any, nd: int = 0) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "-"


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(run_dir, "benchmark-*.json")))
    if not files:
        raise ReportError(f"no benchmark-*.json files found in {run_dir}")

    rows: list[dict[str, Any]] = []
    for file_name in files:
        path = Path(file_name)
        try:
            with path.open(encoding="utf-8") as f:
                row = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportError(f"cannot read {path}: {exc}") from exc
        if not isinstance(row, dict):
            raise ReportError(f"{path} must contain one JSON object")
        row["_source_file"] = path.name
        rows.append(row)

    rows.sort(key=lambda r: (r.get("concurrency", 0), r.get("tag", "")))
    return rows


def expected_run_spec(run_dir: Path) -> tuple[set[int], set[str]]:
    path = run_dir / "experiment.json"
    if not path.exists():
        return set(), set()
    try:
        with path.open(encoding="utf-8") as f:
            experiment = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc

    benchmark = g(experiment, "benchmark", default={})
    if not isinstance(benchmark, dict):
        raise ReportError(f"{path}: benchmark must be an object")

    points = benchmark.get("concurrency_points", [])
    if not isinstance(points, list) or not all(
        isinstance(point, int) and not isinstance(point, bool) and point > 0
        for point in points
    ):
        raise ReportError(f"{path}: benchmark.concurrency_points must be positive integers")

    tags = benchmark.get("expected_tags", [])
    if not isinstance(tags, list) or not all(
        isinstance(tag, str) and tag for tag in tags
    ):
        raise ReportError(f"{path}: benchmark.expected_tags must be non-empty strings")
    return set(points), set(tags)


def expected_request_count(row: dict[str, Any]) -> int | None:
    concurrency = row.get("concurrency")
    measured = row.get("measured_per_worker")
    if (
        isinstance(concurrency, int)
        and not isinstance(concurrency, bool)
        and concurrency > 0
        and isinstance(measured, int)
        and not isinstance(measured, bool)
        and measured > 0
    ):
        return concurrency * measured
    return None


def validate_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if row.get("ok") is not True:
        reasons.append("benchmark did not report ok=true")

    expected = expected_request_count(row)
    requests_ok = row.get("requests_ok")
    requests_failed = row.get("requests_failed")
    if expected is None:
        reasons.append("missing/invalid concurrency or measured_per_worker")
    elif requests_ok != expected:
        reasons.append(f"requests_ok={requests_ok!r}, expected {expected}")
    if requests_failed != 0:
        reasons.append(f"requests_failed={requests_failed!r}, expected 0")

    client_tokens = row.get("completion_tokens_total")
    server_tokens = row.get("server_predicted_tokens_delta")
    if (
        isinstance(client_tokens, (int, float))
        and not isinstance(client_tokens, bool)
        and isinstance(server_tokens, (int, float))
        and not isinstance(server_tokens, bool)
        and abs(float(client_tokens) - float(server_tokens)) > 0.5
    ):
        reasons.append(
            "server/client token mismatch "
            f"({server_tokens!r} != {client_tokens!r})"
        )

    return not reasons, reasons


def build_report(
    run_dir: Path,
    rows: list[dict[str, Any]],
    expected_points: set[int],
    expected_tags: set[str],
) -> tuple[str, list[dict[str, Any]], bool]:
    validated = [(row, *validate_row(row)) for row in rows]
    observed_points = {
        row["concurrency"]
        for row, _ok, _reasons in validated
        if isinstance(row.get("concurrency"), int)
        and not isinstance(row.get("concurrency"), bool)
    }
    observed_tags = {
        row["tag"]
        for row, _ok, _reasons in validated
        if isinstance(row.get("tag"), str) and row.get("tag")
    }
    missing_points = sorted(expected_points - observed_points)
    missing_tags = sorted(expected_tags - observed_tags)
    run_ok = (
        all(ok for _row, ok, _reasons in validated)
        and not missing_points
        and not missing_tags
    )

    lines: list[str] = []
    lines.append("# Concurrency benchmark summary\n")
    lines.append(f"Run status: **{'PASS' if run_ok else 'FAILED / PARTIAL'}**.\n")
    if missing_points:
        lines.append(
            "Missing expected concurrency point(s): "
            + ", ".join(str(point) for point in missing_points)
            + ".\n"
        )
    if missing_tags:
        lines.append(
            "Missing expected benchmark tag(s): "
            + ", ".join(f"`{tag}`" for tag in missing_tags)
            + ".\n"
        )
    failed_files = [row["_source_file"] for row, ok, _reasons in validated if not ok]
    if failed_files:
        lines.append(
            "Failed benchmark artifact(s): " + ", ".join(f"`{name}`" for name in failed_files)
            + ". Their metrics are diagnostic only and are excluded from scaling.\n"
        )

    lines.append(
        "Model: Qwen3-4B-Instruct-2507 + legal-ops LoRA, merged, **Q6_K** GGUF, "
        "llama.cpp CUDA on a single RTX A5000. Short-chat: ~256-tok prompt, "
        "256-tok generation (`ignore_eos`), f16 KV, `--no-kv-unified`, "
        "`--parallel == concurrency`.\n"
    )
    lines.append("Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).\n")
    hdr = (
        "| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) "
        "| Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | "
        "VRAM peak MiB | OK/Fail |"
    )
    lines.append(hdr)
    lines.append("|" + "---|" * 12)

    summary: list[dict[str, Any]] = []
    for row, row_ok, reasons in validated:
        c = row.get("concurrency")
        tpm = row.get("output_tokens_per_min")
        tps = row.get("output_tokens_per_s")
        lines.append(
            f"| {c} | {'PASS' if row_ok else 'FAIL'} | {fnum(tpm, 0)} | {fnum(tps, 1)} "
            f"| {fnum(g(row, 'ttft_s', 'p50'), 3)} | {fnum(g(row, 'ttft_s', 'p95'), 3)} "
            f"| {fnum(g(row, 'latency_s', 'p50'), 2)} | {fnum(g(row, 'latency_s', 'p95'), 2)} "
            f"| {fnum(g(row, 'telemetry', 'gpu_util_pct', 'median'), 0)} "
            f"| {fnum(g(row, 'telemetry', 'power_w', 'median'), 0)} "
            f"| {fnum(g(row, 'telemetry', 'mem_used_mib', 'peak'), 0)} "
            f"| {row.get('requests_ok')}/{row.get('requests_failed')} |"
        )
        summary.append({
            "source_file": row["_source_file"],
            "ok": row_ok,
            "failure_reasons": reasons,
            "concurrency": c,
            "output_tokens_per_min": tpm,
            "output_tokens_per_s": tps,
            "ttft_p50_s": g(row, "ttft_s", "p50"),
            "ttft_p95_s": g(row, "ttft_s", "p95"),
            "latency_p50_s": g(row, "latency_s", "p50"),
            "latency_p95_s": g(row, "latency_s", "p95"),
            "gpu_util_median_pct": g(row, "telemetry", "gpu_util_pct", "median"),
            "mem_ctrl_util_median_pct_proxy": g(
                row, "telemetry", "mem_controller_util_pct_proxy", "median"
            ),
            "power_median_w": g(row, "telemetry", "power_w", "median"),
            "sm_clock_median_mhz": g(row, "telemetry", "sm_clock_mhz", "median"),
            "vram_peak_mib": g(row, "telemetry", "mem_used_mib", "peak"),
            "requests_ok": row.get("requests_ok"),
            "requests_failed": row.get("requests_failed"),
            "server_predicted_tokens_delta": row.get("server_predicted_tokens_delta"),
        })

    valid_summary = [item for item in summary if item["ok"]]
    base = next((item for item in valid_summary if item["concurrency"] == 1), None)
    if base and base["output_tokens_per_s"]:
        lines.append("\n## Throughput scaling vs single stream\n")
        lines.append("| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |")
        lines.append("|---|---|---|---|")
        for item in valid_summary:
            c = item["concurrency"]
            tps = item["output_tokens_per_s"] or 0
            speedup = tps / base["output_tokens_per_s"]
            per_stream = tps / c if c else 0
            lines.append(f"| {c} | {tps:.1f} | {speedup:.1f}x | {per_stream:.1f} |")

    return "\n".join(lines) + "\n", summary, run_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir

    try:
        if not run_dir.is_dir():
            raise ReportError(f"run directory does not exist: {run_dir}")
        rows = load_rows(run_dir)
        points, tags = expected_run_spec(run_dir)
        markdown, summary, run_ok = build_report(run_dir, rows, points, tags)
    except ReportError as exc:
        print(f"[report] ERROR: {exc}", file=sys.stderr)
        return 2

    summary_md = run_dir / "summary.md"
    summary_json = run_dir / "summary.json"
    try:
        summary_md.write_text(markdown, encoding="utf-8")
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[report] ERROR: cannot write report: {exc}", file=sys.stderr)
        return 2

    print(markdown, end="")
    print(f"\n[report] wrote {summary_md} and {summary_json}")
    return 0 if run_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

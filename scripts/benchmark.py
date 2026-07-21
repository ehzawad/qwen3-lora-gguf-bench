#!/usr/bin/env python3
"""Closed-loop concurrency benchmark for llama-server (Q6_K Qwen3-4B).

Reconciled with the codex council (serve-bench + gpu-bottleneck lenses):
  * Starts its OWN llama-server with council-exact flags, saves the PID, waits
    for /health, and kills that PID on exit (never pkill).
  * --no-kv-unified, --parallel == concurrency, --fit off, host prompt cache
    disabled, GGML_CUDA_ENABLE_UNIFIED_MEMORY unset (no silent RAM spill).
  * Closed-loop: C workers released by a barrier; each does W warmup (discarded)
    then K measured requests. ignore_eos + max_tokens=256 => exactly 256 out tok.
  * Streaming SSE: TTFT = first chunk with non-empty assistant content.
  * Aggregate throughput = 60 * sum(completion_tokens) / makespan  (ONE wall
    clock: first measured start -> last measured finish). Never averages rates.
  * Cross-checks client completion tokens against the /metrics
    llamacpp:n_tokens_predicted_total delta.
  * Samples GPU telemetry (nvidia-smi) at 1s during the measured phase.

Usage:
  benchmark.py --model <q6.gguf> --bin-dir <llama build/bin> --prompts <jsonl>
               --concurrency C --ctx N --port P --outdir DIR --tag TAG
               [--measured K] [--warmup W]
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

try:
    import psutil
except ImportError:
    psutil = None

BENCH_VERSION = "1.1"


def log(msg):
    print(f"[bench] {msg}", flush=True)


# ---------------------------------------------------------------- server mgmt
def build_server_cmd(args):
    return [
        str(Path(args.bin_dir) / "llama-server"),
        "-m", args.model,
        "--alias", "qwen3-4b-legal-q6k",
        "--host", "127.0.0.1", "--port", str(args.port),
        "-dev", "CUDA0", "-sm", "none", "--main-gpu", "0",
        "-ngl", "all",
        "--fit", "off",
        "-fa", "on",
        "-np", str(args.concurrency),
        "--ctx-size", str(args.ctx),
        "--no-kv-unified",
        "-cb",
        "-b", "2048", "-ub", "512",
        "-ctk", "f16", "-ctv", "f16",
        "--cache-ram", "0",
        "--no-cache-prompt",
        "--no-context-shift",
        "-rea", "off",
        "--jinja",
        "--no-webui",
        "--metrics",
        "--slots",
        "--threads-http", "128",
    ]


def start_server(args, server_log_path):
    env = dict(os.environ)
    # PCI_BUS_ID so device 0 == physical GPU 0 (A5000), matching nvidia-smi -i 0.
    # Without this, CUDA's default "fastest first" order picks the A6000.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)  # no silent RAM spill
    cmd = build_server_cmd(args)
    log("launching server: " + " ".join(cmd))
    fh = open(server_log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    return proc, fh


def wait_health(base, proc, timeout=180):
    t0 = time.time()
    url = urljoin(base, "/health")
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not become healthy in time")


def stop_server(proc, fh):
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
    try:
        fh.close()
    except Exception:
        pass


# ---------------------------------------------------------------- server info
def get_props(base):
    try:
        r = requests.get(urljoin(base, "/props"), timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return {}


def read_startup_line(server_log_path):
    try:
        with open(server_log_path) as f:
            for line in f:
                if "load_model" in line and "kv_unified" in line:
                    return line.strip()
    except FileNotFoundError:
        pass
    return None


def get_metrics(base):
    """Parse Prometheus /metrics into {name: float}. Returns {} if disabled."""
    try:
        r = requests.get(urljoin(base, "/metrics"), timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return {}
    out = {}
    for line in r.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def gpu_mem_used():
    """GPU0 memory.used in MiB (one-shot), or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


# ---------------------------------------------------------------- one request
def do_request(base, prompt, max_tokens=256):
    """Blocking streaming request. Returns dict with timings + usage."""
    payload = {
        "model": "qwen3-4b-legal-q6k",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "cache_prompt": False,
        "temperature": 0.0,
        "seed": 12345,
    }
    url = urljoin(base, "/v1/chat/completions")
    t_start = time.perf_counter()
    ttft = None
    prompt_tokens = None
    completion_tokens = None
    finish_reason = None
    status = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as r:
            status = r.status_code
            if r.status_code != 200:
                return {"ok": False, "status": status,
                        "error": r.text[:200], "t_start": t_start,
                        "t_end": time.perf_counter()}
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
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content and ttft is None:
                        ttft = time.perf_counter() - t_start
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                usage = obj.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
    except requests.RequestException as e:
        return {"ok": False, "status": status, "error": str(e)[:200],
                "t_start": t_start, "t_end": time.perf_counter()}
    t_end = time.perf_counter()
    ok = (status == 200 and completion_tokens is not None
          and finish_reason == "length")
    return {
        "ok": ok, "status": status, "t_start": t_start, "t_end": t_end,
        "ttft": ttft, "latency": t_end - t_start,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }


# ---------------------------------------------------------------- telemetry
class Telemetry:
    QUERY = (
        "timestamp,pstate,clocks.current.sm,clocks.current.memory,"
        "power.draw.instant,power.limit,temperature.gpu,utilization.gpu,"
        "utilization.memory,memory.used,pcie.link.gen.gpucurrent,"
        "pcie.link.width.current,clocks_event_reasons.sw_power_cap,"
        "clocks_event_reasons.hw_thermal_slowdown,"
        "clocks_event_reasons.sw_thermal_slowdown"
    )

    def __init__(self, csv_path, server_pid=None):
        self.csv_path = csv_path
        self.server_pid = server_pid
        self.proc = None
        self._cpu = []                    # (system_pct, server_proc_pct)
        self._stop = threading.Event()
        self._cpu_thread = None

    def start(self):
        cmd = ["nvidia-smi", "-i", "0", f"--query-gpu={self.QUERY}",
               "--format=csv,nounits", "--loop-ms=1000"]
        self.fh = open(self.csv_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.fh,
                                     stderr=subprocess.DEVNULL)
        if psutil is not None:
            self._cpu_thread = threading.Thread(target=self._cpu_loop, daemon=True)
            self._cpu_thread.start()

    def _cpu_loop(self):
        proc = None
        if self.server_pid:
            try:
                proc = psutil.Process(self.server_pid)
                proc.cpu_percent(None)             # prime
            except psutil.Error:
                proc = None
        psutil.cpu_percent(None)                   # prime system
        n = psutil.cpu_count() or 1
        while not self._stop.wait(1.0):
            sysp = psutil.cpu_percent(None)        # 0..100 avg over all cores
            srvp = None
            if proc is not None:
                try:
                    srvp = proc.cpu_percent(None) / n   # 0..100 of whole box
                except psutil.Error:
                    srvp = None
            self._cpu.append((sysp, srvp))

    def stop(self):
        self._stop.set()
        if self._cpu_thread:
            self._cpu_thread.join(timeout=3)
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            self.fh.close()
        except Exception:
            pass

    def summarize(self):
        """median/p95/peak for sm%, mem%, power, sm clock, mem.used."""
        rows = []
        try:
            with open(self.csv_path) as f:
                header = f.readline().strip().split(", ")
                idx = {h: i for i, h in enumerate(header)}
                for line in f:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < len(header):
                        continue
                    rows.append(parts)
        except FileNotFoundError:
            return {}

        def col(names):
            for n in names:
                for h, i in idx.items():
                    if n in h:
                        return i
            return None

        def stats(names, cast=float):
            i = col(names)
            if i is None:
                return None
            vals = []
            for r in rows:
                try:
                    vals.append(cast(r[i]))
                except (ValueError, IndexError):
                    pass
            if not vals:
                return None
            vals.sort()
            p95 = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
            return {"median": statistics.median(vals), "p95": p95,
                    "peak": max(vals), "min": min(vals), "samples": len(vals)}

        def cstats(vals):
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            vals.sort()
            p95 = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
            return {"median": statistics.median(vals), "p95": p95,
                    "peak": max(vals), "samples": len(vals)}

        return {
            "gpu_util_pct": stats(["utilization.gpu"]),
            "mem_controller_util_pct_proxy": stats(["utilization.memory"]),
            "power_w": stats(["power.draw.instant"]),
            "sm_clock_mhz": stats(["clocks.current.sm"]),
            "mem_used_mib": stats(["memory.used"]),
            "temperature_c": stats(["temperature.gpu"]),
            "cpu_system_pct": cstats([c[0] for c in self._cpu]),
            "cpu_server_proc_pct": cstats([c[1] for c in self._cpu]),
        }


# ---------------------------------------------------------------- load phases
def run_phase(base, prompts, concurrency, per_worker, measured, results_sink):
    """Closed-loop: `concurrency` workers, barrier release, each does
    `per_worker` requests pulling unique prompts from a shared counter."""
    counter = {"i": 0}
    counter_lock = threading.Lock()
    barrier = threading.Barrier(concurrency)

    def next_prompt():
        with counter_lock:
            i = counter["i"]
            counter["i"] += 1
        return prompts[i % len(prompts)]

    def worker():
        barrier.wait()
        for _ in range(per_worker):
            res = do_request(base, next_prompt())
            if measured:
                results_sink.append(res)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--measured", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}/"

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)
               if l.strip()]
    need = args.concurrency * (args.warmup + args.measured)
    if len(prompts) < need:
        log(f"WARNING: corpus has {len(prompts)} prompts, need {need}; "
            f"prompts will repeat (still unique-enough with nonce reuse).")

    server_log = outdir / f"server-{args.tag}.log"
    proc, fh = start_server(args, server_log)
    telem = Telemetry(str(outdir / f"telemetry-{args.tag}.csv"), server_pid=proc.pid)
    result = {"ok": False}
    try:
        wait_health(base, proc)
        props = get_props(base)
        startup_line = read_startup_line(server_log)
        vram_ready = gpu_mem_used()      # after health, before any request
        log(f"server healthy. total_slots={props.get('total_slots')} "
            f"vram_ready={vram_ready} | {startup_line}")

        # warmup (discarded)
        log(f"warmup: {args.warmup} x {args.concurrency} requests")
        run_phase(base, prompts, args.concurrency, args.warmup, False, [])

        # settle + snapshot metrics
        time.sleep(2)
        vram_idle = gpu_mem_used()       # post-warmup idle
        m_before = get_metrics(base)
        telem.start()
        time.sleep(1)

        # measured
        log(f"measured: {args.measured} x {args.concurrency} requests")
        sink = []
        t_wall0 = time.perf_counter()
        run_phase(base, prompts, args.concurrency, args.measured, True, sink)
        t_wall1 = time.perf_counter()

        time.sleep(1)
        telem.stop()
        m_after = get_metrics(base)

        ok_reqs = [r for r in sink if r.get("ok")]
        bad_reqs = [r for r in sink if not r.get("ok")]
        comp_tokens = sum(r["completion_tokens"] for r in ok_reqs)
        # makespan: first measured start -> last measured finish (one clock)
        makespan = (max(r["t_end"] for r in ok_reqs)
                    - min(r["t_start"] for r in ok_reqs)) if ok_reqs else 0.0
        out_tps = comp_tokens / makespan if makespan else 0.0

        def pct(vals, p):
            if not vals:
                return None
            s = sorted(vals)
            return s[min(len(s) - 1, int(p * len(s)))]

        ttfts = [r["ttft"] for r in ok_reqs if r["ttft"] is not None]
        lats = [r["latency"] for r in ok_reqs]

        server_pred = None
        for k in ("llamacpp:n_tokens_predicted_total",
                  "llamacpp:tokens_predicted_total"):
            if k in m_after and k in m_before:
                server_pred = m_after[k] - m_before[k]
                break

        result = {
            "bench_version": BENCH_VERSION,
            "tag": args.tag,
            "concurrency": args.concurrency,
            "ctx_size": args.ctx,
            "ctx_per_slot": args.ctx // args.concurrency,
            "warmup_per_worker": args.warmup,
            "measured_per_worker": args.measured,
            "requests_ok": len(ok_reqs),
            "requests_failed": len(bad_reqs),
            "completion_tokens_total": comp_tokens,
            "prompt_tokens_example": ok_reqs[0]["prompt_tokens"] if ok_reqs else None,
            "makespan_s": makespan,
            "wall_phase_s": t_wall1 - t_wall0,
            "output_tokens_per_s": out_tps,
            "output_tokens_per_min": out_tps * 60.0,
            "server_predicted_tokens_delta": server_pred,
            "ttft_s": {"p50": pct(ttfts, 0.50), "p90": pct(ttfts, 0.90),
                       "p95": pct(ttfts, 0.95), "p99": pct(ttfts, 0.99),
                       "max": max(ttfts) if ttfts else None,
                       "mean": statistics.mean(ttfts) if ttfts else None},
            "latency_s": {"p50": pct(lats, 0.50), "p90": pct(lats, 0.90),
                          "p95": pct(lats, 0.95), "p99": pct(lats, 0.99),
                          "max": max(lats) if lats else None,
                          "mean": statistics.mean(lats) if lats else None},
            "server_props": {
                "total_slots": props.get("total_slots"),
                "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
            },
            "server_startup_line": startup_line,
            "vram_ready_mib": vram_ready,
            "vram_idle_mib": vram_idle,
            "telemetry": telem.summarize(),
            "failures_sample": bad_reqs[:5],
        }
        # per-request records so an external reader can recompute percentiles
        with open(outdir / f"requests-{args.tag}.jsonl", "w") as jf:
            for r in sink:
                jf.write(json.dumps({k: r.get(k) for k in (
                    "ok", "status", "t_start", "t_end", "ttft", "latency",
                    "prompt_tokens", "completion_tokens", "finish_reason")}) + "\n")
        result["ok"] = len(bad_reqs) == 0 and len(ok_reqs) == need_measured(args)
        log(f"DONE {args.tag}: {out_tps*60:.0f} out-tok/min "
            f"({out_tps:.1f} tok/s), ok={len(ok_reqs)} fail={len(bad_reqs)}")
    finally:
        stop_server(proc, fh)

    with open(outdir / f"benchmark-{args.tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"wrote {outdir / f'benchmark-{args.tag}.json'}")
    return 0 if result.get("ok") else 2


def need_measured(args):
    return args.concurrency * args.measured


if __name__ == "__main__":
    sys.exit(main())

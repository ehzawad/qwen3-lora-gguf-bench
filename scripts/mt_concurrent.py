#!/usr/bin/env python3
"""Concurrent multi-turn load: N simultaneous users, each holding a multi-turn
conversation at a target context window, against a running llama-server. This is
the "30 users chatting at the same time" test. Each worker pins id_slot=worker_id
(prefix caching per slot) and replays a composite session turn-by-turn, appending
the model's own replies (a live-ish session), measuring per-turn TTFT/decode and
aggregate output throughput over a single wall clock.

Usage: mt_concurrent.py --url URL --sessions data/composite_sessions.jsonl
        --target 8192 --concurrency 30 --turns 4 --max-tokens 256 --outdir DIR --tag TAG
"""
import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import requests


def turn(base, messages, slot, max_tokens):
    payload = {"model": "qwen3-4b-legal-q6k", "messages": messages, "stream": True,
               "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
               "temperature": 0.7, "cache_prompt": True, "id_slot": slot}
    t0 = time.perf_counter(); ttft = None; text = []
    ptok = ctok = cached = None; fr = None
    try:
        with requests.post(base + "/v1/chat/completions", json=payload, stream=True,
                           timeout=600) as r:
            if r.status_code != 200:
                return None
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                d = raw[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    o = json.loads(d)
                except json.JSONDecodeError:
                    continue
                ch = o.get("choices") or []
                if ch:
                    if (ch[0].get("delta") or {}).get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text.append(ch[0]["delta"]["content"])
                    if ch[0].get("finish_reason"):
                        fr = ch[0]["finish_reason"]
                if o.get("usage"):
                    ptok = o["usage"].get("prompt_tokens")
                    ctok = o["usage"].get("completion_tokens")
                    cached = (o["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    except requests.RequestException:
        return None
    t1 = time.perf_counter()
    if ctok is None or ttft is None:
        return None
    return {"t_start": t0, "t_end": t1, "ttft_s": ttft,
            "decode_tok_s": ctok / (t1 - t0 - ttft) if t1 - t0 > ttft else 0,
            "prompt_tokens": ptok, "cached_tokens": cached or 0,
            "completion_tokens": ctok, "reply": "".join(text), "finish_reason": fr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--sessions", default="data/composite_sessions.jsonl")
    ap.add_argument("--target", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--turns", type=int, default=4)     # measured turns per worker
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    base = args.url.rstrip("/")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    sessions = [json.loads(l) for l in open(args.sessions) if l.strip()
                and json.loads(l)["target"] == args.target]
    if not sessions:
        print(f"no sessions for target {args.target}"); return 2

    recs = []
    lock = threading.Lock()
    barrier = threading.Barrier(args.concurrency)

    def vram():
        import subprocess
        try:
            return int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                text=True).strip().splitlines()[0])
        except Exception:
            return None

    followups = ["Can you elaborate on that?", "Give a concrete example.",
                 "What are the caveats?", "Summarize the key point in one line.",
                 "How would you apply that in practice?"]

    def worker(wid):
        sess = sessions[wid % len(sessions)]
        # seed with the full composite history (ends on a user turn, ~target tokens)
        # so EVERY measured turn happens at deep context; follow-ups grow it further.
        messages = list(sess["messages"])
        barrier.wait()
        for k in range(args.turns):
            res = turn(base, messages, wid, args.max_tokens)
            if res is None:
                break
            res.update({"worker": wid, "turn": k + 1})
            with lock:
                recs.append(res)
            messages.append({"role": "assistant", "content": res["reply"]})
            messages.append({"role": "user", "content": followups[k % len(followups)]})

    v0 = vram()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    # sample peak VRAM while running
    peak = v0 or 0
    while any(t.is_alive() for t in threads):
        peak = max(peak, vram() or peak)
        time.sleep(0.5)
    for t in threads:
        t.join()
    t1 = time.perf_counter()

    ok = [r for r in recs if r.get("completion_tokens")]
    comp = sum(r["completion_tokens"] for r in ok)
    makespan = (max(r["t_end"] for r in ok) - min(r["t_start"] for r in ok)) if ok else 0
    tps = comp / makespan if makespan else 0

    def pct(v, p):
        s = sorted(v); return s[min(len(s)-1, int(p*len(s)))] if s else None

    ttfts = [r["ttft_s"] for r in ok]
    result = {
        "tag": args.tag, "target_ctx": args.target, "concurrency": args.concurrency,
        "measured_turns": len(ok),
        "aggregate_output_tok_s": round(tps, 1),
        "aggregate_output_tok_min": round(tps * 60, 0),
        "per_user_output_tok_s": round(tps / args.concurrency, 2) if args.concurrency else None,
        "prompt_tok_median": statistics.median(r["prompt_tokens"] for r in ok) if ok else None,
        "cache_hit_frac_median": round(statistics.median(
            r["cached_tokens"] / r["prompt_tokens"] for r in ok if r["prompt_tokens"]), 3) if ok else None,
        "ttft_s": {"p50": pct(ttfts, .5), "p95": pct(ttfts, .95), "max": max(ttfts) if ttfts else None},
        "decode_tok_s_median": round(statistics.median(r["decode_tok_s"] for r in ok), 1) if ok else None,
        "vram_start_mib": v0, "vram_peak_mib": peak, "wall_s": round(t1 - t0, 1),
    }
    json.dump(result, open(outdir / f"concurrent-{args.tag}.json", "w"), indent=2)
    print(f"[mt_concurrent] {args.tag}: C={args.concurrency} ctx={args.target} -> "
          f"{result['aggregate_output_tok_s']} tok/s agg, "
          f"{result['per_user_output_tok_s']} per-user, TTFT p50={result['ttft_s']['p50']}, "
          f"cache_hit={result['cache_hit_frac_median']}, VRAM peak={peak} MiB, "
          f"turns={len(ok)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

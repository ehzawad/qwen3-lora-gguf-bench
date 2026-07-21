#!/usr/bin/env python3
"""Minimal interactive multi-turn chatbot against a running llama-server.

Keeps a growing conversation, streams the reply, and prints per-turn TTFT,
decode tok/s, and how many prompt tokens the server served from its prefix cache
(cached_tokens) — so you can watch prefix caching keep TTFT low as history grows.

Usage:  python3 scripts/chat.py --url http://127.0.0.1:8199 [--system "..."]
Start a cache-enabled long-context server first, e.g.:
    ./run.sh chat-serve 8192       # 1 slot, 8k context, f16 KV, prompt cache ON
Type your message; '/reset' clears history; '/quit' exits.
"""
import argparse
import json
import time
import sys

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8199")
    ap.add_argument("--model", default="qwen3-4b-legal-q6k")
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    base = args.url.rstrip("/") + "/v1/chat/completions"

    history = []
    if args.system:
        history.append({"role": "system", "content": args.system})
    print("Chat ready. '/reset' to clear, '/quit' to exit.\n")

    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print("(history cleared)\n"); continue

        history.append({"role": "user", "content": user})
        payload = {"model": args.model, "messages": history, "stream": True,
                   "stream_options": {"include_usage": True},
                   "max_tokens": args.max_tokens, "temperature": 0.7,
                   "cache_prompt": True}
        t0 = time.perf_counter(); ttft = None; text = []
        ptok = ctok = cached = None
        sys.stdout.write("bot › ")
        try:
            with requests.post(base, json=payload, stream=True, timeout=600) as r:
                if r.status_code != 200:
                    print(f"[HTTP {r.status_code}] {r.text[:200]}"); history.pop(); continue
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
                    if ch and (ch[0].get("delta") or {}).get("content"):
                        c = ch[0]["delta"]["content"]
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text.append(c); sys.stdout.write(c); sys.stdout.flush()
                    if o.get("usage"):
                        ptok = o["usage"].get("prompt_tokens")
                        ctok = o["usage"].get("completion_tokens")
                        cached = (o["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
        except requests.RequestException as e:
            print(f"\n[error] {e}"); history.pop(); continue
        dt = time.perf_counter() - t0
        reply = "".join(text)
        history.append({"role": "assistant", "content": reply})
        dec = (ctok / (dt - (ttft or 0))) if ctok and dt > (ttft or 0) else 0
        print(f"\n  \033[90m[TTFT {ttft*1000:.0f} ms · {dec:.0f} tok/s · "
              f"prompt {ptok} tok ({cached or 0} cached) · reply {ctok} tok]\033[0m\n")


if __name__ == "__main__":
    main()

"""Synthetic finite-user (closed-loop, think-time) load for the llama-server.

NOT open-loop: each Locust user waits for its response, thinks (wait_time), then
sends the next request. When latency rises, arrival rate falls. Label results
"synthetic interactive user simulation with response-dependent think time," not
"real-world" and not "open-loop" (per codex-council load-methodology review).

Two profiles (select on the CLI):
  ShortChatUser    - independent short-chat, think-time 1-4s, natural sampling
                     (temp 0.7, max 160 tok).
  ConversationUser - multi-turn: accumulates the REAL assistant reply into a
                     growing history (up to MAX_TURNS) then resets, so prompt
                     tokens (repeated prefill) grow each turn.

Token throughput is counted ONLY inside a declared steady-state window
[test_start + LOCUST_WARMUP_S, + LOCUST_MEASURE_S] so ramp-up and drain are
excluded. Streamed response_time is corrected to full start->[DONE] time.
"""
import json
import os
import random
import threading
import time

from locust import HttpUser, between, events, task

MAX_TOKENS = int(os.environ.get("LOCUST_MAX_TOKENS", "160"))
MAX_TURNS = int(os.environ.get("LOCUST_MAX_TURNS", "8"))
TEMPERATURE = float(os.environ.get("LOCUST_TEMPERATURE", "0.7"))
WARMUP_S = float(os.environ.get("LOCUST_WARMUP_S", "60"))
MEASURE_S = float(os.environ.get("LOCUST_MEASURE_S", "300"))
CONNECT_TIMEOUT = float(os.environ.get("LOCUST_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.environ.get("LOCUST_READ_TIMEOUT", "180"))

_PROMPTS = [
    "How do I reverse a linked list in Python?",
    "Explain the difference between TCP and UDP.",
    "Give me three ideas for a weekend project.",
    "What causes rainbows?",
    "Write a short professional email declining a meeting.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What's a good way to learn to cook?",
    "Explain recursion with a simple example.",
    "Suggest a name for a coffee shop and why.",
    "What are the benefits of unit testing?",
]
_FOLLOWUPS = [
    "Can you make that shorter?", "Give me an example.", "Why does that matter?",
    "What are the tradeoffs?", "Explain it to a beginner.", "What would you change?",
]

_lock = threading.Lock()
_win = {"start": None, "end": None}
_agg = {"completion_tokens": 0, "prompt_tokens": 0, "requests": 0,
        "ttft_ms": [], "e2e_ms": []}


@events.test_start.add_listener
def _on_start(environment, **kw):
    t0 = time.time()
    _win["start"] = t0 + WARMUP_S
    _win["end"] = t0 + WARMUP_S + MEASURE_S


def _in_window(t):
    return _win["start"] is not None and _win["start"] <= t <= _win["end"]


def _stream_chat(client, messages, name):
    """Streaming chat request. Returns (completion_tokens, prompt_tokens,
    assistant_text, ok). Corrects response_time to full body time, applies strict
    success criteria, and accounts tokens only within the steady-state window."""
    payload = {
        "model": "qwen3-4b-legal-q6k", "messages": messages, "stream": True,
        "stream_options": {"include_usage": True}, "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    t0 = time.perf_counter()
    ttft_ms = None
    ctoks = ptoks = None
    finish_reason = None
    saw_done = False
    err_obj = None
    text_parts = []
    try:
        with client.post("/v1/chat/completions", json=payload, stream=True,
                         name=name, catch_response=True,
                         timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return 0, None, "", False
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    r.failure("malformed SSE json")
                    return 0, None, "", False
                if obj.get("error"):
                    err_obj = obj["error"]
                    break
                ch = obj.get("choices") or []
                if ch:
                    delta = ch[0].get("delta") or {}
                    c = delta.get("content")
                    if c:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        text_parts.append(c)
                    if ch[0].get("finish_reason"):
                        finish_reason = ch[0]["finish_reason"]
                if obj.get("usage"):
                    ctoks = obj["usage"].get("completion_tokens")
                    ptoks = obj["usage"].get("prompt_tokens")
            # correct the streamed response_time to full start->done duration
            full_ms = (time.perf_counter() - t0) * 1000
            r.request_meta["response_time"] = full_ms
            ok = (err_obj is None and saw_done and ctoks is not None
                  and finish_reason is not None)
            if ok:
                r.success()
            else:
                r.failure(f"incomplete stream err={err_obj} done={saw_done} "
                          f"usage={ctoks} finish={finish_reason}")
    except Exception as e:  # noqa: BLE001
        return 0, None, "", False

    if ok and _in_window(time.time()):
        with _lock:
            _agg["completion_tokens"] += ctoks
            _agg["prompt_tokens"] += ptoks or 0
            _agg["requests"] += 1
            if ttft_ms is not None:
                _agg["ttft_ms"].append(ttft_ms)
            _agg["e2e_ms"].append(full_ms)
    return (ctoks or 0), ptoks, "".join(text_parts), ok


class ShortChatUser(HttpUser):
    wait_time = between(1, 4)

    @task
    def chat(self):
        _stream_chat(self.client, [{"role": "user", "content": random.choice(_PROMPTS)}],
                     name="short-chat")


class ConversationUser(HttpUser):
    wait_time = between(2, 6)

    def on_start(self):
        self.history = []
        self.turn = 0

    @task
    def converse(self):
        if self.turn == 0 or not self.history:
            self.history = [{"role": "user", "content": random.choice(_PROMPTS)}]
        else:
            self.history.append({"role": "user", "content": random.choice(_FOLLOWUPS)})
        _c, _p, reply, ok = _stream_chat(
            self.client, self.history, name=f"convo-turn-{min(self.turn + 1, MAX_TURNS)}")
        if ok:
            self.history.append({"role": "assistant", "content": reply})  # REAL reply
            self.turn += 1
            if self.turn >= MAX_TURNS:
                self.turn = 0
                self.history = []
        else:
            self.turn = 0
            self.history = []   # reset on failure (recorded policy)


@events.quitting.add_listener
def _summary(environment, **kw):
    with _lock:
        toks = _agg["completion_tokens"]
        reqs = _agg["requests"]
        ttfts = sorted(_agg["ttft_ms"])
        e2es = sorted(_agg["e2e_ms"])

    def pct(a, p):
        return a[min(len(a) - 1, int(p * len(a)))] if a else None

    tps = toks / MEASURE_S if MEASURE_S else 0.0
    print(f"\n[locust] window={MEASURE_S:.0f}s tokens={toks} reqs={reqs} "
          f"=> {tps:.1f} tok/s ({tps*60:.0f} tok/min) | "
          f"TTFT p50={pct(ttfts,0.5)} p95={pct(ttfts,0.95)} ms | "
          f"e2e p50={pct(e2es,0.5)} p95={pct(e2es,0.95)} ms")
    out = os.environ.get("LOCUST_TOKEN_JSON")
    if out:
        json.dump({"window_s": MEASURE_S, "completion_tokens": toks,
                   "prompt_tokens": _agg["prompt_tokens"], "requests_in_window": reqs,
                   "tokens_per_s": tps, "tokens_per_min": tps * 60,
                   "ttft_ms_p50": pct(ttfts, 0.5), "ttft_ms_p95": pct(ttfts, 0.95),
                   "e2e_ms_p50": pct(e2es, 0.5), "e2e_ms_p95": pct(e2es, 0.95)},
                  open(out, "w"), indent=2)

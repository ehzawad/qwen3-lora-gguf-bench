#!/usr/bin/env python3
"""Build length-controlled COMPOSITE multi-turn sessions from real UltraChat
transcripts. UltraChat dialogues are short (p99 ~2.3k tokens, max ~3.3k), so no
single conversation fills 8k/16k/32k. We deterministically concatenate COMPLETE
dialogue blocks (never slicing a message) until the rendered prompt approaches a
per-window budget. Honest label: "composite sessions assembled from real (but
synthetically-generated) UltraChat transcripts," NOT single natural conversations.

Output: data/composite_sessions.jsonl  (each: target, messages[], prompt_tokens)
Usage:  composite_sessions.py [--per-target 32]
"""
import argparse
import json
import os

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models", "base"),
                                    use_fast=True, local_files_only=True)

# per-window prompt budget = ctx - max_gen(256) - margin(32)
TARGETS = {8192: 7904, 16384: 16096, 32768: 32480}


def rendered_len(messages):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-target", type=int, default=40)
    args = ap.parse_args()

    src = [json.loads(l) for l in open(os.path.join(DATA, "ultrachat_multiturn.jsonl"))
           if l.strip()]
    # keep only clean alternating dialogues with >=1 exchange and nonempty content
    blocks = []
    for conv in src:
        ms = conv["messages"]
        if all(m.get("content", "").strip() for m in ms) and len(ms) >= 2:
            blocks.append(ms)
    print(f"[composite] {len(blocks)} usable source dialogues", flush=True)

    out = []
    for target, budget in TARGETS.items():
        made = 0
        bi = 0
        while made < args.per_target and bi < len(blocks):
            session = []
            # append whole dialogue blocks until the next block would exceed budget
            while bi < len(blocks):
                cand = session + blocks[bi]
                # ensure it ends on a user turn (drop trailing assistant for measuring)
                trial = cand
                if trial and trial[-1]["role"] == "assistant":
                    trial = trial[:-1]
                if not trial or trial[-1]["role"] != "user":
                    bi += 1
                    continue
                n = rendered_len(trial)
                if n <= budget:
                    session = cand
                    bi += 1
                    if n >= budget * 0.85:  # close enough to the window
                        break
                else:
                    break
            # finalize: trim to end on a user turn under budget
            while session and session[-1]["role"] == "assistant":
                session = session[:-1]
            if session and session[-1]["role"] == "user":
                n = rendered_len(session)
                if n >= budget * 0.6:   # only keep sessions that actually fill the window
                    out.append({"target": target, "prompt_tokens_est": n,
                                "n_messages": len(session), "messages": session})
                    made += 1
            bi += 1
        print(f"[composite] target {target}: built {made} sessions", flush=True)

    path = os.path.join(DATA, "composite_sessions.jsonl")
    with open(path, "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    print(f"[composite] wrote {len(out)} composite sessions -> {path}", flush=True)


if __name__ == "__main__":
    main()

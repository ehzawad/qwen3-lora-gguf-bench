#!/usr/bin/env python3
"""Download the evaluation datasets from HuggingFace (git-ignored under data/):
  - wikitext-2-raw test -> data/wikitext-2-raw-test.txt  (for llama-perplexity)
  - a slice of HuggingFaceH4/ultrachat_200k -> data/ultrachat_multiturn.jsonl
    (real multi-turn conversations for the context-window replay)
"""
import json
import os

from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

# --- wikitext-2-raw test (accuracy / perplexity) ---
wt_path = os.path.join(DATA, "wikitext-2-raw-test.txt")
if not os.path.exists(wt_path):
    print("[data] wikitext-2-raw-v1 test ...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    with open(wt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r["text"] for r in ds))
    print(f"[data] wrote {wt_path} ({os.path.getsize(wt_path)//1024} KiB)", flush=True)
else:
    print(f"[data] wikitext exists", flush=True)

# --- ultrachat_200k slice (multi-turn conversations) ---
uc_path = os.path.join(DATA, "ultrachat_multiturn.jsonl")
if not os.path.exists(uc_path):
    print("[data] ultrachat_200k test_sft (streaming slice) ...", flush=True)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
    n, kept = 0, 0
    with open(uc_path, "w", encoding="utf-8") as f:
        for row in ds:
            msgs = row.get("messages") or []
            # keep genuinely multi-turn conversations (>= 6 messages = >= 3 turns)
            if len(msgs) >= 6:
                f.write(json.dumps({"messages": [{"role": m["role"],
                                                  "content": m["content"]}
                                                 for m in msgs]}) + "\n")
                kept += 1
            n += 1
            if kept >= 400 or n >= 4000:
                break
    print(f"[data] wrote {uc_path}: {kept} multi-turn conversations", flush=True)
else:
    print(f"[data] ultrachat exists", flush=True)

print("[data] DONE", flush=True)

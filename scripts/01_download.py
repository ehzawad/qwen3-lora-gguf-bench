#!/usr/bin/env python3
"""Download the official Qwen3-4B-Instruct-2507 base and the LoRA adapter.

Public/ungated repos -> no token required. Weights land under models/ and are
gitignored; only this script is committed.
"""
import os
import sys
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

BASE_REPO = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER_REPO = "narcolepticchicken/qwen3-4b-legal-ops-contract-intake-lora"
ADAPTER_REV = "8e5c6a9d99c9079fb775f2df5957d57a619659f9"
ADAPTER_SHA256 = "4848bae64fa74ba689c5300730aaed0f339d850589f08f59af4fe59808c15763"

BASE_DIR = os.path.join(MODELS, "base")
ADAPTER_DIR = os.path.join(MODELS, "adapter")


def dl(repo, dest, revision, allow=None):
    print(f"[download] {repo}@{revision} -> {dest}", flush=True)
    p = snapshot_download(
        repo_id=repo, revision=revision, local_dir=dest, allow_patterns=allow,
    )
    print(f"[download] done: {p}", flush=True)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    os.makedirs(MODELS, exist_ok=True)
    dl(BASE_REPO, BASE_DIR, BASE_REV,
       allow=["*.json", "*.txt", "*.safetensors", "*.jinja", "merges.txt", "vocab.json"])
    dl(ADAPTER_REPO, ADAPTER_DIR, ADAPTER_REV)
    got = sha256(os.path.join(ADAPTER_DIR, "adapter_model.safetensors"))
    assert got == ADAPTER_SHA256, f"adapter SHA mismatch: {got} != {ADAPTER_SHA256}"
    print(f"[download] adapter SHA-256 verified: {got}", flush=True)
    print("[download] ALL DONE", flush=True)

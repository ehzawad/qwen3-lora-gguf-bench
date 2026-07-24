#!/usr/bin/env python3
"""Download the official Qwen3-4B-Instruct-2507 base and the LoRA adapter.

Public/ungated repos -> no token required. Weights land under models/ and are
gitignored; only this script is committed.
"""
from __future__ import annotations

import hashlib
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

BASE_REPO = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER_REPO = "narcolepticchicken/qwen3-4b-legal-ops-contract-intake-lora"
ADAPTER_REV = "8e5c6a9d99c9079fb775f2df5957d57a619659f9"
ADAPTER_SHA256 = "4848bae64fa74ba689c5300730aaed0f339d850589f08f59af4fe59808c15763"

BASE_DIR = os.path.join(MODELS, "base")
ADAPTER_DIR = os.path.join(MODELS, "adapter")


def dl(repo: str, dest: str, revision: str, allow: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    print(f"[download] {repo}@{revision} -> {dest}", flush=True)
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=dest,
        allow_patterns=allow,
    )
    print(f"[download] done: {path}", flush=True)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"adapter SHA mismatch: {actual} != {expected}")
    return actual


def main() -> int:
    os.makedirs(MODELS, exist_ok=True)
    dl(
        BASE_REPO,
        BASE_DIR,
        BASE_REV,
        allow=["*.json", "*.txt", "*.safetensors", "*.jinja", "merges.txt", "vocab.json"],
    )
    dl(ADAPTER_REPO, ADAPTER_DIR, ADAPTER_REV)
    adapter_path = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    actual = verify_sha256(adapter_path, ADAPTER_SHA256)
    print(f"[download] adapter SHA-256 verified: {actual}", flush=True)
    print("[download] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

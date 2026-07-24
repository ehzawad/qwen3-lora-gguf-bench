#!/usr/bin/env python3
"""Capture a sanitized environment/hardware manifest for a benchmark run.

Allowlist only: never dumps tokens, environment variables, username, or hostname.
Git provenance is resolved relative to this script's repository, not the caller's
working directory. If no Git worktree is available, both source fields are null
instead of incorrectly reporting a clean tree.

Usage: capture_env.py <out_json>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent


def sh(command: str) -> str | None:
    try:
        return subprocess.check_output(
            command,
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def command_output(command: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            list(command),
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def pyver(module: str) -> str | None:
    try:
        imported = __import__(module)
        return getattr(imported, "__version__", None)
    except Exception:
        return None


def git_provenance(root: Path = ROOT) -> tuple[str | None, bool | None]:
    commit = command_output(["git", "rev-parse", "HEAD"], cwd=root)
    if not commit:
        return None, None
    status = command_output(["git", "status", "--porcelain"], cwd=root)
    if status is None:
        return commit, None
    return commit, bool(status)


def read_llama_commit(root: Path = ROOT) -> str | None:
    path = root / "results" / "llamacpp_commit.txt"
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def build_manifest(root: Path = ROOT) -> dict[str, object]:
    source_commit, source_dirty = git_provenance(root)
    gpu = sh(
        "nvidia-smi --query-gpu=name,uuid,memory.total,power.limit,"
        "driver_version,pcie.link.gen.max,pcie.link.width.max "
        "--format=csv,noheader,nounits -i 0"
    )

    return {
        "gpu_query": gpu,
        "cuda_umd": sh(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0"
        ),
        "nvcc": sh("nvcc --version | tail -1"),
        "cmake": sh("cmake --version | head -1"),
        "kernel": sh("uname -r"),
        "cpu_model": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2 | xargs"),
        "cpu_threads": sh("nproc"),
        "ram_gib": sh("free -g | awk '/Mem:/{print $2}'"),
        "python": sys.version.split()[0],
        "packages": {
            "torch": pyver("torch"),
            "transformers": pyver("transformers"),
            "peft": pyver("peft"),
            "safetensors": pyver("safetensors"),
            "huggingface_hub": pyver("huggingface_hub"),
            "numpy": pyver("numpy"),
            "requests": pyver("requests"),
        },
        "llama_cpp_commit": read_llama_commit(root),
        "source_git_commit": source_commit,
        "source_dirty": source_dirty,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0]) if args else Path("results/environment.json")
    manifest = build_manifest()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[env] ERROR: cannot write {out}: {exc}", file=sys.stderr)
        return 2
    print(f"[env] wrote {out}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

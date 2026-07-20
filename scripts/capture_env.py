#!/usr/bin/env python3
"""Capture a sanitized environment/hardware manifest for a benchmark run.

Allowlist only: never dumps tokens, env, username, or hostname.
Usage: capture_env.py <out_json>
"""
import json
import subprocess
import sys


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None


def pyver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", None)
    except Exception:
        return None


gpu = sh("nvidia-smi --query-gpu=name,uuid,memory.total,power.limit,"
         "driver_version,pcie.link.gen.max,pcie.link.width.max "
         "--format=csv,noheader,nounits -i 0")

manifest = {
    "gpu_query": gpu,
    "cuda_umd": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0"),
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
    "llama_cpp_commit": sh("cat results/llamacpp_commit.txt 2>/dev/null"),
    "source_git_commit": sh("git rev-parse HEAD 2>/dev/null"),
    "source_dirty": sh("test -n \"$(git status --porcelain 2>/dev/null)\" && echo true || echo false"),
}

out = sys.argv[1] if len(sys.argv) > 1 else "results/environment.json"
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[env] wrote {out}")
print(json.dumps(manifest, indent=2))

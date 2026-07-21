#!/usr/bin/env bash
# E1: fine concurrency sweep on the A5000 (closed-loop, barrier-synced). For each
# concurrency C, a fresh server with --parallel C, ctx=C*768, --no-kv-unified.
# Measures the throughput-vs-active-sequences curve, service latency, CPU/GPU
# telemetry, and the per-CONFIGURED-SLOT VRAM footprint (KV is preallocated at
# startup, so this is VRAM per provisioned slot, NOT per live request).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

MODEL="$ROOT/models/gguf/qwen3-4b-legal-ops-Q6_K.gguf"
BIN="$ROOT/vendor/llama.cpp/build/bin"
PROMPTS="$ROOT/prompts/short-chat.jsonl"
RUN="${1:-$ROOT/results/sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN"
python3 scripts/capture_env.py "$RUN/manifest.json" >/dev/null

# more samples at low C (cheap) for credible percentiles; fewer at high C
meas_for() { local c=$1
  if   [ "$c" -le 1 ]; then echo 60
  elif [ "$c" -le 2 ]; then echo 40
  elif [ "$c" -le 4 ]; then echo 30
  elif [ "$c" -le 8 ]; then echo 20
  else echo 10; fi; }

fail=0
run_point() {
  local c=$1 tag=$2 ctx=$(( $1 * 768 )) m; m=$(meas_for "$1")
  # abort the point if an unexpected process already holds GPU0
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  if [ "$used" -gt 1500 ]; then
    echo "### C=$c SKIPPED: GPU0 already has ${used} MiB used (unexpected) ###"; fail=1; return
  fi
  echo "### SWEEP C=$c ctx=$ctx measured=$m tag=$tag $(date -u +%H:%M:%S) ###"
  python3 scripts/benchmark.py --model "$MODEL" --bin-dir "$BIN" --prompts "$PROMPTS" \
    --concurrency "$c" --ctx "$ctx" --port 8199 --outdir "$RUN" --tag "$tag" \
    --measured "$m" --warmup 2 || { echo "### C=$c FAILED ###"; fail=1; }
}

for c in 1 2 4 8 16 24 32 48 64 96 128; do run_point "$c" "$(printf 'c%03d' "$c")"; done
run_point 32 c032b     # anchor repeat: detect thermal / host-load drift

python3 scripts/report.py "$RUN"
echo "### SWEEP DONE ($RUN) fail=$fail ###"

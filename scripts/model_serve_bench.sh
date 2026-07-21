#!/usr/bin/env bash
# Model-scaling concurrency benchmark: 3 text-only Q4_K_M GGUFs (self-quantized
# with identical llama-quantize Q4_K_M, no imatrix) on ONE A5000, SAME llama.cpp
# binary. Fixed-server / vary-client, prefix-cache OFF, full-GPU (-ngl 99).
# Reports native-token throughput + actual post-template prompt tokens per model.
# NOT iso-precision (matched nominal Q4_K_M tier); NOT a quality comparison.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
BIN="$ROOT/vendor/llama.cpp/build/bin"
Q4="$ROOT/models/gguf-ext/self-q4km"
PORT=8300
CLIENTS="${CLIENTS:-1 8 16 32 64}"
NP="${NP:-64}"; SLOTCTX="${SLOTCTX:-1024}"
MEASURED="${MEASURED:-15}"; WARMUP="${WARMUP:-2}"
RUN="${1:-$ROOT/results/modelscale-$(date -u +%Y%m%dT%H%M%SZ)}"; mkdir -p "$RUN"

declare -a MODELS=(
  "qwen3-4b|$Q4/qwen3-4b-legal-Q4_K_M.gguf"
  "qwen3.5-9b|$Q4/qwen3.5-9b-Q4_K_M.gguf"
  "gemma-4-e2b|$Q4/gemma-4-e2b-Q4_K_M.gguf"
)
CTX=$(( NP * SLOTCTX ))
free_port(){ fuser -k "${PORT}/tcp" 2>/dev/null || true; sleep 3; }
wait_health(){ local t0=$SECONDS; while (( SECONDS-t0 < ${2:-300} )); do curl -sf "$1/health" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
vram(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1; }

for entry in "${MODELS[@]}"; do
  name="${entry%%|*}"; gguf="${entry##*|}"
  echo "### $name  (np=$NP slotctx=$SLOTCTX ctx=$CTX) $(date -u +%H:%M:%S) ###"
  free_port
  v_idle=$(vram)
  setsid "$BIN/llama-server" -m "$gguf" --alias qwen3-4b-legal-q6k \
    --host 127.0.0.1 --port "$PORT" -dev CUDA0 -sm none -ngl 99 -fa on \
    -np "$NP" --ctx-size "$CTX" --no-kv-unified -cb -b 2048 -ub 512 \
    -ctk f16 -ctv f16 --no-cache-prompt --no-context-shift \
    --jinja --no-webui --metrics --slots > "$RUN/server-$name.log" 2>&1 &
  pid=$!
  if ! wait_health "http://127.0.0.1:$PORT" 300; then
    echo "### $name FAILED TO START (likely OOM at np=$NP) — see server-$name.log ###"
    grep -iE "error|oom|out of memory|failed|cuda" "$RUN/server-$name.log" | tail -5
    kill "$pid" 2>/dev/null; free_port; echo "{\"model\":\"$name\",\"status\":\"boot_failed\"}" > "$RUN/status-$name.json"; continue
  fi
  v_load=$(vram)
  echo "  static VRAM: idle=$v_idle load=$v_load (weights+KV alloc for $NP slots)"
  for c in $CLIENTS; do
    python3 scripts/bench_external.py --url "http://127.0.0.1:$PORT" --engine "$name" \
      --concurrency "$c" --outdir "$RUN" --tag "$name-c$(printf '%03d' "$c")" \
      --measured "$MEASURED" --warmup "$WARMUP" || echo "### $name c$c FAILED ###"
    echo "    c=$c VRAM=$(vram) MiB"
  done
  echo "{\"model\":\"$name\",\"status\":\"ok\",\"vram_idle_mib\":$v_idle,\"vram_load_mib\":$v_load,\"np\":$NP,\"slotctx\":$SLOTCTX}" > "$RUN/status-$name.json"
  kill "$pid" 2>/dev/null; pkill -f "llama-server" 2>/dev/null; free_port
done
echo "### MODEL-SCALE DONE ($RUN) ###"

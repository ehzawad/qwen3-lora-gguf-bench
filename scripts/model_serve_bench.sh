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
fail=0
active_pid=""

declare -a MODELS=(
  "qwen3-4b|$Q4/qwen3-4b-legal-Q4_K_M.gguf"
  "qwen3.5-9b|$Q4/qwen3.5-9b-Q4_K_M.gguf"
  "gemma-4-e2b|$Q4/gemma-4-e2b-Q4_K_M.gguf"
)
CTX=$(( NP * SLOTCTX ))

if ! python3 scripts/capture_env.py "$RUN/manifest.json" >/dev/null; then
  echo "### environment capture FAILED ###" >&2
  fail=1
fi

free_port(){ fuser -k "${PORT}/tcp" 2>/dev/null || true; sleep 3; }
wait_health(){ # $1=pid $2=timeout
  local pid="$1" timeout="${2:-300}" t0=$SECONDS
  while (( SECONDS-t0 < timeout )); do
    kill -0 "$pid" 2>/dev/null || return 1
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}
stop_group(){
  local pid="${1:-}"
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
  [ "$active_pid" = "$pid" ] && active_pid=""
}
cleanup(){ stop_group "$active_pid"; }
on_signal(){
  trap - EXIT INT TERM
  cleanup
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

vram(){
  local value
  if value=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1) \
    && [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$value"
  else
    printf 'null\n'
  fi
}

write_status(){ # model status idle load
  local model="$1" status="$2" idle="$3" load="$4"
  printf '{"model":"%s","status":"%s","vram_idle_mib":%s,"vram_load_mib":%s,"np":%s,"slotctx":%s}\n' \
    "$model" "$status" "$idle" "$load" "$NP" "$SLOTCTX" > "$RUN/status-$model.json"
}

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
  active_pid="$pid"
  if ! wait_health "$pid" 300; then
    echo "### $name FAILED TO START (likely OOM at np=$NP) — see server-$name.log ###" >&2
    grep -iE "error|oom|out of memory|failed|cuda" "$RUN/server-$name.log" | tail -5 >&2
    fail=1
    write_status "$name" "boot_failed" "$v_idle" "null"
    stop_group "$pid"
    free_port
    continue
  fi

  v_load=$(vram)
  echo "  static VRAM: idle=$v_idle load=$v_load (weights+KV alloc for $NP slots)"
  model_failed=0
  for c in $CLIENTS; do
    if ! python3 scripts/bench_external.py --url "http://127.0.0.1:$PORT" --engine "$name" \
      --concurrency "$c" --outdir "$RUN" --tag "$name-c$(printf '%03d' "$c")" \
      --measured "$MEASURED" --warmup "$WARMUP"; then
      echo "### $name c$c FAILED ###" >&2
      fail=1
      model_failed=1
    fi
    echo "    c=$c VRAM=$(vram) MiB"
  done

  if [ "$model_failed" -eq 0 ]; then
    write_status "$name" "ok" "$v_idle" "$v_load"
  else
    write_status "$name" "benchmark_failed" "$v_idle" "$v_load"
  fi
  stop_group "$pid"
  free_port
done

echo "### MODEL-SCALE DONE ($RUN) fail=$fail ###"
exit "$fail"

#!/usr/bin/env bash
# E5: engine comparison on ONE A5000 -- llama.cpp (Q6_K) vs vLLM (bf16 merged),
# same closed-loop harness (bench_external.py), same 256-in/256-out payload,
# client concurrency 1/30/100. FIXED-deployment framing: each engine is started
# ONCE with a production-style config and the client load is varied (unlike E1's
# per-C provisioning). Precision differs by engine (llama.cpp 6-bit vs vLLM
# bf16) -- label it; the point is engine architecture (PagedAttention + native
# continuous batching) under concurrency.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

MODEL_Q6="$ROOT/models/gguf/qwen3-4b-legal-ops-Q6_K.gguf"
MERGED="$ROOT/models/merged"
BIN="$ROOT/vendor/llama.cpp/build/bin"
VLLM="$ROOT/.venv-vllm/bin/vllm"
PORT=8199
RUN="${1:-$ROOT/results/engines/eng-$(date -u +%Y%m%dT%H%M%SZ)}"
CLIENTS="${CLIENTS:-1 30 100}"
MEASURED="${MEASURED:-20}"
WARMUP="${WARMUP:-2}"
mkdir -p "$RUN"
fail=0
active_pid=""

if ! python3 scripts/capture_env.py "$RUN/manifest.json" >/dev/null; then
  echo "### environment capture FAILED ###" >&2
  fail=1
fi

free_port() { fuser -k "${PORT}/tcp" 2>/dev/null || true; sleep 3; }
wait_health(){ # $1=pid $2=timeout
  local pid="$1" timeout="${2:-600}" t0=$SECONDS endpoint
  while (( SECONDS-t0 < timeout )); do
    kill -0 "$pid" 2>/dev/null || return 1
    for endpoint in /health /v1/models; do
      curl -sf "http://127.0.0.1:$PORT$endpoint" >/dev/null 2>&1 && return 0
    done
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

sweep(){ # $1=engine-label $2=tag-prefix
  local engine="$1" prefix="$2" c
  for c in $CLIENTS; do
    if ! python3 scripts/bench_external.py --url "http://127.0.0.1:$PORT" --engine "$engine" \
      --concurrency "$c" --outdir "$RUN" --tag "$prefix-c$(printf '%03d' "$c")" \
      --measured "$MEASURED" --warmup "$WARMUP"; then
      echo "### $engine c$c FAILED ###" >&2
      fail=1
    fi
  done
}

run_llamacpp(){
  echo "### E5 llama.cpp (Q6_K, -np 100 fixed) $(date -u +%H:%M:%S) ###"
  free_port
  setsid "$BIN/llama-server" -m "$MODEL_Q6" --alias qwen3-4b-legal-q6k \
    --host 127.0.0.1 --port "$PORT" -dev CUDA0 -sm none -ngl all -fa on \
    -np 100 --ctx-size 76800 --no-kv-unified -cb -b 2048 -ub 512 \
    -ctk f16 -ctv f16 --cache-ram 0 --no-cache-prompt --no-context-shift \
    -rea off --jinja --no-webui --metrics --slots > "$RUN/server-llamacpp.log" 2>&1 &
  local pid=$!
  active_pid="$pid"
  if wait_health "$pid" 300; then
    sweep "llamacpp-q6k" "llamacpp"
  else
    echo "### llama.cpp never ready (see server-llamacpp.log) ###" >&2
    fail=1
  fi
  stop_group "$pid"
  free_port
}

run_vllm(){
  echo "### E5 vLLM (bf16 merged) $(date -u +%H:%M:%S) ###"
  free_port
  if [ ! -x "$VLLM" ]; then
    echo "### missing vLLM executable: $VLLM ###" >&2
    fail=1
    return
  fi
  setsid "$VLLM" serve "$MERGED" --served-model-name qwen3-4b-legal-q6k \
    --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization 0.9 \
    --max-model-len 2048 --dtype bfloat16 \
    > "$RUN/server-vllm.log" 2>&1 &
  local pid=$!
  active_pid="$pid"
  if wait_health "$pid" 600; then
    sweep "vllm-bf16" "vllm"
  else
    echo "### vLLM never ready (see server-vllm.log) ###" >&2
    fail=1
  fi
  stop_group "$pid"
  free_port
}

run_llamacpp
run_vllm

echo "### ENGINE COMPARE DONE ($RUN) fail=$fail ###"
exit "$fail"

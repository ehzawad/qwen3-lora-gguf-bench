#!/usr/bin/env bash
# FAIR engine comparison on ONE A5000 -- iso-precision bf16.
#
# Both engines serve the SAME merged weights at the SAME numerical precision
# (bf16): llama.cpp uses the bf16 GGUF, vLLM/SGLang use the bf16 safetensors.
# Identical client harness (bench_external.py), identical payload (256-token
# prompt-ish chat / exactly 256 generated via ignore_eos, temp 0, seed fixed),
# identical concurrency sweep. Prefix caching is DISABLED on every engine so we
# measure raw continuous-batching throughput, not caching (caching is studied
# separately in the context section). Fixed-server framing: each engine is
# started ONCE with a production-style config (up to 100 concurrent seqs) and
# the client concurrency is swept -- this is how you actually benchmark a
# serving engine, and it exposes each engine's memory model honestly.
#
# Usage: engine_fair.sh [run_dir] [engine]   engine in {llamacpp,vllm,sglang,all}
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

BF16_GGUF="$ROOT/models/gguf/qwen3-4b-legal-ops-bf16.gguf"
MERGED="$ROOT/models/merged"
BIN="$ROOT/vendor/llama.cpp/build/bin"
VLLM="$ROOT/.venv-vllm-stable/bin/vllm"
PORT=8300
ALIAS="qwen3-4b-legal-q6k"      # client sends this model name; serve under it
CLIENTS="${CLIENTS:-1 8 16 30 60 100}"
MEASURED="${MEASURED:-15}"; WARMUP="${WARMUP:-2}"
RUN="${1:-$ROOT/results/enginefair-$(date -u +%Y%m%dT%H%M%SZ)}"
WHICH="${2:-all}"
mkdir -p "$RUN"
fail=0
active_pid=""

if ! python3 scripts/capture_env.py "$RUN/manifest.json" >/dev/null 2>&1; then
  echo "### environment capture FAILED ###" >&2
  fail=1
fi

free_port(){ fuser -k "${PORT}/tcp" 2>/dev/null || true; sleep 3; }
wait_health(){ # $1=pid $2=url $3=timeout
  local pid="$1" url="$2" timeout="${3:-600}" t0=$SECONDS ep
  while (( SECONDS-t0 < timeout )); do
    kill -0 "$pid" 2>/dev/null || return 1
    for ep in /health /v1/models; do
      curl -sf "$url$ep" >/dev/null 2>&1 && return 0
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
cleanup(){
  stop_group "$active_pid"
}
on_signal(){
  trap - EXIT INT TERM
  cleanup
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

sweep(){ # $1=engine-tag
  local engine="$1" c
  for c in $CLIENTS; do
    if ! python3 scripts/bench_external.py --url "http://127.0.0.1:$PORT" --engine "$engine" \
      --concurrency "$c" --outdir "$RUN" --tag "$engine-c$(printf '%03d' "$c")" \
      --measured "$MEASURED" --warmup "$WARMUP"; then
      echo "### $engine c$c FAILED ###" >&2
      fail=1
    fi
  done
}

run_llamacpp(){
  echo "### llama.cpp bf16 (-np 100 fixed, prefix-cache off) $(date -u +%H:%M:%S) ###"
  free_port
  setsid "$BIN/llama-server" -m "$BF16_GGUF" --alias "$ALIAS" \
    --host 127.0.0.1 --port "$PORT" -dev CUDA0 -sm none -ngl all -fa on \
    -np 100 --ctx-size 76800 --no-kv-unified -cb -b 2048 -ub 512 \
    -ctk f16 -ctv f16 --cache-ram 0 --no-cache-prompt --no-context-shift \
    -rea off --jinja --no-webui --metrics --slots > "$RUN/server-llamacpp.log" 2>&1 &
  local pid=$!
  active_pid="$pid"
  if wait_health "$pid" "http://127.0.0.1:$PORT" 300; then
    sweep "llamacpp-bf16"
  else
    echo "### llamacpp never ready (see server-llamacpp.log) ###" >&2
    fail=1
  fi
  stop_group "$pid"
  free_port
}

run_vllm(){
  echo "### vLLM bf16 (paged KV, max-num-seqs 100, prefix-cache off) $(date -u +%H:%M:%S) ###"
  free_port
  if [ ! -x "$VLLM" ]; then
    echo "### missing vLLM executable: $VLLM ###" >&2
    fail=1
    return
  fi
  setsid "$VLLM" serve "$MERGED" --served-model-name "$ALIAS" \
    --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
    --max-model-len 1024 --max-num-seqs 100 --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --disable-log-requests \
    > "$RUN/server-vllm.log" 2>&1 &
  local pid=$!
  active_pid="$pid"
  if wait_health "$pid" "http://127.0.0.1:$PORT" 600; then
    sweep "vllm-bf16"
  else
    echo "### vllm never ready (see server-vllm.log) ###" >&2
    fail=1
  fi
  stop_group "$pid"
  free_port
}

run_sglang(){
  echo "### SGLang bf16 (RadixAttention, prefix-cache off) $(date -u +%H:%M:%S) ###"
  free_port
  local sg="$ROOT/.venv-sglang/bin/python"
  if [ ! -x "$sg" ]; then
    echo "### missing SGLang environment: $sg ###" >&2
    fail=1
    return
  fi
  setsid "$sg" -m sglang.launch_server --model-path "$MERGED" --served-model-name "$ALIAS" \
    --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --context-length 1024 \
    --max-running-requests 100 --disable-radix-cache --disable-cuda-graph \
    > "$RUN/server-sglang.log" 2>&1 &
  local pid=$!
  active_pid="$pid"
  if wait_health "$pid" "http://127.0.0.1:$PORT" 600; then
    sweep "sglang-bf16"
  else
    echo "### sglang never ready (see server-sglang.log) ###" >&2
    fail=1
  fi
  stop_group "$pid"
  free_port
}

case "$WHICH" in
  llamacpp) run_llamacpp;;
  vllm) run_vllm;;
  sglang) run_sglang;;
  all) run_llamacpp; run_vllm; run_sglang;;
  *) echo "usage: $0 [run_dir] {llamacpp|vllm|sglang|all}" >&2; exit 2;;
esac

echo "### ENGINE-FAIR DONE ($RUN) fail=$fail ###"
exit "$fail"

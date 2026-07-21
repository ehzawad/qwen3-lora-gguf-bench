#!/usr/bin/env bash
# Top-level runner for the Qwen3-4B LoRA -> GGUF Q6_K -> llama.cpp concurrency
# benchmark. Single RTX A5000 (GPU 0). See README.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_Q6="$ROOT/models/gguf/qwen3-4b-legal-ops-Q6_K.gguf"
GGUF_BF16="$ROOT/models/gguf/qwen3-4b-legal-ops-bf16.gguf"
BIN="$ROOT/vendor/llama.cpp/build/bin"
PROMPTS="$ROOT/prompts/short-chat.jsonl"
PORT="${PORT:-8199}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # device 0 == physical GPU 0 (A5000)
export CUDA_VISIBLE_DEVICES=0

corpus()   { python3 scripts/make_corpus.py "$PROMPTS" 2400; }
download() { python3 scripts/01_download.py; }
build()    { bash scripts/03_build_llamacpp.sh; }
merge()    { HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
             python3 scripts/merge.py models/base models/adapter models/merged; }
convert()  { bash scripts/convert.sh; }
quantize() { bash scripts/quantize.sh; }
verify()   { python3 scripts/verify_artifacts.py vendor/llama.cpp "$GGUF_BF16" "$MODEL_Q6"; }

preflight() {
  echo "== preflight =="
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader -i 0
  python3 - <<'PY'
import torch, transformers, peft
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
PY
  df -h "$ROOT" | tail -1
}

benchmark() {
  local run="${1:-$ROOT/results/a5000-$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$run"
  python3 scripts/capture_env.py "$run/manifest.json"
  cp config/experiment.json "$run/experiment.json"
  # concurrency -> total ctx (768 tokens/slot, --no-kv-unified)
  for c in 1 30 100; do
    local ctx=$(( c * 768 ))
    local tag=$(printf "c%03d" "$c")
    echo "== benchmark C=$c ctx=$ctx =="
    python3 scripts/benchmark.py \
      --model "$MODEL_Q6" --bin-dir "$BIN" --prompts "$PROMPTS" \
      --concurrency "$c" --ctx "$ctx" --port "$PORT" \
      --outdir "$run" --tag "$tag" --measured 20 --warmup 2 || true
  done
  python3 scripts/report.py "$run"
  echo "== results in $run =="
}

report() { python3 scripts/report.py "${1:?usage: run.sh report <run_dir>}"; }

serve() {   # short-chat benchmark server (prompt cache OFF, for measurement)
  local c="${1:-30}"; local ctx=$(( c * 768 ))
  exec "$BIN/llama-server" -m "$MODEL_Q6" --alias qwen3-4b-legal-q6k \
    --host 127.0.0.1 --port "$PORT" -dev CUDA0 -sm none --main-gpu 0 \
    -ngl all --fit off -fa on -np "$c" --ctx-size "$ctx" --no-kv-unified -cb \
    -b 2048 -ub 512 -ctk f16 -ctv f16 --cache-ram 0 --no-cache-prompt \
    --no-context-shift -rea off --jinja --no-webui --metrics --slots
}

# chatbot server: per-slot context = $1 (default 8192), $2 slots (default 1),
# $3 KV dtype (default f16). Prompt caching is ENABLED for multi-turn reuse.
chat-serve() {
  local ctx="${1:-8192}" c="${2:-1}" kv="${3:-f16}"
  exec "$BIN/llama-server" -m "$MODEL_Q6" --alias qwen3-4b-legal-q6k \
    --host 127.0.0.1 --port "$PORT" -dev CUDA0 -sm none --main-gpu 0 \
    -ngl all --fit off -fa on -np "$c" --ctx-size "$(( c * ctx ))" --no-kv-unified -cb \
    -b 2048 -ub 512 -ctk "$kv" -ctv "$kv" --cache-reuse 256 \
    -rea off --jinja --no-webui --metrics --slots
}

context() { bash scripts/context_sweep.sh; }        # E6
accuracy() { bash scripts/perplexity.sh; }          # E7
data() { python3 scripts/prepare_data.py; }

reproduce() {
  preflight; corpus; download; merge; build; convert; quantize; verify; benchmark
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  preflight|corpus|download|merge|build|convert|quantize|verify|benchmark|report|\
  serve|chat-serve|context|accuracy|data|reproduce)
    "$cmd" "$@";;
  *) echo "usage: ./run.sh {preflight|corpus|download|merge|build|convert|quantize|verify|benchmark|report|serve|chat-serve <ctx> <slots> <kv>|context|accuracy|data|reproduce}";;
esac

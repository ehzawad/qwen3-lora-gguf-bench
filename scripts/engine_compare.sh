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
PORT=8199
RUN="${1:-$ROOT/results/engines/eng-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN"
python3 scripts/capture_env.py "$RUN/manifest.json" >/dev/null
CLIENTS="1 30 100"

free_port() { fuser -k "${PORT}/tcp" 2>/dev/null || true; sleep 3; }

echo "### E5 llama.cpp (Q6_K, -np 100 fixed) $(date -u +%H:%M:%S) ###"
setsid "$BIN/llama-server" -m "$MODEL_Q6" --alias qwen3-4b-legal-q6k \
  --host 127.0.0.1 --port "$PORT" -dev CUDA0 -sm none -ngl all -fa on \
  -np 100 --ctx-size 76800 --no-kv-unified -cb -b 2048 -ub 512 \
  -ctk f16 -ctv f16 --cache-ram 0 --no-cache-prompt --no-context-shift \
  -rea off --jinja --no-webui --metrics --slots > "$RUN/server-llamacpp.log" 2>&1 &
LPID=$!
for c in $CLIENTS; do
  python3 scripts/bench_external.py --url "http://127.0.0.1:$PORT" --engine llamacpp-q6k \
    --concurrency "$c" --outdir "$RUN" --tag "llamacpp-c$(printf '%03d' "$c")" \
    --measured 20 --warmup 2 || echo "### llamacpp c$c FAILED ###"
done
kill "$LPID" 2>/dev/null; free_port

echo "### E5 vLLM (bf16 merged) $(date -u +%H:%M:%S) ###"
setsid .venv-vllm/bin/vllm serve "$MERGED" --served-model-name qwen3-4b-legal-q6k \
  --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization 0.9 \
  --max-model-len 2048 --dtype bfloat16 \
  > "$RUN/server-vllm.log" 2>&1 &
VPID=$!
for c in $CLIENTS; do
  python3 scripts/bench_external.py --url "http://127.0.0.1:$PORT" --engine vllm-bf16 \
    --concurrency "$c" --outdir "$RUN" --tag "vllm-c$(printf '%03d' "$c")" \
    --measured 20 --warmup 2 || echo "### vllm c$c FAILED ###"
done
kill "$VPID" 2>/dev/null; free_port

echo "### ENGINE COMPARE DONE ($RUN) ###"

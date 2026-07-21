#!/usr/bin/env bash
# E4: precision / hyperparameter speed sweep with llama-bench (raw decode/prefill
# tok/s, no HTTP server). Tests whether lower-bit weights, KV-cache dtype, and
# flash-attention change generation speed on the A5000. Decode is expected to be
# weight-bandwidth-bound, so tg tok/s should rise as the weight file shrinks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

BENCH="$ROOT/vendor/llama.cpp/build/bin/llama-bench"
G="$ROOT/models/gguf"
RUN="${1:-$ROOT/results/precision-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN"
python3 scripts/capture_env.py "$RUN/manifest.json" >/dev/null

MODELS=()
for t in Q4_K_M Q5_K_M Q6_K Q8_0 bf16; do
  f="$G/qwen3-4b-legal-ops-$t.gguf"
  [ -f "$f" ] && MODELS+=(-m "$f")
done

# (1) weight quant x flash-attn, f16 KV, pp512 (prefill) + tg128 (decode)
echo "### E4a: weight quant x flash-attn (f16 KV) $(date -u +%H:%M:%S) ###"
"$BENCH" "${MODELS[@]}" -ngl 99 -fa on,off -p 512 -n 128 -r 3 -o json \
  > "$RUN/e4a_quant_fa.json" 2> "$RUN/e4a_quant_fa.err" || cat "$RUN/e4a_quant_fa.err"
"$BENCH" "${MODELS[@]}" -ngl 99 -fa on -p 512 -n 128 -r 3 -o md > "$RUN/e4a_summary.md" 2>/dev/null || true

# (2) KV-cache dtype at fixed Q6_K (matched K/V pairs; quantized KV needs fa on)
echo "### E4b: KV-cache dtype at Q6_K (fa on) ###"
: > "$RUN/e4b_kv_dtype.jsonl"
for kv in f16 bf16 q8_0; do
  "$BENCH" -m "$G/qwen3-4b-legal-ops-Q6_K.gguf" -ngl 99 -fa on \
    -ctk "$kv" -ctv "$kv" -p 512 -n 128 -r 3 -o jsonl \
    >> "$RUN/e4b_kv_dtype.jsonl" 2>> "$RUN/e4b_kv_dtype.err" || cat "$RUN/e4b_kv_dtype.err"
done

echo "### E4a summary (weight quant, fa on, f16 KV) ###"; cat "$RUN/e4a_summary.md" 2>/dev/null || true
echo "### PRECISION SWEEP DONE ($RUN) ###"

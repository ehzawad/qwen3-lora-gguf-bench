#!/usr/bin/env bash
# Quantize BF16 GGUF -> Q6_K (no --pure, no --allow-requantize).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BF16="$ROOT/models/gguf/qwen3-4b-legal-ops-bf16.gguf"
Q6="$ROOT/models/gguf/qwen3-4b-legal-ops-Q6_K.gguf"
QUANT="$ROOT/vendor/llama.cpp/build/bin/llama-quantize"

"$QUANT" "$BF16" "$Q6" Q6_K "$(nproc)"

echo "[quantize] wrote $Q6"
ls -la "$BF16" "$Q6"
sha256sum "$BF16" "$Q6" | tee "$ROOT/results/gguf_sha256.txt"

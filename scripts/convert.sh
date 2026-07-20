#!/usr/bin/env bash
# Convert the merged BF16 HF model -> BF16 GGUF (explicit --outtype bf16).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED="$ROOT/models/merged"
LLAMA="$ROOT/vendor/llama.cpp"
OUT="$ROOT/models/gguf/qwen3-4b-legal-ops-bf16.gguf"
mkdir -p "$ROOT/models/gguf"

export PYTHONPATH="$LLAMA/gguf-py:${PYTHONPATH:-}"
python3 "$LLAMA/convert_hf_to_gguf.py" \
  --outtype bf16 \
  --outfile "$OUT" \
  "$MERGED"

echo "[convert] wrote $OUT"
ls -la "$OUT"

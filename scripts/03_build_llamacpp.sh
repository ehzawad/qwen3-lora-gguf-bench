#!/usr/bin/env bash
# Build llama.cpp (pinned commit) with CUDA for Ampere (RTX A5000, CC 8.6).
# Reproducibility flags per council: pin the SHA, GGML_NATIVE=OFF, no Web UI,
# no tests, no libcurl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"
LLAMA_DIR="$VENDOR/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
LLAMA_CPP_REV="91d2fc387529940230555abd297a8b5e99737d3f"
mkdir -p "$VENDOR" "$ROOT/results"

if [ ! -d "$LLAMA_DIR/.git" ]; then
  echo "[build] fetching pinned llama.cpp $LLAMA_CPP_REV ..."
  git init "$LLAMA_DIR"
  git -C "$LLAMA_DIR" remote add origin https://github.com/ggml-org/llama.cpp.git
  git -C "$LLAMA_DIR" fetch --depth=1 origin "$LLAMA_CPP_REV"
  git -C "$LLAMA_DIR" checkout --detach FETCH_HEAD
fi

test "$(git -C "$LLAMA_DIR" rev-parse HEAD)" = "$LLAMA_CPP_REV" \
  || { echo "[build] ERROR: llama.cpp HEAD != pinned $LLAMA_CPP_REV"; exit 1; }
echo "$LLAMA_CPP_REV" > "$ROOT/results/llamacpp_commit.txt"

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --config Release -j "$(nproc)" \
  --target llama-server llama-quantize llama-cli llama-bench

echo "[build] binaries:"
ls -la "$BUILD_DIR/bin/" | grep -E "llama-(server|quantize|cli|bench)" || true
echo "[build] DONE"

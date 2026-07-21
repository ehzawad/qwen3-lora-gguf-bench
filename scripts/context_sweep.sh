#!/usr/bin/env bash
# E6: how the CONTEXT WINDOW impacts inference on one 24 GB A5000.
#   A) decode tok/s vs KV depth (llama-bench -d): attention reads more KV/token.
#   B) per-slot VRAM at each context size (confirms 144 KiB/token f16).
#   C) the fit/no-fit FRONTIER: max concurrent slots that fit at 8k/16k/32k for
#      f16 vs q8_0 KV -> the answer to "can I serve 30/100 users at 8k/16k/32k?"
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
BIN="$ROOT/vendor/llama.cpp/build/bin"
MODEL="$ROOT/models/gguf/qwen3-4b-legal-ops-Q6_K.gguf"
PORT=8199
RUN="${1:-$ROOT/results/context-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN"
python3 scripts/capture_env.py "$RUN/manifest.json" >/dev/null

echo "### A) decode-vs-KV-depth + prefill cost  $(date -u +%H:%M:%S) ###"
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -ctk f16 -ctv f16 \
  -d 0,2048,8192,16384,32768 -n 64 -p 0 -r 3 -o json > "$RUN/depth_f16.json" 2>"$RUN/depth_f16.err"
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -d 0,2048,8192,16384,32768 -n 64 -p 0 -r 3 -o json > "$RUN/depth_q8.json" 2>"$RUN/depth_q8.err"
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -p 512,2048,8192,16384,32768 -n 0 -r 3 -o json \
  > "$RUN/prefill.json" 2>"$RUN/prefill.err"
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -ctk f16 -ctv f16 -d 0,8192,32768 -n 64 -o md \
  > "$RUN/depth_summary.md" 2>/dev/null || true
echo "--- decode tok/s vs depth (f16 KV) ---"; cat "$RUN/depth_summary.md" 2>/dev/null || true

# launch a server with (np, total ctx, kv dtype); echo FIT<vram> / NOFIT / TIMEOUT
try_fit() {  # $1=np  $2=ctx_total  $3=kv
  fuser -k "${PORT}/tcp" 2>/dev/null || true; sleep 2
  setsid "$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
    -dev CUDA0 -sm none -ngl all --fit off -fa on -np "$1" --ctx-size "$2" \
    --no-kv-unified -cb -ctk "$3" -ctv "$3" --no-webui > "$RUN/srv.log" 2>&1 &
  local SV=$! code v
  for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:$PORT/health" || echo 000)
    if [ "$code" = "200" ]; then
      v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
      kill "$SV" 2>/dev/null; fuser -k "${PORT}/tcp" 2>/dev/null || true; echo "FIT ${v}MiB"; return
    fi
    kill -0 "$SV" 2>/dev/null || { fuser -k "${PORT}/tcp" 2>/dev/null || true; echo "NOFIT(OOM)"; return; }
    sleep 1
  done
  kill "$SV" 2>/dev/null; fuser -k "${PORT}/tcp" 2>/dev/null || true; echo "TIMEOUT"
}

echo "### B) per-slot VRAM at fixed 4 slots (f16) ###" | tee "$RUN/frontier.txt"
for ctx in 2048 4096 8192 16384 32768; do
  r=$(try_fit 4 $((ctx*4)) f16); echo "  per-slot-ctx=$ctx np=4 f16 -> $r" | tee -a "$RUN/frontier.txt"
done

echo "### C) fit/no-fit frontier (per-slot-ctx, np, kv) ###" | tee -a "$RUN/frontier.txt"
for spec in "8192 16 f16" "8192 20 f16" "16384 8 f16" "16384 12 f16" "32768 4 f16" "32768 6 f16" \
            "8192 30 q8_0" "8192 40 q8_0" "16384 16 q8_0" "32768 8 q8_0"; do
  set -- $spec; per=$1; np=$2; kv=$3
  r=$(try_fit "$np" $((per*np)) "$kv")
  echo "  per-slot-ctx=$per np=$np kv=$kv (30-user? $([ "$np" -ge 30 ] && echo yes || echo no)) -> $r" | tee -a "$RUN/frontier.txt"
done
echo "### CONTEXT SWEEP DONE ($RUN) ###"

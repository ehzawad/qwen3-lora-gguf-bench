#!/usr/bin/env bash
# E2/E3: synthetic finite-user (closed-loop, think-time) load via Locust against
# llama-server. NOT open-loop. E2 = stateless short-chat at a fixed deployment
# (-np 64). E3 = multi-turn growing-context (-np 20, 4096 tok/slot) measuring the
# repeated-prefill / context-memory cost. Windows are reduced from the council's
# full protocol to fit the session; labeled accordingly.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
BIN="$ROOT/vendor/llama.cpp/build/bin"
MODEL="$ROOT/models/gguf/qwen3-4b-legal-ops-Q6_K.gguf"
PORT=8199
RUN="${1:-$ROOT/results/locust/loc-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN"
python3 scripts/capture_env.py "$RUN/manifest.json" >/dev/null

free_port() { fuser -k "${PORT}/tcp" 2>/dev/null || true; sleep 3; }
wait_health() {
  for _ in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health)" = "200" ] && return 0
    sleep 1
  done; return 1
}
serve() {  # $1=np $2=ctx
  setsid "$BIN/llama-server" -m "$MODEL" --alias qwen3-4b-legal-q6k \
    --host 127.0.0.1 --port "$PORT" -dev CUDA0 -sm none -ngl all -fa on \
    -np "$1" --ctx-size "$2" --no-kv-unified -cb -b 2048 -ub 512 \
    -ctk f16 -ctv f16 --cache-ram 0 --no-cache-prompt --no-context-shift \
    -rea off --jinja --no-webui --metrics --slots > "$RUN/server-$3.log" 2>&1 &
  echo $!
}

### E2: stateless short-chat, fixed -np 64 ###
echo "### E2 serve (-np 64) $(date -u +%H:%M:%S) ###"
P=$(serve 64 49152 e2); wait_health || { echo "E2 server not ready"; }
for U in 20 60; do
  echo "### E2 users=$U ###"
  LOCUST_WARMUP_S=30 LOCUST_MEASURE_S=120 LOCUST_MAX_TOKENS=160 \
  LOCUST_TOKEN_JSON="$RUN/short-u${U}.json" \
  python3 -m locust -f scripts/locustfile.py ShortChatUser --headless \
    -u "$U" -r 20 -t 200s -s 60 --host "http://127.0.0.1:$PORT" \
    --csv "$RUN/short-u${U}" --html "$RUN/short-u${U}.html" --only-summary \
    > "$RUN/locust-short-u${U}.log" 2>&1 || echo "E2 u$U failed"
done
kill "$P" 2>/dev/null; free_port

### E3: multi-turn growing context, -np 20 @ 4096 tok/slot ###
echo "### E3 serve (-np 20, 4096/slot) $(date -u +%H:%M:%S) ###"
P=$(serve 20 81920 e3); wait_health || { echo "E3 server not ready"; }
echo "### E3 users=20 (multi-turn) ###"
LOCUST_WARMUP_S=60 LOCUST_MEASURE_S=180 LOCUST_MAX_TOKENS=160 LOCUST_MAX_TURNS=8 \
LOCUST_TOKEN_JSON="$RUN/convo-u20.json" \
python3 -m locust -f scripts/locustfile.py ConversationUser --headless \
  -u 20 -r 4 -t 300s -s 120 --host "http://127.0.0.1:$PORT" \
  --csv "$RUN/convo-u20" --html "$RUN/convo-u20.html" --only-summary \
  > "$RUN/locust-convo-u20.log" 2>&1 || echo "E3 failed"
kill "$P" 2>/dev/null; free_port

echo "### LOCUST DONE ($RUN) ###"

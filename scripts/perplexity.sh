#!/usr/bin/env bash
# E7: accuracy vs speed. Run llama-perplexity on wikitext-2-raw test for each
# weight quant of the SAME merged model. Report absolute PPL and Δ vs the bf16
# baseline (quantization-induced degradation) -- the RELATIVE Δ isolates quant
# quality since the legal-ops LoRA is baked into every quant equally.
# NOTE: wikitext PPL is a general-LM proxy, NOT domain/task accuracy.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
BIN="$ROOT/vendor/llama.cpp/build/bin/llama-perplexity"
G="$ROOT/models/gguf"
CORPUS="$ROOT/data/wikitext-2-raw-test.txt"
RUN="${1:-$ROOT/results/accuracy-$(date -u +%Y%m%dT%H%M%SZ)}"
CHUNKS="${CHUNKS:-200}"   # 512-token chunks; identical for every quant
mkdir -p "$RUN"
python3 scripts/capture_env.py "$RUN/manifest.json" >/dev/null

[ -f "$CORPUS" ] || { echo "missing corpus $CORPUS (run scripts/prepare_data.py)"; exit 1; }

: > "$RUN/ppl.jsonl"
for t in bf16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  f="$G/qwen3-4b-legal-ops-$t.gguf"
  [ -f "$f" ] || { echo "skip $t (no gguf)"; continue; }
  echo "### perplexity $t  $(date -u +%H:%M:%S) ###"
  "$BIN" -m "$f" -f "$CORPUS" -c 512 --chunks "$CHUNKS" -ngl 99 -fa on \
    > "$RUN/ppl-$t.log" 2>&1
  ppl=$(grep -oE "Final estimate: PPL = [0-9.]+" "$RUN/ppl-$t.log" | grep -oE "[0-9.]+$" | tail -1)
  echo "  $t: PPL = ${ppl:-FAILED}"
  python3 -c "import json;print(json.dumps({'quant':'$t','ppl':float('${ppl:-nan}') if '${ppl:-}' else None}))" >> "$RUN/ppl.jsonl"
done

echo "### accuracy vs speed summary ###"
python3 -c "
import json
rows=[json.loads(l) for l in open('$RUN/ppl.jsonl') if l.strip()]
d={r['quant']:r['ppl'] for r in rows if r.get('ppl')==r.get('ppl')}
base=d.get('bf16')
# decode tok/s from E4 (llama-bench, f16 KV, fa on)
speed={'Q4_K_M':168.0,'Q5_K_M':155.2,'Q6_K':133.3,'Q8_0':125.4,'bf16':77.0}
size={'Q4_K_M':2.32,'Q5_K_M':2.69,'Q6_K':3.07,'Q8_0':3.98,'bf16':7.49}
print(f\"{'quant':<8}{'GiB':>6}{'PPL':>9}{'dPPL vs bf16':>14}{'decode tok/s':>14}\")
out=[]
for q in ['Q4_K_M','Q5_K_M','Q6_K','Q8_0','bf16']:
    p=d.get(q); dp=(p-base) if (p and base) else None
    print(f\"{q:<8}{size[q]:>6.2f}{(p if p else float('nan')):>9.4f}{('%+.4f'%dp if dp is not None else 'n/a'):>14}{speed[q]:>14.1f}\")
    out.append({'quant':q,'gib':size[q],'ppl':p,'dppl_vs_bf16':dp,'decode_tok_s':speed[q]})
json.dump(out, open('$RUN/accuracy_speed.json','w'), indent=2)
"
echo "### ACCURACY DONE ($RUN) ###"

export const meta = {
  name: 'analyze-llm-inference',
  description: 'Analyze all A5000 LLM-inference experiments (concurrency, precision, engines, LoRA/SVD, Locust) into one fair comparative report',
  phases: [
    { title: 'Analyze', detail: 'parallel deep-dives: concurrency, precision+engines, lora+svd, locust' },
    { title: 'Synthesize', detail: 'reconcile into a fair, comprehensive report draft' },
  ],
}

const R = '/mnt/sdb/arafat/ehz/llm/qwen3-lora-gguf-bench/results'
const d = (args && args.dirs) ? args.dirs : {}
const workA = d.workA || `${R}/a5000-20260720T162318Z`
const sweep = d.sweep || ''      // resolved by the agent via glob if empty
const precision = d.precision || ''
const engines = d.engines || ''
const locust = d.locust || ''
const extraction = `${R}/extraction`

const FINDINGS = {
  type: 'object', additionalProperties: false,
  required: ['headline', 'findings', 'caveats'],
  properties: {
    headline: { type: 'string' },
    findings: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['claim', 'evidence'],
      properties: { claim: { type: 'string' }, evidence: { type: 'string' },
                    numbers: { type: 'string' } } } },
    caveats: { type: 'array', items: { type: 'string' } },
    table_markdown: { type: 'string' },
  },
}

phase('Analyze')
const [concurrency, precEngines, loraSvd, loadtest] = await parallel([
  () => agent(
    `Read the JSON result files (use Bash/Read; ls the dir first) in the E1 concurrency sweep ` +
    `dir (glob ${R}/sweep-*) and Work-A run ${workA}. These are a single RTX A5000 (24GB), ` +
    `Qwen3-4B Q6_K on llama.cpp, closed-loop barrier benchmark (256 prompt / 256 gen tok, ` +
    `ignore_eos). Produce the CONCURRENCY-CAPACITY story (this is the crux): aggregate output ` +
    `tok/s vs concurrency C={1,2,4,8,16,24,32,48,64,96,128}; where throughput PEAKS and where it ` +
    `DECLINES; fair-share throughput (agg/C) and latency growth; VRAM per configured slot (fit ` +
    `MiB vs slots, report slope+intercept — KV is preallocated so it's per-slot not per-live-request); ` +
    `and the bottleneck evidence: GPU util and power FALL while server-process CPU RISES as C grows ` +
    `(cite the cpu_server_proc_pct and gpu_util_pct numbers). Be precise and quote numbers. ` +
    `Do NOT claim 'knee at exactly 30'; report where it actually plateaus/peaks. Note nvidia-smi ` +
    `utilization.memory is a controller-busy proxy, not GB/s.`,
    { label: 'concurrency-capacity', phase: 'Analyze', schema: FINDINGS }),
  () => agent(
    `Read result JSONs in the E4 precision sweep (glob ${R}/precision-*: e4a_quant_fa.json, ` +
    `e4b_kv_dtype.jsonl, e4a_summary.md) and the E5 engine comparison (glob ${R}/engines/eng-*). ` +
    `E4 is llama-bench raw decode(tg)/prefill(pp) tok/s across weight quant {Q4_K_M,Q5_K_M,Q6_K,Q8_0,bf16}, ` +
    `KV-cache dtype {f16,bf16,q8_0}, flash-attn on/off, single-stream. E5 is llama.cpp Q6_K vs vLLM ` +
    `bf16 (same merged Qwen3-4B, fixed deployment, client C=1/30/100, same closed-loop harness). ` +
    `Findings: (1) does lower-bit weight quant speed up DECODE (bandwidth-bound) and is PREFILL flat ` +
    `(compute-bound)? quote tok/s. (2) KV dtype speed vs memory tradeoff. (3) llama.cpp vs vLLM ` +
    `throughput/latency at each C — expect vLLM (PagedAttention + native continuous batching) to win ` +
    `big at high concurrency. FAIRNESS CAVEATS: precision differs (Q6_K 6-bit vs bf16 16-bit); ` +
    `fixed-deployment (E5) vs per-C provisioning (E1); label them. Quote numbers.`,
    { label: 'precision-engines', phase: 'Analyze', schema: FINDINGS }),
  () => agent(
    `Read ${extraction}/ (control_merged_minus_instruct.json, instruct_minus_base.json) and Work-A ` +
    `merge/quant provenance in ${workA} and the repo README/NOTICE. Summarize: (1) the LoRA->merge->` +
    `GGUF Q6_K pipeline correctness (merge bit-exact base+(alpha/r)BA; Q6_K 3.3GB 6.56bpw). (2) The ` +
    `SVD LoRA-extraction study: control (known rank-16 legal-ops delta: embed/norm delta ~0, energy-` +
    `entropy effective rank ~43/2560) vs the Instruct-2507-minus-Base delta (high-rank: rank-16 SVD ` +
    `retains only ~3.55% of the 252 target-linear Frobenius energy; MLP higher-rank than attention; ` +
    `embeddings shifted ~20%). CRITICAL WORDING (per prior council review): call it an INTER-CHECKPOINT ` +
    `delta NOT a proven FullFT delta (Instruct-2507 declares no base_model); say any rank-r SVD is ` +
    `'LoRA-REPRESENTABLE' not 'a working extracted LoRA'; the high-rank delta does NOT refute ` +
    `'LoRA can match FullFT' (Thinking Machines: a TRAINED low-rank adapter stores task info, not the ` +
    `full weight delta; needs all layers esp MLP + adequate capacity). Quote numbers.`,
    { label: 'lora-svd', phase: 'Analyze', schema: FINDINGS }),
  () => agent(
    `Read the Locust E2/E3 results (glob ${R}/locust/*): tokens-*.json / *.json token-window files, ` +
    `*_stats.csv, locust-*.log. E2 = ShortChatUser finite-user think-time load (fixed -np 64 server, ` +
    `user levels ~20/60). E3 = ConversationUser multi-turn growing-context (fixed -np 20, 4096 tok/slot). ` +
    `Report: steady-state completion tokens/s, TTFT and end-to-end p50/p95 per user level; for E3 the ` +
    `effect of growing conversation history (repeated prefill) on TTFT/latency. CRITICAL LABELING ` +
    `(per prior council): this is a CLOSED-loop finite-user think-time workload, NOT open-loop and NOT ` +
    `real production traffic; user count is an UPPER BOUND on in-flight concurrency (start RPS ~= ` +
    `users/(response+think)). Do NOT compare its tok/s head-to-head with the E1 saturation numbers. ` +
    `If files are missing/empty, say so. Quote numbers.`,
    { label: 'locust', phase: 'Analyze', schema: FINDINGS }),
])

phase('Synthesize')
const report = await agent(
  `Write a COMPREHENSIVE, teacher/mentor-style report titled "LLM Inference on a Single RTX A5000: ` +
  `A Field Guide" as GitHub-flavored markdown. Audience: a smart learner who wants the FULL PICTURE. ` +
  `Base it strictly on these four reconciled analyses (JSON). Requirements: ` +
  `(a) an intro on the setup (Qwen3-4B + legal-ops LoRA merged -> Q6_K GGUF, one A5000 24GB); ` +
  `(b) sections: Quantization & GGUF; LoRA (+ the "is a LoRA just matmul / SVD extraction" study, ` +
  `honestly labeled); Concurrency capacity (THE core section — throughput vs C, the peak-then-decline, ` +
  `VRAM per configured slot, and the CPU-rises-while-GPU-falls host-bound evidence); Precision/` +
  `hyperparameter knobs; Engine choice (llama.cpp vs vLLM) with fairness caveats; Real-world finite-` +
  `user load (Locust, correctly labeled closed-loop); (c) a "when to use what" decision guide; ` +
  `(d) a prominent LIMITATIONS/caveats section (single A5000, hardware-specific, util.memory is a ` +
  `proxy not GB/s, closed-vs-open-loop not equivalent, precision differences in the engine comparison, ` +
  `SVD study measures an inter-checkpoint delta). Be fair, quantitative, and honest; never present ` +
  `closed-loop and finite-user numbers as directly comparable. Use tables. Output ONLY the markdown.` +
  `\n\nCONCURRENCY: ${JSON.stringify(concurrency)}\n\nPRECISION+ENGINES: ${JSON.stringify(precEngines)}` +
  `\n\nLORA+SVD: ${JSON.stringify(loraSvd)}\n\nLOCUST: ${JSON.stringify(loadtest)}`,
  { label: 'synthesize-report', phase: 'Synthesize' })

return { concurrency, precEngines, loraSvd, loadtest, report }

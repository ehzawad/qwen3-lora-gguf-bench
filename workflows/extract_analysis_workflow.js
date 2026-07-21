export const meta = {
  name: 'analyze-qwen3-bench',
  description: 'Adversarially audit + analyze + synthesize the Qwen3-4B Q6_K llama.cpp concurrency benchmark results',
  phases: [
    { title: 'Analyze', detail: 'parallel: number-audit, bottleneck classification, scaling analysis' },
    { title: 'Synthesize', detail: 'reconcile into README interpretation prose' },
  ],
}

const RUN = (args && args.runDir) ? args.runDir
  : '/mnt/sdb/arafat/ehz/llm/qwen3-lora-gguf-bench'

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['trustworthy', 'checks', 'issues'],
  properties: {
    trustworthy: { type: 'boolean' },
    checks: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['name', 'pass', 'detail'],
        properties: {
          name: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' },
        },
      },
    },
    issues: { type: 'array', items: { type: 'string' } },
  },
}

const BOTTLENECK_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['classification', 'confidence', 'evidence', 'honest_caveat'],
  properties: {
    classification: { type: 'string', description: 'e.g. decode memory-bandwidth-bound (inferred), compute-bound, slot/queue-bound, KV-VRAM-bound' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    evidence: { type: 'array', items: { type: 'string' } },
    per_concurrency: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['concurrency', 'note'],
        properties: { concurrency: { type: 'integer' }, note: { type: 'string' } },
      },
    },
    honest_caveat: { type: 'string' },
  },
}

const SCALING_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['summary', 'points', 'knee'],
  properties: {
    summary: { type: 'string' },
    points: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['concurrency', 'agg_tok_s', 'speedup_vs_c1', 'per_stream_tok_s', 'ttft_p50_s', 'latency_p50_s'],
        properties: {
          concurrency: { type: 'integer' }, agg_tok_s: { type: 'number' },
          speedup_vs_c1: { type: 'number' }, per_stream_tok_s: { type: 'number' },
          ttft_p50_s: { type: 'number' }, latency_p50_s: { type: 'number' },
        },
      },
    },
    knee: { type: 'string' },
  },
}

phase('Analyze')
const [audit, bottleneck, scaling] = await parallel([
  () => agent(
    `You are an adversarial metrics auditor. Read the benchmark JSON files in ${RUN} ` +
    `(benchmark-c001.json, benchmark-c030.json, benchmark-c100.json) with the Read or Bash tool. ` +
    `Verify these anti-"lying-number" checks and report pass/fail with the actual values: ` +
    `(1) every run has requests_failed==0; ` +
    `(2) completion_tokens_total == concurrency * measured_per_worker * 256 (exactly 256/req); ` +
    `(3) server_predicted_tokens_delta == completion_tokens_total (server counter cross-check); ` +
    `(4) throughput uses makespan not summed per-request rates (output_tokens_per_min ~= output_tokens_per_s*60 and ~= completion_tokens_total*60/makespan_s); ` +
    `(5) prompt_tokens_example is plausible (~200-260) and consistent; ` +
    `(6) server_startup_line shows kv_unified='false' and n_ctx_slot=768. ` +
    `Set trustworthy=false if any critical check fails. Be skeptical and precise.`,
    { label: 'number-audit', phase: 'Analyze', schema: AUDIT_SCHEMA }),
  () => agent(
    `You are a GPU performance analyst. Read ${RUN}/benchmark-c0*.json and ${RUN}/telemetry-c0*.csv ` +
    `(and ${RUN}/summary.json if present) with Read/Bash. The GPU is a single RTX A5000 (Ampere GA102, ` +
    `CC 8.6, ~768 GB/s, 230 W, 24 GB), serving a 3.3 GB Q6_K Qwen3-4B via llama.cpp with f16 KV, ` +
    `--no-kv-unified, 768 tokens/slot. For short-chat (256 prompt + 256 gen) at C=1/30/100, classify the ` +
    `dominant bottleneck at each concurrency using: gpu_util median, mem-controller-util proxy median, ` +
    `power median vs 230W limit, sm_clock, VRAM peak vs 24GB, and requests_failed. ` +
    `Remember: nvidia-smi utilization.memory is a controller-busy PROXY, not GB/s; do NOT claim ` +
    `"memory-bandwidth-bound" as proven — mark it inferred. Note the llama.cpp MMQ->cuBLAS quantized-matmul ` +
    `crossover at active batch 64 (C=30 uses MMQ, C=100 uses the dequantize+cuBLAS path). ` +
    `Give evidence-backed, honestly-hedged conclusions.`,
    { label: 'bottleneck', phase: 'Analyze', schema: BOTTLENECK_SCHEMA }),
  () => agent(
    `You are a throughput-scaling analyst. Read ${RUN}/summary.json (or the benchmark-c0*.json files) with ` +
    `Read/Bash. Build the scaling story from C=1 -> 30 -> 100: aggregate output tok/s, speedup vs C=1, ` +
    `per-stream tok/s (agg/concurrency), and how TTFT p50 and latency p50 grow with concurrency. ` +
    `Identify the knee (where aggregate throughput stops scaling and latency dominates). ` +
    `Report exact numbers from the files.`,
    { label: 'scaling', phase: 'Analyze', schema: SCALING_SCHEMA }),
])

phase('Synthesize')
const synthesis = await agent(
  `Write the "Interpretation & bottlenecks" prose for a benchmark README (GitHub markdown, 2-4 short ` +
  `paragraphs, NO table — the table is generated separately). Base it strictly on these reconciled analyses ` +
  `and the numbers in ${RUN}/summary.md (read it). Be honest and precise: report the single-stream vs ` +
  `aggregate throughput, where concurrency stops paying off, the TTFT/latency tradeoff, and the VRAM/CPU ` +
  `situation (weights ~3.3GB + f16 KV; C=100 fits comfortably in 24GB so it is NOT KV-VRAM-bound). ` +
  `State the decode bottleneck as an INFERENCE (util.memory is a proxy, not GB/s; a profiler would be ` +
  `needed to prove bandwidth saturation). Mention the MMQ->cuBLAS batch-64 crossover as context. ` +
  `\n\nAUDIT: ${JSON.stringify(audit)}\n\nBOTTLENECK: ${JSON.stringify(bottleneck)}\n\nSCALING: ${JSON.stringify(scaling)}`,
  { label: 'synthesize', phase: 'Synthesize' })

return { audit, bottleneck, scaling, synthesis }

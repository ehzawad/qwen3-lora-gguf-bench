export const meta = {
  name: 'analyze-context-accuracy-chatbot',
  description: 'Analyze the context-window (E6), accuracy/perplexity (E7), and concurrent multi-turn chatbot experiments into a fair report section',
  phases: [{ title: 'Analyze' }, { title: 'Synthesize' }],
}
const R = '/mnt/sdb/arafat/ehz/llm/qwen3-lora-gguf-bench/results'
const d = (args && args.dirs) || {}
const F = {
  type: 'object', additionalProperties: false, required: ['headline', 'findings', 'caveats'],
  properties: {
    headline: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['claim', 'evidence'], properties: { claim: { type: 'string' },
      evidence: { type: 'string' }, numbers: { type: 'string' } } } },
    caveats: { type: 'array', items: { type: 'string' } },
  },
}
phase('Analyze')
const [ctx, acc, chat] = await parallel([
  () => agent(
    `Read ${d.context || R + '/context-*'} (depth_summary.md, depth_f16.json, depth_q8.json, prefill.json, frontier.txt). ` +
    `Single RTX A5000 24 GB, Qwen3-4B Q6_K, llama.cpp. Report the CONTEXT-WINDOW impact: ` +
    `(1) decode & prefill tok/s vs KV depth (0/8k/32k) — how deep context slows every token; ` +
    `(2) the measured VRAM fit/no-fit FRONTIER: max concurrent slots at 8k/16k/32k for f16 vs q8_0 KV ` +
    `(report as brackets, e.g. 16 fit / 20 OOM). Note q8_0 KV = 53% of f16 (76.5 KiB/token), not 50%. ` +
    `(3) The answer to "can I serve 30 or 100 users at 8k/16k/32k": f16 caps ~16/8/4; q8_0 lets 30 fit ONLY at 8k (edge, 23 GB); ` +
    `16k/32k @30 and 100 @ anything are infeasible. Frame '30 users' = 30 KV-resident dedicated slots (queueing/offload can serve more connected users). ` +
    `This is a VRAM-bound regime, opposite the 768-ctx short-chat result. Quote exact numbers; flag any overreach.`,
    { label: 'context', phase: 'Analyze', schema: F }),
  () => agent(
    `Read ${d.accuracy || R + '/accuracy-full-*'} (accuracy_speed.json, ppl.jsonl). Full wikitext-2-raw test (583 chunks), ` +
    `llama-perplexity legacy 512-ctx method, per quant of the SAME merged model. Report the speed↔accuracy tradeoff: ` +
    `absolute PPL, ΔPPL vs the bf16 GGUF baseline, relative % change, paired with E4 decode tok/s (Q4_K_M 168 ... bf16 77). ` +
    `Findings: Q6_K and Q8_0 are within-noise of bf16 (lossless); Q5_K_M +~1.5%; Q4_K_M +~2.4% PPL for 2.18x decode. ` +
    `MANDATORY honest wording: this is speed vs general-domain (wikitext) next-token likelihood, NOT legal-task accuracy; ` +
    `the bf16 GGUF is a same-precision reference, not lossless ground truth; do NOT say 'Q4 loses X% quality' or 'adapter preserved'. ` +
    `Quote exact numbers.`,
    { label: 'accuracy', phase: 'Analyze', schema: F }),
  () => agent(
    `Read ${d.ctxchat || R + '/ctxchat-*'} (concurrent-*.json, single-*.json) and ${d.chatbot || R + '/chatbot-*'} (replay-cache-on.json, replay-cache-off.json). ` +
    `These are multi-turn CHATBOT tests on real (synthetic) ultrachat transcripts + composite sessions. Report: ` +
    `(1) single-user multi-turn decode/TTFT at 8k/16k/32k (decode 81->59->29 tok/s; TTFT jumps to ~9s at 32k first turn); ` +
    `(2) THE concurrent test: 30 users each at 8k with q8_0 KV FIT at 23 GB but per-user throughput ~3.8 tok/s (aggregate 113); ` +
    `16 users @8k f16 -> 8.4 tok/s/user; so 30 simultaneous 8k chat users is feasible-but-slow, 100 impossible. ` +
    `(3) PREFIX CACHING (the key multi-turn lever): cache ON -> hit 0.32->0.95 as history grows, TTFT DROPS 137->75 ms; ` +
    `cache OFF -> hit 0, TTFT GROWS 205->505 ms. Label honestly: teacher-forced replay, composite sessions from synthetic ultrachat, ` +
    `'30 users' = 30 resident slots; these do NOT estimate production-arrival latency. Quote exact numbers.`,
    { label: 'chatbot', phase: 'Analyze', schema: F }),
])
phase('Synthesize')
const section = await agent(
  `Write a REPORT.md section titled "## 10. Context window, concurrency, and the accuracy↔speed tradeoff" (GitHub markdown, ` +
  `mentor tone, tables). Cover, from these three reconciled analyses only: (A) context window is the VRAM constraint — the ` +
  `144 KiB/token KV math, the measured fit frontier (f16 16/8/4, q8_0 30-at-8k-only), and the direct answer that 30 concurrent ` +
  `multi-turn users need q8_0 KV at 8k or f16 at <=4k, while 16k/32k@30 and 100@anything don't fit — the opposite of the ` +
  `short-chat finding; (B) deep context slows decode (81->29 tok/s 8k->32k) and per-user throughput at 30x8k is only ~3.8 tok/s; ` +
  `(C) prefix caching flips the TTFT trend (drops with depth when on, grows when off) — the multi-turn must-have; ` +
  `(D) the MEASURED accuracy↔speed table (Q6_K/Q8_0 lossless, Q4 +2.4% PPL for 2.18x decode) with the 'wikitext not task accuracy' caveat. ` +
  `End with a short 'which LoRA' note: narcolepticchicken/qwen3-4b-legal-ops-contract-intake-lora (r16 a32), merged. ` +
  `Be honest and quantitative; label composite sessions and teacher-forced replay; '30 users'=30 resident slots. Output ONLY the markdown.` +
  `\n\nCONTEXT: ${JSON.stringify(ctx)}\n\nACCURACY: ${JSON.stringify(acc)}\n\nCHATBOT: ${JSON.stringify(chat)}`,
  { label: 'synthesize', phase: 'Synthesize' })
return { ctx, acc, chat, section }

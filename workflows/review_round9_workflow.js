export const meta = {
  name: 'review-round9',
  description: 'Review the new §9 (context/accuracy/chatbot) report section + context.html in a skeptical/editorial lens and produce a prioritized fix list',
  phases: [{ title: 'Review' }, { title: 'Synthesize' }],
}
const ROOT = '/mnt/sdb/arafat/ehz/llm/qwen3-lora-gguf-bench'
const F = {
  type: 'object', additionalProperties: false, required: ['lens', 'findings'],
  properties: {
    lens: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['severity', 'where', 'issue', 'fix'],
      properties: {
        severity: { type: 'string', enum: ['must-fix', 'should-fix', 'nice-to-have', 'cut'] },
        where: { type: 'string' }, issue: { type: 'string' }, fix: { type: 'string' } } } },
  },
}
phase('Review')
const [honesty, editor] = await parallel([
  () => agent(
    `You are a SKEPTICAL honesty/overclaim auditor. Read ${ROOT}/REPORT.md — especially the NEW "## 9. Context window..." ` +
    `section and "## 10. LIMITATIONS" and the one-paragraph summary — and ${ROOT}/report/context.html. ` +
    `The rig: single RTX A5000 24GB, Qwen3-4B Q6_K, llama.cpp, merged legal LoRA r16/a32. ` +
    `Your job: catch anything that OVERCLAIMS or is unsupported by measured data. Specifically hunt for: ` +
    `(1) any claim stated as fact that the data only weakly supports; (2) a number quoted without its necessary caveat ` +
    `(error bar, allocation-only frontier, single-stream vs per-slot, teacher-forced/composite, closed-loop TTFT); ` +
    `(3) INTERNAL CONTRADICTIONS — e.g. a place elsewhere in REPORT.md that still says 'we measured only speed' or contradicts §9; ` +
    `(4) any wording that implies legal-task accuracy from wikitext perplexity, or 'lossless ground truth' from a same-precision reference; ` +
    `(5) the vLLM comparison must remain VOID — flag any place it is treated as done. ` +
    `Be concrete: cite the exact sentence. Do NOT invent problems; if a section is honest, say so with few/zero findings. ` +
    `Return findings with severity must-fix/should-fix/nice-to-have/cut.`,
    { label: 'honesty-lens', phase: 'Review', schema: F }),
  () => agent(
    `You are a ruthless EDITOR reviewing for redundancy, clarity, and what to CUT. Read ${ROOT}/REPORT.md (all sections) and ` +
    `list ${ROOT}/report/ (index.html, concurrency.html, context.html, precision.html, engines.html, lora.html, locust.html). ` +
    `The report just grew a large §9 plus a new context.html page. Your job: (1) find DUPLICATION between §9 in REPORT.md and ` +
    `report/context.html that is fine to keep vs redundant; (2) flag anything now STALE elsewhere given §9 exists ` +
    `(other sections/pages that predate the accuracy+context measurements and should point to §9 or be softened); ` +
    `(3) identify filler/ceremony or over-long passages that could be tightened WITHOUT losing a measured fact or caveat; ` +
    `(4) check nav/cross-links are consistent across all 7 HTML pages (each should link context.html); ` +
    `(5) note anything genuinely MISSING a reader would expect. Prefer a few high-value cuts over nitpicks. ` +
    `Return findings with severity must-fix/should-fix/nice-to-have/cut. Reward the report for being honest; don't manufacture work.`,
    { label: 'editor-lens', phase: 'Review', schema: F }),
])
phase('Synthesize')
const plan = await agent(
  `Two reviewers examined a benchmark report's new context-window/accuracy section. Merge their findings into ONE ` +
  `de-duplicated, PRIORITIZED action list. Drop anything that is a matter of taste or that would add unmeasured claims. ` +
  `Keep only changes that make the report more correct, more honest, or materially tighter. For each kept item give: ` +
  `severity, where, the concrete edit to make, and one line of why. Put must-fix first. If a reviewer finding is wrong ` +
  `or not worth doing, say so explicitly in a short 'rejected' list with the reason.` +
  `\n\nHONESTY LENS: ${JSON.stringify(honesty)}\n\nEDITOR LENS: ${JSON.stringify(editor)}`,
  { label: 'synthesize', phase: 'Synthesize' })
return { honesty, editor, plan }

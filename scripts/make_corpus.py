#!/usr/bin/env python3
"""Generate a deterministic, seeded short-chat prompt corpus (~256 tokens each).

Domain matches the adapter (legal contract-intake ops) and is benign. Each
prompt carries a unique nonce so prompt caching cannot silently dedupe them.
Committed to prompts/short-chat.jsonl for reproducibility.

Usage: make_corpus.py <out_jsonl> <count>
"""
import json
import sys

out_path = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 2400

# A pool of benign, domain-flavored sentences. Deterministic selection by index.
SENTENCES = [
    "Review the following contract passage and identify the controlling clause family.",
    "The vendor agrees to maintain production content within the specified jurisdictions.",
    "Payment terms are net thirty days from the date of the accepted invoice.",
    "Any escrow arrangement must name a neutral third-party agent acceptable to both sides.",
    "Data residency obligations require storage and processing to remain within named regions.",
    "The parties acknowledge that confidential information survives termination of the agreement.",
    "Indemnification is limited to direct damages and excludes consequential losses.",
    "Renewal is automatic for successive twelve-month terms unless notice is given in writing.",
    "Termination for convenience requires sixty days of advance written notice to the counterparty.",
    "Service levels are measured monthly and reported through the standard operations dashboard.",
    "Intellectual property created under this statement of work vests in the commissioning party.",
    "Warranties are provided on an as-is basis except where local law prohibits such exclusion.",
    "The governing law is that of the stated jurisdiction without regard to conflict rules.",
    "Assignment of the agreement requires the prior written consent of the non-assigning party.",
    "Force majeure suspends performance for events beyond the reasonable control of a party.",
    "Audit rights permit one inspection per year upon reasonable advance written notice.",
    "Limitation of liability caps aggregate damages at the fees paid in the prior twelve months.",
    "The processor shall implement appropriate technical and organizational security measures.",
    "Change orders must be documented and signed before any additional work is undertaken.",
    "Dispute resolution proceeds first through good-faith negotiation and then binding arbitration.",
]

CLAUSE_CHOICES = (
    "Choose exactly one clause family: data_residency, escrow, payment_terms, "
    "indemnification, termination, confidentiality, liability_cap, audit_rights."
)


def build_prompt(i: int) -> str:
    # Deterministic subset (~11 sentences ~= 256 tokens after templating).
    n = len(SENTENCES)
    picks = [SENTENCES[(i * 7 + k * 3) % n] for k in range(11)]
    body = " ".join(picks)
    nonce = f"[req-{i:06d}]"
    return (
        f"{nonce} You are an intake assistant for a contracts operations team. "
        f"{body} Based strictly on the passage above, decide the single best label. "
        f"{CLAUSE_CHOICES} Answer with the label and one short sentence of justification."
    )


with open(out_path, "w", encoding="utf-8") as f:
    for i in range(count):
        f.write(json.dumps({"id": i, "prompt": build_prompt(i)}) + "\n")

print(f"[corpus] wrote {count} prompts -> {out_path}")

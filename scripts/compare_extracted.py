#!/usr/bin/env python3
"""Apply SVD-extracted rank-r LoRAs (no training) to the base and measure how
close the result gets to the real fine-tune (Instruct), vs rank.

Configs compared against Instruct (ground truth):
  base (r=0), base+rank{16,64,256} extracted delta on the 7 target linears,
  base+full-linear delta (target linears = Instruct's; isolates the residual
  that lives in embeddings/norms, which a linear-only LoRA cannot capture).

Metrics per config (last-token, over eval prompts): logit cosine to Instruct,
top-1 token agreement, and a couple greedy sample generations.

Usage: compare_extracted.py <base_dir> <ft_dir> <out_json>
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

base_dir, ft_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
RANKS = [16, 64]
TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
device = torch.device("cuda:0")

PROMPTS = [
    "Explain what a LoRA adapter is in two sentences.",
    "Write a haiku about the ocean.",
    "What is 17 * 23? Answer with just the number.",
    "List three uses for a paperclip.",
    "Translate 'good morning' into French.",
    "Summarize why the sky is blue in one sentence.",
    "Give one tip for writing clean code.",
    "What is the capital of Japan?",
]

tok = AutoTokenizer.from_pretrained(ft_dir, use_fast=True)


def rendered(p):
    return tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True)


enc = [tok(rendered(p), return_tensors="pt", add_special_tokens=False).to(device)
       for p in PROMPTS]


def last_logits(model):
    outs = []
    for b in enc:
        with torch.inference_mode():
            lg = model(**b).logits[0, -1].float().cpu()
        outs.append(lg)
    return outs


def greedy(model, p, n=48):
    b = tok(rendered(p), return_tensors="pt", add_special_tokens=False).to(device)
    with torch.inference_mode():
        out = model.generate(**b, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, b["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# --- ground truth: Instruct ---
inst = AutoModelForCausalLM.from_pretrained(ft_dir, dtype=torch.bfloat16,
                                            local_files_only=True).to(device).eval()
ref = last_logits(inst)
ref_gen = {p: greedy(inst, p) for p in PROMPTS[:3]}
del inst
torch.cuda.empty_cache()

# --- base weights + SVD cache of the target-linear deltas ---
def build_index(md):
    md = Path(md)
    wm = json.loads((md / "model.safetensors.index.json").read_text())["weight_map"]
    return {n: md / f for n, f in wm.items()}


bidx, fidx = build_index(base_dir), build_index(ft_dir)
_cache = {}

def gt(idx, name):
    h = _cache.get(idx[name])
    if h is None:
        h = safe_open(str(idx[name]), framework="pt"); _cache[idx[name]] = h
    return h.get_tensor(name)


layer_ids = sorted({int(n.split(".")[2]) for n in bidx if n.startswith("model.layers.")})
target_names = [f"model.layers.{li}.{suf}.weight" for li in layer_ids for suf in TARGETS]

svd_cache = {}   # name -> (U256, S256, Vh256) on CPU
full_delta = {}  # name -> (Wf - Wb) small? no, store Wf for full-linear
print(f"[cmp] computing SVD cache for {len(target_names)} target linears ...", flush=True)
for i, name in enumerate(target_names):
    Wb = gt(bidx, name).to(device, torch.float32)
    Wf = gt(fidx, name).to(device, torch.float32)
    dW = Wf - Wb
    q = min(64, min(dW.shape))
    U, S, V = torch.svd_lowrank(dW, q=q, niter=2)   # top-q only: fast & enough
    svd_cache[name] = (U.cpu(), S.cpu(), V.cpu())   # U(m,q) S(q) V(n,q)
    del Wb, Wf, dW, U, S, V
    if (i + 1) % 40 == 0:
        print(f"[cmp]   svd cache {i+1}/{len(target_names)}", flush=True)


def delta_r(name, r):
    U, S, V = svd_cache[name]
    r = min(r, S.numel())
    return (U[:, :r] * S[:r]) @ V[:, :r].T      # (m x n) on CPU, fp32


# base full state dict on CPU (bf16)
base_sd = {}
for name in bidx:
    base_sd[name] = gt(bidx, name)


def eval_config(tag, mutate):
    sd = {k: v.clone() for k, v in base_sd.items()}
    mutate(sd)
    cfg = AutoConfig.from_pretrained(ft_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
    model.load_state_dict(sd, strict=False)   # lm_head is tied (absent in sd)
    model.tie_weights()
    model = model.to(device).eval()
    lg = last_logits(model)
    cos = sum(F.cosine_similarity(a, b, dim=0).item() for a, b in zip(lg, ref)) / len(ref)
    top1 = sum(int(a.argmax() == b.argmax()) for a, b in zip(lg, ref)) / len(ref)
    gens = {p: greedy(model, p) for p in PROMPTS[:3]}
    del model
    torch.cuda.empty_cache()
    return {"tag": tag, "logit_cosine_vs_instruct": cos,
            "top1_agreement_vs_instruct": top1, "sample_generations": gens}


results = []
# r=0 : plain base
results.append(eval_config("base (r=0)", lambda sd: None))
# rank-r extracted LoRA on target linears
for r in RANKS:
    def mut(sd, r=r):
        for name in target_names:
            sd[name] = (sd[name].float() + delta_r(name, r)).to(torch.bfloat16)
    results.append(eval_config(f"base + extracted rank {r}", mut))
# full-linear : target linears = Instruct's (captures ALL linear delta, no emb/norm)
def mut_full(sd):
    for name in target_names:
        sd[name] = gt(fidx, name)
results.append(eval_config("base + full linear delta", mut_full))

out = {"prompts": PROMPTS, "instruct_sample_generations": ref_gen, "configs": results}
Path(out_json).write_text(json.dumps(out, indent=2))
print("\n[cmp] logit cosine / top-1 agreement vs Instruct:")
for r in results:
    print(f"  {r['tag']:<28} cos={r['logit_cosine_vs_instruct']:.4f} "
          f"top1={r['top1_agreement_vs_instruct']:.3f}")
print(f"[cmp] wrote {out_json}")

#!/usr/bin/env python3
"""Merge the LoRA adapter into Qwen3-4B-Instruct-2507 (BF16), verified.

Reconciled with the codex council (merge-convert lens). Key decisions:
  * Use the OFFICIAL base tokenizer, never the adapter's tokenizer_config.json
    (its extra_special_tokens is a list -> crashes transformers 4.56.2).
  * Keep the adapter in FP32; PEFT computes B@A in FP32 then casts the delta to
    the base BF16 dtype. safe_merge=True validates finiteness.
  * Tied embeddings: do NOT resize embeddings, do NOT synthesize lm_head.
  * Prove the merge is real: adapter-on != adapter-off logits, and the merged
    model remains closer to adapter-on than adapter-off with the same argmax.

Usage: merge.py <base_dir> <adapter_dir> <out_dir>
"""
import hashlib
import sys
from pathlib import Path

if not __debug__:
    raise RuntimeError(
        "merge.py uses assertions as integrity gates; optimized Python disables "
        "them. Rerun without -O and unset PYTHONOPTIMIZE."
    )

import torch
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from peft.tuners.lora.layer import LoraLayer
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir = Path(sys.argv[1]).resolve()
adapter_dir = Path(sys.argv[2]).resolve()
out_dir = Path(sys.argv[3]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

expected_targets = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}
EXPECTED_TEMPLATE_SHA = (
    "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"
)

torch.cuda.set_device(0)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device("cuda:0")

adapter_cfg = PeftConfig.from_pretrained(adapter_dir, local_files_only=True)
assert adapter_cfg.r == 16
assert adapter_cfg.lora_alpha == 32
assert set(adapter_cfg.target_modules) == expected_targets
assert adapter_cfg.modules_to_save is None
assert adapter_cfg.bias == "none"

# Official base tokenizer only.
tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True, local_files_only=True)
assert tokenizer.vocab_size == 151643
assert len(tokenizer) == 151669
assert tokenizer.convert_tokens_to_ids("<|endoftext|>") == 151643
assert tokenizer.convert_tokens_to_ids("<|im_start|>") == 151644
assert tokenizer.convert_tokens_to_ids("<|im_end|>") == 151645

adapter_template = (adapter_dir / "chat_template.jinja").read_text(encoding="utf-8")
assert tokenizer.chat_template == adapter_template, "base template != adapter template"
assert hashlib.sha256(adapter_template.encode()).hexdigest() == EXPECTED_TEMPLATE_SHA

base = AutoModelForCausalLM.from_pretrained(
    base_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True,
    attn_implementation="eager",
)
assert base.config.architectures == ["Qwen3ForCausalLM"]
assert base.config.vocab_size == 151936
assert base.config.tie_word_embeddings is True
assert base.get_input_embeddings().weight.shape[0] == 151936
assert (base.get_input_embeddings().weight.data_ptr()
        == base.get_output_embeddings().weight.data_ptr())

base.to(device)
base.eval()

model = PeftModel.from_pretrained(
    base, adapter_dir, is_trainable=False, autocast_adapter_dtype=False,
)
model.eval()

# PEFT 0.19.1 creates the LoRA matrices in the base module's dtype (bf16) when
# autocast is disabled, so loading the FP32 adapter downcasts them. Force them
# back to FP32 so B@A is computed in FP32 before the single cast into BF16.
for _n, _p in model.named_parameters():
    if ".lora_A." in _n or ".lora_B." in _n:
        _p.data = _p.data.float()

lora_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer)]
assert len(lora_layers) == 36 * 7, f"expected 252 LoRA layers, got {len(lora_layers)}"
assert {n.rsplit(".", 1)[-1] for n, _ in lora_layers} == expected_targets

lora_params = [p for n, p in model.named_parameters()
               if ".lora_A." in n or ".lora_B." in n]
assert len(lora_params) == 504
assert {p.dtype for p in lora_params} == {torch.float32}

prompt = (
    "For intake coding, choose the clause family for this passage:\n"
    "production content remains in Ireland and Canada.\n"
    "Choose one: data_residency, escrow, payment_terms."
)
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
)
batch = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)


def last_logits(current_model):
    with torch.inference_mode():
        return current_model(
            **batch, use_cache=False, logits_to_keep=1,
        ).logits[0, -1].float().cpu()


with model.disable_adapter():
    logits_off = last_logits(model)
logits_on = last_logits(model)

adapter_effect = (logits_on - logits_off).abs().max().item()
assert adapter_effect > 0.0, "adapter-on and adapter-off logits are identical"

probe = model.base_model.model.model.layers[0].self_attn.q_proj
weight_before = probe.base_layer.weight.detach().clone()
expected_weight = weight_before.clone()
expected_weight += probe.get_delta_weight("default").detach().to(weight_before.dtype)

merged = model.merge_and_unload(safe_merge=True, progressbar=True)
# Guarantee a uniformly BF16 checkpoint (safe_merge may promote a merged
# tensor to FP32 during the add); re-tie the head after the cast.
merged = merged.to(torch.bfloat16)
merged.eval()
merged.tie_weights()

assert not any(isinstance(m, LoraLayer) for m in merged.modules())
assert not any("lora_" in n for n in merged.state_dict())

weight_after = merged.model.layers[0].self_attn.q_proj.weight.detach()
changed_elements = torch.count_nonzero(weight_after != weight_before).item()
assert changed_elements > 0, "merged q_proj identical to base weight"
# expected_weight = base + (FP32 delta cast to bf16). Allow bf16 rounding-order
# slack between our reference and PEFT's internal merge order.
max_abs_dev = (weight_after.float() - expected_weight.float()).abs().max().item()
assert max_abs_dev < 1e-2, f"merged weight far from base+delta: {max_abs_dev}"

assert merged.config.tie_word_embeddings is True
assert (merged.get_input_embeddings().weight.data_ptr()
        == merged.get_output_embeddings().weight.data_ptr())
assert {p.dtype for p in merged.parameters() if p.is_floating_point()} == {torch.bfloat16}

logits_merged = last_logits(merged)
cos_on = F.cosine_similarity(logits_on, logits_merged, dim=0).item()
cos_off = F.cosine_similarity(logits_off, logits_merged, dim=0).item()
cos_on_off = F.cosine_similarity(logits_on, logits_off, dim=0).item()
rel_l2 = (torch.linalg.vector_norm(logits_on - logits_merged)
          / torch.linalg.vector_norm(logits_on)).item()
top1_on = logits_on.argmax().item()
top1_merged = logits_merged.argmax().item()
top1_off = logits_off.argmax().item()

diag = {
    "adapter_effect_max_abs": adapter_effect,
    "probe_changed_elements": changed_elements,
    "probe_max_abs_dev": max_abs_dev,
    "cosine_merged_vs_on": cos_on,
    "cosine_merged_vs_off": cos_off,
    "cosine_on_vs_off": cos_on_off,
    "relative_l2_merged_vs_on": rel_l2,
    "argmax_on": top1_on, "argmax_merged": top1_merged, "argmax_off": top1_off,
}
print(diag)

# Meaningful correctness for a BF16 merge: the merged model must behave like
# adapter-ON (closer to ON than OFF) and preserve the ON top-1 token. Comparing
# a bf16 merged forward against the fp32-delta forward is not a tight bound.
assert cos_on > cos_off, f"merged closer to OFF than ON: on={cos_on} off={cos_off}"
assert cos_on > 0.95, f"merged vs ON cosine unexpectedly low: {cos_on}"
assert top1_merged == top1_on, f"top1 mismatch merged={top1_merged} on={top1_on}"

merged.to("cpu")
merged.tie_weights()
torch.cuda.empty_cache()

merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")
merged.generation_config.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

assert (out_dir / "chat_template.jinja").read_text(encoding="utf-8") == adapter_template

saved_keys = set()
for shard in out_dir.glob("*.safetensors"):
    with safe_open(str(shard), framework="pt", device="cpu") as h:
        saved_keys.update(h.keys())
assert "model.embed_tokens.weight" in saved_keys
assert "lm_head.weight" not in saved_keys
assert not any("lora_" in k for k in saved_keys)

print("[merge] OK ->", out_dir)

#!/usr/bin/env python3
"""Extract a LoRA from a weight delta by SVD -- NO training. Measures how
low-rank a full fine-tune's weight update actually is.

For each linear weight W in the 7 standard LoRA target modules, computes
  dW = W_ft - W_base
and its singular values, then reports the relative Frobenius reconstruction
error of the best rank-r approximation vs r. A trained rank-r LoRA is a rank-r
delta; this asks the inverse: how much of an *existing* full-FT delta a rank-r
LoRA can even represent.

Context (Thinking Machines, "LoRA Without Regret"): the rank needed to *learn*
a task (capacity vs dataset bits) is NOT the spectral rank of the full-FT delta
measured here -- a heavy post-train delta can be high-rank yet still be matchable
by a *trained* low-rank LoRA. This script measures the spectral side.

Usage: svd_extract.py <base_dir> <ft_dir> <out_json> [label]
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

base_dir = Path(sys.argv[1])
ft_dir = Path(sys.argv[2])
out_json = sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else "delta"

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
TARGET_SUFFIXES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
device = torch.device("cuda:0")


def build_index(model_dir):
    """name -> shard path, using the safetensors index (or single file)."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {n: model_dir / f for n, f in wm.items()}
    single = model_dir / "model.safetensors"
    handles = {}
    with safe_open(str(single), framework="pt") as h:
        for n in h.keys():
            handles[n] = single
    return handles


base_idx = build_index(base_dir)
ft_idx = build_index(ft_dir)
_open_cache = {}


def get_tensor(idx, name):
    shard = idx[name]
    h = _open_cache.get(shard)
    if h is None:
        h = safe_open(str(shard), framework="pt")
        _open_cache[shard] = h
    return h.get_tensor(name)


# Which layers exist?
layer_ids = sorted({int(n.split(".")[2]) for n in base_idx
                    if n.startswith("model.layers.")})

per_matrix = []
for li in layer_ids:
    for suf in TARGET_SUFFIXES:
        name = f"model.layers.{li}.{suf}.weight"
        if name not in base_idx or name not in ft_idx:
            continue
        Wb = get_tensor(base_idx, name).to(device, torch.float32)
        Wf = get_tensor(ft_idx, name).to(device, torch.float32)
        dW = Wf - Wb
        sv = torch.linalg.svdvals(dW)                    # sorted desc
        total = float((sv ** 2).sum())
        base_fro = float(torch.linalg.matrix_norm(Wb))
        d_fro = float(torch.linalg.matrix_norm(dW))
        # reconstruction error (relative Frobenius) at each rank
        err = {}
        csum = torch.cumsum(sv ** 2, 0)
        for r in RANKS:
            if r >= sv.numel():
                err[r] = 0.0
            else:
                err[r] = float(((total - csum[r - 1]) / total).clamp(min=0).sqrt())
        stable_rank = float((sv ** 2).sum() / (sv[0] ** 2))      # ||.||_F^2/sigma_max^2
        p = (sv ** 2) / total
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))  # entropy eff. rank
        mod = suf.split(".")[-1]
        per_matrix.append({
            "layer": li, "module": mod, "kind": "attn" if mod in ATTN else "mlp",
            "shape": list(dW.shape), "delta_fro": d_fro, "base_fro": base_fro,
            "rel_delta": d_fro / base_fro if base_fro else None,
            "sigma_max": float(sv[0]), "stable_rank": stable_rank,
            "eff_rank": eff_rank, "full_rank": int(sv.numel()), "recon_err": err,
        })
        del Wb, Wf, dW, sv


def agg(rows):
    """Frobenius-energy-weighted aggregate reconstruction error vs rank."""
    if not rows:
        return None
    w = [m["delta_fro"] ** 2 for m in rows]
    wsum = sum(w)
    out = {}
    for r in RANKS:
        num = sum(wi * (m["recon_err"][r] ** 2) for wi, m in zip(w, rows))
        out[r] = float((num / wsum) ** 0.5)
    return {
        "recon_err_weighted": out,
        "mean_stable_rank": sum(m["stable_rank"] for m in rows) / len(rows),
        "mean_eff_rank": sum(m["eff_rank"] for m in rows) / len(rows),
        "mean_rel_delta": sum(m["rel_delta"] for m in rows if m["rel_delta"]) / len(rows),
        "n_matrices": len(rows),
    }


# Non-target params that a linear-only LoRA cannot capture (embeddings, norms).
extra = {}
for name in ["model.embed_tokens.weight", "model.norm.weight"]:
    if name in base_idx and name in ft_idx:
        Wb = get_tensor(base_idx, name).to(device, torch.float32)
        Wf = get_tensor(ft_idx, name).to(device, torch.float32)
        extra[name] = {
            "rel_delta": float(torch.linalg.norm(Wf - Wb) / torch.linalg.norm(Wb)),
        }
        del Wb, Wf

by_kind = {
    "all": agg(per_matrix),
    "attn": agg([m for m in per_matrix if m["kind"] == "attn"]),
    "mlp": agg([m for m in per_matrix if m["kind"] == "mlp"]),
}
by_module = {mod: agg([m for m in per_matrix if m["module"] == mod])
             for mod in ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]}

result = {
    "label": label, "base_dir": str(base_dir), "ft_dir": str(ft_dir),
    "ranks": RANKS, "aggregate_by_kind": by_kind, "aggregate_by_module": by_module,
    "non_lora_params": extra, "per_matrix": per_matrix,
}
Path(out_json).write_text(json.dumps(result, indent=2))

# Console summary
print(f"[svd] {label}: {len(per_matrix)} target matrices")
print(f"  mean rel delta ||dW||/||W||: {by_kind['all']['mean_rel_delta']:.4f}")
print(f"  mean stable rank: {by_kind['all']['mean_stable_rank']:.1f} | "
      f"mean eff rank: {by_kind['all']['mean_eff_rank']:.1f} "
      f"(full dim ~{per_matrix[0]['full_rank']})")
print("  Frobenius-weighted reconstruction error vs rank (all / attn / mlp):")
for r in RANKS:
    a = by_kind["all"]["recon_err_weighted"][r]
    at = by_kind["attn"]["recon_err_weighted"][r]
    ml = by_kind["mlp"]["recon_err_weighted"][r]
    print(f"    r={r:>4}:  all={a:.3f}  attn={at:.3f}  mlp={ml:.3f}")
for n, v in extra.items():
    print(f"  non-LoRA {n}: rel delta {v['rel_delta']:.4f}")
print(f"[svd] wrote {out_json}")

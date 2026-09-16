#!/usr/bin/env python3
"""Merge an H3 Turbo LoRA into a bf16 transformer offline, writing a plain bf16 checkpoint.

    SRC=<hf-snapshot>/transformer_ref \
    LORA=<hf-snapshot>/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors \
    DST=/opt/dlami/nvme/vdn/ref2v_turbo_bf16 \
    /opt/dlami/nvme/sglang/.venv/bin/python lora_merge_h3.py

WHY MERGE AT ALL, i.e. why not just --lora-path. Only because of fp8. SGLang accepts `--lora-path`
on this model and does its own key mapping (see LAYOUTS below), but the dynamic-LoRA wrapper is not
quantization-aware on this build:

    AttributeError: 'RowParallelLinearWithLoRA' object has no attribute 'quant_method'

-- repeated across ranks, ending in "Server warmup failed; aborting startup". So `--quantization
fp8 --lora-path ...` does not start. A *merged* checkpoint is plain bf16 with no LoRA wrapper
anywhere, so SGLang quantizes it online exactly like base weights and the fp8 speed comes back.
Serve DST with `--model-variant hybrid --component-weights-paths.transformer $DST`, which is the
flag pair SGLang documents for merged weights (`runtime/pipelines/minimax_h3_pipeline.py:97`).

For *pictures only*, skip this script: bf16 `--lora-path` works today with zero mapping work. Its
latency is not comparable to any fp8 number in this repo.

LAYOUTS. There are two naming conventions in play and a merge is only trivial when both sides use
the same one. This script requires that and refuses otherwise, because each mismatch fails silently
in a way shapes alone do not catch.

    diffusers   transformer_blocks.N.attn.to_{q,k,v}   attn.to_out.0
                ff.net.0.proj (fc1, halves [value, gate])   ff.net.2 (fc2)   norm_out.linear
                index file: diffusion_pytorch_model.safetensors.index.json   638 tensors
    native      blocks.N.attn.qkv_proj (per-head q,k,v row interleave)   attn.out_proj
                mlp.fc1 (halves [gate, value])   mlp.fc2   adaln_proj.linear   final_layer
                index file: model.safetensors.index.json                       535 tensors

`MiniMaxAI/MiniMax-H3` ships **both** for ref2va -- `transformer_ref/` is diffusers-named,
`Ref2VA/transformer` is native-named, same weights. So point SRC at whichever matches the LoRA:

    lightx2v/Minimax-h3-Turbo   diffusers PEFT  ->  SRC=transformer_ref
    g7e's minimax_h3_turbo_v4   native          ->  SRC=FL2VA/transformer   (259/259, proven)

The remaining hard case is a *native* LoRA into a *diffusers* tree (which is every t2va tree,
including `OpenVDN/.../h3-base/transformer`). That needs three translations, all readable in
SGLang's own loader rather than guessed, and none of them written here:
  1. names + fusion, via `get_param_names_mapping(MiniMaxH3DiTArchConfig.param_names_mapping)`,
     which returns `(native_name, merge_index, merge_count)` -- to_q/to_k/to_v are indices 0/1/2 of
     the fused `qkv_proj`, so the reverse direction is a row-slice of `lora_B`.
  2. SwiGLU half order, `[value, gate]` vs `[gate, value]` (`runtime/models/dits/minimax_h3.py:128`;
     lightx2v's own ComfyUI conversions record the same swap in their metadata).
  3. the per-head qkv row interleave, which SGLang undoes at load with
     `_reorder_grouped_qkv_to_qkv`; a native-named LoRA carries the same interleave.

SCALE. `W_eff = W + scale * (B @ A)` in fp32, cast back to the weight dtype.
`scale = STRENGTH * alpha / rank` when the file declares an alpha, else `STRENGTH` alone. This
matters more than anything else in the file: lightx2v's LoRAs are all rank 128 but declare
alpha **8** (scale 0.0625) or **128** (scale 1.0) depending on the file, and their README's
`--lora-alpha 128` is correct for exactly two of them. See REF2VA.md, cause 1. The resolved scale is
printed; check it before trusting the output. Mixed-rank files (g7e's: 16 for adaln_proj, 64
elsewhere) have no single global alpha, and that one declares
`application: W_eff = W + lora_B @ lora_A` outright, so STRENGTH=1.0 with no alpha is right there.
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

src = os.environ["SRC"]
lora_path = os.environ["LORA"]
dst = os.environ["DST"]
strength = float(os.environ.get("STRENGTH", "1.0"))
# ALPHA= forces a value; ALPHA=none ignores the file's own. Unset reads the metadata.
alpha_env = os.environ.get("ALPHA")

_ALPHA_KEYS = ("lora_alpha", "network_alpha", "alpha")
# PEFT writes the adapter name into the key ("...lora_A.default.weight"); plain exports do not.
_SLOTS = (".lora_A.weight", ".lora_B.weight", ".lora_A.default.weight", ".lora_B.default.weight")


def layout_of(keys) -> str:
    """diffusers | native, decided on markers that cannot coexist."""
    ks = list(keys)
    if any(".attn.to_q" in k or ".ff.net.0.proj" in k or k.startswith("transformer_blocks.") for k in ks):
        return "diffusers"
    if any(".attn.qkv_proj" in k or ".mlp.fc1" in k for k in ks):
        return "native"
    return "unknown"


os.makedirs(dst, exist_ok=True)
index_name = next((n for n in os.listdir(src) if n.endswith(".safetensors.index.json")), None)
if index_name is None:
    print(f"no *.safetensors.index.json in {src}", file=sys.stderr)
    sys.exit(1)
with open(os.path.join(src, index_name)) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
src_layout = layout_of(index["weight_map"])
print(f"index: {index_name}  shards: {len(shards)}  tree layout: {src_layout}")

# The whole LoRA fits in memory (bf16, ~1.3 GB); group it by target weight name.
lora, ranks = {}, set()
with safe_open(lora_path, framework="pt") as f:
    meta = f.metadata() or {}
    for k in f.keys():
        slot = next((s for s in _SLOTS if k.endswith(s)), None)
        if slot is None:
            print(f"UNEXPECTED lora key {k} (expected one of {_SLOTS})", file=sys.stderr)
            sys.exit(1)
        base = k[: -len(slot)]
        # Diffusers PEFT exports sometimes carry a "transformer." component prefix; the checkpoint
        # shard keys never do.
        if base.startswith("transformer."):
            base = base[len("transformer.") :]
        side = "A" if ".lora_A" in slot else "B"
        t = f.get_tensor(k)
        if side == "A" and t.ndim == 2:
            ranks.add(t.shape[0])
        lora.setdefault(base + ".weight", {})[side] = t
one_sided = [k for k, v in lora.items() if set(v) != {"A", "B"}]
if one_sided:
    print(f"LoRA modules missing a side: {one_sided[:5]}", file=sys.stderr)
    sys.exit(1)

lora_layout = layout_of(lora)
if lora_layout != src_layout or lora_layout == "unknown":
    print(
        f"LAYOUT MISMATCH: tree is {src_layout!r}, LoRA is {lora_layout!r}. Merging across layouts "
        "needs the three translations in the docstring and would otherwise match 0 modules or, "
        "worse, match by name and be numerically wrong. For a diffusers LoRA point SRC at "
        "transformer_ref/ (ref2va) or transformer/ (t2va); for a native LoRA point it at "
        "Ref2VA/transformer or FL2VA/transformer.",
        file=sys.stderr,
    )
    sys.exit(1)

# scale = strength * alpha / rank, and every part of that is worth printing.
declared = {k: meta[k] for k in _ALPHA_KEYS if k in meta}
if len(set(declared.values())) > 1:
    print(f"conflicting alpha metadata: {declared}", file=sys.stderr)
    sys.exit(1)
alpha = None
if alpha_env is not None and alpha_env.lower() != "none":
    alpha = float(alpha_env)
elif alpha_env is None and declared:
    alpha = float(next(iter(declared.values())))
if alpha is not None:
    if len(ranks) != 1:
        print(
            f"alpha={alpha:g} given but ranks are mixed {sorted(ranks)}; alpha/rank has no single "
            "value. Re-run with ALPHA=none and set STRENGTH to the scale you want.",
            file=sys.stderr,
        )
        sys.exit(1)
    rank = next(iter(ranks))
    scale = strength * alpha / rank
    print(f"lora modules: {len(lora)}  layout: {lora_layout}  rank {rank}  alpha {alpha:g}  "
          f"strength {strength:g}  ->  SCALE {scale:.6g}")
else:
    scale = strength
    print(f"lora modules: {len(lora)}  layout: {lora_layout}  ranks {sorted(ranks)}  no alpha "
          f"declared  ->  SCALE {scale:.6g} (= STRENGTH)")
print(f"lora metadata: {meta if meta else '(none)'}", flush=True)

applied, worst = set(), 0.0
for sh in shards:
    tensors = {}
    with safe_open(os.path.join(src, sh), framework="pt") as f:
        for k in f.keys():
            w = f.get_tensor(k)
            lw = lora.get(k)
            if lw is not None:
                a, b = lw["A"].float(), lw["B"].float()
                # A=[rank, in], B=[out, rank], W=[out, in]
                if a.shape[0] != b.shape[1] or (b.shape[0], a.shape[1]) != tuple(w.shape):
                    print(
                        f"SHAPE MISMATCH {k}: W{tuple(w.shape)} "
                        f"A{tuple(a.shape)} B{tuple(b.shape)}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                delta = (b @ a) * scale
                w32 = w.float()
                # |delta|/|W| is the one cheap sanity number: a distill LoRA should move the
                # weights a few percent, so a value near 0 means nothing merged and a value
                # near 1 means the adapter does not belong to these weights. At scale 0.0625
                # expect it small -- that is the point of cause 1 in REF2VA.md, not a bug.
                worst = max(worst, (delta.norm() / w32.norm()).item())
                w = (w32 + delta).to(w.dtype)
                applied.add(k)
                del delta, w32, a, b
            tensors[k] = w
    save_file(tensors, os.path.join(dst, sh), metadata={"format": "pt"})
    print(f"  {sh}: {len(tensors)} tensors, merged {len(applied)} so far", flush=True)
    del tensors

for extra in (index_name, "config.json"):
    p = os.path.join(src, extra)
    if os.path.isfile(p):
        shutil.copy(p, os.path.join(dst, extra))

unmatched = sorted(set(lora) - applied)
print(f"merged {len(applied)}/{len(lora)} modules, max |delta|/|W| = {worst:.4f}")
if unmatched:
    print(f"NOT MATCHED ({len(unmatched)}): {unmatched[:8]}", file=sys.stderr)
    sys.exit(1)
print(f"wrote {dst}")

#!/usr/bin/env python3
"""Merge an H3 Turbo LoRA into a bf16 transformer offline, writing a plain bf16 checkpoint.

    SRC=<hf-snapshot>/FL2VA/transformer \
    LORA=/opt/dlami/nvme/vdn/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
    DST=/opt/dlami/nvme/vdn/turbo_bf16 \
    /opt/dlami/nvme/sglang/.venv/bin/python lora_merge_h3.py

STATUS: WORKS ONLY ON A NATIVE-NAMED TREE, WHICH MEANS NOT ON t2va. Read this before using it.

This LoRA is named for the *native* MiniMax checkpoint layout:

    blocks.N.attn.qkv_proj   blocks.N.attn.out_proj   blocks.N.mlp.fc{1,2}
    blocks.N.adaln_proj.linear   token_refiner.blocks.N.*   final_layer.adaln_proj.linear

Every t2va transformer on disk is named for the *diffusers* layout instead -- both
`MiniMaxAI/MiniMax-H3/transformer` and `OpenVDN/vdn-minimax-h3/h3-base/transformer`, 638 tensors
each, `diffusion_pytorch_model-*.safetensors`:

    transformer_blocks.N.attn.to_{q,k,v}   transformer_blocks.N.attn.to_out.0
    transformer_blocks.N.ff.net.0.proj (fc1)   transformer_blocks.N.ff.net.2 (fc2)   norm_out.linear

Only `FL2VA/` and `Ref2VA/` ship native names (535 tensors, `model-*.safetensors`). That is why
the g7e project merged against `FL2VA/transformer` and hit 259/259, and why pointing this script
at a t2va tree hits **0/259** and exits 1 rather than writing a silently unmerged 62 GB copy.

Making it work for t2va needs three translations, and all three are readable in SGLang's own
loader (`runtime/models/dits/minimax_h3.py:_diffusers_h3_checkpoint`) rather than guessed:
  1. names, via `get_param_names_mapping(MiniMaxH3DiTArchConfig.param_names_mapping)`, which
     returns `(native_name, merge_index, merge_count)` -- so to_q/to_k/to_v are indices 0/1/2 of
     the fused `qkv_proj`, i.e. the reverse direction is a row-slice of `lora_B`.
  2. SwiGLU half order: diffusers `ff.net.0.proj` is `[value, gate]`, native `mlp.fc1` is
     `[gate, value]` (minimax_h3.py:128). The swap is its own inverse.
  3. qkv row order: the native checkpoint interleaves each head's q,k,v rows and SGLang undoes
     that with `_reorder_grouped_qkv_to_qkv` at load. A native-named LoRA carries the same
     interleave, so a merge into split diffusers tensors has to un-interleave first.
Each of those fails silently in a way shapes alone will not catch, which is why this is not
written speculatively.

WHY OFFLINE AT ALL, i.e. why not just --lora-path. SGLang accepts `--lora-path` on this model and
no longer dies the way g7e recorded (a runtime "merge" doing an in-place add on [out, in] against
an fp8 weight stored transposed, their fc1 reporting 21504 vs 5376): upstream's
`_should_merge_lora_for_layers` now sees `can_merge_base_weight == False` on a quantized layer and
falls back to dynamic LoRA by itself. It fails one layer later instead --

    AttributeError: 'RowParallelLinearWithLoRA' object has no attribute 'quant_method'

-- because the dynamic-LoRA wrapper is not quantization-aware. So `--quantization fp8
--lora-path ...` aborts during server warmup. The bf16 route (`QUANT= bash sglang_base_arm.sh`)
does run `--lora-path` with zero mapping work, and is the right way to get a *picture* out of
base+Turbo; it is the wrong way to get a *latency*, because bf16 is not the precision every other
arm here is measured at. For latency no merge is needed at all: a merged LoRA changes weight
values, not tensor shapes or the graph, so fp8 base H3 at 8 steps already IS fp8 Turbo at 8 steps.

SGLang documents this same offline-merge shape itself: `--model-variant hybrid` refuses to start
without "explicit merged weights via --component-weights-paths.transformer"
(`runtime/pipelines/minimax_h3_pipeline.py:97`), which is also the flag to serve DST with.

`W_eff = W + strength * (B @ A)`, with NO alpha/rank scaling: the file's own safetensors metadata
says `application: W_eff = W + lora_B @ lora_A`, and its ranks are mixed (16 for adaln_proj, 64
elsewhere) so there is no single global alpha to apply anyway. strength defaults to 1.0, which is
what the adapter was tuned at. The delta is computed in fp32 and cast back to the weight dtype.

This differs from the g7e original only in not hardcoding the index filename, since the two trees
disagree on it (`model.safetensors.index.json` vs `diffusion_pytorch_model.safetensors.index.json`).
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

os.makedirs(dst, exist_ok=True)
index_name = next(
    (n for n in os.listdir(src) if n.endswith(".safetensors.index.json")), None
)
if index_name is None:
    print(f"no *.safetensors.index.json in {src}", file=sys.stderr)
    sys.exit(1)
with open(os.path.join(src, index_name)) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
print(f"index: {index_name}  shards: {len(shards)}")

# Refuse before reading 62 GB. A diffusers-named tree would match nothing and exit 1 at the end
# anyway, only after a full read/write pass -- see the layout note in the module docstring.
if any(k.startswith("transformer_blocks.") for k in index["weight_map"]):
    print(
        f"{src} is a diffusers-named tree (transformer_blocks.* / attn.to_q / ff.net.0.proj); "
        "this LoRA is named for the native layout (blocks.* / attn.qkv_proj / mlp.fc1) and would "
        "match 0/259 modules. Point SRC at FL2VA/transformer or Ref2VA/transformer, or implement "
        "the three translations listed in the docstring.",
        file=sys.stderr,
    )
    sys.exit(1)

# The whole LoRA fits in memory (bf16, ~780 MB); group it by target weight name.
lora = {}
with safe_open(lora_path, framework="pt") as f:
    for k in f.keys():
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            print(f"UNEXPECTED lora key {k}", file=sys.stderr)
            sys.exit(1)
        base, side = k.rsplit(".lora_", 1)
        lora.setdefault(base + ".weight", {})[side[0]] = f.get_tensor(k)
one_sided = [k for k, v in lora.items() if set(v) != {"A", "B"}]
if one_sided:
    print(f"LoRA modules missing a side: {one_sided[:5]}", file=sys.stderr)
    sys.exit(1)
print(f"lora modules: {len(lora)}  strength={strength}", flush=True)

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
                delta = (b @ a) * strength
                w32 = w.float()
                # |delta|/|W| is the one cheap sanity number: a distill LoRA should move the
                # weights a few percent, so a value near 0 means nothing merged and a value
                # near 1 means the adapter does not belong to these weights.
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

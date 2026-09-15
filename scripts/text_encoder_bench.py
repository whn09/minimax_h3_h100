#!/usr/bin/env python3
"""What the Qwen3-VL conditioner would cost if it were IN the request path.

Every latency in RESULTS.md excludes text encoding, because upstream's design puts it
outside: `src/inference/encode_prompt.py` runs the 63 GB conditioner once, offline, on one
GPU, and writes `prompt_embeds` to a .pt that `load_prompt()` then `torch.load`s. Upstream's
published 18.3 s excludes it for the same reason. A real prompt-to-video service cannot,
so this measures the three things that decide what it costs:

  1. **How much of it is actually needed.** VDN reads `hidden_states[50]`, which HF fills
     with the INPUT to layer 50, so layers 52-64 -- 13 of 64 -- and the 5120x151936 LM head
     never contribute. Dropping them is the difference between what the checkpoint weighs
     and what has to be resident.
  2. **The forward.** A prompt is ~1300 tokens through 51 layers. This is small.
  3. **What offloading it to the host would cost**, measured as the host<->device transfer
     of the weights that a per-request offload would have to move. Compare against the
     26.69 s that patch 0003's 45.2 GiB `transformer.to("cpu")` measured -- 1.7 GB/s, far
     off PCIe Gen5, because a state dict is thousands of separate unpinned tensors.

    python scripts/text_encoder_bench.py                      # 1300-token synthetic prompt
    python scripts/text_encoder_bench.py --tokens 3485        # the fl2va prompt's length

The conclusion this is meant to settle: 8-way sharding versus host offload. Sharded, the
needed weights are ~6 GiB per rank and sit alongside the 45.2 GiB fp8 DiT on an 80 GiB
card, and the forward is dominated by nothing. Offloaded, the transfer below is the floor.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Qwen3VLForConditionalGeneration

from src.paths import upstream_snapshot

TEXT_ENCODER_LAYER = 50
GIB = 1024 ** 3


def resident_gib(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / GIB


def forward_seconds(model, input_ids, repeats):
    mm_token_type_ids = torch.zeros_like(input_ids)
    best = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            out = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=True,
            )
        embeds = out.hidden_states[TEXT_ENCODER_LAYER]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)
    return best, tuple(embeds.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1300,
                   help="prompt length in tokens; 1299 is the t2va example, 3485 the fl2va one")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--model_root", type=str, default=None)
    args = p.parse_args()

    root = args.model_root or upstream_snapshot("processor", "text_encoder")

    # Random ids rather than real text: this measures FLOPs and bytes, both of which depend
    # only on the token count. Nothing here checks the embeddings, so the ids do not have to
    # spell anything -- and tying the number to a length keeps it comparable across prompts.
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100000, (1, args.tokens), device=args.device)

    load_started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        root, subfolder="text_encoder", dtype=torch.bfloat16
    ).to(args.device)
    model.eval().requires_grad_(False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    full_gib = resident_gib(model)
    print(f"load (NVMe -> GPU, bf16): {load_seconds:.1f}s for {full_gib:.1f} GiB "
          f"({full_gib / load_seconds:.2f} GiB/s)", flush=True)

    full_forward, shape = forward_seconds(model, input_ids, args.repeats)
    print(f"full 64-layer forward, {args.tokens} tokens: {full_forward * 1000:.0f} ms "
          f"-> hidden_states[{TEXT_ENCODER_LAYER}] {shape}", flush=True)

    # Everything above layer 50 is dead weight for this pipeline. Deleting it in place is
    # the cheap way to measure what a truncated checkpoint would be resident; a real
    # deployment would simply not save those shards.
    # The decoder stack has moved around between transformers versions
    # (Qwen3VLModel.language_model.layers today), so find it rather than assume it.
    holder = next((h for h in (getattr(model.model, "language_model", None), model.model, model)
                   if h is not None and isinstance(getattr(h, "layers", None), torch.nn.ModuleList)),
                  None)
    if holder is None:
        raise RuntimeError("could not find the decoder layer list on this text encoder")
    print(f"decoder layers: {len(holder.layers)} on {type(holder).__name__}; "
          f"VDN reads layer {TEXT_ENCODER_LAYER}", flush=True)
    # KEEP 51, not 50. HF appends `hidden_states` at the TOP of each layer's iteration and
    # appends the final-norm output once after the loop, so `hidden_states[50]` is the input
    # to layer 50 -- the un-normed output of layer 49. Truncating to exactly 50 layers would
    # make index 50 the post-norm output instead, i.e. a different tensor under the same
    # name. With 51 layers present, index 50 is byte-for-byte what the full model produced.
    num_layers = len(holder.layers)
    holder.layers = torch.nn.ModuleList(list(holder.layers[: TEXT_ENCODER_LAYER + 1]))
    try:
        del model.lm_head
    except AttributeError:
        pass
    torch.cuda.empty_cache()
    trimmed_gib = resident_gib(model)
    print(f"resident after dropping layers {TEXT_ENCODER_LAYER + 2}-{num_layers} "
          f"({num_layers - TEXT_ENCODER_LAYER - 1} layers) and the LM head: "
          f"{trimmed_gib:.1f} GiB (was {full_gib:.1f}, saved {full_gib - trimmed_gib:.1f})",
          flush=True)

    trimmed_forward, shape = forward_seconds(model, input_ids, args.repeats)
    print(f"trimmed {TEXT_ENCODER_LAYER + 1}-layer forward, {args.tokens} tokens: "
          f"{trimmed_forward * 1000:.0f} ms -> {shape}", flush=True)

    # What a per-request host offload would cost, in both directions. This is the number
    # that decides against offloading; compare 45.2 GiB / 26.69 s from patch 0003.
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.to("cpu")
    torch.cuda.synchronize()
    to_host = time.perf_counter() - started
    started = time.perf_counter()
    model.to(args.device)
    torch.cuda.synchronize()
    to_device = time.perf_counter() - started
    print(f"offload round trip for {trimmed_gib:.1f} GiB: "
          f"to host {to_host:.1f}s ({trimmed_gib / to_host:.2f} GiB/s), "
          f"back {to_device:.1f}s ({trimmed_gib / to_device:.2f} GiB/s)", flush=True)

    world = 8
    print(f"\nsharded over {world} ranks: {trimmed_gib / world:.1f} GiB per rank, "
          f"alongside a 45.2 GiB fp8 DiT = {45.2 + trimmed_gib / world:.1f} GiB of 79.2",
          flush=True)


if __name__ == "__main__":
    main()

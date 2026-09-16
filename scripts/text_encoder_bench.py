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

There is a fourth number, and it is the one the first three make load-bearing:

  4. **What a per-request upload of ONE SHARD costs, with all eight ranks doing it at once.**
     The offload figures above are single-process, one card at a time. A request path that
     keeps the conditioner in pinned host memory and uploads 1/8 of it per rank per request
     has eight concurrent H2D streams competing for host memory bandwidth, so the per-rank
     rate is not necessarily the single-rank rate. `--transfer` measures exactly that and
     needs no checkpoint at all:

    torchrun --standalone --nproc_per_node=8 scripts/text_encoder_bench.py --transfer 6.1

The conclusion this is meant to settle: where the conditioner lives. Resident is the obvious
answer and it does NOT fit -- 6.1 GiB per rank against the ~2.1 GiB the 480p pipeline leaves
spare at its decode peak (RESULTS.md, "Text encoding is not in any of these numbers"). Host
offload of the whole thing per request is 37 s and never in question. What is left is a
pinned host copy uploaded per request, whose cost is (4) plus (2), and that is why (4) is
here.
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


def transfer_bench(gib, repeats):
    """Per-request cost of uploading one conditioner shard from a pinned host copy, with
    every rank uploading at the same time -- the only version of this number that a request
    path would actually see. Run under torchrun; falls back to one rank if not.

    Three arms, because two of them are the alternatives being rejected:
      pinned, all ranks    what the design costs
      pinned, rank 0 only  the same transfer with no contention -- the difference IS the
                           contention, and it is what a single-process benchmark hides
      pageable, all ranks  what you get if the host copy is a plain state dict rather than
                           page-locked, i.e. the mistake this measurement exists to price
    """
    import torch.distributed as dist

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        rank, world, device = 0, 1, "cuda:0"

    elements = int(gib * GIB) // 2                      # bf16
    host_pinned = torch.empty(elements, dtype=torch.bfloat16, pin_memory=True)
    host_pageable = torch.empty(elements, dtype=torch.bfloat16)
    card = torch.empty(elements, dtype=torch.bfloat16, device=device)

    def timed(source, participating):
        """Best of `repeats`, measured only on the ranks that copy. The barrier is what makes
        this concurrent rather than eight staggered transfers."""
        best = None
        for _ in range(repeats):
            if distributed:
                dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            if participating:
                card.copy_(source, non_blocking=True)
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            best = elapsed if best is None else min(best, elapsed)
        return best

    arms = [("pinned, all ranks", host_pinned, True),
            ("pinned, rank 0 only", host_pinned, rank == 0),
            ("pageable, all ranks", host_pageable, True)]
    results = []
    for name, source, participating in arms:
        seconds = timed(source, participating)
        results.append((name, seconds if participating else None))

    if rank == 0:
        print(f"H2D of a {gib:.2f} GiB shard, world={world}, best of {repeats}:", flush=True)
        for name, seconds in results:
            if seconds is None:
                continue
            per_rank = gib / seconds
            note = f", {per_rank * world:.2f} GiB/s aggregate" if "all ranks" in name else ""
            print(f"  {name:22s} {seconds:6.3f} s  ({per_rank:5.2f} GiB/s per rank{note})",
                  flush=True)
        print(f"\nA request path paying this once per request adds it to the 135 ms forward; "
              f"compare 11.45 s (480p) and 33.21 s (768p).", flush=True)
    if distributed:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1300,
                   help="prompt length in tokens; 1299 is the t2va example, 3485 the fl2va one")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--model_root", type=str, default=None)
    p.add_argument("--transfer", type=float, default=None, metavar="GIB",
                   help="skip the model entirely and measure a GIB-per-rank pinned H2D under "
                        "torchrun (6.1 = the trimmed conditioner over 8 ranks)")
    args = p.parse_args()

    if args.transfer is not None:
        transfer_bench(args.transfer, max(args.repeats, 3))
        return

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

    # The accounting that matters is not "shard + DiT" -- that comparison omits the decoders
    # and both stages' activations, and it is how an earlier version of RESULTS.md talked
    # itself into "28 GiB of headroom". The real ceiling is the card minus the non-PyTorch
    # floor, and the real occupant is the pipeline's measured peak.
    world = 8
    shard = trimmed_gib / world
    ceiling, floor, peak_480p = 79.18, 13.92, 63.14   # measured: OOM log, s2_480p_rep10
    spare = ceiling - floor - peak_480p
    print(f"\nsharded over {world} ranks: {shard:.1f} GiB per rank.", flush=True)
    print(f"  card gives PyTorch {ceiling - floor:.2f} GiB ({ceiling} total - {floor} "
          f"non-PyTorch floor); the 480p pipeline peaks at {peak_480p:.2f} "
          f"-> {spare:.2f} GiB spare", flush=True)
    print(f"  resident conditioner {'fits' if shard <= spare else 'DOES NOT FIT'}: "
          f"needs {shard:.1f}, has {spare:.2f}", flush=True)
    print(f"  so price the per-request upload instead: "
          f"torchrun --standalone --nproc_per_node={world} {sys.argv[0]} "
          f"--transfer {shard:.2f}", flush=True)


if __name__ == "__main__":
    main()

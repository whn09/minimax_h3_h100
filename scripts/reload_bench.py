"""How long does the DiT take to come BACK after parallel.transformer_before_decode
frees it?

patch 10 releases the transformer's 45.2 GiB with to_empty(device="meta") so rank 0 can
hold the decoders and a full-length 768p decode. That is the right trade for a one-shot
render -- nothing downstream of the denoise reads the weights. It is NOT obviously the
right trade for a long-lived API process, which will denoise again, and to_empty leaves
no copy anywhere: the storages are gone and the module's parameters point at meta.

So this measures every way of getting them back, on one GPU, at the fp8 numerics the
768p config actually runs:

  release            to_empty(device="meta") + empty_cache      -- what patch 10 costs
  restore  pinned    to_empty(device="cuda") + load_state_dict from a PINNED host copy
  restore  pageable  the same from an ordinary host copy, i.e. what you get if you keep
                     patch 2's CPU assembly around without pinning it
  snapshot pinned    making that pinned copy (once, at build; off the request path)
  snapshot pageable  .to("cpu"), the ~1.7 GiB/s path that costs "offload" its 26.33 s

The fourth restore path -- rebuild from the checkpoint -- is not run here because every
grid arm already measures it: `setup` in the Ulysses summary line, 217-232 s.

Parity is checked by byte-comparing the restored CUDA weights against the host copy they
came from. That is the right check here and an end-to-end diff would not be: the
multi-rank render is not reproducible run to run (RESULTS.md), but a weight copy is, so
a copy either landed or it did not.

    source .venv/bin/activate
    python scripts/reload_bench.py --config configs/inference/8nfe_768p_345f_ulysses_h100.yaml \
        checkpoint=ckpts/stage-dmd-step-250
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.config.inference import (
    InferenceConfig,
    validate_ablation,
    validate_kernels,
    validate_parallel,
)
from src.inference.utils.assemble import build_inference_model


GIB = 1024 ** 3


def resident_bytes(module) -> int:
    """Bytes of CUDA storage the module's parameters and buffers actually hold. Counted
    over storages, not tensors, so a shared storage is not double counted."""
    seen, total = {}, 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.device.type != "cuda":
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen[key] = True
            total += storage.nbytes()
    return total


def full_state(module):
    """Every parameter and buffer by name, including non-persistent buffers.

    NOT state_dict(): a non-persistent buffer is absent from state_dict, so a
    state_dict round trip would silently drop it -- to_empty() repoints it at meta and
    nothing would ever put it back. The window-softmax and vdn branches both carry cached
    index buffers, so this is a real hazard rather than a hypothetical one.
    """
    out = {}
    for name, param in module.named_parameters():
        out[name] = param.detach()
    for name, buf in module.named_buffers():
        out[name] = buf.detach()
    return out


def assign_from(module, host):
    """Copy host tensors into the module's existing CUDA storages, by name."""
    for name, param in module.named_parameters():
        param.data.copy_(host[name], non_blocking=True)
    for name, buf in module.named_buffers():
        buf.copy_(host[name], non_blocking=True)


def timed(label, fn, results):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    results[label] = seconds
    print(f"  {label:<26} {seconds:8.2f} s", flush=True)
    return value


def main():
    cfg = load_config(
        InferenceConfig,
        extra_validators=[validate_ablation, validate_kernels, validate_parallel],
    )
    device = "cuda:0"
    torch.set_grad_enabled(False)
    results = {}

    print("building (this is the 217-232 s the grid logs report as `setup`)", flush=True)
    build_started = time.perf_counter()
    model = build_inference_model(cfg, device, load_decoders=False, log=True)
    results["build_from_checkpoint"] = time.perf_counter() - build_started
    print(f"  build_from_checkpoint      {results['build_from_checkpoint']:8.2f} s",
          flush=True)

    dit = model.transformer
    resident = resident_bytes(dit)
    n_params = sum(1 for _ in dit.parameters())
    n_buffers = sum(1 for _ in dit.buffers())
    print(f"  DiT on card: {resident / GIB:.2f} GiB over {n_params} parameters "
          f"+ {n_buffers} buffers", flush=True)
    print(f"  torch reserved: {torch.cuda.memory_reserved() / GIB:.2f} GiB, "
          f"allocated {torch.cuda.memory_allocated() / GIB:.2f} GiB", flush=True)
    results["resident_gib"] = resident / GIB
    results["n_params"] = n_params
    results["n_buffers"] = n_buffers

    # --- the two ways to hold a host copy -------------------------------------------
    # pageable first, because it is the one whose cost is the headline of patch 10's
    # comparison: this is what model.transformer.to("cpu") does internally.
    print("\nsnapshotting to the host", flush=True)
    pageable = timed("snapshot_pageable",
                     lambda: {k: v.to("cpu", copy=True) for k, v in full_state(dit).items()},
                     results)

    # A pinned copy is made FROM the pageable one, host to host, so this timing is not
    # polluted by a second D2H. In a real API the pinned copy would be made once at
    # build time out of patch 2's CPU assembly, before the weights ever go to the card.
    def make_pinned():
        out = {}
        for name, tensor in pageable.items():
            slot = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu",
                               pin_memory=True)
            slot.copy_(tensor)
            out[name] = slot
        return out

    pinned = timed("snapshot_pinned_h2h", make_pinned, results)

    # --- release, then restore, twice ------------------------------------------------
    for label, host in (("pinned", pinned), ("pageable", pageable)):
        print(f"\nrelease + restore from a {label} host copy", flush=True)
        timed(f"release_to_meta_{label}", lambda: (dit.to_empty(device="meta"),
                                                  torch.cuda.empty_cache()), results)
        after = torch.cuda.memory_reserved() / GIB
        print(f"  reserved after release: {after:.2f} GiB", flush=True)
        results[f"reserved_after_release_{label}_gib"] = after

        timed(f"alloc_on_cuda_{label}", lambda: dit.to_empty(device=device), results)
        timed(f"copy_in_{label}", lambda: assign_from(dit, host), results)
        results[f"restore_total_{label}"] = (results[f"alloc_on_cuda_{label}"]
                                            + results[f"copy_in_{label}"])
        print(f"  {'restore_total_' + label:<26} "
              f"{results['restore_total_' + label]:8.2f} s   "
              f"({resident / GIB / results['restore_total_' + label]:.2f} GiB/s)",
              flush=True)

        # parity: the restored card weights must equal the host copy they came from
        bad = []
        for name, tensor in full_state(dit).items():
            if not torch.equal(tensor.cpu(), host[name].cpu()):
                bad.append(name)
        print(f"  parity: {len(bad)} of {len(host)} tensors differ"
              + (f" -- {bad[:4]}" if bad else " (bit-identical)"), flush=True)
        results[f"parity_mismatches_{label}"] = len(bad)

    print("\n" + json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

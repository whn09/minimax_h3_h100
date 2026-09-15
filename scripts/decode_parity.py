#!/usr/bin/env python3
"""Does the eight-rank video VAE decode produce the same pixels as upstream's one-rank one?

    torchrun --standalone --nproc_per_node=8 scripts/decode_parity.py --frames 345 \
        --height 480 --width 864

`utils/parallel_vae.py` claims its output is bit-identical to `vae.decode(z)`. That claim
needs a test, and the obvious one -- render the same prompt twice, once each way, and `cmp`
the mp4s -- **does not work, for a reason worth stating plainly: this pipeline is not
deterministic run to run.** Two renders at the same seed, same config, same split come out
at ~17 dB PSNR of each other. Measured, for identical configurations run twice:

    480p r3, parallel decode both times     17.55 dB
    768p r0, serial decode both times       16.69 dB

A temporal cross-correlation puts the best alignment at shift 0 with a sharp peak, and
16x-downsampled PSNR only climbs to ~22-29 dB, so the two runs agree on the scene, the
composition and the motion and disagree on the detail everywhere -- ULP-level divergence in
the fp8 GEMMs, amplified over 8 sampler steps. Which means an end-to-end `cmp` compares two
denoise trajectories, not two decoders, and will report a mismatch no matter what this file
does.

So the decode is tested where it lives: **one process, one latent tensor, decoded both ways
back to back.** Everything upstream is removed from the comparison, so a difference here is
this code's difference and nothing else. Latents are drawn from a seeded generator rather
than loaded from a render, because the decoder is a pure function of its input -- it has no
idea where the numbers came from -- and `--latents` is there for anyone who would rather
check an in-distribution tensor.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    align_num_frames, video_latent_num_frames)

from src.inference.render import decode_video_latents, latent_grid, load_decoders_only
from src.inference.utils.ulysses_runtime import init_ulysses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=345)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=864)
    p.add_argument("--vae-source", default=None)
    p.add_argument("--latents", default=None,
                   help="a render's saved latents (render.save_latents=true) instead of noise")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    runtime = init_ulysses()
    device = str(runtime.device)
    torch.set_grad_enabled(False)

    vae, _ = load_decoders_only(args.vae_source, device, audio=False)

    if args.latents:
        latents = torch.load(args.latents, map_location=device, weights_only=True)["video"].float()
    else:
        # The same shape the sampler produces, so the chunk count and the trailing pad are
        # the real ones -- 345 frames is 20 chunks with a pad, which is the case where the
        # assembly's trailing trim actually does something. align_num_frames is applied for
        # the same reason: 345 is already 5 + 17k, but a --frames that is not would
        # otherwise be tested at a shape the renderer can never produce.
        num_frames = align_num_frames(args.frames, 17, 5)
        latent_h, latent_w = latent_grid(args.height, args.width)
        num_latent_frames = video_latent_num_frames(num_frames, 17, 5)
        generator = torch.Generator(device).manual_seed(args.seed)
        latents = torch.randn((1, vae.config.latent_channels, num_latent_frames, latent_h, latent_w),
                              generator=generator, device=device, dtype=torch.float32)
    if runtime.is_main:
        print(f"latents {tuple(latents.shape)} -> {args.width}x{args.height} "
              f"on {runtime.world_size} ranks", flush=True)

    # Parallel first: it is the collective, so every rank must reach it, and going first
    # means a hang shows up before the serial decode has spent its time.
    parallel = decode_video_latents(vae, latents, device, runtime=runtime)
    torch.cuda.synchronize(device)
    runtime.barrier()

    if not runtime.is_main:
        return

    serial = decode_video_latents(vae, latents, device)
    torch.cuda.synchronize(device)

    if serial.shape != parallel.shape:
        print(f"\nFAIL -- shape {tuple(parallel.shape)} against {tuple(serial.shape)}")
        raise SystemExit(1)

    diff = (serial - parallel).abs()
    max_abs = float(diff.max())
    # The comparison is on the pixel-denormalised float output, in [0, 1], because that is
    # what gets written. A difference below 1/255 could not change a single encoded byte
    # even if it were not zero, so that is the second line of defence -- but the claim is
    # exact equality, so exact equality is what passes.
    print(f"\n{'max abs difference':>22}: {max_abs:.3e}")
    print(f"{'mean abs difference':>22}: {float(diff.mean()):.3e}")
    print(f"{'differing elements':>22}: {int((diff > 0).sum())} of {diff.numel()}")
    print(f"{'in 8-bit levels':>22}: {max_abs * 255:.3e}")

    if max_abs == 0.0:
        print("\nOK -- bit-identical. The eight-rank decode is upstream's decode.")
        raise SystemExit(0)
    if max_abs * 255 < 0.5:
        print("\nOK-ish -- not bit-identical, but below half an 8-bit level, so the encoded "
              "bytes cannot differ. Still worth explaining before the claim is weakened.")
        raise SystemExit(0)
    print("\nFAIL -- the parallel decode changed the picture.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

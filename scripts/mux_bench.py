#!/usr/bin/env python3
"""How much of the mux's 2.66 s is avoidable?

    .venv/bin/python scripts/mux_bench.py out/v_480p_345f_serial.mp4
    .venv/bin/python scripts/mux_bench.py out/c_768p_345f_r0.mp4 --nvenc   # needs a free GPU

After patch 5 took the video VAE from 11.2 s to 3.0 s, the largest single item left in a
480p request is the **mp4 mux at 2.66 s** -- larger than the eight-way GPU decode it
follows, and running entirely on the host. `diffusers.utils.export_utils.encode_video`
builds its stream with `container.add_stream("libx264", rate=fps)` and sets only `width`,
`height` and `pix_fmt`: no `preset`, no `thread_count`, no `crf`. 345 frames of 864x480 in
2.66 s is 130 fps, which on a 192-vCPU box is what one thread looks like.

Frames come from **decoding a real render** rather than from noise, because x264's speed
depends on the content: random pixels defeat motion estimation and would make every
configuration look equally slow, which is exactly the effect being measured. The frames are
decoded once, held as a list of `(H, W, 3)` uint8 arrays, and every arm encodes that same
list -- so the numbers differ only in encoder settings.

Reports wall time and the output size, because these trade off: `ultrafast` is not free, it
is bits. A latency number that ships a 3x larger file is not obviously a win, so both are
printed and the choice is left to whoever reads it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import av
import numpy as np


def load_frames(path: str) -> tuple[list[np.ndarray], float]:
    """Decode `path` to a list of rgb24 arrays. Returns the frames and the source fps."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    return frames, fps


def encode(frames, fps, path, codec="libx264", options=None, thread_count=None):
    """One encode, timed. Mirrors encode_video's loop exactly apart from the settings."""
    started = time.perf_counter()
    with av.open(path, mode="w") as container:
        stream = container.add_stream(codec, rate=int(fps), options=options or {})
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        if thread_count is not None:
            # 0 is FFmpeg's "decide for me", which for libx264 hands off to x264's own
            # auto (1.5x cores). PyAV leaves this at 1 unless asked, which is the whole
            # finding here.
            stream.codec_context.thread_count = thread_count
        for array in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return time.perf_counter() - started, os.path.getsize(path)


def verify_against_swscale(frames) -> int:
    """Is `utils/yuv.rgb_to_yuv420p` the same conversion swscale does?

    The GPU conversion replaces a lossy operation with another implementation of it, so
    "looks right" is not good enough -- a wrong matrix (BT.709 instead of BT.601) or a wrong
    range (full instead of limited) both produce a picture, just not this picture. swscale is
    the reference because it is what every previously rendered clip went through.

    The two planes get different tests, because they can fail in different ways and only one
    of those ways matters:

      * **Y is checked on max absolute error, and must be within 1.** Y is where a wrong
        matrix or a wrong range shows up, and it shows up big: BT.709 instead of BT.601 moves
        Y by tens of levels on saturated colour, and full range instead of limited moves it
        by 16 everywhere. A max error of 1 is rounding and nothing else.
      * **Chroma is checked on PSNR, and must clear 50 dB.** Measured, the 2x2 box average
        is the closest match to swscale of the three plausible filters (54.8 dB against
        53.9 dB for taking the top row), so the filter is right and the residual is
        swscale's own fixed-point arithmetic -- it accumulates integer coefficients and
        shifts, where this does the matrix in float. The difference is max 6 levels, mean
        0.17, and it is *ours* that is the more accurate of the two. It also lands well below
        the chroma quantiser step x264 applies at its default CRF, so it cannot survive into
        the file. A max-error test on chroma would be testing swscale's rounding, not this
        code.
    """
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.inference.utils.yuv import rgb_to_yuv420p

    # swscale's answer, read straight out of the converted frame's planes.
    reference = []
    for array in frames:
        converted = av.VideoFrame.from_ndarray(array, format="rgb24").reformat(format="yuv420p")
        reference.append([np.frombuffer(bytes(plane), dtype=np.uint8).reshape(plane.height, plane.width)
                          for plane in converted.planes])

    # Ours. The renderer hands over (frames, 3, H, W) float in [0, 1], so rebuild that shape
    # exactly rather than feeding uint8 -- the quantisation of the input is part of what is
    # being compared.
    stacked = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    y, u, v = rgb_to_yuv420p(stacked)

    failures = []
    for name, ours, index in (("Y", y, 0), ("U", u, 1), ("V", v, 2)):
        mine = ours.numpy()
        theirs = np.stack([r[index][:, : mine.shape[2]] for r in reference])
        diff = mine.astype(np.int16) - theirs.astype(np.int16)
        max_abs = int(np.abs(diff).max())
        mse = float((diff.astype(np.float64) ** 2).mean())
        psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        print(f"  {name}: max abs error {max_abs:3d}   PSNR {psnr:6.2f} dB   "
              f"mean abs {np.abs(diff).mean():.4f}")
        if name == "Y" and max_abs > 1:
            failures.append(f"Y is off by {max_abs}, so the matrix or the range is wrong")
        elif name != "Y" and psnr < 50:
            failures.append(f"{name} is only {psnr:.1f} dB, too far for a rounding difference")

    if not failures:
        print("\nOK -- Y matches to rounding, chroma clears 50 dB. Same conversion.")
        return 0
    for message in failures:
        print(f"\nFAIL -- {message}")
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="an existing render to take frames from")
    p.add_argument("--verify", action="store_true",
                   help="check utils/yuv.rgb_to_yuv420p against swscale instead of timing")
    p.add_argument("--nvenc", action="store_true",
                   help="also try h264_nvenc. Needs a GPU with free memory for a CUDA "
                        "context -- do NOT pass this while a render is using all eight.")
    p.add_argument("--out", default="/tmp/mux_bench")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames, fps = load_frames(args.source)
    h, w = frames[0].shape[:2]
    print(f"{args.source}: {len(frames)} frames of {w}x{h} at {fps:g} fps\n")

    if args.verify:
        # A handful of frames is enough: the conversion is per pixel, so 20 frames is
        # 8.3 million samples per plane and the error distribution is already settled.
        raise SystemExit(verify_against_swscale(frames[:20]))

    arms = [
        # The baseline. Exactly what encode_video does today, settings and all.
        ("libx264, as diffusers ships it", "libx264", None, None),
        # One line of change: let x264 use the box.
        ("libx264, thread_count=0 (auto)", "libx264", None, 0),
        # Threads plus a faster preset. veryfast is the usual "still looks fine" stop.
        ("libx264, auto + preset=veryfast", "libx264", {"preset": "veryfast"}, 0),
        ("libx264, auto + preset=ultrafast", "libx264", {"preset": "ultrafast"}, 0),
        # Threads alone but keeping quality, to separate the two effects.
        ("libx264, auto + preset=medium", "libx264", {"preset": "medium"}, 0),
    ]
    if args.nvenc:
        arms.append(("h264_nvenc, preset=p4", "h264_nvenc", {"preset": "p4"}, None))

    baseline = None
    print(f'| {"arm":34} | {"seconds":>8} | {"fps":>7} | {"size":>9} | {"vs base":>8} |')
    print("|" + "-" * 36 + "|" + "-" * 10 + "|" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 10 + "|")
    for name, codec, options, threads in arms:
        path = os.path.join(args.out, name.replace(" ", "_").replace(",", "") + ".mp4")
        try:
            seconds, size = encode(frames, fps, path, codec, options, threads)
        except Exception as exc:  # a codec may simply not be usable here
            print(f"| {name:34} | {'FAILED':>8} | {'':>7} | {'':>9} | {type(exc).__name__} |")
            continue
        baseline = baseline or seconds
        print(f"| {name:34} | {seconds:8.2f} | {len(frames) / seconds:7.0f} | "
              f"{size / 1e6:8.2f}M | {baseline / seconds:7.2f}x |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decode a rendered clip end to end and report what is actually in it.

    .venv/bin/python scripts/clipinfo.py out/n_480p_seg4.mp4

Written for patch 0009, which encodes the video as several independent segments and
concatenates the bitstreams. That is the one change in this series that can produce a file
which *opens* fine and is still wrong -- a dropped packet, a timestamp that goes backwards,
a segment whose parameter sets do not match -- and none of those show up in a PSNR against
another render, because two renders of the same prompt diverge ~17 dB anyway (see
RESULTS.md). So the check is structural: every frame is decoded, the presentation
timestamps are required to be strictly increasing and evenly spaced, and the key frames are
counted, because the segment boundaries are exactly where the extra ones should be.

Exits non-zero if the timestamps are not a clean 0, 1, 2, ... run at the stream's frame
rate, which is the failure the re-stamping in _concat_segments could plausibly introduce.
"""
from __future__ import annotations

import os
import sys

import av


def main() -> int:
    path = sys.argv[1]
    problems = []
    with av.open(path) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0] if container.streams.audio else None
        fps = float(video.average_rate)
        pts, key_frames = [], []
        count = 0
        for frame in container.decode(video):
            if frame.pts is not None:
                pts.append(float(frame.pts * video.time_base))
            if frame.key_frame:
                key_frames.append(count)
            count += 1
        width, height = video.width, video.height

    name = os.path.basename(path)
    size = os.path.getsize(path) / 1e6
    audio_desc = (f"{audio.codec_context.name} {audio.rate} Hz x{audio.channels}"
                  if audio is not None else "NONE")
    print(f"{name}: {count} frames {width}x{height} @ {fps:g} fps, {count / fps:.3f} s, "
          f"{size:.2f} MB, audio {audio_desc}")
    print(f"  key frames at {key_frames}")

    if audio is None:
        problems.append("no audio stream")
    if len(pts) != count:
        problems.append(f"{count - len(pts)} frames carry no pts")
    else:
        # Evenly spaced and increasing. A tolerance of a tenth of a frame is generous
        # enough for the time-base rounding in the concatenation and far tighter than any
        # real reordering, which would be a whole frame or more.
        step = 1.0 / fps
        for i in range(1, len(pts)):
            delta = pts[i] - pts[i - 1]
            if abs(delta - step) > 0.1 * step:
                problems.append(f"frame {i}: pts step {delta * fps:.3f} frames, not 1 "
                                f"(pts {pts[i - 1]:.4f} -> {pts[i]:.4f})")
                break

    if problems:
        print("  FAIL -- " + "; ".join(problems))
        return 1
    print("  OK -- every frame decoded, timestamps a clean run, audio present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

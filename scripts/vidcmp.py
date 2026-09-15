#!/usr/bin/env python3
"""Two mp4s differ byte-for-byte. By how much do the *pixels* differ?

`cmp -s` on an encoded file answers "are these the same bytes", which is not the question
the parallel-decode A/B is asking. Reordering a floating-point reduction changes the last
bits of a few pixels, x264 turns that into a different macroblock decision, and the file
diverges from there -- so a byte mismatch is consistent with both "the decode is fine" and
"the decode is broken". Per-frame PSNR separates them: >~55 dB is arithmetic reassociation,
<40 dB is different content, and a mismatch confined to particular frame indices points at
the chunk boundaries.
"""
import sys
import av
import numpy as np

def frames(path):
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            yield f.to_ndarray(format="rgb24")

a, b = sys.argv[1], sys.argv[2]
worst, per_frame, total_se, n_px = (0.0, -1), [], 0.0, 0
for i, (fa, fb) in enumerate(zip(frames(a), frames(b))):
    d = fa.astype(np.int32) - fb.astype(np.int32)
    se = float((d.astype(np.float64) ** 2).sum())
    total_se += se; n_px += d.size
    mse = se / d.size
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    per_frame.append((psnr, int(np.abs(d).max())))
    if psnr < (worst[0] or 1e9) or worst[1] < 0:
        pass
print(f"{len(per_frame)} frames compared")
mse = total_se / n_px
print(f"overall PSNR: {10*np.log10(255.0**2/mse):.2f} dB" if mse else "overall: IDENTICAL pixels")
psnrs = [p for p, _ in per_frame]
maxes = [m for _, m in per_frame]
print(f"max abs diff over all frames: {max(maxes)}")
print(f"per-frame PSNR: min {min(psnrs):.2f}  median {sorted(psnrs)[len(psnrs)//2]:.2f}  max {max(psnrs):.2f}")
ident = [i for i, m in enumerate(maxes) if m == 0]
print(f"bit-identical frames: {len(ident)} of {len(per_frame)}")
worst_idx = sorted(range(len(psnrs)), key=lambda i: psnrs[i])[:12]
print("worst frames (index, PSNR, max abs):")
for i in sorted(worst_idx):
    print(f"  {i:4d}  {psnrs[i]:6.2f} dB  {maxes[i]:3d}")

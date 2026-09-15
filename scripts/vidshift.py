#!/usr/bin/env python3
"""Is B the same video as A, just at a different point in time?

The divergence is low-frequency, which has two very different explanations: the two runs
built different scenes, or they built the same scene with the motion at a different phase.
A temporal cross-correlation separates them -- if some shift d makes A[i] ~ B[i+d], it is
timing; if every shift is equally bad, it is content. Run on a 16x-downsampled copy so the
match is on structure and not on texture.
"""
import sys
import av
import numpy as np

def load(path, k=16):
    with av.open(path) as c:
        v = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]).astype(np.float64)
    f, h, w, ch = v.shape
    h, w = h - h % k, w - w % k
    return v[:, :h, :w].reshape(f, h//k, k, w//k, k, ch).mean(axis=(2, 4))

a, b = load(sys.argv[1]), load(sys.argv[2])
n = min(len(a), len(b))

def psnr(x, y):
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)

print(f"{'shift':>6} {'PSNR':>8}")
best = (0, -1)
for d in range(-24, 25, 2):
    lo, hi = max(0, -d), min(n, n - d)
    p = psnr(a[lo:hi], b[lo + d:hi + d])
    if p > best[1]:
        best = (d, p)
    if abs(d) <= 8 or d % 8 == 0:
        print(f"{d:>6} {p:7.2f} dB")
print(f"\nbest shift {best[0]} at {best[1]:.2f} dB (shift 0 is the first table row for d=0)")

# A different question: does each frame of A have *some* good match anywhere in B?
# If the scene is the same but re-timed non-uniformly, per-frame nearest neighbours are
# good while no single global shift is.
sub_a, sub_b = a[::20], b
d = ((sub_a[:, None] - sub_b[None]) ** 2).mean(axis=(2, 3, 4))
print("\nper-frame nearest match in B (every 20th frame of A):")
print(f"{'A idx':>6} {'best B idx':>11} {'PSNR':>9} {'PSNR at same idx':>18}")
for row, i in enumerate(range(0, len(a), 20)):
    j = int(d[row].argmin())
    p_best = 10 * np.log10(255.0**2 / d[row][j])
    p_same = 10 * np.log10(255.0**2 / d[row][min(i, len(sub_b) - 1)])
    print(f"{i:>6} {j:>11} {p_best:8.2f} dB {p_same:15.2f} dB")

#!/usr/bin/env python3
"""Same scene diverging in detail, or two different videos?

17 dB is ambiguous on its own. Two renders of one prompt that agree on composition but
differ in fine texture score about the same as two unrelated clips, because PSNR is
dominated by high-frequency energy either way. Downsampling strips that: if the structure
agrees, PSNR climbs steeply with the blur factor, and if the content genuinely differs it
stays flat. Also reported is the per-frame mean colour, which is a scene fingerprint no
amount of detail noise moves.
"""
import sys
import av
import numpy as np

def load(path):
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])])

a, b = load(sys.argv[1]), load(sys.argv[2])
n = min(len(a), len(b)); a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)

def psnr(x, y):
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)

def down(v, k):
    f, h, w, c = v.shape
    h, w = h - h % k, w - w % k
    return v[:, :h, :w].reshape(f, h//k, k, w//k, k, c).mean(axis=(2, 4))

print(f"{n} frames\n{'downsample':>12} {'PSNR':>8}")
for k in (1, 2, 4, 8, 16, 48):
    print(f"{k:>10}x   {psnr(down(a, k), down(b, k)):7.2f} dB")

ma, mb = a.mean(axis=(1, 2)), b.mean(axis=(1, 2))   # (frames, 3) mean colour
print(f"\nper-frame mean RGB: max abs difference {np.abs(ma - mb).max():.2f} levels, "
      f"mean {np.abs(ma - mb).mean():.2f}")
print("global mean RGB   A", np.round(a.mean(axis=(0,1,2)), 2), " B", np.round(b.mean(axis=(0,1,2)), 2))

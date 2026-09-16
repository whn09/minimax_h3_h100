#!/usr/bin/env python3
"""Is it melting? Per-frame detail, colour and half-frame asymmetry, so the claim is measured.

    python3 melt_metrics.py a.mp4 b.mp4 ...
    python3 melt_metrics.py --csv curves.csv a.mp4 b.mp4

"融化 / melting" is a temporal claim -- a subject losing structural integrity **while it moves** --
so a single-frame look at the output cannot confirm or refute it, and neither can PSNR between two
arms (`vidcmp.py`), which answers a different question: are these the same pixels. What melting
does to a clip is specific and each part of it has a number:

  sharp   variance of the 3x3 Laplacian on luma, the standard focus measure. Blurred hands, smeared
          faces and dissolved edges all lower it. This is the headline column.
  decay   mean(sharp) over the LAST third of the clip / over the FIRST third. **< 1 is melting**:
          detail present at the start and gone by the end, which is exactly what "melts" describes
          and what a plain per-clip average hides. ~1.0 means the clip holds together.
  flick   mean |sharp(t) - sharp(t-1)| / mean(sharp). Distinguishes a steadily soft clip (low) from
          one breaking down and recovering frame to frame (high) -- the "artifacts on fight scenes"
          failure rather than the "everything is soft" one.
  sat     mean HSV saturation, and `clip` the fraction of pixels at >= 250 in any channel. These
          catch the *other* end of a wrong LoRA scale: 16x too strong reads as lightx2v's own
          threads describe it ("extremely over saturated plastic look", "colors blow out"), too
          weak reads as "the color became dull" (#9). Compare arms B and C on these two columns.
  up/lo   sharp computed on the top and bottom half separately, printed as the ratio. #44 reports
          ref2v ghosting confined to "the upper half of the screen" in 9:16 and clean 16:9, so a
          per-half number is the difference between reproducing that report and guessing at it.

None of these is a quality score, and none replaces watching the file. They exist so that "arm C
melts and arm B does not" is a number that survives being questioned, and so that a regression can
be spotted across seven arms without seven careful viewings. Read them only *between arms rendered
at one seed, one prompt and one reference* -- the absolute values depend on content, so a busy
courtyard scores higher than a static portrait for reasons that have nothing to do with the model.

Needs only `av` and `numpy`, which `vidcmp.py` already uses.
"""
import sys

import av
import numpy as np


def curves(path: str) -> dict[str, np.ndarray]:
    sharp, sharp_up, sharp_lo, grad, sat, clipped = [], [], [], [], [], []
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            a = f.to_ndarray(format="rgb24").astype(np.float32)
            y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

            def lap_var(v: np.ndarray) -> float:
                if v.shape[0] < 3 or v.shape[1] < 3:
                    return float("nan")
                l = (4.0 * v[1:-1, 1:-1] - v[:-2, 1:-1] - v[2:, 1:-1]
                     - v[1:-1, :-2] - v[1:-1, 2:])
                return float(l.var())

            h = y.shape[0] // 2
            sharp.append(lap_var(y))
            sharp_up.append(lap_var(y[:h]))
            sharp_lo.append(lap_var(y[h:]))
            gx = np.diff(y, axis=1)[:-1, :]
            gy = np.diff(y, axis=0)[:, :-1]
            grad.append(float(np.sqrt(gx * gx + gy * gy).mean()))
            mx = a.max(axis=2)
            mn = a.min(axis=2)
            sat.append(float(((mx - mn) / np.maximum(mx, 1.0)).mean()))
            clipped.append(float((a >= 250.0).any(axis=2).mean()))
    return {k: np.asarray(v, dtype=np.float64) for k, v in
            dict(sharp=sharp, sharp_up=sharp_up, sharp_lo=sharp_lo,
                 grad=grad, sat=sat, clipped=clipped).items()}


def summarize(name: str, c: dict[str, np.ndarray]) -> dict:
    s = c["sharp"]
    n = len(s)
    third = max(n // 3, 1)
    first, last = s[:third].mean(), s[-third:].mean()
    return {
        "name": name,
        "frames": n,
        "sharp": s.mean(),
        "decay": last / first if first else float("nan"),
        "flick": np.abs(np.diff(s)).mean() / s.mean() if n > 1 and s.mean() else float("nan"),
        "grad": c["grad"].mean(),
        "sat": c["sat"].mean(),
        "clip": c["clipped"].mean(),
        "up_lo": c["sharp_up"].mean() / c["sharp_lo"].mean() if c["sharp_lo"].mean() else float("nan"),
        "worst": int(np.argmin(s)),
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    csv_path = None
    if args and args[0] == "--csv":
        csv_path = args[1]
        args = args[2:]
    if not args:
        raise SystemExit(__doc__.splitlines()[2].strip())
    rows, all_curves = [], {}
    for p in args:
        c = curves(p)
        all_curves[p] = c
        rows.append(summarize(p.rsplit("/", 1)[-1], c))
    w = max(len(r["name"]) for r in rows)
    print(f"{'clip':{w}s} {'frames':>6} {'sharp':>9} {'decay':>6} {'flick':>6} "
          f"{'grad':>6} {'sat':>6} {'clip%':>6} {'up/lo':>6} {'worst f':>7}")
    for r in rows:
        print(f"{r['name']:{w}s} {r['frames']:6d} {r['sharp']:9.1f} {r['decay']:6.3f} "
              f"{r['flick']:6.3f} {r['grad']:6.2f} {r['sat']:6.3f} {100*r['clip']:6.2f} "
              f"{r['up_lo']:6.3f} {r['worst']:7d}")
    if len(rows) > 1:
        b = rows[0]
        print(f"\nvs {b['name']} (first argument):")
        for r in rows[1:]:
            print(f"  {r['name']:{w}s} sharp {100*(r['sharp']/b['sharp']-1):+7.1f} %   "
                  f"sat {100*(r['sat']/b['sat']-1):+7.1f} %   "
                  f"clip {100*(r['clip']-b['clip']):+6.2f} pp   decay {r['decay']-b['decay']:+.3f}")
    if csv_path:
        keys = list(next(iter(all_curves.values())))
        with open(csv_path, "w") as fh:
            fh.write("clip,frame," + ",".join(keys) + "\n")
            for p, c in all_curves.items():
                for i in range(len(c["sharp"])):
                    fh.write(f"{p},{i}," + ",".join(f"{c[k][i]:.6g}" for k in keys) + "\n")
        print(f"\nper-frame curves -> {csv_path}")

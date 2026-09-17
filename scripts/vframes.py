#!/usr/bin/env python3
"""Contact sheet of a frame range, optionally cropped and zoomed -- for looking, not for scoring.

    python3 vframes.py --clip a.mp4 --range 105:140 --step 2 --cols 6 --out /tmp/hand.png
    python3 vframes.py --clip a.mp4 --range 105:140 --crop 0.55,0.35,0.95,0.85 --zoom 2 --out z.png

WHY A SEPARATE TOOL. `melt_frames.py` decides *which* frames to look at, and every time its score
was wrong the only thing that caught it was looking at frames it had not picked. When someone says
"the hand at 5 s is mush", the right move is to render 4.4-5.8 s and look, not to tune a threshold
until it agrees. --crop takes fractions of width/height so the same call works at 864x480 and
1344x768, and --zoom is nearest-neighbour on purpose: interpolation invents the texture that is
exactly what is in question.
"""
import sys

import cv2
import numpy as np


def lap_var(y: np.ndarray) -> float:
    l = (4.0 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:])
    return float(l.var())


def main(argv: list[str]) -> None:
    opt = {"--step": "1", "--cols": "6", "--zoom": "1", "--scale": "1"}
    for i in range(0, len(argv) - 1, 2):
        opt[argv[i]] = argv[i + 1]
    # --clips LABEL=path,... renders one ROW per clip over the same range and crop. Comparing arms
    # at the same moment is the only way to attribute an artifact, and the arms share seed and
    # prompt, so the same frame index is roughly the same moment.
    clips = ([kv.split("=", 1) for kv in opt["--clips"].split(",")] if "--clips" in opt
             else [["", opt["--clip"]]])
    out = opt["--out"]
    a, b = (int(v) for v in opt["--range"].split(":"))
    step, cols = int(opt["--step"]), int(opt["--cols"])
    zoom, scale = float(opt["--zoom"]), float(opt["--scale"])
    crop = [float(v) for v in opt["--crop"].split(",")] if "--crop" in opt else None

    rows = []
    for label, clip in clips:
      cap = cv2.VideoCapture(clip)
      cells, f = [], 0
      while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if a <= f <= b and (f - a) % step == 0:
            h, w = bgr.shape[:2]
            if crop:
                x0, y0, x1, y1 = (int(crop[0] * w), int(crop[1] * h),
                                  int(crop[2] * w), int(crop[3] * h))
                bgr = bgr[y0:y1, x0:x1]
            if zoom != 1 or scale != 1:
                k = zoom * scale
                bgr = cv2.resize(bgr, None, fx=k, fy=k, interpolation=(
                    cv2.INTER_NEAREST if k > 1 else cv2.INTER_AREA))
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cv2.putText(bgr, f"{label} f{f} t{f / 24.0:.2f}s sharp {lap_var(g):.0f}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cells.append(bgr)
        f += 1
      cap.release()
      if not cells:
        raise SystemExit(f"no frames in range {a}:{b} of {clip}")
      while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
      rows.extend(np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols))
    w0 = min(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == w0 else
            cv2.resize(r, (w0, max(1, round(r.shape[0] * w0 / r.shape[1]))),
                       interpolation=cv2.INTER_AREA) for r in rows]
    cells = rows
    grid = np.vstack(rows)
    cv2.imwrite(out, grid)
    print(f"{len(cells)} cells -> {out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main(sys.argv[1:])

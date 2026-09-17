#!/usr/bin/env python3
"""Which frames melt, and by how much -- the per-frame, per-region view `melt_metrics.py` averages away.

    python3 melt_frames.py A=a.mp4 B=b.mp4
    python3 melt_frames.py --top 8 --sheet /tmp/dips.png A=a.mp4 B=b.mp4

WHY THIS EXISTS. `melt_metrics.py` answers "is the whole clip soft, and does it get softer over
time" (sharp / decay / flick), and on the customer's actual complaint it says "no melting": arm B
scores sharp 40.7 and decay 0.954, indistinguishable from a clean clip. But the complaint is not
"the clip is soft" -- it is "*this frame* has a blurred face while the rest of the clip is fine".
A clip-mean cannot see that, and neither can `decay`, which is a mean over the last third: one
bad frame in 345 moves a 345-frame mean by 0.3 %. Worse, the blur is *local* -- a face is ~5 % of
a 864x480 frame, so even a per-frame global Laplacian variance barely dips when the face
dissolves.

SO THE METRIC IS LOCAL, RELATIVE AND TRANSIENT. Each frame is cut into a TILES grid; every tile's
sharpness is divided by a ROLLING median of that same tile over +-WINDOW//2 frames, not by the
clip median. The rolling baseline is the whole point: a handheld camera and a spinning subject
change *what is in a tile* constantly, so tile-vs-clip-median flags a tile that merely panned onto
flat sky -- the first version of this script scored 330 of 345 frames as events on the 50-step
control, which is visibly stable, and that is what a wrong baseline looks like. Against a rolling
baseline a content change moves the baseline with it, and only a detail collapse that RECOVERS
stands out, which is exactly what "melts and comes back" means.

AND THE SCORE IS CONDITIONAL ON MOTION, which is the third and decisive calibration. Detail loss
alone *is* motion blur: ranking frames by raw loss boxed whipping hair in every arm including the
50-step control, and hair tips at 24 fps genuinely lose their high frequencies -- correctly so. So
every (frame, tile) sample is binned by its own local motion (mean |luma(t) - luma(t-1)| inside
that tile), and the score is a robust z of the loss WITHIN its motion bin: "this region lost more
detail than regions of this clip normally lose when they move this fast". Motion blur is the
per-bin median and scores ~0; a face that dissolves while barely moving is a large positive z.
That is the customer's complaint, stated as a number.

dip = 1.0 means every region is as sharp as it just was; dip = 0.4 means some region lost 60 % of
its detail for a moment. Reported per clip: the z of the worst frames, how many frames clear
Z_EVENT, and the local motion at the hit so a hair-whip hit is recognisable on sight. Tiles flatter
than DETAIL_PCTL of the clip's tile medians are excluded -- a flat wall's ratio is noise either way.

The sheet is the evidence, not the numbers: one row per clip, one column per worst frame, with
the offending tile boxed in red. Frame indices are per clip, because two arms at the same seed
still drift -- so the columns are that clip's own worst frames, printed in the caption.
"""
import sys

import cv2
import numpy as np

TILES = (4, 6)          # rows x cols. 864x480 -> 120x144 px tiles, a face is ~1 tile
THRESH = 0.55           # "this region lost 45 % of the detail it had a moment ago" == a blur event
DETAIL_PCTL = 40        # tiles flatter than this percentile of median sharpness are noise
WINDOW = 9              # rolling-baseline length in frames (0.37 s at 24 fps); odd, centred
MOT_BINS = 8            # motion deciles-ish; the loss z-score is computed within each bin
Z_EVENT = 4.0           # robust z above the same-motion median that counts as a blur event
MOT_STATIC = 2.0        # mean |dluma| per pixel below which a tile counts as "not moving"
LUMA_TOL = 6.0          # 8-bit mean-luma drift allowed before "same content" stops being true
BASE_MIN = 0.0          # absolute Laplacian-variance floor on the pre-collapse baseline
STRIP_HALF = 4          # --strip shows the hit tile over +-this many frames, to prove it recovers


def lap_var(y: np.ndarray) -> float:
    if y.shape[0] < 3 or y.shape[1] < 3:
        return float("nan")
    l = (4.0 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:])
    return float(l.var())


def read(path: str):
    """-> frames BGR, and per tile [n, rows, cols]: sharpness, motion, mean luma.

    Motion is per tile, not per frame: the whole point of the score is that a tile is compared
    against tiles that moved as much as it did, and a global frame mean cannot tell whipping hair
    from the static wall behind it in the same frame.
    """
    cap = cv2.VideoCapture(path)
    frames, tiles, motion, luma, prev = [], [], [], [], None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
        y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = y.shape
        rs, cs = TILES
        t = np.empty(TILES, dtype=np.float64)
        m = np.zeros(TILES, dtype=np.float64)
        b = np.empty(TILES, dtype=np.float64)
        d = None if prev is None else np.abs(y - prev)
        for r in range(rs):
            for c in range(cs):
                sl = (slice(r * h // rs, (r + 1) * h // rs), slice(c * w // cs, (c + 1) * w // cs))
                t[r, c] = lap_var(y[sl])
                b[r, c] = float(y[sl].mean())
                if d is not None:
                    m[r, c] = float(d[sl].mean())
        tiles.append(t)
        motion.append(m)
        luma.append(b)
        prev = y
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {path}")
    mot = np.asarray(motion)
    if len(mot) > 1:
        mot[0] = mot[1]                                  # frame 0 has no predecessor
    return frames, np.asarray(tiles), mot, np.asarray(luma)


def analyse(tiles: np.ndarray, motion: np.ndarray, luma: np.ndarray):
    n = len(tiles)
    med = np.median(tiles, axis=0)                       # [rows, cols], only to pick detail tiles
    keep = med >= np.percentile(med, DETAIL_PCTL)        # detail-carrying tiles only
    half = WINDOW // 2
    # Rolling median per tile, centred, edges clamped. Small n and small tile count, so the
    # straightforward stack beats being clever.
    idx = np.clip(np.arange(n)[:, None] + np.arange(-half, half + 1)[None, :], 0, n - 1)
    base = np.median(tiles[idx], axis=1)                 # [n, rows, cols]
    # Absolute loss, not the ratio: a ratio picks the darkest, flattest tile in the frame -- the
    # back of a head against a wall drops to 0.10 because its baseline is tiny -- and the sheet it
    # produced boxed hair and cobblestones in every arm.
    scale = float(np.median(med[keep])) if keep.any() else 1.0
    loss = (base - tiles) / max(scale, 1e-6)
    # ... but absolute loss alone IS motion blur, and ranking by it boxed whipping hair even on the
    # visibly stable 50-step control. So compare each sample only against samples that moved as
    # much as it did: bin by local motion, take a robust z within the bin. Motion blur sits at the
    # bin median (z ~ 0); a region that dissolves without moving is a large positive z.
    ok = keep[None] & np.isfinite(loss) & np.isfinite(motion)
    z = np.full(loss.shape, -np.inf)
    mv = motion[ok]
    if mv.size:
        edges = np.unique(np.percentile(mv, np.linspace(0, 100, MOT_BINS + 1)))
        which = np.clip(np.digitize(motion, edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            sel = ok & (which == b)
            if sel.sum() < 20:                           # too few to estimate a spread
                continue
            v = loss[sel]
            m = np.median(v)
            mad = np.median(np.abs(v - m)) * 1.4826
            z[sel] = (v - m) / max(mad, 1e-6)
    z = z.reshape(n, -1)
    where = z.argmax(axis=1)
    rows = np.arange(n)
    hit_z = z[rows, where]
    flat_t, flat_b = tiles.reshape(n, -1), base.reshape(n, -1)
    # The headline count, and the only number that is comparable BETWEEN clips: a detail-carrying
    # tile that is essentially still (mot < MOT_STATIC) yet lost more than 1-THRESH of the detail it
    # had 4 frames ago. The z above is a within-clip robust score -- good for ranking frames inside
    # one clip, not for ranking arms, because its MAD denominator is the clip's own spread. `static`
    # has no such denominator: it is a ratio against the tile's own recent past plus an absolute
    # motion gate, so hair whipping at mot 28 can never enter it, and 30 arms can be sorted by it.
    # SAME CONTENT, NOT JUST SAME PLACE. Without this gate the count is occlusion, not melting: the
    # deepest "static" events in the base 50-step control were the tile going black under the
    # subject's hair, in the 8-step base clip the sun blowing it out to white, and in the 2048 LoRA
    # clip the dark jacket sweeping in. All three lose Laplacian energy honestly, and none is what
    # the customer sees. Melting keeps the region's brightness and loses its texture, so require the
    # tile's mean luma to sit within LUMA_TOL of its own rolling baseline.
    # And the floor is ABSOLUTE, on the baseline, not on the tile's clip median. That distinction is
    # what the luma gate could not fix: the base model's deepest "melts" were dark hair going 15 -> 1
    # and blown-out sky going 4 -> 1, both at constant brightness, both a ratio of 0.10 on a region
    # that had nothing to lose. The delivered LoRA clip's are textured pavement going 52 -> 6. So the
    # region has to have carried real detail a moment earlier -- half the clip's typical tile -- or
    # the ratio is dividing noise by noise.
    lbase = np.median(luma[idx], axis=1)
    same = np.abs(luma - lbase) <= LUMA_TOL
    still = (keep[None] & same & (motion < MOT_STATIC) & (tiles < THRESH * base)
             & (base >= BASE_MIN))
    # Rank the still events by depth, and hand back which tile each one is, so --strip can show the
    # events that the count is actually made of. Anything else lets the headline number and the
    # picture disagree, which is how the first two versions of this script went wrong.
    sr = np.where(still, tiles / np.maximum(base, 1e-6), np.inf).reshape(n, -1)
    still_where = sr.argmin(axis=1)
    still_ratio = sr[rows, still_where]
    still_mot = motion.reshape(n, -1)[rows, still_where]
    still_base = flat_b[rows, still_where]
    return hit_z, flat_t, flat_b, still_ratio, still_where, still_mot, still_base


def main(argv: list[str]) -> None:
    top, sheet, strip, pick, clips = 6, None, None, {}, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--top":
            top = int(argv[i + 1]); i += 2
        elif a == "--sheet":
            sheet = argv[i + 1]; i += 2
        elif a == "--strip":
            strip = argv[i + 1]; i += 2
        elif a == "--pick":
            # LABEL=frame,LABEL=frame -- strip THIS frame instead of the clip's deepest event, so a
            # named candidate can be checked by eye. The detector nominates; the eye decides.
            pick = dict((k, int(v)) for k, v in
                        (kv.split("=") for kv in argv[i + 1].split(","))); i += 2
        elif "=" in a:
            clips.append(tuple(a.split("=", 1))); i += 1
        else:
            raise SystemExit(f"unexpected argument {a!r}; use LABEL=path.mp4")
    if not clips:
        raise SystemExit(__doc__)

    rows, worst_all = [], {}
    print(f"{'clip':>10s} {'frames':>6s} {'still melt':>10s} {'per 10 s':>8s} "
          f"{'z max':>7s} {'z p99':>7s} {'mot mean':>8s}")
    for label, path in clips:
        frames, tiles, motion, luma = read(path)
        z, flat_t, flat_b, sratio, swhere, smot, sbase = analyse(tiles, motion, luma)
        # Rank and display the STILL events, deepest first -- they are what the count counts.
        order = np.argsort(sratio)[:top]
        order = order[np.isfinite(sratio[order])]
        worst_all[label] = (frames, sratio, swhere, order, flat_t, flat_b)
        nst = int(np.isfinite(sratio).sum())
        print(f"{label:>10s} {len(frames):>6d} {nst:>10d} "
              f"{nst / (len(frames) / 240.0):>8.1f} {z.max():>7.1f} "
              f"{np.percentile(z, 99):>7.1f} {motion.mean():>8.2f}")
        rows.append((label, [(int(f), float(sratio[f]), int(swhere[f]), float(z[f]),
                              float(smot[f]), float(sbase[f])) for f in order]))

    print()
    for label, ws in rows:
        print(f"{label:>10s} still melts: " +
              ("  ".join(f"f{f}(t{f / 24.0:.2f}s) kept {d:.2f} of base {bs:.0f} mot {m:.1f}"
                         for f, d, _, zz, m, bs in ws) or "none"))

    if strip:
        # The one check the sheet cannot make: is the tile's own content sharp before and after?
        # A melt is "sharp -> mush -> sharp" on the SAME content; a leaking baseline is the subject's
        # hair sweeping through the tile in the neighbour frames and lifting the median. Only a strip
        # of that same tile across +-STRIP_HALF frames tells the two apart, so it is the evidence.
        rs, cs = TILES
        cells = []
        for label, ws in rows:
            frames, dip, where, order, flat_t, flat_b = worst_all[label]
            if not ws:
                continue
            f0, _, w, zz, _, _ = ws[0]
            if label in pick:
                f0 = pick[label]
                w = int(where[f0]) if np.isfinite(dip[f0]) else w
            r, c = divmod(w, cs)
            row = []
            for f in range(max(0, f0 - STRIP_HALF), min(len(frames), f0 + STRIP_HALF + 1)):
                img = frames[f]
                h, wd = img.shape[:2]
                th, tw = h // rs, wd // cs
                crop = cv2.resize(img[r * th:(r + 1) * th, c * tw:(c + 1) * tw], (tw * 2, th * 2),
                                  interpolation=cv2.INTER_NEAREST)
                cv2.putText(crop, f"{label} f{f} sharp {flat_t[f, w]:.0f}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 255) if f == f0 else (0, 255, 255), 1, cv2.LINE_AA)
                row.append(crop)
            if row:
                cells.append(np.hstack(row))
        w0 = min(c.shape[1] for c in cells)
        cells = [c if c.shape[1] == w0 else
                 cv2.resize(c, (w0, max(1, round(c.shape[0] * w0 / c.shape[1]))),
                            interpolation=cv2.INTER_AREA)
                 for c in cells]
        cv2.imwrite(strip, np.vstack(cells))
        print(f"\nstrip -> {strip}")

    if sheet:
        rs, cs = TILES
        tile_h = tile_w = None
        cells = []
        for label, ws in rows:
            frames, dip, where, order, flat_t, flat_b = worst_all[label]
            row = []
            for f, d, w, zz, m, bs in ws:
                img = frames[f].copy()
                h, wd = img.shape[:2]
                tile_h, tile_w = h // rs, wd // cs
                r, c = divmod(w, cs)
                y0, x0 = r * tile_h, c * tile_w
                # 2x zoom of the offending tile, pasted top-right: the whole point is to see the
                # texture, and a 144x120 tile inside a 6-column sheet is too small to judge.
                crop = cv2.resize(img[y0:y0 + tile_h, x0:x0 + tile_w], (tile_w * 2, tile_h * 2),
                                  interpolation=cv2.INTER_NEAREST)
                zh, zw = crop.shape[:2]
                zh, zw = min(zh, h), min(zw, wd)
                cv2.rectangle(img, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (0, 0, 255), 2)
                img[0:zh, wd - zw:wd] = crop[0:zh, 0:zw]
                cv2.rectangle(img, (wd - zw, 0), (wd - 1, zh - 1), (0, 255, 255), 2)
                cv2.putText(img, f"{label} f{f} kept {d:.2f} mot {m:.1f}", (6, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                row.append(img)
            if row:
                cells.append(np.hstack(row))
        # Rows can come from different canvases (864x480 next to 1344x768), so normalise the row
        # width before stacking rather than refusing to mix arms -- comparing 480p against 768p is
        # the reason this is one sheet.
        w0 = min(c.shape[1] for c in cells)
        cells = [c if c.shape[1] == w0 else
                 cv2.resize(c, (w0, max(1, round(c.shape[0] * w0 / c.shape[1]))),
                            interpolation=cv2.INTER_AREA)
                 for c in cells]
        cv2.imwrite(sheet, np.vstack(cells))
        print(f"\nsheet -> {sheet}")


if __name__ == "__main__":
    main(sys.argv[1:])

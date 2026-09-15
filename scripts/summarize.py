#!/usr/bin/env python3
"""Turn the grid's <tag>.mp4.inference.json records into one markdown table.

    python3 summarize.py /opt/dlami/nvme/vdn/out

Reads only what infer_ulysses.py wrote, so the table cannot disagree with the run: the
canvas and frame count come from the resolved config in the record, not from the tag.
s/NFE is `denoise_seconds / num_steps` -- the same convention as upstream's published
table, i.e. after render.warmup_steps NFE in the same process and excluding model load,
VAE decode and mp4 mux. E2E is the whole process, which is what a user waits.
"""
import json
import sys
from pathlib import Path

FPS = 24


def aligned_frames(n: int) -> int:
    """`render.num_frames` is a REQUEST; the render uses diffusers'
    align_num_frames(n, 17, 5), which snaps up to the VAE's 17-frames-to-5-latents
    chunking. The reachable lengths are therefore 5 + 17k frames -- around 15 s that is
    345 (14.375 s) or 362 (15.083 s), nothing between. Reproduced here rather than
    imported so the table can be regenerated without the venv; verified against
    diffusers for n in 1..363."""
    if n <= 5:
        return 5
    return 5 + 17 * -(-(n - 5) // 17)


def rows(out_dir: Path):
    for path in sorted(out_dir.glob("*.mp4.inference.json")):
        record = json.loads(path.read_text())
        # "overlay" is render_record's name for resolved_dict(cfg) -- the fully resolved
        # config, defaults included, so the canvas here is what the render actually used.
        cfg, timings = record["overlay"], record.get("timings", {})
        parallel = record.get("parallel", {})
        render = cfg["render"]
        frames = aligned_frames(render["num_frames"])
        yield {
            "tag": path.name[: -len(".mp4.inference.json")],
            "canvas": f'{render.get("width", 1344)}x{render.get("height", 768)}',
            "frames": frames,
            "seconds": frames / FPS,
            "gpus": parallel.get("world_size", 1),
            "split": (f'{parallel["softmax_ranks"]}+'
                      f'{parallel["world_size"] - parallel["softmax_ranks"]}'
                      if parallel.get("softmax_ranks") else "standard"),
            # The packed sequence length, summed over the Ulysses shards: text + audio +
            # video rows, which is the quantity the DiT's cost actually scales with.
            "rows": sum(parallel.get("sequence_splits", [])) or None,
            "nfe": render["num_steps"],
            "s_per_nfe": timings.get("seconds_per_step"),
            "denoise": timings.get("denoise_seconds"),
            "decode": timings.get("decode_and_encode_seconds"),
            "setup": timings.get("model_setup_seconds"),
            "e2e": timings.get("end_to_end_seconds"),
        }


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/dlami/nvme/vdn/out")
    table = list(rows(out_dir))
    if not table:
        print(f"no records under {out_dir}")
        return
    header = ("| tag | canvas | clip | rows | GPUs | split | NFE | s/NFE | denoise | "
              "decode+mux | E2E | realtime |")
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for r in table:
        # "realtime" = clip seconds per wall second of denoising: >1 means the model
        # produces video faster than it plays, which is VDN's headline claim.
        realtime = r["seconds"] / r["denoise"] if r["denoise"] else 0
        print(f'| {r["tag"]} | {r["canvas"]} | {r["frames"]}f / {r["seconds"]:.2f}s | '
              f'{r["rows"] or "-"} | '
              f'{r["gpus"]} | {r["split"]} | {r["nfe"]} | {r["s_per_nfe"]:.3f} | '
              f'{r["denoise"]:.2f}s | {r["decode"]:.2f}s | {r["e2e"]:.2f}s | '
              f'{realtime:.2f}x |')


if __name__ == "__main__":
    main()

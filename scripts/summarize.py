#!/usr/bin/env python3
"""Turn the grid's <tag>.mp4.inference.json records into markdown tables.

    python3 summarize.py /opt/dlami/nvme/vdn/out

Reads only what infer_ulysses.py wrote, so the tables cannot disagree with the run: the
canvas and frame count come from the resolved config in the record, not from the tag.

Two conventions, both printed, because they answer different questions:

  s/NFE = denoise_seconds / num_steps. Upstream's published convention -- after
    render.warmup_steps NFE in the same process, and EXCLUDING model load, VAE decode and
    mp4 mux. Use this and only this to compare against their 2.29 (H200) / 1.40 (B200).

  post-warmup E2E = denoise + transformer offload + decoder load + decode&mux. What a
    request costs once the process is up, which is the number that was actually asked for.
    `end_to_end_seconds` in the record is NOT this: it starts before the ~200 s host-side
    fp8 assembly, which happens once per process and never again.

  steady = post-warmup E2E minus the decoder load, since the decoders stay resident. That
    is the per-request figure for a server that has already served one request.
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
        # infer.py (the single-GPU entrypoint, used for the 1-GPU point of the scaling
        # curve) records only step_seconds -- no aggregates, no decode or E2E, and no
        # "parallel" section at all. Derive what can be derived and leave the rest blank
        # rather than dropping the row: the s/NFE is the comparable quantity anyway.
        steps = timings.get("step_seconds") or []
        denoise = timings.get("denoise_seconds") or (sum(steps) if steps else None)
        s_per_nfe = timings.get("seconds_per_step") or (
            denoise / len(steps) if denoise and steps else None)
        # transformer_release_seconds since patch 10; the old key was
        # transformer_offload_seconds, when copying to the host was the only mode.
        offload = (timings.get("transformer_release_seconds")
                   or timings.get("transformer_offload_seconds"))
        loading = timings.get("decoder_load_seconds")
        decode = timings.get("decode_and_encode_seconds")
        # The collective decode is timed OUTSIDE decode_and_save (a collective cannot live
        # in a rank-0-only branch), so it is a sibling key, not a stage. Fold it in, and
        # keep it visible in the stage table under its own name.
        parallel_vae = timings.get("parallel_video_vae_seconds")
        request = None
        if denoise is not None and decode is not None:
            request = denoise + decode + (offload or 0) + (parallel_vae or 0) + (loading or 0)
        stages = dict(timings.get("decode_stages") or {})
        if parallel_vae is not None:
            world = parallel.get("world_size", 1)
            stages = {f"video_vae_x{world}": parallel_vae,
                      **{k.removesuffix("_seconds"): v for k, v in stages.items()}}
        else:
            stages = {k.removesuffix("_seconds"): v for k, v in stages.items()}
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
            "s_per_nfe": s_per_nfe,
            "denoise": denoise,
            "offload": offload,
            "load": loading,
            "decode": (decode + parallel_vae) if (decode is not None and parallel_vae) else decode,
            "request": request,
            "steady": (request - loading) if (request is not None and loading is not None) else request,
            "setup": timings.get("model_setup_seconds"),
            "stages": stages,
        }


def secs(value):
    return f"{value:.2f}s" if value is not None else "-"


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/dlami/nvme/vdn/out")
    table = list(rows(out_dir))
    if not table:
        print(f"no records under {out_dir}")
        return

    header = ("| tag | canvas | clip | rows | GPUs | split | NFE | s/NFE | denoise | "
              "offload | load | decode+mux | post-warmup E2E | steady | realtime |")
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for r in table:
        # "realtime" = clip seconds per wall second of denoising: >1 means the model
        # produces video faster than it plays, which is VDN's headline claim.
        realtime = f'{r["seconds"] / r["denoise"]:.2f}x' if r["denoise"] else "-"
        per_nfe = f'{r["s_per_nfe"]:.3f}' if r["s_per_nfe"] is not None else "-"
        print(f'| {r["tag"]} | {r["canvas"]} | {r["frames"]}f / {r["seconds"]:.2f}s | '
              f'{r["rows"] or "-"} | {r["gpus"]} | {r["split"]} | {r["nfe"]} | '
              f'{per_nfe} | {secs(r["denoise"])} | {secs(r["offload"])} | '
              f'{secs(r["load"])} | {secs(r["decode"])} | {secs(r["request"])} | '
              f'{secs(r["steady"])} | {realtime} |')

    # Which part of "decode" is which. Only the arms measured after patch 0004 have this,
    # so the table is built from the union of the keys that are actually present rather
    # than from a fixed list -- an older record simply has no stage columns.
    names = []
    for r in table:
        for name in r["stages"]:
            if name not in names:
                names.append(name)
    if not names:
        return
    print()
    print("| tag | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))
    for r in table:
        if not r["stages"]:
            continue
        print(f'| {r["tag"]} | '
              + " | ".join(secs(r["stages"].get(n)) for n in names) + " |")


if __name__ == "__main__":
    main()

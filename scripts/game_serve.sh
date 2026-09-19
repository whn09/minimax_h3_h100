#!/usr/bin/env bash
# Serve ONE replica of the interactive-game workload: 480p, 345 frames, VDN-H3-8-step, over HTTP.
#
#   bash game_serve.sh                      # auto-detect GPUs, 864x480, 345 f, port 30010
#   GPUS=8 PORT=30010 bash game_serve.sh    # pin them
#   FRAMES=362 bash game_serve.sh           # 15.083 s, the other length measured in RESULTS.md
#   FRAMES=243 bash game_serve.sh           # 10.125 s, if a shorter beat suits the game better
#   bash game_serve.sh stop
#
# WHY NOT JUST sglang_arm.sh. That file is the measurement driver: its defaults (345 frames,
# GPUS=8) are the published workload, and every number in RESULTS.md and RUNBOOK.md is quoted
# against it. Retuning its defaults for a game would make those numbers refer to a command that no
# longer exists. The serve flags below are transcribed from it unchanged -- only the shape, the GPU
# count and the warmup differ. If sglang_arm.sh gains a flag, this file needs the same flag.
#
# THE FRAME COUNT IS A LATTICE, NOT A DIAL. `align_num_frames(n, 17, 5)` snaps to the VAE's
# 17-frames-to-5-latents chunking, so the reachable lengths are 5 + 17k: ... 226, 243, 260, 277 ...
# 328, 345, 362. A request off the lattice is not an error, it is silently rounded, which is worse --
# you get a clip of a length you did not ask for and no log line saying so. This script REFUSES
# instead. 345 f = 14.375 s is the default because it is the shape every number in RESULTS.md was
# measured at; 362 f = 15.083 s is the answer when the requirement is literally "15 seconds", and
# there is nothing in between.
#
# WHAT THIS IS NOT. It is not a real-time renderer. The fastest measured configuration in this repo
# is 8.02 s end-to-end for this exact shape on eight H100s, of which ~6.9 s is server-side inference.
# A game can pipeline that (render ahead of the player, or treat each clip as a scene beat) but it
# cannot wait on it inside a frame loop. GAME.md has the numbers per GPU count and what they mean for
# the loop.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/sglang}
VDNROOT=${VDNROOT:-/opt/dlami/nvme/vdn}
MODEL=${MODEL:-OpenVDN/vdn-minimax-h3}
PORT=${PORT:-30010}
FRAMES=${FRAMES:-345}          # 14.375 s at 24 fps -- the shape RESULTS.md measured at 8.02 s
CANVAS=${CANVAS:-864x480}
export HF_HOME=${HF_HOME:-$VDNROOT/hf}
export SGLANG_DIFFUSION_CACHE_ROOT=${SGLANG_DIFFUSION_CACHE_ROOT:-$ROOT/cache}

mode=${1:-serve}
# shellcheck source=_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"   # CUDA_HOME, NCCL_NET_PLUGIN=none, expandable_segments

if [ "$mode" = stop ]; then
  # The [s] is not decoration -- see sglang_arm.sh: an unbracketed pattern matches the ssh command
  # line that carries it and kills the session issuing the stop.
  pkill -f '[s]glang.*serve'
  sleep 5; echo stopped
  exit 0
fi

# ---------------------------------------------------------------- the shape has to be reachable
if [ $(( (FRAMES - 5) % 17 )) -ne 0 ] || [ "$FRAMES" -lt 22 ]; then
  echo "FRAMES=$FRAMES is not reachable. align_num_frames(n, 17, 5) gives 5 + 17k only."
  echo "Near 10 s: 226 (9.417 s), 243 (10.125 s), 260 (10.833 s)."
  echo "Near 15 s: 328 (13.667 s), 345 (14.375 s, measured), 362 (15.083 s, measured)."
  exit 1
fi
SECONDS_OUT=$(python3 -c "print($FRAMES/24)")

# ---------------------------------------------------------------- however many cards this box has
# sglang_arm.sh hardcodes GPUS=8 because it only ever ran on a p5.48xlarge. This is meant to be
# copied onto whatever box exists, so it counts. --ulysses-degree tracks --num-gpus: Ulysses shards
# the SEQUENCE, so every rank replicates the whole DiT and the degree is just "how many ways".
GPUS=${GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 1)}
[ "$GPUS" -ge 1 ] || { echo "no GPU visible"; exit 1; }

# The measured 8-card peak is 54.7 GB/GPU at 480p/345f with the DiT, both VAEs and the Qwen3-VL
# conditioner all resident (RESULTS.md, "SGLang Diffusion is faster than all of it"). The weights do
# not shard, so fewer ranks means the SAME resident footprint and a LARGER activation share -- 1/G of
# the sequence instead of 1/8. On one card that share is 8x, and by arithmetic off the 8-card peak it
# lands within a couple of GB of what an H100 gives PyTorch. NOT MEASURED, on either side: treat a
# single-card OOM during warmup as expected-but-unconfirmed and climb the ladder in RUNBOOK 1c
# (--performance-mode auto, then --layerwise-offload-components text_encoder), recording which rung
# you used, because "it serves" and "it serves while offloading" are different claims.
if [ "$GPUS" -le 1 ]; then
  echo "NOTE: one rank holds the whole sequence's activations on top of a footprint measured at"
  echo "      54.7 GB/GPU with 8 ranks. If warmup OOMs, that is the reason -- see RUNBOOK 1c."
fi

mkdir -p "$ROOT/logs" "$VDNROOT/outputs"
LOG=$ROOT/logs/game_${CANVAS}_${FRAMES}f_g${GPUS}.log
shift || true

# --quantization fp8            SM90 per-channel fp8, online. The cookbook's mxfp8 default is SM100+.
# --attention-backend hybrid_window_attn_h3
#                               REQUIRED. A dense backend does not error -- it silently skips the
#                               linear branch and the gates, i.e. serves a different model.
# --encoder-parallel auto       folds the Qwen3-VL conditioner across ranks that are otherwise idle.
# --performance-mode speed      keeps every component resident. First rung to trade if it OOMs.
# --warmup-num-frames           honoured: the warmup renders FRAMES frames, so the temporal shape and
#                               the allocator growth that goes with it are paid before request 1.
# --warmup-resolutions          DOES NOT DO WHAT IT SAYS ON H3, and it is passed anyway because the
#                               aspect ratio is the part it does control. Upstream's
#                               _synthetic_warmup_target() (configs/sample/minimax_h3.py) uses this
#                               string ONLY to pick the nearest entry in MINIMAX_H3_FINITE_ASPECT_-
#                               RATIOS; short_edge is MINIMAX_H3_RECOMMENDED_SHORT_EDGE = 768,
#                               hardcoded. So a 480p server warms at 768p -- the log line to check is
#                               "server warmup req (1344x768x345f)", which is what it says here. Cost
#                               measured on this box: the first 480p request took 7.58 s of server
#                               time against a 6.70 s steady state. Fix it from the client side:
#                               `game_client.py --warm` / `Renderer.warm()` throws away one clip per
#                               replica at the real shape.
# --host 127.0.0.1              loopback ONLY, deliberately. The way in from a laptop is the ssh
#                               tunnel in game_tunnel.sh; binding 0.0.0.0 would put an unauthenticated
#                               video generator on the VPC (and on the internet, if the security group
#                               is ever widened) for no gain, since -L reaches loopback either way.
set -x
sglang serve \
  --model-path "$MODEL" \
  --num-gpus "$GPUS" \
  --ulysses-degree "$GPUS" \
  --quantization fp8 \
  --attention-backend hybrid_window_attn_h3 \
  --encoder-parallel auto \
  --performance-mode speed \
  --warmup-num-frames "$FRAMES" \
  --warmup-resolutions "$CANVAS" \
  --host 127.0.0.1 --port "$PORT" \
  "$@" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"

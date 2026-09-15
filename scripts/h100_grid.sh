#!/usr/bin/env bash
# The driver behind RESULTS.md: how fast can 8x H100 render a 480p clip?
#
# Clip length is 345 frames (14.375 s) everywhere except one arm, so the comparison with
# upstream's published 768p number moves ONE variable -- the canvas. 362 frames (15.083 s)
# is the smallest clip that actually reaches 15 s and gets its own arm.
#
#   setsid nohup bash h100_grid.sh > /opt/dlami/nvme/vdn/grid.log 2>&1 < /dev/null &
#   PHASES=c bash h100_grid.sh        # one phase
#
# Phases (PHASES, default "cbsg"):
#   c  CONTROL     768p / 345 frames, the shape upstream publishes (8x H200 =
#                  2.29 s/NFE), at both splits -- so the H100 gap is measured, not
#                  assumed. Needs patch 0003 as well: rank 0 is the only rank carrying
#                  decoders, and at 768p those ~11 GiB are what it does not have.
#   b  BASELINE    480p / 345 frames at the H200-tuned 6+2, plus the 362-frame
#                  (15.083 s) variant -- the literal "15 second" answer
#   s  SPLIT       480p / 345 frames sweeping parallel.softmax_ranks: 0 (standard Ulysses),
#                  4, 5, 6, 7. The 6+2 in upstream's H200 file was tuned on a 104k-row
#                  768p sequence; 480p/345f is 44k rows, so the balance point moves.
#                  Measured: it does -- standard Ulysses wins at this size.
#   g  SCALE       the best 480p split at 1 / 2 / 4 / 8 GPUs -- the Ulysses efficiency
#                  curve, which is what says whether 8 GPUs is the right answer at all
#   n  STEPS       4 / 6 / 8 NFE at the best config (quality falls off below 8; this
#                  only measures what the step budget costs)
#
# Timing convention, matching upstream's table: `denoise_seconds / num_steps` is the
# s/NFE, measured AFTER render.warmup_steps=2 NFE have run in the same process, so the
# kernels are compiled and the second-timestep re-specialisation has happened. E2E
# additionally carries model load, VAE decode and mp4 mux; both are recorded.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/vdn}
REPO=$ROOT/vdn-minimax-h3
OUT=${OUT:-$ROOT/out}
CKPT=${CKPT:-ckpts/stage-dmd-step-250}
PROMPT=${PROMPT:-prompts/example_2.pt}
PHASES=${PHASES:-cbsg}

mkdir -p "$OUT"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
# The 63 GB Qwen3-VL text encoder lives on the NVMe, not under $HOME -- / is 484 GB and the
# checkpoint alone is 82. Only phase f's keyframe re-encode reads it, but the default would
# silently start a second 63 GB download onto the small disk.
export HF_HOME=${HF_HOME:-$ROOT/hf}

# run <tag> <config> <gpus> [overrides...]
run() {
  local tag=$1 config=$2 gpus=$3; shift 3
  local log=$OUT/$tag.log
  if [ -s "$OUT/$tag.mp4.inference.json" ]; then echo "SKIP $tag (already done)"; return; fi
  echo "=== [$(date -u +%H:%M:%S)] $tag  (gpus=$gpus, $config, $*)"
  # A dead rank from a previous arm leaves the port bound; --standalone picks a free one,
  # but stale processes still hold HBM, so make each arm start from a clean box.
  pkill -f 'infer_ulysses\.py' 2>/dev/null; sleep 3
  # OMP_NUM_THREADS: torchrun's own default is 1, which is right for the render but makes
  # patch 2's host-side assembly (LoRA merge + fp8 quantise on the CPU) single-threaded.
  # Measured on this box: 1 thread was still going after 6 minutes; 192/nproc finishes the
  # whole assembly in ~200 s. The render itself is unaffected -- it is all on the device.
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpus - 1))) \
  OMP_NUM_THREADS=$((192 / gpus)) MKL_NUM_THREADS=$((192 / gpus)) \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 3600 torchrun --standalone --nproc_per_node="$gpus" src/inference/infer_ulysses.py \
      --config "configs/inference/$config" \
      checkpoint="$CKPT" \
      render.prompt_file="${PROMPT_OVERRIDE:-$PROMPT}" \
      render.out="$OUT/$tag.mp4" \
      render.record=true \
      "$@" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "    FAILED rc=$rc -- last lines:"; tail -6 "$log" | sed 's/^/    /'
    # A large canvas fails in the decode, AFTER the denoise has been measured. That number
    # is the point of the arm, so surface it even though no record was written.
    grep -E '^(canvas|denoise )' "$log" | sed 's/^/    salvaged: /'
    return
  fi
  grep -E '^(canvas|Ulysses timing|branch-parallel|standard Ulysses)' "$log" | sed 's/^/    /'
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ' - | sed 's/^/    HBM after: /'
}

# run1 <tag> <config> [overrides...] -- the single-GPU entrypoint, for the 1-GPU point of
# the scaling curve. No torchrun, so nothing forces OMP_NUM_THREADS down; set it anyway so
# the host-side assembly matches the other arms.
#
# The parallel.* resets are not redundant. infer.py's validate_single_process refuses to
# start if cfg.parallel differs from the dataclass default AT ALL, on the grounds that a
# parallel knob would silently do nothing on a single process -- and the 480p config now
# carries softmax_ranks: 3 and parallel_vae_decode: true, because those are the right
# answers on 8 cards. Overriding them back is what lets the same file serve both arms.
#
# softmax_ranks=6, not 0, and that is not a typo. The guard compares the whole parallel
# section against `asdict(ParallelConfig())`, whose softmax_ranks default is 6 -- upstream's
# H200 value. So 0, which is what "no branch parallelism" means everywhere else in this
# file, is itself a non-default and gets the config refused. The value is inert here (there
# is one process and no branch to split); it is written to match the dataclass, nothing more.
run1() {
  local tag=$1 config=$2; shift 2
  set -- parallel.softmax_ranks=6 parallel.parallel_vae_decode=false "$@"
  local log=$OUT/$tag.log
  if [ -s "$OUT/$tag.mp4.inference.json" ]; then echo "SKIP $tag (already done)"; return; fi
  echo "=== [$(date -u +%H:%M:%S)] $tag  (gpus=1, infer.py, $config, $*)"
  pkill -f 'infer_ulysses\.py' 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=192 MKL_NUM_THREADS=192 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 3600 python src/inference/infer.py \
      --config "configs/inference/$config" \
      checkpoint="$CKPT" \
      render.prompt_file="${PROMPT_OVERRIDE:-$PROMPT}" \
      render.out="$OUT/$tag.mp4" \
      render.record=true \
      "$@" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "    FAILED rc=$rc -- last lines:"; tail -6 "$log" | sed 's/^/    /'; return
  fi
  grep -E '^(canvas|timing)' "$log" | sed 's/^/    /'
}

C480=8nfe_480p_345f_ulysses_h100.yaml    # 14.375 s, upstream's clip length
C480L=8nfe_480p_362f_ulysses_h100.yaml   # 15.083 s, the literal 15-second ask
C768=8nfe_768p_345f_ulysses_h100.yaml    # the control
C2K=8nfe_2k_ulysses_h100.yaml            # 2560x1440, MiniMax's 2K canvas

case $PHASES in *c*)
  echo "##### PHASE c: control, 768p / 345 frames (upstream's published shape)"
  run c_768p_345f_r6 $C768 8 parallel.softmax_ranks=6
  run c_768p_345f_r0 $C768 8 parallel.softmax_ranks=0
;; esac

case $PHASES in *b*)
  echo "##### PHASE b: baseline, 480p / 345 frames at the H200-tuned 6+2 split"
  run b_480p_345f_r6 $C480 8 parallel.softmax_ranks=6
  run b_480p_362f_r0 $C480L 8 parallel.softmax_ranks=0
;; esac

case $PHASES in *s*)
  echo "##### PHASE s: softmax_ranks sweep at 480p / 345 frames"
  for r in 0 4 5 7; do
    run "s_480p_345f_r$r" $C480 8 parallel.softmax_ranks=$r
  done
;; esac

case $PHASES in *g*)
  echo "##### PHASE g: GPU-count curve at 480p / 345 frames"
  # 1 GPU is a different entrypoint: infer_ulysses.py refuses to start with one rank
  # ("must be launched with torchrun and at least two ranks"), which is correct -- there is
  # no sequence to shard. src/inference/infer.py is the single-GPU path, same kernels, same
  # fp8, no Ulysses. Its record carries only step_seconds, which summarize.py handles.
  run1 g_480p_345f_g1 $C480
  # Run it a SECOND time, which is not a duplicate: phase v established that this pipeline
  # does not reproduce run to run (~17 dB between two renders at the same seed), and the
  # open question is whether that comes from the parallelism -- the Ulysses all-to-alls, the
  # branch-parallel dispatch, patch 0005's all-gather -- or from the fp8 GEMMs underneath
  # all of it. One GPU and infer.py has no collectives at all, so this pair answers it: two
  # matching files mean the divergence is something the patches introduced and needs fixing;
  # two differing files mean it is the arithmetic and no amount of parallelism care removes
  # it. Nothing else in the grid can tell those apart.
  run1 g_480p_345f_g1_again $C480
  if cmp -s "$OUT/g_480p_345f_g1.mp4" "$OUT/g_480p_345f_g1_again.mp4"; then
    echo "    1 GPU is deterministic -- run-to-run divergence comes from the parallel paths"
  else
    echo "    1 GPU also diverges -- the nondeterminism is beneath the parallelism"
  fi
  run g_480p_345f_g2 $C480 2 parallel.softmax_ranks=0
  run g_480p_345f_g2_r1 $C480 2 parallel.softmax_ranks=1
  run g_480p_345f_g4 $C480 4 parallel.softmax_ranks=0
  run g_480p_345f_g4_r3 $C480 4 parallel.softmax_ranks=3
;; esac

case $PHASES in *n*)
  echo "##### PHASE n: step budget at 480p / 345 frames (8 GPUs)"
  for n in 4 6; do
    run "n_480p_345f_s$n" $C480 8 render.num_steps=$n
  done
;; esac

case $PHASES in *x*)
  echo "##### PHASE x: the left half of the split sweep at 480p / 345 frames"
  # The first pass sampled 0/4/5/6/7 and found the minimum at 4 -- an interior point, not
  # an endpoint. So the curve has a real optimum and 1/2/3 are needed to know it is 4.
  for r in 1 2 3; do
    run "x_480p_345f_r$r" $C480 8 parallel.softmax_ranks=$r
  done
;; esac

case $PHASES in *w*)
  echo "##### PHASE w: the 4-GPU split done properly"
  # Phase g tested 4 GPUs at 3+1 and it came out WORSE than standard Ulysses (2.19 vs
  # 1.94 s/NFE), the same way 7+1 lost at 8 GPUs. Both starve the linear branch. The
  # 8-GPU optimum is the even 4+4, so the 4-GPU analogue to test is 2+2, which phase g
  # never did -- without it the scaling curve compares 8 GPUs at its best split against
  # 4 GPUs at a bad one.
  for r in 1 2; do
    run "w_480p_345f_g4_r$r" $C480 4 parallel.softmax_ranks=$r
  done
;; esac

case $PHASES in *y*)
  echo "##### PHASE y: the same split sweep at 768p, so the control has its OWN best"
  # 480p's best split (4+4) beats the 6+2 that upstream tuned on H200. Comparing 480p's
  # best against 768p's r0 would flatter 480p, so 768p gets the same sweep. r6 is here
  # because it OOMed in the first pass, before the transformer offload existed.
  for r in 3 4 5 6; do
    run "y_768p_345f_r$r" $C768 8 parallel.softmax_ranks=$r
  done
;; esac

case $PHASES in *z*)
  echo "##### PHASE z: headline reruns at the winning split, clean E2E"
  # The b_* arms paid an unconditional ~27 s transformer offload that 480p does not need
  # (it is a config field now, off for 480p). Their denoise is unaffected, but their E2E is
  # inflated, so the two numbers that get quoted are measured again with the final code.
  run z_480p_345f_best $C480 8
  # 362 frames is NOT re-run here. It was, once, and the result is in RESULTS.md; what it
  # showed is that at the same split it is +4.2 % denoise over 345 (matching +4.9 % rows) and
  # identical in decode, because 21 chunks and 20 chunks both give ceil(n/8) = 3. So the
  # 362-frame numbers are derivable from the 345-frame ones and the arm buys nothing but
  # another 215 s of host-side fp8 assembly. The config stays -- it is the only length that
  # actually reaches 15 s, so someone will want to render it -- but it is not a grid arm.
;; esac

case $PHASES in *f*)
  echo "##### PHASE f: fl2va vs t2va, same canvas, same split -- what conditioning costs"
  # Both arms at 768p because that is the canvas upstream's example_fl2va.pt was encoded
  # for: its condition_latents are (1, 24, 1, 48, 84), which only fits a 1344x768 render.
  # Re-encoding them at 480p needs the 63 GB Qwen3-VL text encoder, which the second half of
  # this phase now does.
  #
  # MEASURED, and it corrects the guess this comment used to carry ("conditioning adds a
  # fixed number of rows, so its relative cost is LARGER on 480p's shorter sequence"). It is
  # the same at both canvases: +8.9 % at 768p, +8.7 % at 480p. Conditioning is NOT additive --
  # the vision rows and the condition latents both scale with the canvas (2016 -> 814 and
  # 2016 -> 810), so the overhead is proportional and the ratio is canvas-invariant.
  #
  # Rows, computed from build_packed_sequence: t2va 768p = 1299 text + 1150 audio +
  # 102,816 video = 105,265. fl2va 768p = 3485 text + 2016 condition + 1150 + 102,816 =
  # 109,467, i.e. +4.0%. If the measured gap is bigger than that, the extra is the window
  # softmax: condition and text rows sit outside the video span, so they are attended
  # densely in both directions and skipped by the linear branch -- and 1299 -> 4295 dense
  # rows is 3.3x, not 1.04x.
  #
  # The t2va arm is re-run here rather than reusing c_768p_345f_r0: that record predates
  # the transformer offload (no transformer_offload_seconds key at all), so its E2E is not
  # comparable with anything measured after patch 3.
  run f_768p_t2va_r0 $C768 8 parallel.softmax_ranks=0
  PROMPT_OVERRIDE=prompts/image/example_fl2va.pt \
    run f_768p_fl2va_r0 $C768 8 parallel.softmax_ranks=0

  # ... and then the same comparison at the canvas this whole exercise is about. That needs
  # a keyframe cache encoded AT 480p, because condition_latents come out at the canvas's
  # latent size. Patch 0006 adds --height/--width for exactly this; the prompt text is read
  # back out of the released cache so the two arms differ only in canvas.
  FL480=prompts/image/example_fl2va_480p.pt
  if [ ! -s "$FL480" ]; then
    echo "=== [$(date -u +%H:%M:%S)] encoding a 864x480 fl2va keyframe cache (63 GB text encoder)"
    pkill -f 'infer_ulysses\.py' 2>/dev/null; sleep 3
    PROMPT_TEXT=$(python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['prompt'])" prompts/image/example_fl2va.pt)
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=192 python src/inference/encode_keyframes.py \
        --prompt "$PROMPT_TEXT" \
        --first prompts/image/first.png --last prompts/image/last.png \
        --height 480 --width 864 --out "$FL480" > "$OUT/encode_fl2va_480p.log" 2>&1 \
      || { echo "    FAILED to encode -- last lines:"; tail -8 "$OUT/encode_fl2va_480p.log" | sed 's/^/    /'; }
  fi
  if [ -s "$FL480" ]; then
    run f_480p_t2va_r3  $C480 8
    PROMPT_OVERRIDE=$FL480 run f_480p_fl2va_r3 $C480 8
  fi
;; esac

case $PHASES in *v*)
  echo "##### PHASE v: the video VAE on 8 ranks instead of 1"
  # The decode is the bigger half of the post-warmup latency at 480p and every bit of it
  # runs on rank 0 upstream. Patch 0005 spreads its 20 temporal chunks over the ranks;
  # these two pairs are the before/after at both canvases. Timing only -- see below for
  # why the correctness check is a separate program and not a diff of these two files.
  run v_480p_345f_serial   $C480 8 parallel.parallel_vae_decode=false
  run v_480p_345f_parallel $C480 8 parallel.parallel_vae_decode=true
  run v_768p_345f_serial   $C768 8 parallel.softmax_ranks=0 parallel.parallel_vae_decode=false
  run v_768p_345f_parallel $C768 8 parallel.softmax_ranks=0 parallel.parallel_vae_decode=true

  # This used to `cmp -s` the serial and parallel mp4s and call a mismatch a bug in the
  # parallel decode. That test was WRONG, and it is worth leaving the reason here so it does
  # not come back: **this pipeline does not reproduce run to run.** Two renders at the same
  # seed, same config, same split measure ~17 dB PSNR of each other (480p r3 twice: 17.55 dB;
  # 768p r0 twice: 16.69 dB; not one frame of 345 bit-identical). Best temporal alignment is
  # shift 0 with a sharp peak and 16x-downsampled PSNR reaches only ~22-29 dB, so the runs
  # agree on scene, composition and motion and disagree on detail everywhere -- ULP
  # divergence in the fp8 GEMMs, amplified over 8 sampler steps. So `cmp` on two renders
  # compares two denoise trajectories and reports a mismatch whatever the decoder does.
  #
  # The decode is therefore tested where it lives: one process, one latent tensor, decoded
  # both ways back to back, with nothing upstream in the comparison.
  echo "=== [$(date -u +%H:%M:%S)] decode parity: one latent tensor, both decode paths"
  for canvas in "480 864" "768 1344"; do
    set -- $canvas
    timeout 900 torchrun --standalone --nproc_per_node=8 scripts/decode_parity.py \
        --frames 345 --height "$1" --width "$2" > "$OUT/decode_parity_$1p.log" 2>&1 \
      && echo "    ${1}p: $(grep -E '^(OK|FAIL)' "$OUT/decode_parity_$1p.log" | head -1)" \
      || echo "    ${1}p: FAILED -- $(tail -3 "$OUT/decode_parity_$1p.log" | tr '\n' ' ')"
  done
;; esac

case $PHASES in *m*)
  echo "##### PHASE m: the tail -- audio VAE + frames-to-host + mux"
  # With the video VAE spread over 8 ranks (patch 0005) the tail is what is left, and at
  # 480p it was 3.18 s against 2.97 s for the whole eight-way decode. scripts/mux_bench.py
  # decomposed the mux and found it is NOT x264: of 2.65 s, 2.03 s is swscale converting
  # rgb24 -> yuv420p in one thread. Patch 0007 does that conversion on the card, which also
  # halves the host copy (1.5 bytes/px instead of 3). These two arms are the before/after.
  #
  # Timing only, deliberately. The conversion is lossy and re-implemented, so the two mp4s
  # WILL differ -- but comparing these two renders could not show that anyway, because the
  # pipeline diverges ~17 dB run to run (see phase v). Equivalence to swscale is checked on
  # fixed frames instead, where the denoise is not in the way:
  #     .venv/bin/python scripts/mux_bench.py $OUT/m_480p_swscale.mp4 --verify
  run m_480p_gpu_yuv $C480 8 render.gpu_color_convert=true
  run m_480p_swscale $C480 8 render.gpu_color_convert=false
  run m_768p_gpu_yuv $C768 8 parallel.softmax_ranks=5 parallel.parallel_vae_decode=true \
      render.gpu_color_convert=true
  echo "=== [$(date -u +%H:%M:%S)] tail stages, gpu convert vs swscale"
  for tag in m_480p_gpu_yuv m_480p_swscale m_768p_gpu_yuv; do
    grep -E 'audio_vae|frames_to_host|mux|decode_stages' "$OUT/$tag.log" | sed "s/^/    $tag: /"
  done
  # And the conversion's numerical check, against swscale's own output on real frames.
  .venv/bin/python scripts/mux_bench.py "$OUT/m_480p_swscale.mp4" --verify 2>&1 | sed 's/^/    /'
;; esac

case $PHASES in *k*)
  echo "##### PHASE k: what a 2K canvas costs on this DiT"
  # DONE AND CLOSED -- not in any default PHASES, and there is no reason to run it again.
  # MiniMax's own model page states H3-Regenerate-2K is not open-sourced (API only), so no
  # arm here can measure the real thing; this was only ever a lower bound on the DiT cost at
  # a 2560x1440 canvas. What it found:
  #
  #   345 frames  OOM in the DENOISE, not the decode: rank 0 asks for 63.61 GiB in one
  #               allocation with 11.94 GiB free. 3600 tokens/frame against 768p's 1008.
  #   243 frames  OOM the same way (ranks 6/7 first, 1.68 GiB short).
  #    90 frames  fits: 3.58 s/NFE, denoise 28.63 s, decode+encode 4.27 s.
  #
  # So 2K on this box tops out around 90 frames = 3.75 s of video, and at 3.58 s/NFE the
  # denoise alone is 2x the 480p/345f figure for a twentieth of the clip. The split sweep
  # below never ran and is left only because the loop is harmless if anyone revisits this.
  for f in 345 243 90; do
    run "k_2k_${f}f_r6" $C2K 8 render.num_frames=$f
  done
  # The balance point moves with tokens_per_frame, so the winning split at 768p is not
  # necessarily the winning split here. Only run the sweep at the length that fit.
  for f in 345 243 90; do
    if [ -s "$OUT/k_2k_${f}f_r6.mp4.inference.json" ]; then
      for r in 0 4 5 7; do run "k_2k_${f}f_r$r" $C2K 8 render.num_frames=$f parallel.softmax_ranks=$r; done
      break
    fi
  done
;; esac

echo "=== [$(date -u +%H:%M:%S)] grid done"
python "$REPO/../summarize.py" "$OUT" 2>/dev/null || true

#!/usr/bin/env bash
# Speed sweep for base MiniMax-H3 on ONE g7.48xlarge -- 8x RTX PRO 4500 Blackwell, 32 GB each.
# Runs INSIDE the h3-serve pod (no docker here: the pod IS the container), one arm at a time:
#
#   bash /data/h3/g7_sweep.sh                       # every arm below
#   ARMS="fa cudnn" bash /data/h3/g7_sweep.sh       # a subset, by label
#
# WHY THIS MACHINE NEEDS ITS OWN SWEEP INSTEAD OF REUSING THE H100 NUMBERS. Three things differ and
# all three change which knob is worth turning:
#   * 32 GB per card, not 80. The fp8 DiT is 30.93 GB resident and --quantization fp8 is ONLINE
#     quantization (the loader lands the 65.65 GiB bf16 checkpoint on the card and casts after), so
#     pure Ulysses cannot even load and TP is mandatory, not optional. See sglang_base_arm.sh.
#   * sm_120, and the sgl-kernel FlashAttention wheels do not claim it:
#         runtime/platforms/cuda.py:696  "Defaulting to Torch SDPA backend on SM12.x"
#     _resolve_default_attn_backend() returns TORCH_SDPA for is_sm120() *before* reaching
#     _prepare_flash_attention_for_blackwell(), whose own comment says "the FA4 CuTe package ships
#     an sm120 forward kernel". So the default is a correctness-first fallback that an explicit
#     --attention-backend can step over -- which is why `fa` is the first arm and not an
#     afterthought. Every H100 number in RESULTS.md was measured on FA; this box was not.
#   * no NVLink. Eight cards on PCIe + 2x EFA, so TP's per-layer all-reduce and Ulysses's
#     per-attention all-to-all are both expensive and their ratio is not the H100's. GPUS = TP *
#     ULYSSES is the only constraint; where the optimum sits has to be measured.
#
# ONE PROMPT, ONE SEED, ONE LENGTH ACROSS EVERY ARM -- the customer's own t2va case at 768p / 121 f /
# 25 steps, seed 42, which is also what p5.4xlarge rendered. t2va and not ref2va because it is the
# cheaper of the two (105 s vs 213 s on the H100) and the topology/backend question does not care
# which weight partition answers it; the winner gets re-run on ref2va afterwards, where the
# reference tower's 7 296 extra rows actually live.
#
# WHICH ARMS PRESERVE QUALITY, because "保证质量的情况下最快" is the actual question:
#   fa / cudnn / sdpa / tp*  -- IDENTICAL MATH. Different attention kernels and different collective
#     placement change accumulation order, nothing else. Same seed, same schedule, same weights.
#   compile                  -- same math, graph-level fusion only.
#   vsa                      -- NOT identical: VSA-H3 drops attention blocks by a sparsity ratio.
#     Kept in the sweep because it is the single largest lever available at 50/25 steps, but it is
#     labelled and its output has to be eyeballed against the sdpa render before it can be quoted.
#   cachedit                 -- NOT identical either: it reuses DiT output across adjacent steps.
# So the sweep reports two tables, not one: exact arms, and approximate arms with a quality gate.
set -uo pipefail

ROOT=${ROOT:-/data/h3/sglang}
VDNROOT=${VDNROOT:-/data/h3}
CASE=${CASE:-$VDNROOT/case.txt}
EDGE=${EDGE:-768}
STEPS=${STEPS:-25}
TASK=${TASK:-t2va}
export FRAMES=${FRAMES:-121}
export OUTDIR=${OUTDIR:-$VDNROOT/pull/case}
export REFDIR=${REFDIR:-$VDNROOT/ref}
LOGDIR=$ROOT/logs
RESULTS=${RESULTS:-$LOGDIR/g7_sweep.txt}
mkdir -p "$LOGDIR" "$OUTDIR"

# `up <log>`: wait for readiness. The readiness line is NOT "Uvicorn running" -- http_server.py runs
# the warmup in _run_server_warmup_after_http_live, i.e. after the socket is open and after
# GET /liveness answers 200, so a client that fires on "Uvicorn running" hits a mid-warmup server.
# A warmup OOM has to be matched explicitly too: this build's _degrade_after_oom is broken
# (ValueError: field num_frames is declared with init=False) so it cannot retry smaller, it aborts.
#
# AND THE FATAL PATTERN HAS TO BE NARROW. Matching "Traceback" or "out of memory" threw away a good
# --enable-torch-compile arm: inductor's GEMM autotuner *prints tracebacks as normal operation* while
# it rejects candidate configs that do not fit the card's shared memory --
#     torch/_inductor/select_algorithm.py: torch.OutOfMemoryError: out of resource: triton_mm
#     Required: 131072 Hardware limit: 101376
# (that 99 KB limit is the RTX PRO 4500's shared memory per block, against an H100's 227 KB, and it
# is the same reason several hand-written kernels do not have sm_120 builds). Those lines are
# expected, so the only things treated as fatal are the two messages that actually end startup, plus
# the server process disappearing -- which every real OOM does produce.
up() {
  local log=$1 i
  for i in $(seq 1 ${UPTRIES:-120}); do
    grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready after ${i}0s"; return 0; }
    if grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null; then
      echo "  FAILED -- tail of $log:"; grep -m3 -i "outofmemory\|Error" "$log" | cut -c1-300; return 1
    fi
    pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE:"; tail -8 "$log" | cut -c1-300; return 1; }
    sleep 10
  done
  echo "  TIMEOUT"; tail -8 "$log" | cut -c1-300; return 1
}

stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }

# arm <label> <TP> <ULYSSES> <QUANT> [extra sglang flags...]     -- t2va, served from the FL2VA
#                                                                   partition by sglang_base_arm.sh
# ref2arm <label> <steps> <TP> <ULYSSES> <QUANT> [flags...]      -- ref2va, via sglang_ref2va_arm.sh
# STEPS is per-arm in ref2arm and global for arm, because the only reason to change it is a distill
# LoRA, and only the ref2va side has one whose training schedule this repo has already validated.
arm() {
  local label=$1 tp=$2 uly=$3 quant=$4; shift 4
  _run "$label" t2va "$STEPS" "$tp" "$uly" "$quant" sglang_base_arm.sh serve_base "$@"
}
ref2arm() {
  local label=$1 steps=$2 tp=$3 uly=$4 quant=$5; shift 5
  _run "$label" ref2va "$steps" "$tp" "$uly" "$quant" sglang_ref2va_arm.sh serve_ref2va "$@"
}
_run() {
  local label=$1 task=$2 steps=$3 tp=$4 uly=$5 quant=$6 script=$7 base=$8; shift 8
  if [ -n "${ARMS-}" ] && [[ " $ARMS " != *" $label "* ]]; then return 0; fi
  local log="$LOGDIR/${base}_${EDGE}p_$label.log"
  echo "=== $label: $task ${steps}step TP=$tp ULYSSES=$uly quant='${quant:-bf16}' lora='${LORA-}' flags='$*'" \
    | tee -a "$RESULTS"
  rm -f "$log"; stop
  ROOT=$ROOT VDNROOT=$VDNROOT QUANT="$quant" GPUS=8 TP="$tp" ULYSSES="$uly" \
    FRAMES="$FRAMES" LOGTAG="$label" \
    setsid nohup bash "$VDNROOT/$script" serve "$EDGE" "$@" \
    > "$LOGDIR/launch_$label.log" 2>&1 < /dev/null &
  sleep 15
  if ! up "$log"; then echo "  $label: SERVER FAILED" | tee -a "$RESULTS"; stop; return 1; fi
  # nvidia-smi at steady state, before the render: the resident footprint is what says whether an
  # arm has room for a longer clip, and peak_memory_mb from the response says whether it nearly died.
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ' | tee -a "$RESULTS"
  echo "" | tee -a "$RESULTS"
  python "$VDNROOT/sglang_case.py" "case=$CASE" "task=$task" "tag=$label" \
    "$EDGE:$steps:$FRAMES" 2>&1 | tee -a "$RESULTS"
  stop
}

# ---- exact-math arms -------------------------------------------------------------------------
# Baseline first only if asked for: tp4u2_sdpa is already measured (105.37 s, 4.21 s/step, peak
# 27 770 MB) and re-running it costs 3.5 minutes to reprint a number.
arm sdpa      4 2 fp8
# `fa` and `cudnn` are MEASURED NO-OPS on this box, kept only so the log says so: both resolve back
# to torch_sdpa (105.68 s and 104.35 s against sdpa's 105.37 s -- the same number three times).
#   platforms/cuda.py:513  _FlashAttentionBackendResolver.resolve()
#       if platform.is_sm120(): "FlashAttention is not supported on SM12.x in this build" -> TORCH_SDPA
# i.e. FA3/FA4 is hard-gated off for sm_120 regardless of what --attention-backend asks for, and the
# cuDNN resolver lands on SDPA too. FA2 is NOT gated: _FlashAttention2BackendResolver has no sm120
# branch and the flash_attn package is installed, so fa2 is the one backend arm that can actually
# change the kernel. That is why it is here and why fa/cudnn are not worth re-running.
arm fa        4 2 fp8 --attention-backend fa
arm cudnn     4 2 fp8 --attention-backend torch_cudnn_sdpa
arm fa2       4 2 fp8 --attention-backend fa2
# fa2 is not a no-op, it is a HARD FAILURE, which is the answer to "is there any faster attention
# kernel on this card": there is not.
#     flash_attn_2.py:13  ImportError: cannot import name 'flash_attn_func' from ...flash_attn
# The wheel in this image exposes no FA2 entry point at all. Together with the sm120 gate on FA3/FA4
# above, torch_sdpa is the ONLY dense attention backend that runs here -- so on g7 there is no
# exact-math attention speedup to be had, and everything below is about topology, weights and steps.

# TP=2 x Ulysses=4 IS THE FASTEST FEASIBLE SHAPE, and the only thing standing between it and running
# is 24 GiB of AdaLN. Measured from the FL2VA/transformer safetensors headers: of the 61.73 GiB DiT,
# `adaln`-named tensors are 24.29 GiB -- 39 % of the checkpoint. That is why plain fp8 dies at TP=2
# with "29.65 GiB is allocated" during transformer load (online quantization has to land 61.73/2 =
# 30.9 GiB of bf16 on a 31.37 GiB card before it can cast), and why --minimax-h3-adaln-online is not
# a memory footnote but the enabling flag: it filters those tensors out of the GPU load and rebuilds
# their outputs from the checkpoint into an 8 GB host slab, leaving (61.73-24.29)/2 = 18.7 GiB to
# load. Same values, so this is an exact-math arm; the open question is what the rebuild costs per
# step, which is what adaln_tp4 (known-good topology, adaln on) isolates before adaln_tp2 changes two
# things at once. It needs the native tensor layout, which FL2VA and Ref2VA both are.
arm adaln_tp4 4 2 fp8 --minimax-h3-adaln-online
arm adaln_tp2 2 4 fp8 --minimax-h3-adaln-online
# Topology, at whichever backend wins above -- BACKEND is substituted by the caller so the topology
# arms are not silently measured on the loser.
BACKEND=${BACKEND:-}
back=(); [ -n "$BACKEND" ] && back=(--attention-backend "$BACKEND")
arm tp8u1     8 1 fp8 "${back[@]}"
arm tp2u4     2 4 fp8 "${back[@]}"
arm compile   4 2 fp8 "${back[@]}" --enable-torch-compile

# ---- the ref2va side, which is the customer's expensive case -----------------------------------
# +68 % over t2va on the H100 (213 s vs 127 s) and the reason is structural, not a knob: the
# refedge-2048 reference tower adds 7 296 rows of conditioning for one 16:9 image. So the ref2va
# baseline has to be measured here rather than scaled from t2va.
ref2arm r_base 25 4 2 fp8

# ---- step-count arms: the only lever worth more than 20 % -------------------------------------
# 25 -> 8 steps is 3.1x, and it is not a quality-free change unless the weights were distilled for
# the short schedule -- which is exactly what lightx2v's Turbo LoRA is. Two things this repo already
# established and that this arm depends on:
#   * pass NO --lora-alpha. The alpha is in the safetensors metadata (0.0625 for ref2v v1.0); the
#     `--lora-alpha 128` in lightx2v's README is for a different file and is a 16x overdrive -- it is
#     the actual cause of the "melting" the customer reported. See REF2VA.md.
#   * bf16, not fp8. --lora-path on a quantized layer falls back to dynamic LoRA and the dynamic
#     wrapper is not quantization-aware, so warmup dies on
#     'RowParallelLinearWithLoRA' object has no attribute 'quant_method'. bf16 merges the adapter at
#     load and costs nothing per step. bf16 also raises the DiT to 61.73 GB, i.e. 15.4 GB per card at
#     TP=4 -- which is why the encoder has to come off the card for this arm and not for the others.
# Exported on its own line rather than as a `LORA=... ref2arm ...` prefix: a prefix assignment on a
# *function* call leaks into the shell afterwards in bash, so the arms below it would silently
# inherit a LoRA. Unset again at the end of the LoRA block.
export LORA=${LORA:-$VDNROOT/lora/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors}
ref2arm r_lora8 8 4 2 "" --text-encoder-cpu-offload true
# r_lora8 above OOMs, and where it OOMs is the point: not in the load, in the warmup forward --
#     Error executing request None: ... this process has 30.92 GiB memory in use
# i.e. bf16 at TP=4 does fit as weights (15.4 GB/card) and then leaves too little for the ~15 GB of
# activations. --minimax-h3-adaln-online is the fix and not a workaround: it takes 24.29/4 = 6.1 GB
# per card of AdaLN out of the resident weights, so the same arm has ~9.4 GB of weights and room to
# run. The adapter targets attn/mlp projections, not AdaLN, so the two do not interact.
ref2arm r_lora8_adaln 8 4 2 "" --text-encoder-cpu-offload true --minimax-h3-adaln-online
# TP=8 as the fallback shape for the same arm: bf16 is 7.7 GB/card there instead of 15.4, so if the
# TP=4 arm cannot fit even with the encoder on the host, this one can -- at TP=8's 35 % step penalty,
# which 8 steps instead of 25 pays for many times over.
ref2arm r_lora8_tp8 8 8 1 ""
unset LORA

# ---- approximate arms, quality gate required -------------------------------------------------
arm vsa       4 2 fp8 --attention-backend video_sparse_attn_h3 --attention-backend-config '{"VSA_sparsity": 0.9}'
# cache-dit is the lever for anyone who will not take a LoRA: DBCache skips the DiT on steps whose
# residual barely moved, so it buys time out of the *schedule* the way the LoRA does, without new
# weights. Defaults from cache_dit's DBCacheConfig (Fn_compute_blocks 8, residual_diff_threshold
# 0.08, max_warmup_steps 8) written to a file because load_configs(path_or_dict) reads a path, not a
# JSON string. It is in the approximate section for the obvious reason: a skipped step is a step that
# did not run.
if [ -n "${ARMS-}" ] && [[ " $ARMS " == *" cachedit "* ]]; then
  cat > "$VDNROOT/cachedit.json" <<'JSON'
{"cache_type": "DBCache", "Fn_compute_blocks": 8, "Bn_compute_blocks": 0,
 "residual_diff_threshold": 0.08, "max_warmup_steps": 8}
JSON
fi
arm cachedit  4 2 fp8 --cache-dit-config "$VDNROOT/cachedit.json"

echo "=== sweep done"; tail -40 "$RESULTS"

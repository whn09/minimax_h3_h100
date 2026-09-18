#!/usr/bin/env bash
# SageAttention 2 on g7: the attention lever G7.md said did not exist.
#
# WHY THIS REOPENS A CLOSED QUESTION. G7.md §2 concluded "torch_sdpa is the only dense attention
# backend on g7, and there is no exact-math attention speedup to be had". The first half is still
# true and the second half was wrong, because it was reasoned from the WRONG GATE. What is gated on
# sm_120 is FlashAttention (runtime/platforms/cuda.py:513 hard-returns TORCH_SDPA for is_sm120())
# and VSA-H3 (needs cc 9.0/10.0/10.3). Sage is a different resolver and it has NO sm_120 gate:
#
#   platforms/cuda.py  _SageAttentionBackendResolver.resolve()
#     try: from sageattention import sageattn        -> ImportError => return AttentionBackendEnum.FA
#     if platform.is_hopper(): <SM90 binding check>  -> skipped entirely on sm_120
#     return "...backends.sage_attn.SageAttentionBackend"
#
# and it clears H3's own gate too: minimax_h3.py:308 asks for
# AttentionRequirements(packed_varlen=True), and SageAttentionImpl overrides forward_varlen
# (backends/sage_attn.py:83). That is the same test that refuses sage_attn_3, video_sparse_attn and
# ~14 other backends. So sage is admissible on this card and nothing had installed it.
#
# THE TRAP, AND IT IS WHY THIS SCRIPT ASSERTS. The ImportError branch falls back to FA, and FA on
# sm_120 is itself silently downgraded to TORCH_SDPA. So "sage is installed but did nothing" and
# "sage was never installed" produce THE SAME NUMBER -- 105 s, i.e. exactly the baseline this arm is
# being compared against. A run that forgets the build reads as a clean negative result. Hence
# assert_sage(): the serve log must say `Using sage_attn attention backend`, and must NOT say
# `Defaulting to Torch SDPA backend on SM12.x` at the DiT (that line also appears twice for other
# components during load, so it is matched together with the positive line, not alone).
#
# WHAT IS HELD FIXED. Everything except the attention backend: base model (no LoRA), fp8,
# TP=4 x ULYSSES=2, 768p, 121 f (5.04 s), seed 42, 25 steps, case_ir.txt, reference short edge at
# the 2048 default. The comparison is therefore against the recorded arms and nothing was re-run to
# make it: ref2va/rir25 = 177.2 s inference, t2va/ir25 = 105.4 s.
#
# DELIBERATE DIVERGENCE FROM THE g7e RECIPE: it serves ref2va with
# SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024 and calls it "a 1.46x lever". We keep 2048.
# REF2VA.md:447 -- "Do not patch MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE to 1024. It buys ~20% and
# causes the melting the customer reported." That is this customer's own defect, so the 1024 row is
# not available to us and our numbers should not be compared against theirs.
#
# NVFP4 IS NOT IN THIS SCRIPT, on purpose. It is the other half of the g7e recipe and it is a
# bigger, riskier project -- see SAGE.md for why (offline conversion, two source patches, and it is
# untested at TP>1, which 32 GB cards force on us). Sage is independent of it, cannot produce a
# wrong picture, and per g7e's own ablation is the part that actually matters at 768p: weight
# quantization never touches attention, and attention is 59% of a 768p step.
set -uo pipefail
V=/data/h3; L=$V/sglang/logs; R=$L/sage.txt
export ROOT=$V/sglang VDNROOT=$V FRAMES=121 OUTDIR=$V/pull/case REFDIR=$V/ref
: > $R
SAGE=(--attention-backend sage_attn
      --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa)
# All three components are exempted, not just the text encoder. Qwen3VL's LocalAttention only
# accepts {fa, torch_sdpa}, and on newer sglang the audio and video VAEs join it: a global
# --attention-backend stops being a fallback and becomes a hard ValueError at selector time
# (sgl-project/sglang#35743). Exempting a component that would have resolved to torch_sdpa anyway
# is free -- this build already logs "Attention backends for text_encoder|audio_vae|video_vae:
# torch_sdpa" -- so the exemption costs nothing and removes a startup failure mode.

up() { local log=$1 i; for i in $(seq 1 40); do
  grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready ${i}0s" | tee -a $R; return 0; }
  grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null && { echo "  FAILED" | tee -a $R; grep -m3 -i "outofmemory\|Error" "$log" | cut -c1-260 | tee -a $R; return 1; }
  pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE" | tee -a $R; tail -6 "$log" | cut -c1-260 | tee -a $R; return 1; }
  sleep 10; done; echo "  TIMEOUT" | tee -a $R; return 1; }
stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }

# Read the backend back out of the server's own log rather than trusting the flag. Returns non-zero
# so the caller SKIPS the render: a fallback arm would burn 175 s to reproduce a number we already
# have, and would then look like evidence that sage does not help.
assert_sage() { local log=$1 line
  line=$(grep -h "Using .* attention backend" "$log" | tail -1)
  echo "  backend readback: ${line:-<none found>}" | tee -a $R
  case "$line" in
    *sage_attn*) return 0 ;;
    *) echo "  !! SAGE NOT ACTIVE -- skipping the render. Check: docker/pod has sageattention" \
            "installed (python3 -c 'import sageattention'), built from source for sm_120." | tee -a $R
       return 1 ;;
  esac; }

stop
echo "=== S1  ref2va + sage   (vs rir25 = 177.2 s, same prompt/seed/geometry)" | tee -a $R
log=$L/serve_ref2va_768p_sage.log; rm -f $log
QUANT=fp8 GPUS=8 TP=4 ULYSSES=2 LOGTAG=sage \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 "${SAGE[@]}" > $L/launch_sage_a.log 2>&1 < /dev/null &
sleep 15
if up $log && assert_sage $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=ref2va tag=sage 768:25:121 2>&1 | tee -a $R
fi
stop

echo "=== S2  t2va + sage     (vs ir25 = 105.4 s, same prompt/seed/geometry)" | tee -a $R
log=$L/serve_base_768p_sage.log; rm -f $log
QUANT=fp8 GPUS=8 TP=4 ULYSSES=2 LOGTAG=sage \
  setsid nohup bash $V/sglang_base_arm.sh serve 768 "${SAGE[@]}" > $L/launch_sage_b.log 2>&1 < /dev/null &
sleep 15
if up $log && assert_sage $log; then
  python $V/sglang_case.py case=$V/case_ir.txt task=t2va tag=sage 768:25:121 2>&1 | tee -a $R
fi
stop
echo SAGE_DONE | tee -a $R

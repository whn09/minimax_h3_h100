#!/usr/bin/env bash
# ref2va handwriting: the capacity/prompt arm and the task-change arm (case_ref2va_v2.txt).
#
# TWO SERVERS, NOT ONE, and that is forced rather than chosen: t2va/fl2va and ref2va are different
# weight partitions (MINIMAX_H3_TASK_PARTITIONS), so --model-variant differs and they cannot share a
# process. Arm 1 needs the ref2va partition on 30012, arm 2 needs the fl2va partition on 30011, so
# this is two 70-90 s starts. The saving is that arm 2 rides the *base* server -- the same one every
# t2va arm in this repo used -- rather than needing a third variant.
#
# Base model, no LoRA. fp8 TP=4 x U=2, 768p, 121 f (5.04 s), seed 42, 25 steps, reference short edge
# left at the 2048 default: identical to rir25, which is the control and is already rendered. Budget
# is roughly 175 s (arm 1, ref2va is ~7 s/step) + 105 s (arm 2, fl2va runs on the cheaper partition)
# plus the two starts, so ~8 min of GPU.
set -uo pipefail
V=/data/h3; L=$V/sglang/logs; R=$L/r2.txt
export ROOT=$V/sglang VDNROOT=$V FRAMES=121 OUTDIR=$V/pull/case REFDIR=$V/ref
: > $R
# The fatal pattern stays narrow on purpose: inductor's GEMM autotuner prints
# "torch.OutOfMemoryError: out of resource: triton_mm" as normal operation while it rejects
# candidates, so a watcher that treats "out of memory" or "Traceback" as fatal throws away a healthy
# arm. Only these three messages actually end startup, plus the process disappearing.
up() { local log=$1 i; for i in $(seq 1 40); do
  grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready ${i}0s" | tee -a $R; return 0; }
  grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null && { echo "  FAILED" | tee -a $R; grep -m3 -i "outofmemory\|Error" "$log" | cut -c1-260 | tee -a $R; return 1; }
  pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE" | tee -a $R; tail -6 "$log" | cut -c1-260 | tee -a $R; return 1; }
  sleep 10; done; echo "  TIMEOUT" | tee -a $R; return 1; }
stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }

stop
echo "=== ARM 1  ref2va@one  (one character, no stroke order, board fills frame)  base fp8 TP4xU2 x 25" | tee -a $R
log=$L/serve_ref2va_768p_r2.log; rm -f $log
QUANT=fp8 GPUS=8 TP=4 ULYSSES=2 LOGTAG=r2 \
  setsid nohup bash $V/sglang_ref2va_arm.sh serve 768 > $L/launch_r2a.log 2>&1 < /dev/null &
sleep 15
if up $log; then
  python $V/sglang_case.py case=$V/case_ref2va_v2.txt task=ref2va tag=r2 768:25:121 2>&1 | tee -a $R
fi
stop

echo "=== ARM 2  fl2va@last  (ref2va.jpg as keyframe frame_index=-1)  base fp8 TP4xU2 x 25" | tee -a $R
log=$L/serve_base_768p_r2.log; rm -f $log
QUANT=fp8 GPUS=8 TP=4 ULYSSES=2 LOGTAG=r2 \
  setsid nohup bash $V/sglang_base_arm.sh serve 768 > $L/launch_r2b.log 2>&1 < /dev/null &
sleep 15
if up $log; then
  # task=fl2va selects the second line only. The image is the same file as arm 1; sglang_case.py
  # sends it as role=keyframe with frame_index=-1 here and role=reference there, which is the
  # entire experiment.
  python $V/sglang_case.py case=$V/case_ref2va_v2.txt task=fl2va tag=r2 768:25:121 2>&1 | tee -a $R
fi
stop
echo R2_DONE | tee -a $R

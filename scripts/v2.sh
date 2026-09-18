#!/usr/bin/env bash
# t2va production candidate: the settled dialogue plus the pose and framing fixes (case_t2va_v2.txt).
#
# The audio grid (aud.sh) answered the pronunciation question and, in doing so, produced a montage
# showing the teacher holding chalk against the board in every frame of every arm -- a writing pose on
# a static equation. Two arms here separate the pose fix from the framing fix; see the case file's
# header for why both are needed and why "what is in frame" beats a shot-size noun.
#
# Same server shape and schedule as the audio grid so the two files are directly comparable:
# t2va, fp8 TP=4 x U=2, 768p, 121 f, seed 42, 25 steps -- the customer's requested config.
set -uo pipefail
V=/data/h3; L=$V/sglang/logs; R=$L/v2.txt
export ROOT=$V/sglang VDNROOT=$V FRAMES=121 OUTDIR=$V/pull/case REFDIR=$V/ref
: > $R
up() { local log=$1 i; for i in $(seq 1 40); do
  grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready ${i}0s" | tee -a $R; return 0; }
  grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null && { echo "  FAILED" | tee -a $R; grep -m3 -i "outofmemory\|Error" "$log" | cut -c1-260 | tee -a $R; return 1; }
  pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE" | tee -a $R; tail -6 "$log" | cut -c1-260 | tee -a $R; return 1; }
  sleep 10; done; echo "  TIMEOUT" | tee -a $R; return 1; }
stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }

stop
echo "=== t2va fp8 TP4xU2, pose + framing fixes x 25 steps" | tee -a $R
log=$L/serve_base_768p_v2.log; rm -f $log
QUANT=fp8 GPUS=8 TP=4 ULYSSES=2 LOGTAG=v2 \
  setsid nohup bash $V/sglang_base_arm.sh serve 768 > $L/launch_v2.log 2>&1 < /dev/null &
sleep 15
if up $log; then
  python $V/sglang_case.py case=$V/case_t2va_v2.txt tag=v2 768:25:121 2>&1 | tee -a $R
fi
stop

# ASR is a regression check here, not the measurement: the dialogue is byte-identical to the audio
# grid's @cn arm, so anything other than a clean reading means the surrounding prose moved the audio.
echo "=== ASR" | tee -a $R
PYTHONPATH=$V/py python $V/asr.py "$OUTDIR/t2va_v2_*/*.mp4" 2>&1 | tee -a $R
echo V2_DONE | tee -a $R

#!/usr/bin/env bash
# t2va audio grid: find a spelling of "2x" that the model actually pronounces.
#
# The customer's report is that the IR prompt's dialogue is intelligible except for `2x`, which comes
# out as "rx". Nothing in the serving path normalises text -- presentation.py:109 tokenizes the prompt
# verbatim -- and neither official prompt guide has a single rule about numerals, digits, letters or
# formulas inside <d> (grepped). So this is a prompt-content question and it is settled by rendering,
# not by reading: four dialogue spellings, ONE server process, everything else held fixed.
#
# ONE SERVER, FOUR RENDERS. The arms differ only in the bytes between <d> and </d>, so they must not
# differ in the process either -- a restart between arms would put weight-load nondeterminism into a
# comparison whose whole subject is 12 characters of text. fp8 TP=4 x U=2 at 25 steps because that is
# what the customer asked for ("我只需要25步的base模型的效果") and it is the exact-math floor from G7.md
# at 105 s/render, so the grid is ~7 min of GPU plus one 90 s start.
#
# @mix is byte-identical to the t2va case in case_ir.txt (1036 chars both), i.e. it is a re-render of
# the exact input that produced "rx" rather than a reconstruction of it -- without that control in the
# same process a difference in arms 2-4 cannot be attributed to the text.
set -uo pipefail
V=/data/h3; L=$V/sglang/logs; R=$L/aud.txt
export ROOT=$V/sglang VDNROOT=$V FRAMES=121 OUTDIR=$V/pull/case REFDIR=$V/ref
: > $R
up() { local log=$1 i; for i in $(seq 1 40); do
  grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready ${i}0s" | tee -a $R; return 0; }
  grep -q "Error while loading component\|Server warmup failed\|processing failed" "$log" 2>/dev/null && { echo "  FAILED" | tee -a $R; grep -m3 -i "outofmemory\|Error" "$log" | cut -c1-260 | tee -a $R; return 1; }
  pgrep -f '[s]glang.*serve' >/dev/null || { echo "  SERVER GONE" | tee -a $R; tail -6 "$log" | cut -c1-260 | tee -a $R; return 1; }
  sleep 10; done; echo "  TIMEOUT" | tee -a $R; return 1; }
stop() { pkill -f '[s]glang.*serve' >/dev/null 2>&1; sleep 12; }

stop
echo "=== t2va fp8 TP4xU2, 4 dialogue spellings x 25 steps" | tee -a $R
log=$L/serve_base_768p_aud.log; rm -f $log
QUANT=fp8 GPUS=8 TP=4 ULYSSES=2 LOGTAG=aud \
  setsid nohup bash $V/sglang_base_arm.sh serve 768 > $L/launch_aud.log 2>&1 < /dev/null &
sleep 15
if up $log; then
  # No task= filter: every line in the file is t2va, and the per-line `t2va@label:` tag is what keeps
  # the four mp4s in four directories (output_path is task_tag_edge_steps_frames, so without the label
  # all four would land in one directory under four uuids and nothing could tell them apart).
  python $V/sglang_case.py case=$V/case_t2va_audio.txt tag=aud 768:25:121 2>&1 | tee -a $R
fi
stop

echo "=== ASR" | tee -a $R
PYTHONPATH=$V/py python $V/asr.py "$OUTDIR/t2va_aud_*/*.mp4" 2>&1 | tee -a $R
echo AUD_DONE | tee -a $R

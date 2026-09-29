#!/bin/bash
# Keep running a gemini_transcribe.py job until all 71 grid files exist, alternating API keys.
# Usage: gemini_loop.sh OUT_DIR LOG [extra gemini_transcribe.py args...]
# Exit 3 from the runner = that key's daily quota is spent; any other failure (503s) = wait and retry.
cd "$(dirname "$0")/../.." || exit 1   # repo root
OUT=$1; LOG=$2; shift 2
PY=C:/Users/user/anaconda3/envs/collaboration-stt/python.exe
M=${M:-.scratch/manifest_all.csv}   # built as in Stage_5_Eval/RUNBOOK.md, "Baseline models" §2
S=Stage_5_Eval/noise_eval/results_2026-09-24/oracle.segments.json
deadline=$(( $(date +%s) + 4 * 86400 ))
STALL_S=${STALL_S:-999999999}   # give up if no new file for this long
last_n=$(ls $OUT/*.txt 2>/dev/null | wc -l); last_t=$(date +%s)
while [ "$(ls $OUT/*.txt 2>/dev/null | wc -l)" -lt 71 ] && [ "$(date +%s)" -lt $deadline ]; do
  n=$(ls $OUT/*.txt 2>/dev/null | wc -l)
  if [ $n -gt $last_n ]; then last_n=$n; last_t=$(date +%s); fi
  if [ $(( $(date +%s) - last_t )) -gt $STALL_S ]; then
    echo "=== GAVE UP $(date): no new file for ${STALL_S}s, $n/71 done" >> $LOG; exit 4
  fi
  spent=0
  for key in ~/.gemini_api_key ~/.gemini_api_key_2; do
    echo "=== $(date) key ${key##*/}" >> $LOG
    $PY Stage_5_Eval/noise_eval/gemini_transcribe.py $M $S $OUT "$@" --key-file $key >> $LOG 2>&1
    rc=$?
    echo "=== rc=$rc, $(ls $OUT/*.txt | wc -l)/71 done" >> $LOG
    [ $rc -eq 0 ] && break
    [ $rc -eq 3 ] && spent=$((spent + 1))
  done
  [ "$(ls $OUT/*.txt | wc -l)" -ge 71 ] && break
  if [ $spent -eq 2 ]; then sleep 3600; else sleep 600; fi
done
echo "=== finished $(date): $(ls $OUT/*.txt | wc -l)/71" >> $LOG

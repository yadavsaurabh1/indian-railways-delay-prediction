#!/bin/zsh
# Restart the XL direct run from where the adaptive schedule stopped it (step 78,000).
#  A. initial rate 2e-4 (ramped over 500 steps), no cuts, 4,000 steps -> step 82,000
#  B. if validation went below the previous best (0.24153): keep going at 2e-4 to the end of 8 epochs
#     otherwise: back to the previous best weights, fixed 2.5e-5, to the end of 8 epochs
# Nothing stops it automatically - stop by hand (Ctrl-C in this window) whenever you want.
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
cd "$(dirname "$0")/../src_v2"
LOG=../logs/final.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }
PRE=0.24153
BASE=(--run final --d 512 --heads 8 --layers 12 --ctx-layers 4 --micro-bs 32 --epochs 8 --eval-every 2000
      --schedule constant --resume)

say "restart queued: waiting for the test-set scoring of the stopped run to finish"
while pgrep -f "train.py --run final" >/dev/null; do sleep 60; done
[ -f train_next.py ] && mv train_next.py train.py
cp ../checkpoints_final/best.pt ../checkpoints_final/best_before_restart.pt
for n in val test; do [ -f ../reports/eval_${n}_final.json ] && cp ../reports/eval_${n}_final.json ../reports/eval_${n}_final_step74k.json; done

say "code check of resume + rate reset (tiny CPU model)"
rm -rf ../checkpoints_smoke
SM=(--smoke --device cpu --d 64 --heads 4 --layers 2 --ctx-layers 1 --workers 0)
caffeinate -i uv run python train.py ${SM[@]} --schedule adaptive 2>&1 | tail -3 | tee -a $LOG || { say "FAILED: check 1"; exit 1; }
caffeinate -i uv run python train.py ${SM[@]} --schedule constant --resume --reset-lr --lr 1e-4 \
  --weights-from ../checkpoints_smoke/best.pt 2>&1 | grep -E "reset|replaced|resumed|Error|Traceback" | tee -a $LOG
grep -q "learning rate reset" $LOG || { say "FAILED: check 2"; exit 1; }
rm -rf ../checkpoints_smoke

# train <first-launch args> -- <crash-retry args>
train() {
  local first=() retry=() seen=0
  for x in "$@"; do if [ "$x" = "--" ]; then seen=1; elif (( seen )); then retry+=$x; else first+=$x; fi; done
  local args=(${first[@]})
  for attempt in 1 2 3 4 5; do
    if caffeinate -i uv run python train.py ${BASE[@]} ${args[@]} 2>&1 | tee -a $LOG; then return 0; fi
    say "stopped (attempt $attempt) - resuming in 2 min"
    sleep 120
    args=(${retry[@]})
  done
  return 1
}

say "A: initial rate 2e-4, steps 78,000 -> 82,000 (previous best val pinball $PRE)"
train --lr 2e-4 --reset-lr --max-steps 82000 -- --lr 2e-4 --max-steps 82000 || { say "FAILED in A"; exit 1; }
A=$(awk -F, '$1>78000 && $1<=82000 {print $4}' ../logs/train_metrics_final.csv | sort -n | head -1)
if [ -n "$A" ] && (( $(echo "$A < $PRE" | bc -l) )); then
  say "A improved: best $A < $PRE -> continuing at 2e-4 to the end"
  train --lr 2e-4 -- --lr 2e-4 || { say "FAILED"; exit 1; }
else
  say "A did not improve (best ${A:-none} vs $PRE) -> back to the previous best weights, fixed 2.5e-5, to the end"
  train --lr 2.5e-5 --reset-lr --weights-from ../checkpoints_final/best_before_restart.pt -- --lr 2.5e-5 \
    || { say "FAILED"; exit 1; }
fi
say "RESTART RUN DONE: checkpoints_final/best.pt, reports/eval_{val,test}_final.json"

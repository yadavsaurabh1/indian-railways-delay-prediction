#!/bin/zsh
# The final model: XL (d=512, 12 route layers, 4 context layers) with every stop-to-stop input (other trains
# per stretch with identity and class), each stop's delay predicted DIRECTLY (summing per-stretch changes
# was tried and was unstable: best 16.40, kept in checkpoints_xl_summed/). 8 epochs or until the adaptive
# schedule finds validation has stopped improving. Peak lr 2e-4, depth-scaled start-up weights.
# Finds a batch chunk size that fits first; resumes after crashes.
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
# cap GPU memory at the recommended working set (~11.8 GB of the 16 GB): overflowing then fails with a
# clean out-of-memory (-> smaller chunks) instead of swapping, which made steps crawl
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
cd "$(dirname "$0")/../src_v2"
LOG=../logs/final.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }
SIZE=(--d 512 --heads 8 --layers 12 --ctx-layers 4 --schedule adaptive --lr 2e-4)
MICRO=$(cat ../checkpoints_final_micro.txt 2>/dev/null)
if [ -z "$MICRO" ]; then
  for m in 64 32 16; do
    if caffeinate -i uv run python train.py --smoke ${SIZE[@]} --micro-bs $m 2>&1 | tee -a $LOG; then MICRO=$m; break; fi
    say "chunk $m did not fit, trying smaller"
  done
  [ -n "$MICRO" ] || { say "FAILED: no chunk size fits"; exit 1; }
  echo $MICRO > ../checkpoints_final_micro.txt
fi
say "final XL run, direct output (chunks of $MICRO)"
for attempt in 1 2 3 4 5; do
  if caffeinate -i uv run python train.py --run final ${SIZE[@]} --micro-bs $MICRO --epochs 8 --eval-every 2000 \
       --resume 2>&1 | tee -a $LOG; then
    say "FINAL RUN DONE: checkpoints_final/best.pt, reports/eval_val_final.json, reports/eval_test_final.json"
    exit 0
  fi
  MICRO=$(( MICRO > 16 ? MICRO / 2 : 16 )); echo $MICRO > ../checkpoints_final_micro.txt
  say "run stopped (attempt $attempt) - resuming in 2 min with chunks of $MICRO"
  sleep 120
done
say "FAILED: final run crashed 5 times"
exit 1

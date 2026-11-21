#!/bin/zsh
# A/B: same small model and new inputs as the stop-to-stop test, but each stop's delay predicted directly
# (no adding up of per-stretch changes). 4,000 steps. Compare val MAE @ 4000:
#   first model 15.91 | stop-to-stop (summed stretches) 20.56
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../src_v2"
LOG=../logs/final.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }
rm -rf ../checkpoints_s2s_direct ../logs/train_metrics_s2s_direct.csv
say "A/B - small model, new inputs, direct output, 4,000 steps"
caffeinate -i uv run python train.py --run s2s_direct --micro-bs 128 --max-steps 4000 --eval-every 1000 2>&1 | tee -a $LOG \
  || { say "FAILED: direct A/B"; exit 1; }
say "A/B DONE - val MAE @ 4000: first model 15.91, summed stretches 20.56, direct:"
grep -E "== val @ 4000:" $LOG | tail -1 | tee -a $LOG

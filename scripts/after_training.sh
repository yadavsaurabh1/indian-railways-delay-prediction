#!/bin/zsh
# Queue: waits for the transformer training to end (for any reason), makes sure it finished
# (resumes up to 2 times if it crashed), then runs the ablation and the LightGBM comparison.
set -u
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../src"
LOG=../logs
say() { echo "=== [$(date '+%F %T')] $*"; }

say "queue armed; waiting for training to end"
while pgrep -f "train.py" >/dev/null; do sleep 60; done
say "training process has exited"

tries=0
while ! grep -q "training done" $LOG/training.log && [ -f ../checkpoints/last.pt ] && [ $tries -lt 2 ]; do
  tries=$((tries + 1))
  say "training did not finish cleanly - resume attempt $tries"
  caffeinate -i uv run python train.py --resume 2>&1 | tee -a $LOG/training.log
done

if [ ! -f ../checkpoints/best.pt ]; then say "no checkpoints/best.pt - nothing to evaluate, stopping"; exit 1; fi
if [ ! -f ../reports/eval_test.json ]; then
  say "test report missing - running eval-only"
  caffeinate -i uv run python train.py --eval-only 2>&1 | tee -a $LOG/training.log
fi

say "ablation: how much the other-train information helps"
caffeinate -i uv run python ablate.py 2>&1 | tee -a $LOG/ablation.log

say "LightGBM comparison"
caffeinate -i uv run python lgbm_baseline.py 2>&1 | tee -a $LOG/lgbm.log

say "queue finished: reports/eval_test.json, reports/ablation_test.json, reports/comparison.json"

#!/bin/zsh
# Short test: small stop-to-stop model (first model's size and recipe + the new inputs), 8,000 steps.
# Batches are split into chunks (same maths) - the per-train stretch inputs need more GPU memory.
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../src_v2"
LOG=../logs/final.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }
for m in 128 64; do
  rm -rf ../checkpoints_s2s_test ../logs/train_metrics_s2s_test.csv
  say "TEST - small stop-to-stop model, 8,000 steps, chunks of $m"
  if caffeinate -i uv run python train.py --run s2s_test --cumulative --micro-bs $m --max-steps 8000 --eval-every 2000 \
       2>&1 | tee -a $LOG; then
    say "TEST DONE - first model val MAE: 15.91 @ 4000, 15.53 @ 8000. Stop-to-stop:"
    grep -E "== val @ (2000|4000|6000|8000):" $LOG | tail -4 | tee -a $LOG
    say "If worth it: scripts/run_final_xl.sh"
    exit 0
  fi
  say "chunks of $m failed"
done
say "FAILED: test"
exit 1

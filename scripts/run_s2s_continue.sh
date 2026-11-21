#!/bin/zsh
# Continue the stop-to-stop (summed stretches) test from its step-8,000 checkpoint to 16,000, after the
# direct A/B is done (one GPU job at a time). First model: 15.19 @ 12000, 15.00 @ 16000.
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../src_v2"
LOG=../logs/final.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }
while pgrep -f "run_s2s_direct.sh" >/dev/null; do sleep 60; done
say "stop-to-stop test continued from step 8,000 to 16,000"
caffeinate -i uv run python train.py --run s2s_test --cumulative --micro-bs 128 --max-steps 16000 --eval-every 2000 \
  --resume 2>&1 | tee -a $LOG || { say "FAILED: continuation"; exit 1; }
say "CONTINUATION DONE - first model val MAE: 15.19 @ 12000, 15.00 @ 16000. Stop-to-stop:"
grep -E "== val @ (10000|12000|14000|16000):" $LOG | tail -4 | tee -a $LOG

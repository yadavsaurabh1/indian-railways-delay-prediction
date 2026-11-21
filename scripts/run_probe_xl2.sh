#!/bin/zsh
# XL probe, restarted after a GPU out-of-memory at step ~1,850 (no checkpoint had been written yet).
# Same model, data, LR schedule and validation subset as before. Batch split into chunks of 64
# (identical maths, lower memory peak). Validation + checkpoint every 1,000 steps; after a crash it
# resumes from checkpoints_probe_XL/last.pt (up to 3 retries).
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../src"
LOG=../logs/probe_xl.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }
ARGS=(--run probe_XL --d 512 --layers 12 --ctx-layers 4 --heads 8 --micro-bs 64 --max-steps 8000 --eval-every 1000)
say "probe restart: XL, chunks of 64, checkpoint every 1,000 steps"
for attempt in 1 2 3 4; do
  R=(); [ -f ../checkpoints_probe_XL/last.pt ] && R=(--resume)
  if caffeinate -i uv run python train.py ${ARGS[@]} ${R[@]} 2>&1 | tee -a $LOG; then
    say "XL probe finished - compare: first run val @ 4000 = 15.91, @ 8000 = 15.53"
    exit 0
  fi
  say "probe stopped (attempt $attempt) - resuming in 2 min"
  sleep 120
done
say "FAILED: probe crashed 4 times"
exit 1

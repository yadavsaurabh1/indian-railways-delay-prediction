#!/bin/zsh
# Scaling probe: does a ~4x bigger transformer beat the current one at the same step?
# Same data, same LR schedule (8-epoch cosine), same validation subset as the first run,
# stopped at 8,000 steps. Compare its "== val @ 4000/8000" lines with the first run's
# (15.91 / 15.53 min MAE). Waits for run_followups.sh to finish so the GPU/RAM are free.
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../src"
LOG=../logs/probe.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }

say "waiting for run_followups.sh (ablation + LightGBM) to finish"
while pgrep -f run_followups.sh >/dev/null; do sleep 60; done

ARGS=(--d 384 --layers 8 --ctx-layers 3 --heads 8 --micro-bs 256)
say "smoke test of the large config"
caffeinate -i uv run python train.py --smoke ${ARGS[@]} 2>&1 | tee -a $LOG || { say "FAILED: smoke"; exit 1; }

say "probe: large model, 8,000 steps"
caffeinate -i uv run python train.py --run probe_L ${ARGS[@]} --max-steps 8000 --eval-every 2000 2>&1 | tee -a $LOG \
  || { say "FAILED: probe"; exit 1; }
say "probe finished - compare with first run: val @ 4000 = 15.91, val @ 8000 = 15.53 min MAE"

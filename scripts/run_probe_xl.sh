#!/bin/zsh
# Second scaling point: XL transformer (d=512, 12 route blocks, 4 context blocks, ~75M params),
# same data / LR schedule / validation subset, stopped at 8,000 steps.
# Full 512 batch first; only if that fails (e.g. GPU memory) it retries with the batch split
# into chunks (identical maths). Runs after run_probe.sh (the L probe) has finished.
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../src"
LOG=../logs/probe_xl.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }

say "waiting for run_followups.sh and run_probe.sh to finish"
while pgrep -f "run_followups.sh|run_probe.sh" >/dev/null; do sleep 60; done

ARGS=(--d 512 --layers 12 --ctx-layers 4 --heads 8)
MICRO=()
say "smoke test of the XL config, full batch"
if ! caffeinate -i uv run python train.py --smoke ${ARGS[@]} 2>&1 | tee -a $LOG; then
  say "full batch failed - retrying with chunks of 128 (same maths)"
  MICRO=(--micro-bs 128)
  caffeinate -i uv run python train.py --smoke ${ARGS[@]} ${MICRO[@]} 2>&1 | tee -a $LOG || { say "FAILED: smoke"; exit 1; }
fi

say "probe: XL model, 8,000 steps ${MICRO[@]}"
caffeinate -i uv run python train.py --run probe_XL ${ARGS[@]} ${MICRO[@]} --max-steps 8000 --eval-every 2000 2>&1 \
  | tee -a $LOG || { say "FAILED: probe"; exit 1; }
say "XL probe finished - compare: first run val @ 4000 = 15.91, @ 8000 = 15.53; L probe in logs/probe.log"

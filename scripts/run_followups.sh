#!/bin/zsh
# After training: (1) ablation - how much the other-train information helps,
# (2) LightGBM quick check on a small slice, (3) full LightGBM comparison + blend.
# Stops at the first failing step.
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../src"
LOG=../logs/followups.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }
run() { caffeinate -i uv run python "$@" 2>&1 | tee -a $LOG || { say "FAILED: $*"; exit 1; }; }

say "step 1/3: ablation (test split, 5 variants)"
run ablate.py

say "step 2/3: LightGBM quick check"
run lgbm_baseline.py --train-samples 20000 --val-samples 3000 --max-test-samples 3000 --rounds 50 --tag _smoke

say "step 3/3: full LightGBM comparison"
run lgbm_baseline.py

say "all done: reports/ablation_test.json, reports/lgbm_test.json, reports/comparison.json"

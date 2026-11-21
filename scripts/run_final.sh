#!/bin/zsh
# The final model: XL (d=512, 12 route layers, 4 context layers), every input, stop-to-stop output.
#
#  1. code check on one small month: stretch features -> tiny pack -> CPU smoke run (minutes)
#  2. stretch features for all months (other trains per stretch, with train identity and class)
#  3. pack them next to data/packed
#  4. TEST: a small stop-to-stop model - same size and training recipe as the first model (d192, 4 layers,
#     cosine lr 4e-4), only the new inputs + stop-to-stop output differ. 8,000 steps, then STOP.
#     Compare with the first model at the same steps: 15.91 @ 4000, 15.53 @ 8000.
#  -> if worth it: scripts/run_final_xl.sh trains the XL stop-to-stop model to the end.
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src_v2"
LOG=../logs/final.log
say() { echo "=== [$(date '+%F %T')] $*" | tee -a $LOG; }
run() { caffeinate -i uv run python "$@" 2>&1 | tee -a $LOG || { say "FAILED: $*"; exit 1; }; }

while pgrep -f "run_probe.sh|run_probe_xl|train.py" >/dev/null; do say "another training job is running - waiting"; sleep 300; done

say "step 1/4: code check on 2026-09 (stretch features -> tiny pack -> CPU smoke)"
export POLARS_MAX_THREADS=8
run build_segments.py 2026-09
rm -rf ../data/packed_tiny ../checkpoints_tiny
run pack.py --months 2026-09 --out ../data/packed_tiny
run pack.py --segments-only --months 2026-09 --out ../data/packed_tiny
run train.py --smoke --packed ../data/packed_tiny --device cpu --d 64 --heads 4 --layers 2 --ctx-layers 1 \
  --cumulative --schedule adaptive --ckpt-dir ../checkpoints_tiny --workers 0

say "step 2/4: stretch features for all months"
run build_segments.py --skip-done

say "step 3/4: pack them next to data/packed"
df -h .. | tail -1 | tee -a $LOG
run pack.py --segments-only --out ../data/packed
du -sh ../data/packed | tee -a $LOG

say "step 4/4: TEST - small stop-to-stop model, 8,000 steps"
rm -rf ../checkpoints_s2s_test ../logs/train_metrics_s2s_test.csv
run train.py --run s2s_test --cumulative --max-steps 8000 --eval-every 2000
say "TEST DONE - first model val MAE: 15.91 @ 4000, 15.53 @ 8000. Stop-to-stop:"
grep -E "== val @ (2000|4000|6000|8000):" $LOG | tail -4 | tee -a $LOG
say "If worth it: scripts/run_final_xl.sh"

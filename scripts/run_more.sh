#!/bin/zsh
# Continue the XL direct run past 8 epochs: resume from step 103,424 at the same fixed 2.5e-5,
# up to 16 epochs. Nothing stops it automatically - stop by hand (Ctrl-C in this window).
set -u
set -o pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
cd "$(dirname "$0")/../src_v2"
LOG=../logs/final.log
say() { echo "=== [$(date "+%F %T")] $*" | tee -a $LOG; }
cp ../checkpoints_final/best.pt ../checkpoints_final/best_step103k.pt
cp ../checkpoints_final/last.pt ../checkpoints_final/last_step103k.pt
for n in val test; do cp ../reports/eval_${n}_final.json ../reports/eval_${n}_final_step103k.json; done
say "MORE: continuing from step 103,424 at fixed 2.5e-5, epochs 8 -> 16 (best val pinball 0.2410)"
ARGS=(--run final --d 512 --heads 8 --layers 12 --ctx-layers 4 --micro-bs 32 --epochs 16 --eval-every 2000
      --schedule constant --resume --lr 2.5e-5)
for attempt in 1 2 3 4 5; do
  if caffeinate -i uv run python train.py ${ARGS[@]} 2>&1 | tee -a $LOG; then say "MORE RUN DONE"; exit 0; fi
  say "stopped (attempt $attempt) - resuming in 2 min"; sleep 120
done
say "FAILED after 5 attempts"

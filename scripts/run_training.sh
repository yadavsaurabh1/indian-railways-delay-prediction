#!/bin/zsh
# Remaining pipeline after build_month: weather -> pack -> smoke test -> full training -> test report.
# Every step is resumable; re-running skips finished work.
set -eu
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../src"
mkdir -p ../logs

echo "=== [$(date '+%F %T')] month tensors (skips finished months)"
uv run python build_month.py --skip-done

echo "=== [$(date '+%F %T')] weather (NASA POWER, cached per cell)"
uv run python fetch_weather.py || echo "!!! weather fetch failed - continuing without weather"

echo "=== [$(date '+%F %T')] pack"
uv run python pack.py

echo "=== [$(date '+%F %T')] smoke test"
uv run python train.py --smoke --workers 2   # writes to checkpoints_smoke/

echo "=== [$(date '+%F %T')] full training"
if [ -f ../checkpoints/last.pt ]; then uv run python train.py --resume; else uv run python train.py; fi

echo "=== [$(date '+%F %T')] done. Reports in reports/eval_val.json and reports/eval_test.json"

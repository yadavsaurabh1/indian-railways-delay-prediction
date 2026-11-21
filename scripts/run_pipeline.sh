#!/bin/zsh
# Download the dataset and convert it to Parquet. Safe to re-run: every step resumes.
set -u
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
mkdir -p logs data/raw reports  # NOTE: when launching with tee, create logs/ first

EXPECTED_CSVS=12935
export HF_XET_HIGH_PERFORMANCE=1

echo "=== [$(date '+%F %T')] download yadavsaurabh/indian-railways-runs -> data/raw"
attempt=0
while :; do
  attempt=$((attempt + 1))
  uv run hf download yadavsaurabh/indian-railways-runs --repo-type dataset \
      --local-dir data/raw --max-workers 16 && \
    have=$(find data/raw/trains data/raw/trains_1 -name '*.csv' 2>/dev/null | wc -l | tr -d ' ') && \
    [ "$have" -ge "$EXPECTED_CSVS" ] && break
  have=$(find data/raw/trains data/raw/trains_1 -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')
  echo "=== download attempt $attempt incomplete ($have/$EXPECTED_CSVS CSVs); retrying in 60s"
  [ "$attempt" -ge 30 ] && { echo "giving up after $attempt attempts"; exit 1; }
  sleep 60
done
echo "=== [$(date '+%F %T')] download complete: $(du -sh data/raw | cut -f1), $have CSVs"

echo "=== [$(date '+%F %T')] ingest: CSV -> typed parquet -> monthly partitions"
uv run python src/ingest.py all --workers 8 || { echo "ingest failed"; exit 1; }

echo "=== [$(date '+%F %T')] pipeline finished. Summary in reports/ingest_summary.json"

"""Per (train, station) delay history, strictly causal.

For a journey starting on day D, a stop's history covers that train's observed
delays at that station over the 90 days ending at the latest run on or before
D-3. The 3-day gap ensures the earlier runs used here (up to about 75 h for
the longest trains) had finished before journey D started.

data/staged/NNNNN.parquet -> data/history_staged/NNNNN.parquet -> data/history_by_month/ym=YYYY-MM/
Columns: train_no, date, seq, station_code, h_mean, h_med, h_std, h_q90, h_cnt, h_last, h_age_days
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common import DATA, HISTORY, STAGED, clean

OUT = DATA / "history_staged"
GAP_DAYS = 3
WINDOW = "90d"


def _one(src: str, dst: str) -> int:
    import polars as pl

    c = clean(pl.scan_parquet(src).with_columns(pl.col("halts_here").fill_null(1))).select(
        "train_no", "date", "seq", "station_code", "target").collect()
    if c.is_empty():
        return 0
    obs = (c.drop_nulls("target").group_by("station_code", "date").agg(pl.col("target").mean())
           .sort("station_code", "date"))
    keys = c.select("train_no", "date", "seq", "station_code").sort("date")
    if obs.is_empty():
        out = keys
    else:
        t = pl.col("target")
        roll = (obs.rolling("date", period=WINDOW, group_by="station_code")
                .agg(t.mean().alias("h_mean"), t.median().alias("h_med"), t.std().alias("h_std"),
                     t.quantile(0.9).alias("h_q90"), pl.len().alias("h_cnt"), t.last().alias("h_last"))
                .with_columns((pl.col("date") + pl.duration(days=GAP_DAYS)).alias("avail_date"),
                              pl.col("date").alias("h_date"))
                .drop("date").sort("avail_date"))
        out = keys.join_asof(roll, left_on="date", right_on="avail_date", by="station_code",
                             strategy="backward", check_sortedness=False)
        out = out.with_columns((pl.col("date") - pl.col("h_date")).dt.total_days().cast(pl.Int16).alias("h_age_days"),
                               pl.col(pl.Float64).cast(pl.Float32), pl.col("h_cnt").cast(pl.Int16)
                               ).drop("avail_date", "h_date")
    out = out.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("ym"))
    tmp = dst + ".tmp"
    out.write_parquet(tmp, compression="zstd")
    os.replace(tmp, dst)
    return out.height


def _init():
    os.environ["POLARS_MAX_THREADS"] = "1"


def main() -> None:
    import pyarrow as pa
    import pyarrow.dataset as ds

    OUT.mkdir(parents=True, exist_ok=True)
    srcs = sorted(STAGED.glob("*.parquet"))
    todo = [(str(s), str(OUT / s.name)) for s in srcs if not (OUT / s.name).exists()]
    print(f"[history] {len(srcs)} trains, {len(todo)} to do", flush=True)
    t0, rows, errs = time.time(), 0, {}
    os.environ["POLARS_MAX_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=8, initializer=_init) as ex:
        futs = {ex.submit(_one, s, d): Path(s).stem for s, d in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows += f.result()
            except Exception as e:
                errs[futs[f]] = repr(e)
            if i % 1000 == 0 or i == len(todo):
                print(f"[history] {i}/{len(todo)} rows={rows:,} errors={len(errs)} {time.time() - t0:.0f}s", flush=True)
    if errs:
        print("[history] errors:", list(errs.items())[:5], flush=True)
        sys.exit(1)
    print("[history] repartitioning by month", flush=True)
    ds.write_dataset(
        ds.dataset(str(OUT), format="parquet"), str(HISTORY), format="parquet",
        partitioning=ds.partitioning(pa.schema([("ym", pa.string())]), flavor="hive"),
        existing_data_behavior="delete_matching",
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        max_open_files=256, min_rows_per_group=50_000, max_rows_per_group=500_000)
    print(f"[history] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

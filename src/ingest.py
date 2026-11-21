"""Convert the raw per-train CSVs into typed Parquet.

stage1: data/raw/{trains,trains_1}/NNNNN.csv -> data/staged/NNNNN.parquet
        One typed file per train. Times become real timestamps (IST, naive),
        Devanagari platform numerals become ASCII. Resumable: a train is
        skipped when its parquet is newer than its CSV.
stage2: data/staged/*.parquet -> data/by_month/ym=YYYY-MM/*.parquet
        Hive-partitioned by journey start month, so network-wide features
        (which trains were on the same stretch at the same time) can be built
        one month at a time within 16 GB of RAM.
summary: writes reports/ingest_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
STAGED = ROOT / "data" / "staged"
BY_MONTH = ROOT / "data" / "by_month"
REPORTS = ROOT / "reports"

RAW_COLUMNS = [
    "date", "train_no", "train_name", "train_type", "seq", "station_code",
    "halts_here", "sch_arrival", "sch_arrival_date", "sch_departure",
    "sch_departure_date", "act_arrival", "act_arrival_date", "act_departure",
    "act_departure_date", "delay_arrival_min", "delay_departure_min",
    "platform", "distance_km", "lat", "lng", "departed", "reported",
    "reported_at",
]

# (output column, date column, time column)
TIMESTAMP_PAIRS = [
    ("sch_arr", "sch_arrival_date", "sch_arrival"),
    ("sch_dep", "sch_departure_date", "sch_departure"),
    ("act_arr", "act_arrival_date", "act_arrival"),
    ("act_dep", "act_departure_date", "act_departure"),
]

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _convert_one(csv_path: str, out_path: str) -> dict:
    import polars as pl

    df = pl.read_csv(csv_path, infer_schema=False)  # everything as String
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    df = df.with_columns(pl.col(pl.String).str.strip_chars().replace("", None))

    def ts(date_col: str, time_col: str) -> pl.Expr:
        return pl.concat_str([pl.col(date_col), pl.col(time_col)], separator=" ").str.strptime(
            pl.Datetime("us"), "%d %b %Y %H:%M", strict=False
        )

    reported_at = pl.coalesce(
        pl.col("reported_at").str.strptime(pl.Datetime("us", "UTC"), "%Y-%m-%dT%H:%M:%S%.f%:z", strict=False),
        pl.col("reported_at").str.strptime(pl.Datetime("us", "UTC"), "%Y-%m-%dT%H:%M:%S%:z", strict=False),
    ).dt.convert_time_zone("Asia/Kolkata").dt.replace_time_zone(None)

    out = df.select(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
        pl.col("train_no").str.zfill(5),
        pl.col("train_name"),
        pl.col("train_type"),
        pl.col("seq").cast(pl.Float32, strict=False),
        pl.col("station_code"),
        pl.col("halts_here").cast(pl.Int8, strict=False),
        *[ts(d, t).alias(name) for name, d, t in TIMESTAMP_PAIRS],
        pl.col("delay_arrival_min").cast(pl.Float32, strict=False),
        pl.col("delay_departure_min").cast(pl.Float32, strict=False),
        pl.col("platform").map_elements(lambda s: s.translate(DEVANAGARI_DIGITS), return_dtype=pl.String),
        pl.col("distance_km").cast(pl.Float64, strict=False),
        pl.col("lat").cast(pl.Float64, strict=False),
        pl.col("lng").cast(pl.Float64, strict=False),
        pl.col("departed").cast(pl.Int8, strict=False),
        pl.col("reported").cast(pl.Int8, strict=False),
        reported_at.alias("reported_at"),
    ).with_columns(pl.col("date").dt.strftime("%Y-%m").alias("ym"))

    # Parse-failure accounting: a value was present in the CSV but became null.
    failures = {}
    raw_present = {
        "date": df["date"].is_not_null(),
        "seq": df["seq"].is_not_null(),
        "halts_here": df["halts_here"].is_not_null(),
        "delay_arrival_min": df["delay_arrival_min"].is_not_null(),
        "delay_departure_min": df["delay_departure_min"].is_not_null(),
        "distance_km": df["distance_km"].is_not_null(),
        "lat": df["lat"].is_not_null(),
        "lng": df["lng"].is_not_null(),
        "departed": df["departed"].is_not_null(),
        "reported": df["reported"].is_not_null(),
        "reported_at": df["reported_at"].is_not_null(),
    }
    for name, d, t in TIMESTAMP_PAIRS:
        raw_present[name] = df[d].is_not_null() & df[t].is_not_null()
    for col, present in raw_present.items():
        n = int((present & out[col].is_null()).sum())
        if n:
            failures[col] = n

    tmp = out_path + ".tmp"
    out.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, out_path)
    return {"rows": out.height, "failures": failures}


def _worker_init():
    os.environ["POLARS_MAX_THREADS"] = "1"


def stage1(workers: int) -> None:
    STAGED.mkdir(parents=True, exist_ok=True)
    csvs = sorted(RAW.glob("trains*/*.csv"))
    if not csvs:
        sys.exit(f"no CSVs under {RAW}")
    todo = []
    for c in csvs:
        out = STAGED / (c.stem + ".parquet")
        if not out.exists() or out.stat().st_mtime < c.stat().st_mtime:
            todo.append((str(c), str(out)))
    print(f"[stage1] {len(csvs)} CSVs, {len(todo)} to convert, {workers} workers", flush=True)

    manifest_path = REPORTS / "stage1_manifest.json"
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    errors = {}
    t0 = time.time()
    os.environ["POLARS_MAX_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        futs = {ex.submit(_convert_one, c, o): Path(c).stem for c, o in todo}
        for i, f in enumerate(as_completed(futs), 1):
            train = futs[f]
            try:
                manifest[train] = f.result()
            except Exception as e:  # keep going, report at the end
                errors[train] = repr(e)
            if i % 500 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"[stage1] {i}/{len(todo)}  {rate:.1f} files/s  errors={len(errors)}", flush=True)
                manifest_path.write_text(json.dumps(manifest))
    manifest_path.write_text(json.dumps(manifest))
    if errors:
        (REPORTS / "stage1_errors.json").write_text(json.dumps(errors, indent=1))
        print(f"[stage1] {len(errors)} files failed, see reports/stage1_errors.json", flush=True)
    print(f"[stage1] done in {time.time() - t0:.0f}s", flush=True)


def stage2() -> None:
    import pyarrow as pa
    import pyarrow.dataset as ds

    t0 = time.time()
    src = ds.dataset(str(STAGED), format="parquet")
    print(f"[stage2] repartitioning {len(src.files)} files by month -> {BY_MONTH}", flush=True)
    ds.write_dataset(
        src,
        str(BY_MONTH),
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("ym", pa.string())]), flavor="hive"),
        existing_data_behavior="delete_matching",
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        max_open_files=256,
        min_rows_per_group=50_000,
        max_rows_per_group=500_000,
        max_rows_per_file=5_000_000,
    )
    print(f"[stage2] done in {time.time() - t0:.0f}s", flush=True)


def summary() -> None:
    import polars as pl

    lf = pl.scan_parquet(str(BY_MONTH / "**/*.parquet"), hive_partitioning=True)
    s = lf.select(
        pl.len().alias("rows"),
        pl.col("train_no").n_unique().alias("trains"),
        pl.col("station_code").n_unique().alias("stations"),
        pl.struct("train_no", "date").n_unique().alias("journeys"),
        pl.col("date").min().cast(pl.String).alias("first_date"),
        pl.col("date").max().cast(pl.String).alias("last_date"),
        pl.col("reported").mean().alias("reported_frac"),
        pl.col("delay_arrival_min").filter(pl.col("reported") == 1).median().alias("median_arr_delay"),
        pl.col("delay_arrival_min").filter(pl.col("reported") == 1).quantile(0.9).alias("p90_arr_delay"),
    ).collect().to_dicts()[0]

    manifest = json.loads((REPORTS / "stage1_manifest.json").read_text())
    fail_totals: dict[str, int] = {}
    for v in manifest.values():
        for k, n in v["failures"].items():
            fail_totals[k] = fail_totals.get(k, 0) + n
    s["parse_failures"] = fail_totals
    s["staged_rows"] = sum(v["rows"] for v in manifest.values())
    (REPORTS / "ingest_summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["stage1", "stage2", "summary", "all"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--one", help="convert a single CSV to stdout-summary (smoke test)")
    a = p.parse_args()
    if a.one:
        out = str(ROOT / "data" / "smoke.parquet")
        print(_convert_one(a.one, out))
        import polars as pl
        with pl.Config(tbl_cols=30, tbl_width_chars=250):
            print(pl.read_parquet(out).head(5))
        return
    if a.step in ("stage1", "all"):
        stage1(a.workers)
    if a.step in ("stage2", "all"):
        stage2()
    if a.step in ("summary", "all"):
        summary()


if __name__ == "__main__":
    main()

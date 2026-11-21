"""Shared paths, time helpers and the single cleaning definition used everywhere.

Cleaning rules (verified against the data, 2026-11-04):
  * delay_*_min columns are NULL whenever the delay is < 2 min (incl. early
    arrivals), so signed delays are recomputed as act - sch from timestamps.
  * act_* on unreported stops are projections from the source -> ignored.
  * act_dep is a projection when the stop's last update happened before it
    (reported_at < act_dep - 1 min) -> departure not treated as observed.
  * rows without `seq` (4.75%, almost never reported) are dropped; exact
    duplicate stops are collapsed keeping the reported/latest copy.
  * information about a stop becomes available AVAIL_LAG_MIN after the event.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BY_MONTH = DATA / "by_month"
STAGED = DATA / "staged"
STATIC = DATA / "static"
HISTORY = DATA / "history_by_month"
TENSORS = DATA / "tensors"
REPORTS = ROOT / "reports"

EPOCH = datetime(2020, 1, 1)
AVAIL_LAG_MIN = 2
MIN_DELAY, MAX_DELAY = -180, 3000  # signed minutes; outside -> treated as bad data

# Suburban services: 63% of journeys. All of them stay as *context* (they share
# track), but only 1 in SUBURBAN_KEEP of their journeys become *targets*.
SUBURBAN_TYPES = {"EMU", "SUB", "DEMU", "MEMU", "EMU - KOLKATA", "EMU - MUMBAI", "EMU - DELHI"}
SUBURBAN_KEEP = 4

TRAIN_END = "2025-09-01"   # train: journeys before this
VAL_END = "2026-03-01"     # val: [TRAIN_END, VAL_END), test: [VAL_END, ...)


def minutes(expr: pl.Expr) -> pl.Expr:
    """Datetime -> int32 minutes since 2020-01-01 (IST, naive)."""
    return (expr - pl.lit(EPOCH)).dt.total_minutes().cast(pl.Int32)


def _signed(a: str, s: str) -> pl.Expr:
    d = (pl.col(a) - pl.col(s)).dt.total_minutes().cast(pl.Float32)
    return pl.when(d.is_between(MIN_DELAY, MAX_DELAY)).then(d).otherwise(None)


def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Raw staged/by_month rows -> one clean row per journey stop, in route order.

    Adds: stop_idx, n_stops, sch_arr_m, sch_dep_m, obs_arr, obs_dep, delay_arr,
    delay_dep, target (arrival delay; departure delay at the origin),
    avail_arr_m, avail_dep_m, avail_m (when `target` became known).
    """
    j = ["train_no", "date"]
    lf = (
        lf.filter(pl.col("seq").is_not_null() & (pl.col("sch_arr").is_not_null() | pl.col("sch_dep").is_not_null()))
        .sort(["train_no", "date", "seq", "reported", "reported_at"],
              descending=[False, False, False, True, True], nulls_last=True)
        .unique(subset=["train_no", "date", "station_code", "seq"], keep="first", maintain_order=True)
        .with_columns(
            pl.coalesce("sch_arr", "sch_dep").alias("sch_arr"),
            pl.coalesce("sch_dep", "sch_arr").alias("sch_dep"),
            pl.int_range(pl.len()).over(j).cast(pl.Int16).alias("stop_idx"),
            pl.len().over(j).cast(pl.Int16).alias("n_stops"),
        )
    )
    rep = pl.col("reported") == 1
    first = pl.col("stop_idx") == 0
    last = pl.col("stop_idx") == pl.col("n_stops") - 1
    lf = lf.with_columns(
        (rep & pl.col("act_arr").is_not_null() & ~first).alias("obs_arr"),
        (rep & pl.col("act_dep").is_not_null() & ~last
         & (pl.col("reported_at") >= pl.col("act_dep") - pl.duration(minutes=1))).alias("obs_dep"),
    ).with_columns(
        pl.when("obs_arr").then(_signed("act_arr", "sch_arr")).alias("delay_arr"),
        pl.when("obs_dep").then(_signed("act_dep", "sch_dep")).alias("delay_dep"),
    ).with_columns(
        # a delay outside the plausible range is bad data, not an observation
        (pl.col("obs_arr") & pl.col("delay_arr").is_not_null()).alias("obs_arr"),
        (pl.col("obs_dep") & pl.col("delay_dep").is_not_null()).alias("obs_dep"),
        minutes(pl.col("sch_arr")).alias("sch_arr_m"),
        minutes(pl.col("sch_dep")).alias("sch_dep_m"),
        (minutes(pl.col("act_arr")) + AVAIL_LAG_MIN).alias("avail_arr_m"),
        (minutes(pl.col("act_dep")) + AVAIL_LAG_MIN).alias("avail_dep_m"),
    ).with_columns(
        pl.when(first).then(pl.col("delay_dep")).otherwise(pl.col("delay_arr")).alias("target"),
        pl.when(first).then(pl.when("obs_dep").then("avail_dep_m"))
          .otherwise(pl.when("obs_arr").then("avail_arr_m")).alias("avail_m"),
    )
    return lf


def scan_months(months: list[str]) -> pl.LazyFrame:
    paths = [str(BY_MONTH / f"ym={m}" / "*.parquet") for m in months if (BY_MONTH / f"ym={m}").exists()]
    return pl.scan_parquet(paths).with_columns(pl.col("halts_here").fill_null(1))


def all_months() -> list[str]:
    return sorted(p.name.split("=", 1)[1] for p in BY_MONTH.glob("ym=*"))


def neighbours(month: str) -> list[str]:
    ms = all_months()
    i = ms.index(month)
    return ms[max(0, i - 1): i + 2]

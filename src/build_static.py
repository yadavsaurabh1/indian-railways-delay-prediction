"""Global tables: station / train / type vocabularies, station coordinates,
paired (rake-link) trains, and daily per-station congestion.

Outputs in data/static/:
  stations.parquet       station_code, station_id, lat, lng, events, reported_frac
  trains.parquet         train_no, train_id, train_type, type_id, src, dst,
                         sch_duration_min, journeys, pair_train_no, pair_id
  types.parquet          train_type, type_id
  station_daily.parquet  station_id, day, gain_*, delay_*, n_* rolling stats,
                         available for use from day+2 (see build_month)
Ids 0 = padding, 1 = unknown; real ids start at 2.
"""

from __future__ import annotations

import time

import polars as pl

from common import STATIC, all_months, clean, minutes, scan_months


def main() -> None:
    t0 = time.time()
    STATIC.mkdir(parents=True, exist_ok=True)
    months = all_months()
    lf = scan_months(months)

    # ---- stations -------------------------------------------------------
    st = (
        lf.group_by("station_code")
        .agg(pl.col("lat").median(), pl.col("lng").median(), pl.len().alias("events"),
             pl.col("reported").mean().alias("reported_frac"))
        .sort("events", descending=True)
        .collect()
        .with_row_index("station_id", offset=2)
        .with_columns(pl.col("station_id").cast(pl.Int32))
    )
    st.write_parquet(STATIC / "stations.parquet")
    print(f"[static] {st.height} stations", flush=True)

    # ---- types ---------------------------------------------------------
    ty = (lf.group_by("train_type").agg(pl.len()).sort("len", descending=True).collect()
          .drop_nulls("train_type").with_row_index("type_id", offset=2)
          .with_columns(pl.col("type_id").cast(pl.Int16)).drop("len"))
    ty.write_parquet(STATIC / "types.parquet")

    # ---- trains: route from the most recent journey ----------------------
    c = clean(lf.select("train_no", "date", "seq", "station_code", "sch_arr", "sch_dep", "reported",
                        "reported_at", "act_arr", "act_dep", "train_type", "halts_here"))
    last_journey = c.group_by("train_no").agg(pl.col("date").max().alias("last_date"))
    route = (
        c.join(last_journey, on="train_no").filter(pl.col("date") == pl.col("last_date"))
        .group_by("train_no")
        .agg(pl.col("station_code").sort_by("stop_idx").first().alias("src"),
             pl.col("station_code").sort_by("stop_idx").last().alias("dst"),
             (pl.col("sch_arr_m").max() - pl.col("sch_dep_m").min()).alias("sch_duration_min"),
             pl.col("train_type").last())
    )
    counts = c.group_by("train_no").agg(pl.struct("date").n_unique().alias("journeys"))
    tr = (route.join(counts, on="train_no").collect().sort("journeys", descending=True)
          .with_row_index("train_id", offset=2).with_columns(pl.col("train_id").cast(pl.Int32))
          .join(ty, on="train_type", how="left").with_columns(pl.col("type_id").fill_null(1)))
    n = tr["train_no"].cast(pl.Int64, strict=False)
    tr = tr.with_columns(
        pl.when(n % 2 == 0).then(n - 1).otherwise(n + 1).cast(pl.String).str.zfill(5).alias("pair_cand"))
    pair = tr.select(pl.col("train_no").alias("pair_cand"), pl.col("train_id").alias("pid"),
                     pl.col("src").alias("psrc"), pl.col("dst").alias("pdst"))
    tr = tr.join(pair, on="pair_cand", how="left").with_columns(
        pl.when(pl.col("pdst") == pl.col("src")).then(pl.col("pair_cand")).alias("pair_train_no"),
        pl.when(pl.col("pdst") == pl.col("src")).then(pl.col("pid")).otherwise(0).cast(pl.Int32).alias("pair_id"),
    ).drop("pair_cand", "pid", "psrc", "pdst")
    tr.write_parquet(STATIC / "trains.parquet")
    print(f"[static] {tr.height} trains, {int((tr['pair_id'] > 0).sum())} with a rake-link pair "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- per-station daily congestion, month by month (bounded memory) ---
    sid = st.select("station_code", "station_id")
    parts = []
    for m in months:
        d = (
            clean(scan_months([m]).select("train_no", "date", "seq", "station_code", "sch_arr", "sch_dep",
                                          "reported", "reported_at", "act_arr", "act_dep", "halts_here"))
            .sort("train_no", "date", "stop_idx")
            .with_columns(
                # delay known just before this stop: last observed dep/arr delay earlier in the journey
                pl.coalesce("delay_dep", "delay_arr").shift(1).forward_fill().over(["train_no", "date"]).alias("prev_delay"))
            .filter("obs_arr")
            .with_columns((pl.col("delay_arr") - pl.col("prev_delay")).alias("gain"),
                          pl.col("act_arr").dt.date().alias("day"))
            .group_by("station_code", "day")
            .agg(pl.col("gain").sum().alias("gain_sum"), pl.col("gain").count().alias("gain_n"),
                 pl.col("delay_arr").sum().alias("delay_sum"), pl.len().alias("n"))
            .collect()
        )
        parts.append(d)
    daily = (pl.concat(parts).group_by("station_code", "day").sum()
             .join(sid, on="station_code").drop("station_code").sort("station_id", "day"))
    out = []
    for w in (7, 28):
        out.append(daily.rolling("day", period=f"{w}d", group_by="station_id").agg(
            (pl.col("gain_sum").sum() / pl.col("gain_n").sum()).alias(f"st_gain_{w}d"),
            (pl.col("delay_sum").sum() / pl.col("n").sum()).alias(f"st_delay_{w}d"),
            (pl.col("n").sum() / w).alias(f"st_traffic_{w}d"),
        ))
    roll = out[0].join(out[1], on=["station_id", "day"])
    roll = roll.with_columns(pl.col(pl.Float64).cast(pl.Float32)).sort("station_id", "day")
    roll.write_parquet(STATIC / "station_daily.parquet")
    print(f"[static] station_daily {roll.height} rows. done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

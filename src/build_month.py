"""Training examples for one month.

A sample = (target journey, cutoff time t). Everything the model sees about a
sample is what was knowable at t; everything after t on the target journey
is to be predicted.

Per month m -> data/tensors/ym=m/
  stops.parquet    one row per stop of each target journey (route order)
  journeys.parquet j_local, train_id, type_id, pair_id, date, n_stops
  samples.parquet  sample_id, j_local, t, cur_delay, cur_known, cur_age, last_known_idx, n_targets
  context.parquet  other trains around the next 6 halts (slots 0-5) + rake-link pair (slot 6),
                   mirroring what live_station / live_status can return at time t.

Context window per station S (matches the live_station view, then capped):
  their scheduled time at S in [t - 8h, t + 6h], not departed S more than 1h before t;
  keep the CTX_PER_SLOT trains closest to our expected time at S.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import polars as pl

from common import (HISTORY, STATIC, SUBURBAN_KEEP, SUBURBAN_TYPES, TENSORS, all_months, clean,
                    neighbours, scan_months)

K_CUTOFFS = 2
N_SLOTS = 6
RAKE_SLOT = 6
CTX_PER_SLOT = 8
WIN_BEFORE, WIN_AFTER = 8 * 60, 6 * 60
DEPARTED_GRACE = 60
BUCKET = 120
CHUNK = 20_000
DAY0 = pl.date(2020, 1, 1)

RAW_COLS = ["train_no", "date", "seq", "station_code", "halts_here", "sch_arr", "sch_dep", "act_arr", "act_dep",
            "reported", "reported_at", "distance_km", "train_type"]


def _rad(e: pl.Expr) -> pl.Expr:
    return e * (math.pi / 180.0)


def haversine_km(lat1, lng1, lat2, lng2) -> pl.Expr:
    p1, p2 = _rad(lat1), _rad(lat2)
    dp, dl = p2 - p1, _rad(lng2) - _rad(lng1)
    a = (dp / 2).sin() ** 2 + p1.cos() * p2.cos() * (dl / 2).sin() ** 2
    return 2 * 6371.0 * a.sqrt().arcsin()


def bearing(lat1, lng1, lat2, lng2) -> pl.Expr:
    p1, p2 = _rad(lat1), _rad(lat2)
    dl = _rad(lng2) - _rad(lng1)
    return pl.arctan2(dl.sin() * p2.cos(), p1.cos() * p2.sin() - p1.sin() * p2.cos() * dl.cos())


def load(month: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Clean rows for month-1..month+1 joined to static ids, plus the target month's mask."""
    st = pl.read_parquet(STATIC / "stations.parquet").select("station_code", "station_id", "lat", "lng")
    tr = pl.read_parquet(STATIC / "trains.parquet").select("train_no", "train_id", "type_id", "pair_id")
    ty = pl.read_parquet(STATIC / "types.parquet")
    c = (clean(scan_months(neighbours(month)).select(RAW_COLS))
         .drop("lat", "lng", strict=False)
         .join(st.lazy(), on="station_code", how="left")
         .join(tr.lazy(), on="train_no", how="left")
         .with_columns(pl.col("station_id").fill_null(1), pl.col("train_id").fill_null(1),
                       pl.col("pair_id").fill_null(0),
                       ((pl.col("date") - DAY0).dt.total_days()).cast(pl.Int64).alias("day"))
         .with_columns((pl.col("train_id").cast(pl.Int64) * 10_000 + pl.col("day")).alias("jk"))
         .collect())
    # per-journey type from the row itself (types change over the years), fall back to static
    c = c.join(ty.rename({"type_id": "row_type_id"}), on="train_type", how="left").with_columns(
        pl.coalesce("row_type_id", "type_id").fill_null(1).cast(pl.Int16).alias("type_id")).drop("row_type_id")
    return c, st


def main_month(month: str, seed: int = 0) -> None:
    t0 = time.time()
    out_dir = TENSORS / f"ym={month}"
    out_dir.mkdir(parents=True, exist_ok=True)
    c, st = load(month)
    y, mo = map(int, month.split("-"))
    in_month = (pl.col("date").dt.year() == y) & (pl.col("date").dt.month() == mo)

    # ---------- target journeys ---------------------------------------
    jt = (c.filter(in_month).group_by("jk").agg(
        pl.col("train_type").first(), pl.col("target").count().alias("n_obs")))
    keep = jt.filter((pl.col("n_obs") > 0) & (
        ~pl.col("train_type").is_in(list(SUBURBAN_TYPES)) | (pl.col("jk").hash(seed) % SUBURBAN_KEEP == 0)))
    tgt = c.join(keep.select("jk"), on="jk", how="semi").sort("jk", "stop_idx")
    if tgt.is_empty():
        print(f"[{month}] no target journeys", flush=True)
        return
    jindex = (tgt.group_by("jk", maintain_order=True)
              .agg(pl.col("train_id").first(), pl.col("type_id").first(), pl.col("pair_id").first(),
                   pl.col("date").first(), pl.len().alias("n_stops").cast(pl.Int16),
                   pl.col("station_id").first().alias("origin_sid"),
                   pl.col("sch_dep_m").first().alias("origin_sch_dep"))
              .with_row_index("j_local"))
    tgt = tgt.join(jindex.select("jk", "j_local"), on="jk")

    # history (train x station) and station congestion (as of D-2)
    hist = (pl.scan_parquet(str(HISTORY / f"ym={month}" / "*.parquet"))
            .select("train_no", "date", "seq", "station_code", "h_mean", "h_med", "h_std", "h_q90", "h_cnt",
                    "h_last", "h_age_days").collect())
    tgt = tgt.join(hist, on=["train_no", "date", "seq", "station_code"], how="left")
    sd = pl.read_parquet(STATIC / "station_daily.parquet").with_columns(
        ((pl.col("day") - DAY0).dt.total_days() + 2).cast(pl.Int64).alias("usable_day")).drop("day").sort("usable_day")
    tgt = tgt.sort("day").join_asof(sd, left_on="day", right_on="usable_day", by="station_id",
                                    strategy="backward", check_sortedness=False).drop("usable_day")
    tgt = tgt.sort("j_local", "stop_idx").with_columns(
        bearing(pl.col("lat").shift(1).over("j_local"), pl.col("lng").shift(1).over("j_local"),
                pl.col("lat"), pl.col("lng")).alias("brg_in"),
        bearing(pl.col("lat"), pl.col("lng"), pl.col("lat").shift(-1).over("j_local"),
                pl.col("lng").shift(-1).over("j_local")).alias("brg_out"),
    ).with_columns(pl.coalesce("brg_in", "brg_out").fill_null(0.0).alias("brg"))

    stops = tgt.select(
        "j_local", "stop_idx", "station_id", pl.col("halts_here").cast(pl.Int8), pl.col("distance_km").cast(pl.Float32),
        "sch_arr_m", "sch_dep_m", "obs_arr", "obs_dep", "delay_arr", "delay_dep",
        pl.col("avail_arr_m"), pl.col("avail_dep_m"), "target", "avail_m",
        pl.col("lat").cast(pl.Float32), pl.col("lng").cast(pl.Float32),
        "h_mean", "h_med", "h_std", "h_q90", "h_cnt", "h_last", "h_age_days",
        "st_gain_7d", "st_delay_7d", "st_traffic_7d", "st_gain_28d", "st_delay_28d", "st_traffic_28d")
    stops.write_parquet(out_dir / "stops.parquet", compression="zstd")
    jindex.drop("jk").write_parquet(out_dir / "journeys.parquet")

    # ---------- samples: K random cutoffs per journey ------------------
    span = tgt.group_by("j_local").agg(
        (pl.col("sch_dep_m").first() - 90).alias("t_lo"),
        (pl.col("avail_m").filter(pl.col("target").is_not_null()).max() - 1).alias("t_hi"))
    rng = np.random.default_rng(abs(hash((month, seed))) % 2**32)
    sp = span.sort("j_local")
    lo = sp["t_lo"].to_numpy().astype(np.int64)
    hi = np.maximum(sp["t_hi"].to_numpy().astype(np.int64), lo + 1)
    reps = np.repeat(np.arange(sp.height), K_CUTOFFS)
    tt = lo[reps] + (rng.random(reps.size) * (hi[reps] - lo[reps] + 1)).astype(np.int64)
    smp = pl.DataFrame({"j_local": sp["j_local"].to_numpy()[reps], "t": tt.astype(np.int32)}).with_row_index("sample_id")

    # observations of every journey, for "what was known at t"
    obs = pl.concat([
        c.filter("obs_arr").select("jk", pl.col("avail_arr_m").alias("avail"), pl.col("delay_arr").alias("delay"),
                                   "station_id", "stop_idx"),
        c.filter("obs_dep").select("jk", pl.col("avail_dep_m").alias("avail"), pl.col("delay_dep").alias("delay"),
                                   "station_id", "stop_idx"),
    ]).sort("avail")

    # sample state at t (own journey)
    sj = smp.join(jindex.select("j_local", "jk"), on="j_local")
    own = (sj.sort("t").join_asof(obs.rename({"station_id": "o_sid", "stop_idx": "o_idx"}), left_on="t",
                                  right_on="avail", by="jk", strategy="backward", check_sortedness=False))
    # last_known_idx must be the furthest stop known, not the latest-timed observation
    known_idx = (sj.join(tgt.select("j_local", "stop_idx", "obs_arr", "obs_dep", "avail_arr_m", "avail_dep_m"),
                         on="j_local")
                 .filter((pl.col("obs_arr") & (pl.col("avail_arr_m") <= pl.col("t")))
                         | (pl.col("obs_dep") & (pl.col("avail_dep_m") <= pl.col("t"))))
                 .group_by("sample_id").agg(pl.col("stop_idx").max().alias("last_known_idx")))
    ntg = (sj.join(tgt.select("j_local", "target", "avail_m"), on="j_local")
           .filter(pl.col("target").is_not_null() & (pl.col("avail_m") > pl.col("t")))
           .group_by("sample_id").agg(pl.len().alias("n_targets")))
    smp = (own.select("sample_id", "j_local", "jk", "t", pl.col("delay").alias("cur_delay"),
                      (pl.col("t") - pl.col("avail")).alias("cur_age"))
           .join(known_idx, on="sample_id", how="left").join(ntg, on="sample_id", how="inner")
           .with_columns(pl.col("cur_delay").is_not_null().cast(pl.Int8).alias("cur_known"),
                         pl.col("cur_delay").fill_null(0.0), pl.col("cur_age").fill_null(0),
                         pl.col("last_known_idx").fill_null(-1).cast(pl.Int16))
           .sort("sample_id"))

    # ---------- context ---------------------------------------------
    ev = c.select("jk", "train_id", "type_id", "station_id", "stop_idx", "n_stops", "sch_arr_m", "sch_dep_m",
                  pl.when("obs_arr").then("avail_arr_m").alias("ev_avail_arr"),
                  pl.when("obs_dep").then("avail_dep_m").alias("ev_avail_dep"))
    dst = (c.filter(pl.col("stop_idx") == pl.col("n_stops") - 1).select("jk", pl.col("station_id").alias("dst_sid")))
    org = (c.filter(pl.col("stop_idx") == 0).select("jk", pl.col("station_id").alias("org_sid")))
    ev = ev.join(dst, on="jk", how="left").join(org, on="jk", how="left").with_columns(
        (pl.col("sch_arr_m") // BUCKET).alias("bucket"))
    ev_last = ev.filter(pl.col("stop_idx") == pl.col("n_stops") - 1)
    stc = st.select("station_id", "lat", "lng")

    up = (smp.select("sample_id", "j_local", "jk", "t", "cur_delay", "last_known_idx")
          .join(tgt.filter(pl.col("halts_here") == 1).select("j_local", "stop_idx", "station_id", "sch_arr_m", "brg"),
                on="j_local")
          .filter(pl.col("stop_idx") > pl.col("last_known_idx"))
          .sort("sample_id", "stop_idx")
          .with_columns(pl.int_range(pl.len()).over("sample_id").alias("slot"))
          .filter(pl.col("slot") < N_SLOTS)
          .with_columns((pl.col("sch_arr_m") + pl.col("cur_delay")).alias("our_exp"))
          .rename({"sch_arr_m": "our_sch"}))

    ctx_parts = []
    ids = smp["sample_id"].to_numpy()
    for a in range(0, ids.size, CHUNK):
        lo_id, hi_id = int(ids[a]), int(ids[min(a + CHUNK, ids.size) - 1])
        u = up.filter(pl.col("sample_id").is_between(lo_id, hi_id))
        ub = u.with_columns(pl.int_ranges((pl.col("t") - WIN_BEFORE) // BUCKET,
                                          (pl.col("t") + WIN_AFTER) // BUCKET + 1).alias("bucket")).explode("bucket")
        cand = (ub.join(ev, on=["station_id", "bucket"], suffix="_c")
                .filter((pl.col("jk_c") != pl.col("jk"))
                        & pl.col("sch_arr_m").is_between(pl.col("t") - WIN_BEFORE, pl.col("t") + WIN_AFTER)
                        & ~(pl.col("ev_avail_dep").is_not_null() & (pl.col("ev_avail_dep") < pl.col("t") - DEPARTED_GRACE))))

        # rake-link pair: its arrival at our origin, scheduled within 36h before our departure (+1h)
        r = (smp.filter(pl.col("sample_id").is_between(lo_id, hi_id) & (pl.col("last_known_idx") <= 0))
             .join(jindex.select("j_local", "pair_id", "origin_sid", "origin_sch_dep"), on="j_local")
             .filter(pl.col("pair_id") > 0)
             .join(ev_last, left_on=["origin_sid", "pair_id"], right_on=["station_id", "train_id"], suffix="_c")
             .filter(pl.col("sch_arr_m").is_between(pl.col("origin_sch_dep") - 36 * 60, pl.col("origin_sch_dep") + 60))
             .sort("sch_arr_m").group_by("sample_id").last()
             .with_columns(pl.lit(RAKE_SLOT).cast(pl.Int64).alias("slot"),
                           pl.col("origin_sid").alias("station_id"), pl.col("pair_id").alias("train_id"),
                           pl.col("origin_sch_dep").alias("our_sch"),
                           (pl.col("origin_sch_dep") + 0.0).alias("our_exp"),
                           pl.lit(0.0).alias("brg")))
        cols = ["sample_id", "slot", "station_id", "t", "our_sch", "our_exp", "brg", "jk_c", "train_id", "type_id",
                "stop_idx_c", "sch_arr_m", "sch_dep_m", "ev_avail_arr", "ev_avail_dep", "dst_sid", "org_sid"]
        cand = cand.rename({"train_id": "train_id_c_"}, strict=False)
        cand = cand.with_columns(pl.col("train_id_c").alias("train_id") if "train_id_c" in cand.columns else pl.col("train_id_c_").alias("train_id"),
                                 pl.col("type_id_c").alias("type_id") if "type_id_c" in cand.columns else pl.col("type_id"))
        r = r.rename({"stop_idx": "stop_idx_c"}) if "stop_idx_c" not in r.columns else r
        cand = pl.concat([cand.select(cols), r.with_columns(pl.col("type_id_c").alias("type_id") if "type_id_c" in r.columns else pl.col("type_id")).select(cols)],
                         how="vertical_relaxed")

        cand = (cand.sort("t").join_asof(obs.rename({"jk": "jk_c", "station_id": "o_sid", "stop_idx": "o_idx"}),
                                         left_on="t", right_on="avail", by="jk_c", strategy="backward",
                                         check_sortedness=False))
        cand = (cand
                .with_columns(pl.col("delay").is_not_null().cast(pl.Int8).alias("known"),
                              pl.col("delay").fill_null(0.0).alias("their_cur"),
                              pl.coalesce("o_sid", "org_sid").alias("cur_sid"))
                .with_columns((pl.col("sch_arr_m") + pl.col("their_cur")).alias("their_exp"),
                              ((pl.col("o_idx").fill_null(-1) >= pl.col("stop_idx_c"))
                               | (pl.col("ev_avail_arr").is_not_null() & (pl.col("ev_avail_arr") <= pl.col("t")))).alias("arrived"),
                              ((pl.col("o_idx").fill_null(-1) > pl.col("stop_idx_c"))
                               | (pl.col("ev_avail_dep").is_not_null() & (pl.col("ev_avail_dep") <= pl.col("t")))).alias("departed"))
                .join(stc.rename({"station_id": "station_id", "lat": "s_lat", "lng": "s_lng"}), on="station_id", how="left")
                .join(stc.rename({"station_id": "cur_sid", "lat": "c_lat", "lng": "c_lng"}), on="cur_sid", how="left")
                .join(stc.rename({"station_id": "dst_sid", "lat": "d_lat", "lng": "d_lng"}), on="dst_sid", how="left")
                .with_columns(haversine_km(pl.col("c_lat"), pl.col("c_lng"), pl.col("s_lat"), pl.col("s_lng")).alias("dist_cur"))
                .with_columns(pl.when(pl.col("arrived") | (pl.col("dist_cur") < 0.5))
                              .then(bearing(pl.col("s_lat"), pl.col("s_lng"), pl.col("d_lat"), pl.col("d_lng")))
                              .otherwise(bearing(pl.col("c_lat"), pl.col("c_lng"), pl.col("s_lat"), pl.col("s_lng")))
                              .alias("their_brg"))
                .with_columns((pl.col("their_exp") - pl.col("our_exp")).alias("headway"),
                              (pl.col("their_exp") - pl.col("t")).alias("rel_t"),
                              (pl.col("our_exp") - pl.col("t")).alias("our_rel"),
                              (pl.col("t") - pl.col("avail")).fill_null(0).alias("age"),
                              (pl.col("sch_dep_m") - pl.col("sch_arr_m")).alias("sch_halt"),
                              (pl.col("their_brg") - pl.col("brg")).cos().fill_nan(0.0).fill_null(0.0).alias("dir_cos"))
                .with_columns(pl.col("headway").abs().alias("_rank"))
                .sort("sample_id", "slot", "_rank")
                .with_columns(pl.int_range(pl.len()).over("sample_id", "slot").alias("_r"))
                .filter(pl.col("_r") < CTX_PER_SLOT)
                .select("sample_id", pl.col("slot").cast(pl.Int8), pl.col("station_id").cast(pl.Int32),
                        pl.col("train_id").cast(pl.Int32), pl.col("type_id").fill_null(1).cast(pl.Int16),
                        "known", pl.col("arrived").cast(pl.Int8), pl.col("departed").cast(pl.Int8),
                        *[pl.col(x).cast(pl.Float32) for x in
                          ("headway", "rel_t", "our_rel", "their_cur", "age", "dist_cur", "dir_cos", "sch_halt")]))
        ctx_parts.append(cand)
    ctx = pl.concat(ctx_parts).sort("sample_id", "slot")
    smp.drop("jk").write_parquet(out_dir / "samples.parquet")
    ctx.write_parquet(out_dir / "context.parquet", compression="zstd")
    print(f"[{month}] journeys={jindex.height:,} stops={stops.height:,} samples={smp.height:,} "
          f"ctx={ctx.height:,} ({ctx.height / max(smp.height, 1):.1f}/sample) {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("months", nargs="*", help="YYYY-MM ... (default: all)")
    p.add_argument("--skip-done", action="store_true")
    a = p.parse_args()
    months = a.months or all_months()
    for m in months:
        if a.skip_done and (TENSORS / f"ym={m}" / "context.parquet").exists():
            continue
        main_month(m)


if __name__ == "__main__":
    main()

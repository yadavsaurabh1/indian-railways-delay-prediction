"""Stop-to-stop traffic: how other trains are doing on each of our next stretches, as of t.

For every sample (journey, cutoff t) and each of its next K_SEG upcoming halts B_k, the stretch
is A_k -> B_k with A_k = the previous halt on our route.

Stretch rows (one per sample x k) - data/tensors/ym=*/segments.parquet
  our_run          our scheduled minutes from leaving A to reaching B
  our_usual_gain   our own typical delay change on this stretch (90-day history, h_med(B) - h_med(A))
  n_sched_before/after   other trains scheduled to leave A for B in [t-3h, t) / [t, t+3h)
  reg_delay, reg_n       mean delay / count of all reports in B's 1x1 degree cell, last 3 full hours

Token rows (one per other train on the stretch) - data/tensors/ym=*/segtok.parquet
  kind 1 done      same direction, reached B in the last 3 h          (N_DONE most recent)
       2 progress  same direction, left A in the last 3 h, not at B yet (N_PROG most recent)
       3 opposite  ran B -> A, reached A in the last 3 h               (N_OPP most recent)
  train_id, type_id   which train and its class (the model learns per-train / per-class behaviour)
  gain                delay it gained on the stretch (so far, for kind 2)
  age                 minutes since it reached the end (kind 1, 3) / left A (kind 2)
  their_run           its scheduled minutes for the stretch (vs our_run: faster or slower train)
  their_delay_in      how late it was entering the stretch
  usual_gain          its own typical delay change on the stretch (its 90-day history)
  halt_a, halt_b      whether it stops at either end

Only reports available by t (+2 min lag, common.AVAIL_LAG_MIN) are used. Same source as
live_status, so the live predictor rebuilds all of this from the trains on the upcoming boards.
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from build_month import load
from common import HISTORY, TENSORS, all_months, neighbours

K_SEG = 20
WIN_PAST, WIN_FUT = 6 * 60, 3 * 60      # candidate's scheduled time at A relative to t
RECENT = 180                            # "last 3 h"
REG_HOURS = 3
N_DONE, N_PROG, N_OPP = 6, 3, 3
N_TOK = K_SEG * (N_DONE + N_PROG + N_OPP)
BUCKET = 120
CHUNK = 5_000
DAY0 = pl.date(2020, 1, 1)

STRETCH_COLS = ["k", "b_idx", "our_run", "our_usual_gain", "n_sched_before", "n_sched_after", "reg_delay", "reg_n"]
TOK_NUM_COLS = ["gain", "age", "their_run", "their_delay_in", "usual_gain", "halt_a", "halt_b"]
TOK_CAT_COLS = ["kind", "k", "b_idx", "train_id", "type_id"]


def _ranked(df: pl.DataFrame, n: int, order: str, kind: int) -> pl.DataFrame:
    return (df.sort(order, descending=True)
            .with_columns(pl.int_range(pl.len()).over("sample_id", "k").alias("_r"))
            .filter(pl.col("_r") < n)
            .with_columns(pl.lit(kind).alias("kind"), pl.col(order).alias("_o")))


def main_month(month: str) -> None:
    t0 = time.time()
    d = TENSORS / f"ym={month}"
    c, st = load(month)
    hist = (pl.scan_parquet([str(HISTORY / f"ym={m}" / "*.parquet") for m in neighbours(month)
                             if (HISTORY / f"ym={m}").exists()])
            .select("train_no", "date", "seq", "station_code", "h_med").collect())
    c = c.join(hist, on=["train_no", "date", "seq", "station_code"], how="left")
    del hist

    jn = pl.read_parquet(d / "journeys.parquet").select(
        "j_local", (pl.col("train_id").cast(pl.Int64) * 10_000
                    + (pl.col("date") - DAY0).dt.total_days().cast(pl.Int64)).alias("jk"))
    smp = pl.read_parquet(d / "samples.parquet").select("sample_id", "j_local", "t", "last_known_idx")
    stops = pl.read_parquet(d / "stops.parquet").select("j_local", "stop_idx", "station_id", "halts_here",
                                                        "sch_arr_m", "sch_dep_m", "h_med")
    halts = (stops.filter(pl.col("halts_here") == 1).sort("j_local", "stop_idx")
             .with_columns(pl.col("station_id").shift(1).over("j_local").alias("A"),
                           pl.col("sch_dep_m").shift(1).over("j_local").alias("our_dep_a"),
                           pl.col("h_med").shift(1).over("j_local").alias("our_h_a")))
    segs = (smp.join(jn, on="j_local").join(halts, on="j_local")
            .filter(pl.col("stop_idx") > pl.col("last_known_idx"))
            .sort("sample_id", "stop_idx")
            .with_columns(pl.int_range(pl.len()).over("sample_id").alias("k"))
            .filter((pl.col("k") < K_SEG) & pl.col("A").is_not_null())
            .select("sample_id", "jk", "t", "k", pl.col("stop_idx").alias("b_idx"), "A",
                    pl.col("station_id").alias("B"),
                    (pl.col("sch_arr_m") - pl.col("our_dep_a")).alias("our_run"),
                    (pl.col("h_med") - pl.col("our_h_a")).alias("our_usual_gain")))

    # every row of every journey: delay leaving (dep preferred) and arriving (arr preferred)
    lv_d = pl.when("obs_dep").then("delay_dep").when("obs_arr").then("delay_arr")
    lv_a = pl.when("obs_dep").then("avail_dep_m").when("obs_arr").then("avail_arr_m")
    ar_d = pl.when("obs_arr").then("delay_arr").when("obs_dep").then("delay_dep")
    ar_a = pl.when("obs_arr").then("avail_arr_m").when("obs_dep").then("avail_dep_m")
    R = c.select("jk", "train_id", "type_id", "station_id", "stop_idx", pl.col("halts_here").cast(pl.Int8).alias("halt"),
                 "sch_arr_m", "sch_dep_m", "h_med", lv_d.alias("lv_d"), lv_a.alias("lv_a"),
                 ar_d.alias("ar_d"), ar_a.alias("ar_a"))
    RA = R.with_columns((pl.col("sch_dep_m") // BUCKET).alias("bucket")).rename(
        {"jk": "jk_o", "train_id": "train_id_o", "type_id": "type_id_o", "station_id": "A", "stop_idx": "ia",
         "halt": "A_halt", "sch_arr_m": "A_sarr", "sch_dep_m": "A_sdep", "h_med": "A_h",
         "lv_d": "A_lv_d", "lv_a": "A_lv_a", "ar_d": "A_ar_d", "ar_a": "A_ar_a"})
    RB = R.drop("train_id", "type_id").rename(
        {"jk": "jk_o", "station_id": "B", "stop_idx": "ib", "halt": "B_halt", "sch_arr_m": "B_sarr",
         "sch_dep_m": "B_sdep", "h_med": "B_h", "lv_d": "B_lv_d", "lv_a": "B_lv_a", "ar_d": "B_ar_d", "ar_a": "B_ar_a"})
    obs = pl.concat([
        c.filter("obs_arr").select(pl.col("jk").alias("jk_o"), pl.col("avail_arr_m").alias("avail"),
                                   pl.col("delay_arr").alias("cur_d"), pl.col("stop_idx").alias("cur_idx")),
        c.filter("obs_dep").select(pl.col("jk").alias("jk_o"), pl.col("avail_dep_m").alias("avail"),
                                   pl.col("delay_dep").alias("cur_d"), pl.col("stop_idx").alias("cur_idx")),
    ]).sort("avail")
    # regional state: per (1 degree cell, clock hour) sum/count of reports that became available in that hour
    cell = st.select("station_id", pl.col("lat").floor().alias("clat"), pl.col("lng").floor().alias("clng"))
    reg = (pl.concat([
        c.filter("obs_arr").select("station_id", pl.col("avail_arr_m").alias("avail"), pl.col("delay_arr").alias("d")),
        c.filter("obs_dep").select("station_id", pl.col("avail_dep_m").alias("avail"), pl.col("delay_dep").alias("d")),
    ]).join(cell, on="station_id").with_columns((pl.col("avail") // 60).alias("hb"))
        .group_by("clat", "clng", "hb").agg(pl.col("d").sum().alias("dsum"), pl.len().alias("n")))
    seg_cell = segs.select("sample_id", "k", "t", "B").join(cell, left_on="B", right_on="station_id", how="left")
    del c

    s_parts, t_parts = [], []
    ids = segs["sample_id"].unique().sort().to_numpy()
    t = pl.col("t")
    for a in range(0, ids.size, CHUNK):
        lo_id, hi_id = int(ids[a]), int(ids[min(a + CHUNK, ids.size) - 1])
        s = segs.filter(pl.col("sample_id").is_between(lo_id, hi_id))
        sb = s.select("sample_id", "jk", "t", "k", "b_idx", "A", "B").with_columns(
            pl.int_ranges((t - WIN_PAST) // BUCKET, (t + WIN_FUT) // BUCKET + 1).alias("bucket")).explode("bucket")
        e = (sb.join(RA, on=["A", "bucket"])
             .filter((pl.col("jk_o") != pl.col("jk")) & pl.col("A_sdep").is_between(t - WIN_PAST, t + WIN_FUT))
             .join(RB, on=["jk_o", "B"]))
        same = e.filter(pl.col("ib") > pl.col("ia")).with_columns(
            (pl.col("B_sarr") - pl.col("A_sdep")).alias("their_run"), (pl.col("B_h") - pl.col("A_h")).alias("usual_gain"),
            pl.col("A_lv_d").alias("their_delay_in"))
        opp = e.filter(pl.col("ib") < pl.col("ia")).with_columns(
            (pl.col("A_sarr") - pl.col("B_sdep")).alias("their_run"), (pl.col("A_h") - pl.col("B_h")).alias("usual_gain"),
            pl.col("B_lv_d").alias("their_delay_in"))

        load_ = same.group_by("sample_id", "k").agg(
            pl.col("A_sdep").is_between(t - RECENT, t, closed="left").sum().alias("n_sched_before"),
            pl.col("A_sdep").is_between(t, t + RECENT, closed="left").sum().alias("n_sched_after"))
        done = (same.filter(pl.col("A_lv_a").is_not_null() & pl.col("B_ar_a").is_not_null()
                            & (pl.col("A_lv_a") <= t) & (pl.col("B_ar_a") <= t) & (pl.col("B_ar_a") >= t - RECENT))
                .with_columns((pl.col("B_ar_d") - pl.col("A_lv_d")).alias("gain"), (t - pl.col("B_ar_a")).alias("age")))
        prog = (same.filter(pl.col("A_lv_a").is_not_null() & (pl.col("A_lv_a") <= t) & (pl.col("A_lv_a") >= t - RECENT)
                            & (pl.col("B_ar_a").is_null() | (pl.col("B_ar_a") > t)))
                .sort("t").join_asof(obs, left_on="t", right_on="avail", by="jk_o", strategy="backward",
                                     check_sortedness=False)
                .filter(pl.col("cur_idx") < pl.col("ib"))   # reported beyond B already -> not on the stretch
                .with_columns((pl.col("cur_d") - pl.col("A_lv_d")).alias("gain"), (t - pl.col("A_lv_a")).alias("age")))
        oppd = (opp.filter(pl.col("B_lv_a").is_not_null() & pl.col("A_ar_a").is_not_null()
                           & (pl.col("B_lv_a") <= t) & (pl.col("A_ar_a") <= t) & (pl.col("A_ar_a") >= t - RECENT))
                .with_columns((pl.col("A_ar_d") - pl.col("B_lv_d")).alias("gain"), (t - pl.col("A_ar_a")).alias("age")))
        tok_cols = ["sample_id", "kind", "k", "b_idx", "_o", pl.col("train_id_o").alias("train_id"),
                    pl.col("type_id_o").alias("type_id"), "gain", "age", "their_run", "their_delay_in", "usual_gain",
                    pl.col("A_halt").alias("halt_a"), pl.col("B_halt").alias("halt_b")]
        toks = pl.concat([_ranked(done.with_columns(o=pl.col("B_ar_a")), N_DONE, "o", 1).select(tok_cols),
                          _ranked(prog.with_columns(o=pl.col("A_lv_a")), N_PROG, "o", 2).select(tok_cols),
                          _ranked(oppd.with_columns(o=pl.col("A_ar_a")), N_OPP, "o", 3).select(tok_cols)],
                         how="vertical_relaxed")
        t_parts.append(toks)

        regk = (seg_cell.filter(pl.col("sample_id").is_between(lo_id, hi_id))
                .with_columns(pl.int_ranges(t // 60 - REG_HOURS, t // 60).alias("hb")).explode("hb")
                .join(reg, on=["clat", "clng", "hb"])
                .group_by("sample_id", "k").agg((pl.col("dsum").sum() / pl.col("n").sum()).alias("reg_delay"),
                                                pl.col("n").sum().alias("reg_n")))
        s_parts.append(s.select("sample_id", "k", "b_idx", "our_run", "our_usual_gain")
                       .join(load_, on=["sample_id", "k"], how="left")
                       .join(regk, on=["sample_id", "k"], how="left")
                       .with_columns(pl.col("n_sched_before", "n_sched_after").fill_null(0)))

    S = (pl.concat(s_parts).sort("sample_id", "k")
         .select("sample_id", *[pl.col(x).cast(pl.Float32) for x in STRETCH_COLS]))
    T = (pl.concat(t_parts).sort(["sample_id", "k", "kind", "_o"], descending=[False, False, False, True])
         .select("sample_id", *[pl.col(x).cast(pl.Int32) for x in TOK_CAT_COLS],
                 *[pl.col(x).cast(pl.Float32) for x in TOK_NUM_COLS]))
    S.write_parquet(d / "segments.parquet", compression="zstd")
    T.write_parquet(d / "segtok.parquet", compression="zstd")
    per = T.group_by("kind").len().sort("kind")
    kinds = dict(zip(per["kind"].to_list(), per["len"].to_list()))
    print(f"[{month}] stretches={S.height:,} ({S.height / max(smp.height, 1):.1f}/sample) other-train tokens={T.height:,} "
          f"({T.height / max(smp.height, 1):.1f}/sample: done {kinds.get(1, 0):,} on-stretch {kinds.get(2, 0):,} "
          f"opposite {kinds.get(3, 0):,}) {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("months", nargs="*", help="YYYY-MM ... (default: all)")
    p.add_argument("--skip-done", action="store_true")
    a = p.parse_args()
    for m in a.months or all_months():
        if not (TENSORS / f"ym={m}" / "samples.parquet").exists():
            continue
        if a.skip_done and (TENSORS / f"ym={m}" / "segtok.parquet").exists():
            continue
        main_month(m)


if __name__ == "__main__":
    main()

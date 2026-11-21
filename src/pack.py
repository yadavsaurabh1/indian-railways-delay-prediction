"""Pack the per-month parquet tensors into flat on-disk arrays for fast training.

data/packed/
  stops.npy     float32 [N_stops, len(STOP_COLS)]   rows of a journey are contiguous
  journeys.npz  stop_off, n_stops, train_id, type_id, pair_id, day, origin_sch_dep, hol_*
  samples.npz   j, t, cur_delay, cur_known, cur_age, last_known_idx, ctx_off, ctx_n, split
  context.npy   float32 [N_ctx, len(CTX_COLS)]      rows of a sample are contiguous
  meta.json     column names, vocab sizes, split boundaries

Weather (if data/static/weather_daily.parquet exists) is attached per stop for the
station's 0.5° cell on the scheduled arrival day.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from common import DATA, STATIC, TENSORS, TRAIN_END, VAL_END

PACKED = DATA / "packed"
WEATHER_COLS = ["t2m", "t2m_min", "dew_dep", "rh2m", "precip", "ws2m"]
STOP_COLS = [
    "stop_idx", "station_id", "halts_here", "distance_km", "sch_arr_m", "sch_dep_m",
    "obs_arr", "obs_dep", "delay_arr", "delay_dep", "avail_arr_m", "avail_dep_m", "target", "avail_m",
    "lat", "lng", "h_mean", "h_med", "h_std", "h_q90", "h_cnt", "h_last", "h_age_days",
    "st_gain_7d", "st_delay_7d", "st_traffic_7d", "st_gain_28d", "st_delay_28d", "st_traffic_28d",
    *WEATHER_COLS,
]
CTX_COLS = ["slot", "station_id", "train_id", "type_id", "known", "arrived", "departed",
            "headway", "rel_t", "our_rel", "their_cur", "age", "dist_cur", "dir_cos", "sch_halt"]
DAY0 = date(2020, 1, 1)


def holiday_features(days: np.ndarray) -> dict[str, np.ndarray]:
    import holidays

    hs = holidays.country_holidays("IN", years=range(2019, 2028))
    hol = np.array(sorted((d - DAY0).days for d in hs.keys()))
    nxt = np.searchsorted(hol, days, side="left")
    prv = np.searchsorted(hol, days, side="right") - 1
    to_next = np.where(nxt < hol.size, hol[np.minimum(nxt, hol.size - 1)] - days, 60)
    since_prev = np.where(prv >= 0, days - hol[np.maximum(prv, 0)], 60)
    return {"hol_is": (to_next == 0).astype(np.float32),
            "hol_to_next": np.minimum(to_next, 60).astype(np.float32),
            "hol_since_prev": np.minimum(since_prev, 60).astype(np.float32)}


def _pack_rows(out: Path, months: list[Path], fname: str, arrays: dict[str, tuple[list[str], type]],
               sort_cols: list[str], index_name: str) -> int:
    """Stack one per-sample table from every month into memmapped arrays, rows of a sample contiguous and
    in the main pack's sample order (months sorted, sample_id sorted). Writes <index_name>.npz (off, n)."""
    import pyarrow.parquet as pq
    t0 = time.time()
    n = sum(pq.ParquetFile(m / fname).metadata.num_rows for m in months)
    print(f"[pack] {fname}: {len(months)} months, {n:,} rows", flush=True)
    mm = {name: np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=dt, shape=(n, len(cols)))
          for name, (cols, dt) in arrays.items()}
    offs, ns, off = [], [], 0
    for m in months:
        sm = pl.read_parquet(m / "samples.parquet", columns=["sample_id"]).sort("sample_id")
        tb = pl.read_parquet(m / fname).sort(sort_cols)
        cnt = (sm.join(tb.group_by("sample_id").len(), on="sample_id", how="left")
               .with_columns(pl.col("len").fill_null(0)).sort("sample_id")["len"].to_numpy().astype(np.int64))
        assert int(cnt.sum()) == tb.height, (m.name, fname, cnt.sum(), tb.height)  # every row belongs to a sample
        for name, (cols, dt) in arrays.items():
            mm[name][off:off + tb.height] = tb.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy().astype(dt)
        offs.append(off + np.concatenate([[0], np.cumsum(cnt)[:-1]]))
        ns.append(cnt.astype(np.int32))
        off += tb.height
        print(f"[pack] {m.name} {fname} done ({time.time() - t0:.0f}s)", flush=True)
    for a in mm.values():
        a.flush()
    o, c = np.concatenate(offs), np.concatenate(ns)
    n_samples = len(np.load(out / "samples.npz")["t"])
    assert len(c) == n_samples, (fname, len(c), n_samples)
    np.savez(out / f"{index_name}.npz", off=o, n=c)
    return off


def pack_segments(out: Path, only_months=None) -> None:
    """Stop-to-stop traffic (build_segments.py) next to an existing pack:
       segments.npy    float32 [N_stretch, STRETCH_COLS]          + seg_index.npz (off, n)
       segtok_num.npy  float32 [N_tok, TOK_NUM_COLS]
       segtok_cat.npy  int32   [N_tok, TOK_CAT_COLS]               + segtok_index.npz (off, n)"""
    from build_segments import STRETCH_COLS, TOK_CAT_COLS, TOK_NUM_COLS
    t0 = time.time()
    months = sorted(p for p in TENSORS.glob("ym=*") if (p / "context.parquet").exists()
                    and (not only_months or p.name.split("=")[1] in only_months))
    missing = [m.name for m in months if not (m / "segtok.parquet").exists()]
    if missing:
        raise SystemExit(f"segtok.parquet missing for {len(missing)} months, e.g. {missing[:5]} - run build_segments.py")
    n_s = _pack_rows(out, months, "segments.parquet", {"segments": (STRETCH_COLS, np.float32)},
                     ["sample_id", "k"], "seg_index")
    n_t = _pack_rows(out, months, "segtok.parquet",
                     {"segtok_num": (TOK_NUM_COLS, np.float32), "segtok_cat": (TOK_CAT_COLS, np.int32)},
                     ["sample_id", "k", "kind"], "segtok_index")
    meta = json.loads((out / "meta.json").read_text())
    meta |= {"stretch_cols": STRETCH_COLS, "tok_num_cols": TOK_NUM_COLS, "tok_cat_cols": TOK_CAT_COLS,
             "n_stretch": int(n_s), "n_tok": int(n_t)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[pack] stop-to-stop arrays done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", help="subset of YYYY-MM (default: all)")
    ap.add_argument("--out", default=str(PACKED))
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--segments-only", action="store_true",
                    help="only add segments.npy + seg_index.npz (stop-to-stop traffic) to an existing packed dir")
    a = ap.parse_args()
    out = Path(a.out)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    if a.segments_only:
        pack_segments(out, a.months)
        return
    months = sorted(p for p in TENSORS.glob("ym=*") if (p / "context.parquet").exists()
                    and (not a.months or p.name.split("=")[1] in a.months))
    import pyarrow.parquet as pq
    n_stops = sum(pq.ParquetFile(m / "stops.parquet").metadata.num_rows for m in months)
    n_ctx = sum(pq.ParquetFile(m / "context.parquet").metadata.num_rows for m in months)
    print(f"[pack] {len(months)} months, {n_stops:,} stops, {n_ctx:,} context rows", flush=True)

    weather = None
    if not a.no_weather and (STATIC / "weather_daily.parquet").exists():
        cells = pl.read_parquet(STATIC / "weather_cells.parquet")
        weather = (pl.read_parquet(STATIC / "weather_daily.parquet")
                   .with_columns((pl.col("t2m_min") - pl.col("t2mdew")).alias("dew_dep"),
                                 pl.col("prectotcorr").alias("precip"))
                   .select("clat", "clng", "day", *WEATHER_COLS))
        print("[pack] weather attached", flush=True)

    S = np.lib.format.open_memmap(out / "stops.npy", mode="w+", dtype=np.float32, shape=(n_stops, len(STOP_COLS)))
    C = np.lib.format.open_memmap(out / "context.npy", mode="w+", dtype=np.float32, shape=(n_ctx, len(CTX_COLS)))
    J, SM = [], []
    s_off = c_off = j_off = 0
    for m in months:
        st = pl.read_parquet(m / "stops.parquet")
        if weather is not None:
            st = (st.join(cells, on="station_id", how="left")
                  .with_columns((pl.lit(np.datetime64("2020-01-01")).cast(pl.Date)
                                 + pl.duration(days=pl.col("sch_arr_m") // 1440)).alias("day"))
                  .join(weather, on=["clat", "clng", "day"], how="left"))
        else:
            st = st.with_columns([pl.lit(None, pl.Float32).alias(c) for c in WEATHER_COLS])
        st = st.select([pl.col(c).cast(pl.Float32) for c in STOP_COLS])
        S[s_off:s_off + st.height] = st.to_numpy()

        jn = pl.read_parquet(m / "journeys.parquet").sort("j_local")
        lens = jn["n_stops"].to_numpy().astype(np.int64)
        starts = s_off + np.concatenate([[0], np.cumsum(lens)[:-1]])
        J.append(pl.DataFrame({
            "stop_off": starts, "n_stops": lens.astype(np.int32),
            "train_id": jn["train_id"].to_numpy(), "type_id": jn["type_id"].to_numpy().astype(np.int32),
            "pair_id": jn["pair_id"].to_numpy(),
            "day": (jn["date"].cast(pl.Int32) - (DAY0 - date(1970, 1, 1)).days).to_numpy().astype(np.int32),
            "origin_sch_dep": jn["origin_sch_dep"].to_numpy()}))

        sm = pl.read_parquet(m / "samples.parquet").sort("sample_id")
        cx = pl.read_parquet(m / "context.parquet").sort("sample_id", "slot")
        counts = (cx.group_by("sample_id").len().rename({"len": "ctx_n"}))
        sm = sm.join(counts, on="sample_id", how="left").with_columns(pl.col("ctx_n").fill_null(0)).sort("sample_id")
        ctx_n = sm["ctx_n"].to_numpy().astype(np.int64)
        # context rows are sorted by sample_id, so offsets follow sample order
        ctx_start = c_off + np.concatenate([[0], np.cumsum(ctx_n)[:-1]])
        SM.append(pl.DataFrame({
            "j": (sm["j_local"].to_numpy().astype(np.int64) + j_off).astype(np.int32),
            "t": sm["t"].to_numpy(), "cur_delay": sm["cur_delay"].to_numpy().astype(np.float32),
            "cur_known": sm["cur_known"].to_numpy().astype(np.int8), "cur_age": sm["cur_age"].to_numpy().astype(np.float32),
            "last_known_idx": sm["last_known_idx"].to_numpy().astype(np.int16),
            "ctx_off": ctx_start, "ctx_n": ctx_n.astype(np.int16)}))
        cxa = cx.filter(pl.col("sample_id").is_in(sm["sample_id"].implode())).select([pl.col(c).cast(pl.Float32) for c in CTX_COLS])
        assert cxa.height == int(ctx_n.sum()), (m, cxa.height, ctx_n.sum())
        C[c_off:c_off + cxa.height] = cxa.to_numpy()
        s_off += st.height
        c_off += cxa.height
        j_off += jn.height
        print(f"[pack] {m.name} done ({time.time() - t0:.0f}s)", flush=True)
    S.flush()
    C.flush()

    J = pl.concat(J)
    SM = pl.concat(SM)
    day = J["day"].to_numpy()
    hol = holiday_features(day)
    jd = {c: J[c].to_numpy() for c in J.columns} | hol
    np.savez(out / "journeys.npz", **jd)
    split_day = lambda s: (date.fromisoformat(s) - DAY0).days
    sday = day[SM["j"].to_numpy()]
    split = np.where(sday < split_day(TRAIN_END), 0, np.where(sday < split_day(VAL_END), 1, 2)).astype(np.int8)
    np.savez(out / "samples.npz", **{c: SM[c].to_numpy() for c in SM.columns}, split=split)

    stations = pl.read_parquet(STATIC / "stations.parquet")
    trains = pl.read_parquet(STATIC / "trains.parquet")
    types = pl.read_parquet(STATIC / "types.parquet")
    meta = {"stop_cols": STOP_COLS, "ctx_cols": CTX_COLS, "n_stops": n_stops, "n_ctx": c_off,
            "n_journeys": int(J.height), "n_samples": int(SM.height),
            "n_station_ids": int(stations["station_id"].max()) + 1, "n_train_ids": int(trains["train_id"].max()) + 1,
            "n_type_ids": int(types["type_id"].max()) + 1, "weather": weather is not None,
            "split_counts": {k: int((split == i).sum()) for i, k in enumerate(["train", "val", "test"])},
            "train_end": TRAIN_END, "val_end": VAL_END}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[pack] done in {time.time() - t0:.0f}s: {json.dumps(meta['split_counts'])}", flush=True)


if __name__ == "__main__":
    main()

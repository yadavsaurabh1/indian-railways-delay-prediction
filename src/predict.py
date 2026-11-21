"""Live delay prediction for a running train.

  uv run python predict.py 12624 04-11-2026            # train number, journey start date (DD-MM-YYYY)
  uv run python predict.py 12624 04-11-2026 --json

Builds exactly the inputs the model was trained on, from live data:
  * live_status(train)           -> the journey so far (same schema as the dataset)
  * live_station(S) x next 6 halts -> other trains around the route (context slots 0-5)
  * live_status(pair train)      -> the rake-link train, before departure (slot 6)
  * this train's history + station congestion from the local dataset, holidays,
    and the Open-Meteo daily forecast in place of the NASA POWER weather used in training.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import torch

import live
from build_month import CTX_PER_SLOT, DEPARTED_GRACE, N_SLOTS, RAKE_SLOT, WIN_AFTER, WIN_BEFORE
from build_segments import K_SEG, N_DONE, N_OPP, N_PROG, RECENT
from common import EPOCH, ROOT, STAGED, STATIC, clean
from features import KEEP_BEHIND, L_MAX, build_batch
from fetch_weather import cell_of
from model import RouteTransformer
from pack import CTX_COLS, DAY0, STOP_COLS, holiday_features

IST_NOW = lambda: datetime.now(live.IST).replace(tzinfo=None)
to_min = lambda dt: None if dt is None else int((dt - EPOCH).total_seconds() // 60)


def _hav(lat1, lng1, lat2, lng2):
    if None in (lat1, lng1, lat2, lng2):
        return 0.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _brg(lat1, lng1, lat2, lng2):
    if None in (lat1, lng1, lat2, lng2):
        return 0.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    return math.atan2(math.sin(dl) * math.cos(p2), math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))


class Static:
    def __init__(self):
        self.st = pl.read_parquet(STATIC / "stations.parquet")
        self.tr = pl.read_parquet(STATIC / "trains.parquet")
        self.ty = pl.read_parquet(STATIC / "types.parquet")
        self.sd = pl.read_parquet(STATIC / "station_daily.parquet")
        self.sid = dict(zip(self.st["station_code"], self.st["station_id"]))
        self.ll = {r[0]: (r[1], r[2]) for r in self.st.select("station_code", "lat", "lng").iter_rows()}
        self.tid = {r["train_no"]: r for r in self.tr.iter_rows(named=True)}
        self.type_id = dict(zip(self.ty["train_type"], self.ty["type_id"]))


def history(train_no: str, d: date, stations: list[str]) -> pl.DataFrame:
    """Same definition as build_history: 90-day window ending at the last run on or before d-3."""
    f = STAGED / f"{train_no}.parquet"
    cols = ["h_mean", "h_med", "h_std", "h_q90", "h_cnt", "h_last", "h_age_days"]
    if not f.exists():
        return pl.DataFrame({"station_code": stations, **{c: [None] * len(stations) for c in cols}})
    c = clean(pl.scan_parquet(f).with_columns(pl.col("halts_here").fill_null(1))).select(
        "date", "station_code", "target").collect().drop_nulls("target")
    c = c.filter(pl.col("date") <= d - timedelta(days=3))
    rows = []
    for s in stations:
        x = c.filter(pl.col("station_code") == s).sort("date")
        if x.is_empty():
            rows.append({"station_code": s, **{k: None for k in cols}})
            continue
        last = x["date"].max()
        w = x.filter(pl.col("date") > last - timedelta(days=90)).group_by("date").agg(pl.col("target").mean()).sort("date")
        v = w["target"]
        rows.append({"station_code": s, "h_mean": v.mean(), "h_med": v.median(), "h_std": v.std(),
                     "h_q90": v.quantile(0.9), "h_cnt": v.len(), "h_last": v[-1], "h_age_days": (d - last).days})
    return pl.DataFrame(rows, schema={"station_code": pl.String, **{k: pl.Float64 for k in cols}})


def weather_forecast(cells: list[tuple[float, float]], day: date) -> dict[tuple[float, float], dict]:
    """Open-Meteo daily forecast mapped to the training variables (NASA POWER names)."""
    import urllib.request
    out = {}
    for i in range(0, len(cells), 50):
        part = cells[i:i + 50]
        url = ("https://api.open-meteo.com/v1/forecast?latitude=" + ",".join(str(a) for a, _ in part)
               + "&longitude=" + ",".join(str(b) for _, b in part)
               + "&daily=temperature_2m_mean,temperature_2m_min,dew_point_2m_mean,relative_humidity_2m_mean,"
                 "precipitation_sum,wind_speed_10m_mean&wind_speed_unit=ms&timezone=Asia%2FKolkata"
               + f"&start_date={day}&end_date={day + timedelta(days=2)}")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": live.USER_AGENT}), timeout=30) as r:
                data = json.load(r)
        except Exception:
            continue  # weather is optional; the model was trained with a missing-weather flag
        for cell, d in zip(part, data if isinstance(data, list) else [data]):
            dd = d["daily"]
            for k, dstr in enumerate(dd["time"]):
                t2m, tmin, dew = dd["temperature_2m_mean"][k], dd["temperature_2m_min"][k], dd["dew_point_2m_mean"][k]
                ws = dd["wind_speed_10m_mean"][k]
                out[(cell, dstr)] = {"t2m": t2m, "t2m_min": tmin,
                                     "dew_dep": None if tmin is None or dew is None else tmin - dew,
                                     "rh2m": dd["relative_humidity_2m_mean"][k], "precip": dd["precipitation_sum"][k],
                                     "ws2m": None if ws is None else ws * 0.75}  # 10 m -> ~2 m wind
    return out


MAX_STATUS_CALLS = 40   # live_status lookups of other trains per prediction (1 request/s -> ~1 min)


def _journey(train_no: str, start, S) -> dict | None:
    """One other train's journey from live_status: ids, date, and per station its schedule, halt flag and
    the delays it left / arrived with (dep- resp. arr-preferred, as in build_segments)."""
    try:
        raw, _ = live.live_status(train_no, start)
    except Exception:
        return None
    c = clean(raw.lazy().with_columns(pl.col("halts_here").fill_null(1))).collect()
    if c.is_empty():
        return None
    tinfo = S.tid.get(train_no, {})
    rows, latest = {}, None
    for r in c.iter_rows(named=True):
        lv = (r["delay_dep"], r["avail_dep_m"]) if r["obs_dep"] else ((r["delay_arr"], r["avail_arr_m"]) if r["obs_arr"] else (None, None))
        ar = (r["delay_arr"], r["avail_arr_m"]) if r["obs_arr"] else ((r["delay_dep"], r["avail_dep_m"]) if r["obs_dep"] else (None, None))
        rows.setdefault(r["station_code"], {"idx": r["stop_idx"], "halt": float(r["halts_here"] or 0),
                                            "sarr": r["sch_arr_m"], "sdep": r["sch_dep_m"],
                                            "lv_d": lv[0], "lv_a": lv[1], "ar_d": ar[0], "ar_a": ar[1]})
        for d_, a_ in (lv, ar):
            if a_ is not None and (latest is None or a_ > latest[0]):
                latest = (a_, d_, r["stop_idx"])
    return {"train_no": train_no, "date": c["date"][0], "rows": rows, "latest": latest,
            "train_id": tinfo.get("train_id", 1),
            "type_id": S.type_id.get(c["train_type"][0], tinfo.get("type_id", 1)) if "train_type" in c.columns else tinfo.get("type_id", 1)}


def live_segments(c: pl.DataFrame, last_known_idx: int, t: int, train_no: str, S):
    """Stop-to-stop inputs for our next K_SEG stretches, with the same definitions as build_segments:
    stretch rows (our schedule + usual gain, load and regional state from the board at the stretch end) and
    one token per other train that just ran / is on / just ran back each stretch. Candidates are the trains
    on the boards of all our upcoming halts (trains ahead keep showing up further down the route; trains
    coming the other way show up before they reach us), fetched most recently updated first."""
    from build_segments import STRETCH_COLS, TOK_CAT_COLS, TOK_NUM_COLS
    halts = c.filter(pl.col("halts_here") == 1).select("stop_idx", "station_code", "sch_arr_m", "sch_dep_m", "h_med").rows()
    stretches = []
    for i in range(1, len(halts)):
        b_idx, B, b_sarr, _, b_h = halts[i]
        if b_idx <= last_known_idx:
            continue
        if len(stretches) >= K_SEG:
            break
        _, A, _, a_sdep, a_h = halts[i - 1]
        stretches.append({"k": len(stretches), "b_idx": b_idx, "A": A, "B": B,
                          "our_run": (b_sarr - a_sdep) if (b_sarr is not None and a_sdep is not None) else None,
                          "our_usual_gain": (b_h - a_h) if (b_h is not None and a_h is not None) else None})
    cands = {}
    for s in stretches:
        try:
            board = live.live_station(s["B"])
        except Exception:
            continue
        board = board.filter((pl.col("train_no") != train_no) & ~pl.col("cancelled").fill_null(False))
        rec = board.filter(pl.col("last_update").is_not_null()).with_columns(
            d=pl.coalesce("delay_departure_min", "delay_arrival_min"),
            lu=pl.col("last_update").map_elements(to_min, return_dtype=pl.Int64))
        rec = rec.filter(pl.col("d").is_not_null() & (pl.col("lu") <= t))
        recent = rec.filter(pl.col("lu") >= t - RECENT)
        if recent.height:  # training: all reports in B's 1 degree cell over the last 3 h
            s["reg_delay"], s["reg_n"] = float(recent["d"].mean()), float(recent.height)
        sch = board.with_columns(x=pl.coalesce("sch_arr", "sch_dep").map_elements(to_min, return_dtype=pl.Int64))
        s["n_sched_before"] = sch.filter((pl.col("x") >= t - RECENT) & (pl.col("x") < t)).height
        s["n_sched_after"] = sch.filter((pl.col("x") >= t) & (pl.col("x") < t + RECENT)).height
        for no, sd, lu in rec.filter(pl.col("lu") >= t - 2 * RECENT).select("train_no", "start_date", "lu").rows():
            if no not in cands or lu > cands[no][1]:
                cands[no] = (sd, lu)
    journeys = []
    for no, (sd, _) in sorted(cands.items(), key=lambda x: -x[1][1])[:MAX_STATUS_CALLS]:
        j = _journey(no, sd, S)
        if j:
            journeys.append(j)

    toks, notes = [], []
    hist_cache = {}
    def usual(j, a, b):  # its own typical delay change from a to b (90-day history, as in training)
        key = (j["train_no"], a, b)
        if key not in hist_cache:
            h = history(j["train_no"], j["date"], [a, b])
            hm = dict(zip(h["station_code"].to_list(), h["h_med"].to_list()))
            hist_cache[key] = (hm[b] - hm[a]) if hm.get(a) is not None and hm.get(b) is not None else None
        return hist_cache[key]

    for s in stretches:
        A, B = s["A"], s["B"]
        found = {1: [], 2: [], 3: []}
        for j in journeys:
            ra, rb = j["rows"].get(A), j["rows"].get(B)
            if not ra or not rb:
                continue
            if rb["idx"] > ra["idx"]:  # same direction
                base = dict(their_run=(rb["sarr"] - ra["sdep"]) if rb["sarr"] is not None and ra["sdep"] is not None else None,
                            their_delay_in=ra["lv_d"], halt_a=ra["halt"], halt_b=rb["halt"])
                if ra["lv_a"] is not None and rb["ar_a"] is not None and ra["lv_a"] <= t and t - RECENT <= rb["ar_a"] <= t:
                    found[1].append((rb["ar_a"], j, dict(base, gain=rb["ar_d"] - ra["lv_d"], age=t - rb["ar_a"],
                                                         usual_gain=usual(j, A, B))))
                elif (ra["lv_a"] is not None and t - RECENT <= ra["lv_a"] <= t
                      and not (rb["ar_a"] is not None and rb["ar_a"] <= t)
                      and j["latest"] and j["latest"][2] < rb["idx"]):
                    found[2].append((ra["lv_a"], j, dict(base, gain=j["latest"][1] - ra["lv_d"], age=t - ra["lv_a"],
                                                         usual_gain=usual(j, A, B))))
            elif rb["idx"] < ra["idx"]:  # opposite direction: ran B -> A
                if rb["lv_a"] is not None and ra["ar_a"] is not None and rb["lv_a"] <= t and t - RECENT <= ra["ar_a"] <= t:
                    found[3].append((ra["ar_a"], j, dict(
                        their_run=(ra["sarr"] - rb["sdep"]) if ra["sarr"] is not None and rb["sdep"] is not None else None,
                        their_delay_in=rb["lv_d"], halt_a=ra["halt"], halt_b=rb["halt"],
                        gain=ra["ar_d"] - rb["lv_d"], age=t - ra["ar_a"], usual_gain=usual(j, B, A))))
        note = {"stop_idx": s["b_idx"], "stretch": f"{A}->{B}", "done": [], "on_now": [], "opposite": []}
        for kind, n, label in ((1, N_DONE, "done"), (2, N_PROG, "on_now"), (3, N_OPP, "opposite")):
            for _, j, f in sorted(found[kind], key=lambda x: -x[0])[:n]:
                toks.append(([f.get(x) for x in TOK_NUM_COLS],
                             [kind, s["k"], s["b_idx"], j["train_id"], j["type_id"]]))
                note[label].append({"train": j["train_no"], "gain": round(float(f["gain"]), 1),
                                    "usual": None if f["usual_gain"] is None else round(float(f["usual_gain"]), 1)})
        notes.append(note)

    seg = np.full((len(stretches), len(STRETCH_COLS)), np.nan, dtype=np.float32)
    for r, s in enumerate(stretches):
        for name in STRETCH_COLS:
            if s.get(name) is not None:
                seg[r, STRETCH_COLS.index(name)] = s[name]
    tok_num = np.array([[np.nan if v is None else v for v in x] for x, _ in toks], dtype=np.float32).reshape(-1, len(TOK_NUM_COLS))
    tok_cat = np.array([cat for _, cat in toks], dtype=np.int32).reshape(-1, len(TOK_CAT_COLS))
    return seg, (tok_num, tok_cat), notes, len(journeys)


def predict(train_no: str, start: str, ckpt=ROOT / "checkpoints" / "best.pt", now: datetime | None = None) -> dict:
    S = Static()
    train_no = train_no.zfill(5)
    now = now or IST_NOW()
    t = to_min(now)
    raw, meta = live.live_status(train_no, start)
    c = clean(raw.lazy().with_columns(pl.col("halts_here").fill_null(1))).collect()
    if c.is_empty():
        raise SystemExit("live_status returned no stops")
    jd = c["date"][0]
    info_t = S.tid.get(train_no, {})
    c = c.with_columns(pl.col("station_code").replace_strict(S.sid, default=1).alias("station_id"),
                       pl.col("station_code").map_elements(lambda s: S.ll.get(s, (None, None))[0], return_dtype=pl.Float64).alias("lat_s"),
                       pl.col("station_code").map_elements(lambda s: S.ll.get(s, (None, None))[1], return_dtype=pl.Float64).alias("lng_s"))
    c = c.with_columns(pl.coalesce("lat_s", "lat").alias("lat"), pl.coalesce("lng_s", "lng").alias("lng"))

    # history, station congestion (latest usable day <= D-2), weather
    # unique codes: a route that passes the same station twice must not get duplicated stops
    h = history(train_no, jd, c["station_code"].unique().to_list())
    c = c.join(h, on="station_code", how="left")
    usable = S.sd.filter(pl.col("day") <= jd - timedelta(days=2)).sort("day").group_by("station_id").last().drop("day")
    c = c.join(usable, on="station_id", how="left").sort("stop_idx")  # joins don't guarantee row order
    clat, clng = cell_of(pl.col("lat"), pl.col("lng"))
    c = c.with_columns(clat.alias("clat"), clng.alias("clng"),
                       (pl.lit(EPOCH) + pl.duration(minutes=pl.col("sch_arr_m"))).dt.date().cast(pl.String).alias("wday"))
    cells = sorted({(a, b) for a, b in c.select("clat", "clng").drop_nulls().iter_rows()})
    wx = weather_forecast(cells, now.date())
    for k in ("t2m", "t2m_min", "dew_dep", "rh2m", "precip", "ws2m"):
        c = c.with_columns(pl.Series(k, [wx.get(((a, b), d), {}).get(k) for a, b, d in
                                         c.select("clat", "clng", "wday").iter_rows()], dtype=pl.Float64))
    stops = c.select([pl.col(x).cast(pl.Float32) for x in STOP_COLS]).to_numpy()

    # own state at t
    known = c.filter((pl.col("obs_arr") & (pl.col("avail_arr_m") <= t)) | (pl.col("obs_dep") & (pl.col("avail_dep_m") <= t)))
    last_known_idx = int(known["stop_idx"].max()) if known.height else -1
    obs = pl.concat([c.filter("obs_arr").select(pl.col("avail_arr_m").alias("a"), pl.col("delay_arr").alias("d")),
                     c.filter("obs_dep").select(pl.col("avail_dep_m").alias("a"), pl.col("delay_dep").alias("d"))]
                    ).filter(pl.col("a") <= t).sort("a")
    cur_known = obs.height > 0
    cur_delay = float(obs["d"][-1]) if cur_known else 0.0
    cur_age = float(t - obs["a"][-1]) if cur_known else 0.0

    # context: next 6 halts via live_station
    brg = {}
    rows = c.select("stop_idx", "station_code", "lat", "lng").rows()
    for i, (_, code, la, ln) in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        brg[code] = _brg(prev[2], prev[3], la, ln) if prev else (_brg(la, ln, nxt[2], nxt[3]) if nxt else 0.0)
    upcoming = c.filter((pl.col("halts_here") == 1) & (pl.col("stop_idx") > last_known_idx)).head(N_SLOTS)

    # live_status only reports halts, but the station boards of the next halts carry this train's own
    # running delay from its latest position (often a pass-through station minutes ago). Use it when fresher.
    last_obs_min = t - cur_age if cur_known else None
    fresh = None
    for code in upcoming["station_code"].to_list():
        try:
            ls = live.live_station(code)  # cached; reused for the context below
        except Exception:
            continue
        for r in ls.filter(pl.col("train_no") == train_no).iter_rows(named=True):
            d = r["delay_departure_min"] if r["departed"] else r["delay_arrival_min"]
            lu = to_min(r["last_update"]) if r["last_update"] else None
            if d is None or lu is None or lu > t or (last_obs_min is not None and lu <= last_obs_min):
                continue
            if fresh is None or lu > fresh[0]:
                fresh = (lu, float(d), r["current_station"])
    if fresh:
        cur_delay, cur_known, cur_age = fresh[1], True, float(t - fresh[0])
    ctx_rows = []
    for slot, (code, our_sch) in enumerate(upcoming.select("station_code", "sch_arr_m").iter_rows()):
        try:
            ls = live.live_station(code)
        except Exception:
            continue
        our_exp = our_sch + cur_delay
        cands = []
        for r in ls.iter_rows(named=True):
            if r["train_no"] == train_no or r["cancelled"]:
                continue
            sch = to_min(r["sch_arr"] or r["sch_dep"])
            if sch is None or not (t - WIN_BEFORE <= sch <= t + WIN_AFTER):
                continue
            dep_at = None
            if r["departed"] and r["sch_dep"] is not None and r["delay_departure_min"] is not None:
                dep_at = to_min(r["sch_dep"]) + r["delay_departure_min"]
            if dep_at is not None and dep_at < t - DEPARTED_GRACE:
                continue
            kn = r["last_update"] is not None and (r["delay_arrival_min"] is not None or r["delay_departure_min"] is not None)
            cur = (r["delay_departure_min"] if r["departed"] else r["delay_arrival_min"]) if kn else 0.0
            cur = float(cur or 0.0)
            s_ll = S.ll.get(code, (None, None))
            c_ll = S.ll.get(r["current_station"] or "", s_ll)
            d_ll = S.ll.get(r["destination_station"] or "", (None, None))
            dist = _hav(*c_ll, *s_ll)
            tb = _brg(*s_ll, *d_ll) if (r["arrived"] or dist < 0.5) else _brg(*c_ll, *s_ll)
            tinfo = S.tid.get(r["train_no"], {})
            their_exp = sch + cur
            sch_halt = (to_min(r["sch_dep"]) - to_min(r["sch_arr"])) if r["sch_dep"] and r["sch_arr"] else 0.0
            cands.append([slot, S.sid.get(code, 1), tinfo.get("train_id", 1),
                          S.type_id.get(r["train_type"], tinfo.get("type_id", 1)), float(kn), float(r["arrived"]),
                          float(r["departed"]), their_exp - our_exp, their_exp - t, our_exp - t, cur,
                          float(t - to_min(r["last_update"])) if kn else 0.0, dist, math.cos(tb - brg.get(code, 0.0)),
                          float(sch_halt)])
        cands.sort(key=lambda x: abs(x[7]))
        ctx_rows += cands[:CTX_PER_SLOT]

    # rake-link pair, before departure
    pair_no = info_t.get("pair_train_no")
    origin_sch_dep = int(c["sch_dep_m"][0])
    if last_known_idx <= 0 and pair_no:
        pinfo = S.tid.get(pair_no, {})
        dur = int(pinfo.get("sch_duration_min") or 0)
        origin = c["station_code"][0]
        for back in range(0, 4):
            pd_ = (EPOCH + timedelta(minutes=origin_sch_dep - dur)).date() - timedelta(days=back)
            try:
                praw, _ = live.live_status(pair_no, pd_)
            except Exception:
                continue
            pc = clean(praw.lazy().with_columns(pl.col("halts_here").fill_null(1))).collect()
            arr = pc.filter(pl.col("station_code") == origin).tail(1)
            if arr.is_empty():
                continue
            sch = int(arr["sch_arr_m"][0])
            if not (origin_sch_dep - 36 * 60 <= sch <= origin_sch_dep + 60):
                continue
            pobs = pl.concat([pc.filter("obs_arr").select(pl.col("avail_arr_m").alias("a"), pl.col("delay_arr").alias("d"), "station_code", "stop_idx"),
                              pc.filter("obs_dep").select(pl.col("avail_dep_m").alias("a"), pl.col("delay_dep").alias("d"), "station_code", "stop_idx")]
                             ).filter(pl.col("a") <= t).sort("a")
            kn = pobs.height > 0
            cur = float(pobs["d"][-1]) if kn else 0.0
            cur_stn = pobs["station_code"][-1] if kn else pc["station_code"][0]
            arrived = bool(kn and pobs["stop_idx"].max() >= int(arr["stop_idx"][0]))
            s_ll, c_ll = S.ll.get(origin, (None, None)), S.ll.get(cur_stn, (None, None))
            ctx_rows.append([RAKE_SLOT, S.sid.get(origin, 1), pinfo.get("train_id", 1), pinfo.get("type_id", 1),
                             float(kn), float(arrived), 0.0, sch + cur - origin_sch_dep, sch + cur - t,
                             float(origin_sch_dep - t), cur, float(t - pobs["a"][-1]) if kn else 0.0,
                             _hav(*c_ll, *s_ll), 0.0, 0.0])
            break
    ctx = np.array(ctx_rows, dtype=np.float32).reshape(-1, len(CTX_COLS))

    day = np.array([(jd - DAY0).days])
    hol = holiday_features(day)
    info = {"t": np.array([t]), "cur_delay": np.array([cur_delay], dtype=np.float32),
            "cur_known": np.array([int(cur_known)]), "cur_age": np.array([cur_age], dtype=np.float32),
            "last_known_idx": np.array([last_known_idx]), "train_id": np.array([info_t.get("train_id", 1)]),
            "type_id": np.array([S.type_id.get(raw["train_type"][0], info_t.get("type_id", 1))]),
            "pair_id": np.array([info_t.get("pair_id", 0)]), "day": day, "origin_sch_dep": np.array([origin_sch_dep]),
            "n_stops": np.array([len(stops)]), **{k: v for k, v in hol.items()}}
    # live "target" is unknown: mark every stop as a prediction target so the model scores all of them
    stops_pred = stops.copy()
    stops_pred[:, STOP_COLS.index("target")] = 0.0
    stops_pred[:, STOP_COLS.index("avail_m")] = t + 1
    ck = torch.load(ckpt, map_location="cpu")
    model = RouteTransformer(**{k: v for k, v in ck["config"].items()})
    model.load_state_dict(ck["model"])
    model.eval()
    segs, toks, seg_notes, n_looked_up = None, None, [], 0
    if ck["config"].get("tokens"):
        segs, toks, seg_notes, n_looked_up = live_segments(c, last_known_idx, t, train_no, S)

    def run(win_start: int) -> np.ndarray:
        b = build_batch([stops_pred], [ctx], {**info, "win_start": np.array([win_start])},
                        segs=None if segs is None else [segs], toks=None if toks is None else [toks])
        with torch.no_grad():
            return model({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()})[0].numpy()

    # The model sees at most L_MAX stops, starting KEEP_BEHIND stops behind the train (as in training).
    # pred is indexed by stop_idx; stops past that window get a second pass ending at the destination.
    n = len(stops_pred)
    pred = np.full((n, 3), np.nan, dtype=np.float32)
    a0 = int(np.clip(last_known_idx - KEEP_BEHIND, 0, max(n - L_MAX, 0)))
    pred[a0:a0 + L_MAX] = run(a0)[:min(L_MAX, n - a0)]
    if a0 + L_MAX < n:
        a1 = n - L_MAX
        pred[a0 + L_MAX:] = run(a1)[a0 + L_MAX - a1:]

    out = []
    prev = None  # (sch_dep, [p10, p50, p90] expected departure) of the previous upcoming stop
    note_by_stop = {n["stop_idx"]: n for n in seg_notes}
    for i, r in enumerate(c.iter_rows(named=True)):
        is_known = r["stop_idx"] <= last_known_idx
        is_origin, is_last = i == 0, i == n - 1
        sch_a, sch_d = r["sch_arr"], r["sch_dep"]
        halt = (sch_d - sch_a).total_seconds() / 60 if sch_a and sch_d else 0.0
        s = {"station": r["station_code"], "stop_idx": i, "halts": bool(r["halts_here"]),
             "status": "passed" if is_known else "upcoming",
             "sch_arr": None if is_origin else str(sch_a), "sch_dep": None if is_last else str(sch_d)}
        if is_known:
            s["actual_arr"] = str(sch_a + timedelta(minutes=r["delay_arr"])) if r["obs_arr"] and sch_a else None
            s["actual_dep"] = str(sch_d + timedelta(minutes=r["delay_dep"])) if r["obs_dep"] and sch_d else None
            s["observed_delay"] = r["target"]
        else:
            p = [float(x) for x in pred[i]]  # p10, p50, p90 delay in minutes (origin: departure delay)
            if is_origin:
                eta_a = None
                eta_d = [sch_d + timedelta(minutes=x) for x in p]
            else:
                eta_a = [sch_a + timedelta(minutes=x) for x in p]
                if prev is not None:
                    # each stop is predicted separately; keep the timeline physically ordered: arrive no
                    # earlier than the previous departure + 30% of the scheduled run (about the fastest
                    # 1-5% of real runs, measured on 2025-26 data)
                    run_min = max((sch_a - prev[0]).total_seconds() / 60, 0.0) if prev[0] else 0.0
                    floor = timedelta(minutes=max(1.0, 0.3 * run_min))
                    eta_a = [max(x, pd + floor) for x, pd in zip(eta_a, prev[1])]
                    p = [(x - sch_a).total_seconds() / 60 for x in eta_a]
                # departure: never before schedule; a late train's dwell follows the measured pattern
                eta_d = None if is_last else [max(sch_d, x + timedelta(minutes=late_dwell(halt))) for x in eta_a]
            prev = (sch_d or sch_a, eta_d or eta_a)
            s["pred_delay_p10"], s["pred_delay_p50"], s["pred_delay_p90"] = (round(x, 1) for x in p)
            m = lambda x: str((x + timedelta(seconds=30)).replace(second=0, microsecond=0))  # nearest minute
            s["eta_arr"] = m(eta_a[1]) if eta_a else None
            s["eta_arr_range"] = [m(eta_a[0]), m(eta_a[2])] if eta_a else None
            s["eta_dep"] = m(eta_d[1]) if eta_d else None
            s["eta_dep_range"] = [m(eta_d[0]), m(eta_d[2])] if eta_d else None
            s["eta"] = s["eta_arr"] or s["eta_dep"]
            if i in note_by_stop:
                s["stretch_traffic"] = {k_: v_ for k_, v_ in note_by_stop[i].items() if k_ != "stop_idx"}
        out.append(s)
    warnings = []
    late_by = (t - origin_sch_dep) / 60
    if not cur_known and late_by > 2:
        warnings.append(f"no live report since the scheduled departure {late_by:.0f} h ago: the train may be untracked, "
                        f"rescheduled or cancelled. The estimates assume it has not left yet.")
    elif cur_known and cur_age > 180:
        warnings.append(f"last live report is {cur_age / 60:.1f} h old; estimates are less reliable")
    return {"train_no": train_no, "start_date": str(jd), "as_of": str(now), "current_delay": cur_delay if cur_known else None,
            "last_known_stop": c["station_code"][last_known_idx] if last_known_idx >= 0 else None,
            "latest_position": ({"station": fresh[2], "delay": fresh[1], "at": str(EPOCH + timedelta(minutes=fresh[0]))}
                                if fresh else None),
            "other_trains_considered": int(len(ctx)), "trains_looked_up_for_stretches": n_looked_up, "rake_link": pair_no, "warnings": warnings, "stops": out}


def late_dwell(halt: float) -> float:
    """Typical minutes a late-running train spends at a stop with this scheduled halt.
    Measured on Nov-2025 / Feb-2026 / May-2026 runs: short halts are kept or slightly
    overrun (2 -> 3, 5 -> 6-7, 10 -> 13 min); long halts (> 20 min) are cut short."""
    if halt <= 1:
        return halt
    if halt <= 10:
        return halt + max(1.0, round(0.25 * halt))
    if halt <= 20:
        return halt
    return max(15.0, 0.6 * halt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_no")
    ap.add_argument("start_date", help="DD-MM-YYYY journey start date")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--all", action="store_true", help="also list stations the train passes without stopping")
    a = ap.parse_args()
    r = predict(a.train_no, a.start_date, ckpt=a.ckpt)
    if a.json:
        print(json.dumps(r, indent=1, default=str))
        return

    def hm(x):  # "Thu 14:32"
        return datetime.fromisoformat(x).strftime("%a %H:%M") if x else "    -    "

    print(f"{r['train_no']} started {r['start_date']} | as of {r['as_of'][:16]} | current delay "
          f"{r['current_delay']} min at {r['last_known_stop']} | {r['other_trains_considered']} other trains considered")
    if r["latest_position"]:
        lp = r["latest_position"]
        print(f"  latest position: {lp['station']} at {lp['at'][11:16]}, running {lp['delay']:+.0f} min (from station boards)")
    for w in r["warnings"]:
        print(f"  WARNING: {w}")
    print(f"  {'stn':<6} {'sch arr':>9} {'sch dep':>9} | {'exp arr':>9} {'exp dep':>9}  {'delay':>6}  likely arrival window")
    for s in r["stops"]:
        if not (s["halts"] or a.all or s["stop_idx"] in (0, len(r["stops"]) - 1)):
            continue
        name = s["station"] if s["halts"] else f"({s['station']})"
        head = f"  {name:<6} {hm(s['sch_arr']):>9} {hm(s['sch_dep']):>9} |"
        if s["status"] == "passed":
            d = s["observed_delay"]
            print(f"{head} {hm(s['actual_arr']):>9} {hm(s['actual_dep']):>9}  {'' if d is None else f'{d:+.0f}':>6}  (actual)")
        else:
            rng = s["eta_arr_range"] or s["eta_dep_range"]
            tr = s.get("stretch_traffic")
            extra = ""
            if tr and (tr["done"] or tr["on_now"] or tr["opposite"]):
                def fmt(e):  # "12189 -8 (usually +1)"
                    return f"{e['train']} {e['gain']:+.0f}" + (f" (usually {e['usual']:+.0f})" if e["usual"] is not None else "")
                parts = []
                if tr["done"]:
                    parts.append("just ran it: " + ", ".join(fmt(e) for e in tr["done"][:3]))
                if tr["on_now"]:
                    parts.append("on it now: " + ", ".join(fmt(e) for e in tr["on_now"][:2]))
                if tr["opposite"]:
                    parts.append("other way: " + ", ".join(fmt(e) for e in tr["opposite"][:2]))
                extra = f"  | {tr['stretch']} " + "; ".join(parts)
            print(f"{head} {hm(s['eta_arr']):>9} {hm(s['eta_dep']):>9}  {s['pred_delay_p50']:+6.0f}  "
                  f"{hm(rng[0])} .. {hm(rng[1])}{extra}")


if __name__ == "__main__":
    main()

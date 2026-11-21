"""Raw arrays -> model inputs. Shared by training and live inference.

Inputs per sample:
  stops  float32 [n, len(STOP_COLS)]  the whole journey (pack.STOP_COLS order)
  ctx    float32 [m, len(CTX_COLS)]   other-train tokens (pack.CTX_COLS order)
  info   dict of scalars: t, cur_delay, cur_known, cur_age, last_known_idx, train_id, type_id,
         pair_id, day, origin_sch_dep, hol_is, hol_to_next, hol_since_prev
Everything is expressed relative to the cutoff t; nothing after t leaks in.
"""

from __future__ import annotations

import numpy as np

from build_segments import K_SEG, N_TOK, STRETCH_COLS, TOK_CAT_COLS, TOK_NUM_COLS
from pack import CTX_COLS, STOP_COLS

L_MAX = 128          # stops kept per journey (99.9% of journeys have <= 85)
KEEP_BEHIND = 16     # when trimming a long journey, passed stops kept before the current position
CTX_MAX = 6 * 8 + 1
SI = {c: i for i, c in enumerate(STOP_COLS)}
CI = {c: i for i, c in enumerate(CTX_COLS)}
GI = {c: i for i, c in enumerate(STRETCH_COLS)}
TNI = {c: i for i, c in enumerate(TOK_NUM_COLS)}
TCI = {c: i for i, c in enumerate(TOK_CAT_COLS)}

# per-stretch features on the stop the stretch ends at (build_segments.py)
SEG_NUM = ["s_has", "s_our_run", "s_our_usual_gain", "s_our_usual_has", "s_load_before", "s_load_after",
           "s_reg_delay", "s_reg_has"]
# one token per other train on one of our next stretches; categorical: kind, k, stop pos, train, class
TOK_NUM = ["gain", "age", "their_run", "run_ratio", "their_delay_in", "usual_gain", "usual_has", "anomaly",
           "halt_a", "halt_b", "our_run"]
STOP_NUM = ["known_arr", "known_dep", "d_arr", "d_dep", "age_obs", "passed", "rel_arr", "rel_dep", "halt",
            "run", "dist", "seg", "tod_s", "tod_c", "lat", "lng", "lat_na", "h_mean", "h_med", "h_std",
            "h_q90", "h_last", "h_cnt", "h_age", "h_na", "st_gain7", "st_delay7", "st_traf7", "st_gain28",
            "st_delay28", "st_traf28", "st_na", "w_t2m", "w_tmin", "w_dewdep", "w_rh", "w_precip", "w_wind",
            "w_na", "halts", "rel_hist", "is_origin", "is_last", *SEG_NUM]
CTX_NUM = ["known", "arrived", "departed", "headway", "rel_t", "our_rel", "their_cur", "age", "dist_cur",
           "dir_cos", "sch_halt", "abs_headway_small"]
GLOB_NUM = ["cur_delay", "cur_known", "cur_age", "t_tod_s", "t_tod_c", "year_s", "year_c", "half_s", "half_c",
            "year", "hol_is", "hol_next", "hol_prev", "n_stops", "since_origin", "has_pair", "progress"]


def _nz(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def stretch_features(segs, toks, starts: np.ndarray, B: int, L: int, rng: np.random.Generator | None,
                     drop_max: float):
    """Stop-to-stop inputs.
    Returns (seg_num [B, L, len(SEG_NUM)] scattered onto the stop each stretch ends at,
             tok_num [B, T, len(TOK_NUM)], tok_cat [B, T, 5], tok_pad [B, T]).
    Training-time dropout (rng given) mimics the live lookup: a random share (up to drop_max) of the other
    trains is unseen; 30% of samples only get the first k' stretches (the live call budget ran out);
    5% get no other trains at all (the API failed)."""
    seg_num = np.zeros((B, L, len(SEG_NUM)), dtype=np.float64)
    tn, tc = [], []
    for b in range(B):
        g = segs[b] if segs is not None else None
        tk = toks[b] if toks is not None else None
        k_cut, keep_p, none = K_SEG, 0.0, False
        if rng is not None and drop_max > 0:
            none = rng.random() < 0.05
            if rng.random() < 0.3:
                k_cut = int(rng.integers(2, K_SEG + 1))
            keep_p = rng.random() * drop_max
        our_run = {}
        if g is not None and len(g):
            p = g[:, GI["b_idx"]].astype(np.int64) - int(starts[b])
            our_run = dict(zip(g[:, GI["k"]].astype(np.int64), g[:, GI["our_run"]]))
            ok = (p >= 0) & (p < L)
            g, p = g[ok], p[ok]
            vis = g[:, GI["k"]] < k_cut
            ug, rd = g[:, GI["our_usual_gain"]], g[:, GI["reg_delay"]]
            col = lambda name: SEG_NUM.index(name)
            seg_num[b, p, col("s_has")] = 1.0
            seg_num[b, p, col("s_our_run")] = np.clip(np.nan_to_num(g[:, GI["our_run"]]) / 60.0, 0, 24) / 2.0
            seg_num[b, p, col("s_our_usual_gain")] = np.clip(np.nan_to_num(ug) / 60.0, -3, 5)
            seg_num[b, p, col("s_our_usual_has")] = np.isfinite(ug)
            # board-derived numbers are only there for stretches the live lookup reached
            seg_num[b, p, col("s_load_before")] = vis * np.log1p(np.nan_to_num(g[:, GI["n_sched_before"]])) / 3.0
            seg_num[b, p, col("s_load_after")] = vis * np.log1p(np.nan_to_num(g[:, GI["n_sched_after"]])) / 3.0
            seg_num[b, p, col("s_reg_delay")] = vis * np.clip(np.nan_to_num(rd) / 60.0, -1, 10)
            seg_num[b, p, col("s_reg_has")] = vis & np.isfinite(rd)
        if tk is None or none:
            continue
        num, cat = tk
        if len(num) == 0:
            continue
        pos = cat[:, TCI["b_idx"]].astype(np.int64) - int(starts[b])
        keep = (pos >= 0) & (pos < L) & (cat[:, TCI["k"]] < k_cut)
        if keep_p > 0:
            keep &= rng.random(len(num)) >= keep_p
        num, cat = num[keep].astype(np.float64), cat[keep]
        if len(num) == 0:
            continue
        x = lambda name: num[:, TNI[name]]
        orun = np.array([our_run.get(int(k), np.nan) for k in cat[:, TCI["k"]]], dtype=np.float64)
        ug = x("usual_gain")
        f = np.stack([
            np.clip(_nz(x("gain")) / 60.0, -3, 5), np.clip(_nz(x("age")) / 180.0, 0, 1.5),
            np.clip(_nz(x("their_run")) / 60.0, 0, 24) / 2.0,
            np.clip(_nz(x("their_run") / np.where(orun > 0, orun, np.nan)), 0, 3),
            np.clip(_nz(x("their_delay_in")) / 60.0, -3, 30) / 5.0,
            np.clip(_nz(ug) / 60.0, -3, 5), np.isfinite(ug),
            np.where(np.isfinite(ug), np.clip(_nz(x("gain") - ug) / 60.0, -3, 5), 0.0),
            _nz(x("halt_a")), _nz(x("halt_b")), np.clip(_nz(orun) / 60.0, 0, 24) / 2.0], axis=-1)
        c = np.stack([cat[:, TCI["kind"]], np.clip(cat[:, TCI["k"]], 0, K_SEG),
                      np.clip(cat[:, TCI["b_idx"]], 0, 255), np.maximum(cat[:, TCI["train_id"]], 1),
                      np.maximum(cat[:, TCI["type_id"]], 1)], axis=-1)
        tn.append((b, f))
        tc.append(c)
    T = max([len(f) for _, f in tn], default=0) or 1
    tok_num = np.zeros((B, T, len(TOK_NUM)), dtype=np.float32)
    tok_cat = np.zeros((B, T, 5), dtype=np.int64)
    tok_pad = np.ones((B, T), dtype=bool)
    for (b, f), c in zip(tn, tc):
        tok_num[b, :len(f)] = f
        tok_cat[b, :len(f)] = c
        tok_pad[b, :len(f)] = False
    return seg_num, tok_num, tok_cat, tok_pad


def build_batch(stops: list[np.ndarray], ctx: list[np.ndarray], info: dict[str, np.ndarray],
                rng: np.random.Generator | None = None, ctx_drop_max: float = 0.0,
                segs: list[np.ndarray] | None = None,
                toks: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict[str, np.ndarray]:
    B = len(stops)
    t = info["t"].astype(np.float64)
    lki = info["last_known_idx"].astype(np.int64)

    # ---- trim + pad journey stops --------------------------------------
    lens = np.array([min(len(s), L_MAX) for s in stops])
    L = int(lens.max())
    raw = np.full((B, L, len(STOP_COLS)), np.nan, dtype=np.float64)
    pad = np.ones((B, L), dtype=bool)
    starts = np.zeros(B, dtype=np.int64)
    for b, s in enumerate(stops):
        n = len(s)
        a = 0
        if n > L_MAX:
            # live inference may ask for a later window (stops beyond the first 128); training never does
            a = int(info["win_start"][b]) if "win_start" in info else int(np.clip(lki[b] - KEEP_BEHIND, 0, n - L_MAX))
        starts[b] = a
        seg = s[a:a + L_MAX]
        raw[b, :len(seg)] = seg
        pad[b, :len(seg)] = False
    g = lambda c: raw[:, :, SI[c]]
    tt = t[:, None]

    known_arr = (g("obs_arr") > 0) & (g("avail_arr_m") <= tt)
    known_dep = (g("obs_dep") > 0) & (g("avail_dep_m") <= tt)
    d_arr = np.where(known_arr, g("delay_arr"), 0.0)
    d_dep = np.where(known_dep, g("delay_dep"), 0.0)
    last_avail = np.fmax(np.where(known_arr, g("avail_arr_m"), np.nan), np.where(known_dep, g("avail_dep_m"), np.nan))
    age_obs = np.where(np.isfinite(last_avail), (tt - last_avail) / 60.0, 0.0)
    sch_arr, sch_dep = g("sch_arr_m"), g("sch_dep_m")
    prev_dep = np.concatenate([sch_dep[:, :1], sch_dep[:, :-1]], axis=1)
    dist = g("distance_km")
    prev_dist = np.concatenate([dist[:, :1], dist[:, :-1]], axis=1)
    tod = (np.mod(sch_arr, 1440.0) / 1440.0) * 2 * np.pi
    stop_idx = g("stop_idx")
    n_st = info["n_stops"].astype(np.float64)[:, None] if "n_stops" in info else np.nanmax(stop_idx, axis=1, keepdims=True) + 1
    h_na = ~np.isfinite(g("h_med"))
    st_na = ~np.isfinite(g("st_delay_7d"))
    w_na = ~np.isfinite(g("t2m"))
    feats = {
        "known_arr": known_arr, "known_dep": known_dep,
        "d_arr": d_arr / 60.0, "d_dep": d_dep / 60.0, "age_obs": np.clip(age_obs, 0, 24) / 6.0,
        "passed": stop_idx <= lki[:, None],
        "rel_arr": np.clip((sch_arr - tt) / 60.0, -48, 96) / 12.0,
        "rel_dep": np.clip((sch_dep - tt) / 60.0, -48, 96) / 12.0,
        "halt": np.clip((sch_dep - sch_arr) / 30.0, 0, 10),
        "run": np.clip((sch_arr - prev_dep) / 60.0, 0, 24) / 2.0,
        "dist": dist / 1000.0, "seg": np.clip(dist - prev_dist, 0, 1000) / 100.0,
        "tod_s": np.sin(tod), "tod_c": np.cos(tod),
        "lat": (g("lat") - 22.0) / 6.0, "lng": (g("lng") - 80.0) / 8.0, "lat_na": ~np.isfinite(g("lat")),
        "h_mean": g("h_mean") / 60.0, "h_med": g("h_med") / 60.0, "h_std": g("h_std") / 60.0,
        "h_q90": g("h_q90") / 60.0, "h_last": g("h_last") / 60.0, "h_cnt": np.log1p(g("h_cnt")) / 4.0,
        "h_age": np.clip(g("h_age_days"), 0, 365) / 90.0, "h_na": h_na,
        "st_gain7": np.clip(g("st_gain_7d"), -120, 120) / 10.0, "st_delay7": g("st_delay_7d") / 60.0,
        "st_traf7": np.log1p(g("st_traffic_7d")) / 5.0, "st_gain28": np.clip(g("st_gain_28d"), -120, 120) / 10.0,
        "st_delay28": g("st_delay_28d") / 60.0, "st_traf28": np.log1p(g("st_traffic_28d")) / 5.0, "st_na": st_na,
        "w_t2m": (g("t2m") - 25.0) / 10.0, "w_tmin": (g("t2m_min") - 20.0) / 10.0, "w_dewdep": g("dew_dep") / 5.0,
        "w_rh": g("rh2m") / 100.0, "w_precip": np.log1p(np.clip(g("precip"), 0, None)) / 3.0,
        "w_wind": g("ws2m") / 5.0, "w_na": w_na,
        "halts": g("halts_here"),
        # current delay relative to this stop's typical delay: "running later than usual here?"
        "rel_hist": np.clip((info["cur_delay"][:, None] - np.nan_to_num(g("h_med"))) / 60.0, -10, 30),
        "is_origin": stop_idx == 0, "is_last": stop_idx == n_st - 1,
    }
    base_num = np.stack([_nz(np.asarray(feats[k], dtype=np.float64)) for k in STOP_NUM if k not in SEG_NUM], axis=-1)
    seg_num, tok_num, tok_cat, tok_pad = stretch_features(segs, toks, starts, B, L, rng, ctx_drop_max)
    stop_num = np.concatenate([base_num, seg_num], axis=-1)
    stop_num[pad] = 0.0
    station = np.where(pad, 0, _nz(g("station_id"))).astype(np.int64)
    pos = np.where(pad, 0, np.clip(_nz(stop_idx), 0, 255)).astype(np.int64)
    target = g("target")
    tmask = np.isfinite(target) & (g("avail_m") > tt) & ~pad
    horizon = (sch_arr - tt)

    # ---- context tokens ------------------------------------------------
    Cn = np.zeros((B, CTX_MAX, len(CTX_NUM)), dtype=np.float64)
    Ccat = np.zeros((B, CTX_MAX, 4), dtype=np.int64)  # slot, station, train, type
    Cpad = np.ones((B, CTX_MAX), dtype=bool)
    for b, c in enumerate(ctx):
        if len(c) == 0:
            continue
        c = c[:CTX_MAX]
        if rng is not None and ctx_drop_max > 0:
            if rng.random() < 0.05:
                continue  # the live API returned nothing
            keep = rng.random(len(c)) >= rng.random() * ctx_drop_max
            c = c[keep]
            if len(c) == 0:
                continue
        k = len(c)
        col = lambda name: c[:, CI[name]].astype(np.float64)
        Cn[b, :k] = np.stack([
            col("known"), col("arrived"), col("departed"),
            np.clip(col("headway") / 60.0, -12, 12), np.clip(col("rel_t") / 60.0, -12, 24) / 2.0,
            np.clip(col("our_rel") / 60.0, -12, 24) / 2.0, np.clip(col("their_cur") / 60.0, -5, 30),
            np.clip(col("age") / 60.0, 0, 24) / 2.0, np.clip(col("dist_cur"), 0, 2000) / 100.0,
            col("dir_cos"), np.clip(col("sch_halt") / 30.0, 0, 10),
            (np.abs(col("headway")) < 20).astype(np.float64)], axis=-1)
        Ccat[b, :k] = np.stack([col("slot"), col("station_id"), col("train_id"), col("type_id")], axis=-1).astype(np.int64)
        Cpad[b, :k] = False
    Cn = _nz(Cn)

    # ---- journey-level ("global") token --------------------------------
    day = info["day"].astype(np.float64)
    doy = np.mod(day + 0.0, 365.25) / 365.25 * 2 * np.pi  # 2020-01-01 = day 0; annual + half-year harmonics
    t_tod = np.mod(t, 1440.0) / 1440.0 * 2 * np.pi
    dow = np.mod(day.astype(np.int64) + 2, 7)  # 2020-01-01 was a Wednesday -> 0 = Monday
    glob = np.stack([
        np.clip(info["cur_delay"] / 60.0, -3, 30), info["cur_known"].astype(np.float64),
        np.clip(info["cur_age"] / 60.0, 0, 24) / 2.0, np.sin(t_tod), np.cos(t_tod),
        np.sin(doy), np.cos(doy), np.sin(2 * doy), np.cos(2 * doy),
        (day / 365.25 - 4.0) / 2.0, info["hol_is"], info["hol_to_next"] / 30.0, info["hol_since_prev"] / 30.0,
        info["n_stops"] / 50.0, np.clip((t - info["origin_sch_dep"]) / 600.0, -1, 10),
        (info["pair_id"] > 0).astype(np.float64),
        np.clip((lki + 1) / np.maximum(info["n_stops"], 1), 0, 1),
    ], axis=-1)

    base = np.where(info["cur_known"] > 0, info["cur_delay"], 0.0).astype(np.float32)
    return {
        "stop_num": stop_num.astype(np.float32), "station": station, "pos": pos, "stop_pad": pad,
        "ctx_num": Cn.astype(np.float32), "ctx_cat": Ccat, "ctx_pad": Cpad,
        "glob_num": _nz(glob).astype(np.float32),
        "glob_cat": np.stack([info["train_id"], info["type_id"], dow], axis=-1).astype(np.int64),
        "base": base, "cur_known": (info["cur_known"] > 0), "target": np.nan_to_num(target, nan=0.0).astype(np.float32), "tmask": tmask,
        "horizon": _nz(horizon).astype(np.float32),
        "h_med_raw": g("h_med").astype(np.float32),
        "tok_num": tok_num, "tok_cat": tok_cat, "tok_pad": tok_pad,
    }

"""LightGBM benchmark on the exact same inputs, scored on the exact same test rows as the
transformer, plus a 50/50 blend of the two.

Each row = (sample, future stop). Features: the transformer's per-stop and journey-level
inputs, categorical ids, and the other-train context summarised per sample and per
upcoming-halt slot (the trees cannot attend over tokens, so the context is aggregated).
Label: delay minus the current known delay (residual), L1 objective (= median).

Writes reports/lgbm_test.json, reports/comparison.json, checkpoints/lgbm.txt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb  # noqa: F401  must load before torch: with torch's OpenMP loaded first, LightGBM segfaults on macOS
import numpy as np

import train as T
from features import CTX_NUM, GLOB_NUM, STOP_NUM

C = {n: i for i, n in enumerate(CTX_NUM)}
SLOT_FEATS = ["slot_n", "slot_known", "slot_cur_mean", "slot_min_abs_headway", "slot_samedir_near_cur"]
SAMPLE_FEATS = ["ctx_n", "ctx_known", "ctx_cur_mean", "ctx_cur_max", "near_n", "near_cur_mean", "samedir_near_n",
                "samedir_near_cur_mean", "samedir_near_headway_mean", "ahead_samedir_n", "ahead_samedir_cur_max",
                "rake_present", "rake_known", "rake_cur", "rake_headway", "rake_arrived"]
CAT_FEATS = ["station", "train_id", "type_id", "dow"]
FEATURES = STOP_NUM + GLOB_NUM + SAMPLE_FEATS + SLOT_FEATS + ["stop_slot"] + CAT_FEATS


def _mmean(x, m):
    s = (x * m).sum(-1)
    n = m.sum(-1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _mmax(x, m):
    return np.where(m.any(-1), np.where(m, x, -np.inf).max(-1), np.nan)


def flatten(b: dict) -> tuple[np.ndarray, dict]:
    m = b["tmask"]
    bi, li = np.nonzero(m)
    cn, cc, valid = b["ctx_num"], b["ctx_cat"], ~b["ctx_pad"]
    slot = cc[..., 0]
    known = valid & (cn[..., C["known"]] > 0)
    cur, hw, dirc = cn[..., C["their_cur"]], cn[..., C["headway"]], cn[..., C["dir_cos"]]
    near = known & (np.abs(hw) < 0.5)
    same_near = near & (dirc > 0.5)
    ahead = valid & (dirc > 0.5) & (hw < 0) & (hw > -1)
    rake = valid & (slot == 6)
    sample = np.stack([
        valid.sum(-1), known.sum(-1), _mmean(cur, known), _mmax(cur, known), near.sum(-1), _mmean(cur, near),
        same_near.sum(-1), _mmean(cur, same_near), _mmean(hw, same_near), ahead.sum(-1), _mmax(cur, ahead & known),
        rake.any(-1), (rake & known).any(-1), _mmean(cur, rake), _mmean(hw, rake),
        _mmean(cn[..., C["arrived"]], rake)], axis=-1).astype(np.float32)

    # per-slot aggregates, then mapped onto the stop that slot refers to
    B = cn.shape[0]
    slot_f = np.full((B, 7, len(SLOT_FEATS)), np.nan, dtype=np.float32)
    for s in range(6):
        ms, mk = valid & (slot == s), known & (slot == s)
        slot_f[:, s] = np.stack([ms.sum(-1), mk.sum(-1), _mmean(cur, mk),
                                 np.where(ms.any(-1), np.where(ms, np.abs(hw), np.inf).min(-1), np.nan),
                                 _mmean(cur, mk & (np.abs(hw) < 0.5) & (dirc > 0.5))], axis=-1)
    # stop -> slot: k-th halting stop after the current position
    halts = b["stop_num"][..., STOP_NUM.index("halts")] > 0
    passed = b["stop_num"][..., STOP_NUM.index("passed")] > 0
    up = halts & ~passed & ~b["stop_pad"]
    stop_slot = np.where(up, np.cumsum(up, axis=1) - 1, 6)
    stop_slot = np.minimum(stop_slot, 6)
    ss = stop_slot[bi, li]
    gc = b["glob_cat"][bi]
    X = np.concatenate([
        b["stop_num"][bi, li], b["glob_num"][bi], sample[bi], slot_f[bi, ss], ss[:, None].astype(np.float32),
        b["station"][bi, li][:, None].astype(np.float32), gc.astype(np.float32)], axis=1)
    extra = {"y": b["target"][bi, li], "base": b["base"][bi], "horizon": b["horizon"][bi, li],
             "h_med": b["h_med_raw"][bi, li], "known": b["cur_known"][bi], "bi": bi, "li": li}
    return X, extra


def build(pk, idx, bs=2048, label="") -> tuple[np.ndarray, np.ndarray]:
    """One contiguous feature matrix (LightGBM's list-of-arrays input segfaults in 4.7), filled
    chunk by chunk into a preallocated block so the matrix is never held twice."""
    Xs, ys = [], []
    t0 = time.time()
    for i in range(0, len(idx), bs):
        X, e = flatten(pk.batch(idx[i:i + bs]))
        Xs.append(X)
        ys.append((e["y"] - e["base"]).astype(np.float32))
        if (i // bs) % 200 == 0:
            print(f"[lgbm] building {label}: {i:,}/{len(idx):,} samples ({time.time() - t0:.0f}s)", flush=True)
    n = sum(len(x) for x in Xs)
    Xall = np.empty((n, Xs[0].shape[1]), dtype=np.float32)
    o = 0
    while Xs:
        x = Xs.pop(0)
        Xall[o:o + len(x)] = x
        o += len(x)
    return Xall, np.concatenate(ys)


def main():
    import torch

    from model import RouteTransformer

    ap = argparse.ArgumentParser()
    ap.add_argument("--packed")
    ap.add_argument("--train-samples", type=int, default=1_200_000)
    ap.add_argument("--val-samples", type=int, default=100_000)
    ap.add_argument("--max-test-samples", type=int, default=0, help="0 = whole test split")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--ckpt", default=str(T.ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.packed:
        os.environ["PACKED_DIR"] = a.packed
    pk = T.Packed()
    rng = np.random.default_rng(0)
    split = pk.SM["split"]
    pick = lambda s, n: np.sort(rng.choice(np.flatnonzero(split == s), min(n, int((split == s).sum())), replace=False))
    tr, va = pick(0, a.train_samples), pick(1, a.val_samples)
    te = np.flatnonzero(split == 2)
    if a.max_test_samples:
        te = np.sort(rng.choice(te, min(a.max_test_samples, len(te)), replace=False))

    Xtr, ytr = build(pk, tr, label="train")
    Xva, yva = build(pk, va, label="val")
    n_train_rows = len(ytr)
    print(f"[lgbm] train rows {n_train_rows:,} x {Xtr.shape[1]} features, val rows {len(yva):,}", flush=True)
    cat_idx = [FEATURES.index(c) for c in CAT_FEATS]
    params = dict(objective="l1", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, cat_smooth=50,
                  min_data_per_group=200, max_cat_threshold=64, num_threads=10, verbose=-1)
    dtr = lgb.Dataset(Xtr, ytr, feature_name=FEATURES, categorical_feature=cat_idx, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=True)
    t0 = time.time()
    booster = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    del Xtr, ytr, dtr
    print(f"[lgbm] trained {booster.best_iteration} rounds in {(time.time() - t0) / 60:.0f} min", flush=True)
    os.makedirs(T.ROOT / "checkpoints", exist_ok=True)
    booster.save_model(str(T.ROOT / "checkpoints" / f"lgbm{a.tag}.txt"))
    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])

    # ---- identical test rows for transformer, LightGBM and the blend -----
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = None
    if os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location=dev)
        model = RouteTransformer(**ck["config"]).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
    names = ["lgbm", "transformer", "blend", "persistence", "persist_or_history"]
    tot = {n: 0.0 for n in names}
    hz = {f"{lo:g}..{hi:g}": {n: [0, 0.0] for n in names} for lo, hi in T.HORIZONS}
    n_rows = 0
    t0 = time.time()
    for i in range(0, len(te), 2048):
        b = pk.batch(te[i:i + 2048])
        X, e = flatten(b)
        p_l = e["base"] + booster.predict(X, num_iteration=booster.best_iteration)
        preds = {"lgbm": p_l, "persistence": e["base"],
                 "persist_or_history": np.where(e["known"], e["base"], np.where(np.isnan(e["h_med"]), e["base"], e["h_med"]))}
        if model is not None:
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                pt = model(T.to_dev({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}, dev))
            p_t = pt.float()[..., 1].cpu().numpy()[e["bi"], e["li"]]
            preds["transformer"], preds["blend"] = p_t, 0.5 * (p_t + p_l)
        y = e["y"]
        n_rows += len(y)
        for n, p in preds.items():
            err = np.abs(p - y)
            tot[n] += float(err.sum())
            for lo, hi in T.HORIZONS:
                s = (e["horizon"] >= lo) & (e["horizon"] < hi)
                h = hz[f"{lo:g}..{hi:g}"][n]
                h[0] += int(s.sum())
                h[1] += float(err[s].sum())
        if (i // 2048) % 100 == 0:
            print(f"[lgbm] test {i:,}/{len(te):,} ({time.time() - t0:.0f}s)", flush=True)
    res = {"test_rows": n_rows, "mae": {n: tot[n] / max(n_rows, 1) for n in names if model is not None or n not in ("transformer", "blend")},
           "by_horizon_mae": {k: {n: v[n][1] / v[n][0] for n in names if v[n][0] and v[n][1]} for k, v in hz.items()},
           "lgbm_rounds": booster.best_iteration, "train_rows": n_train_rows,
           "top_features_by_gain": [[f, float(g)] for f, g in imp[:30]]}
    with open(T.REPORTS / f"lgbm_test{a.tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    with open(T.REPORTS / f"comparison{a.tag}.json", "w") as f:
        json.dump({"test_mae_minutes": res["mae"], "by_horizon_mae": res["by_horizon_mae"]}, f, indent=1)
    print("[lgbm] test MAE (minutes):", json.dumps(res["mae"], indent=1), flush=True)


if __name__ == "__main__":
    main()

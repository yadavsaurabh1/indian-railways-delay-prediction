"""How much does the other-train information help? Scores the trained model on the
test months with parts of the context removed (no retraining).

Variants:
  full                    everything (the normal model)
  no_other_trains         all context removed: only this train's own journey + history
  no_rake_link            only the incoming paired train (slot 6) removed
  no_route_traffic        trains around the next 6 halts removed, rake link kept
  half_other_trains       a random half of the context dropped (simulates a sparse live_station)

Writes reports/ablation_test.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import train as T
from model import RouteTransformer

VARIANTS = ["full", "no_other_trains", "no_rake_link", "no_route_traffic", "half_other_trains"]


def masked(loader, variant: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    for b in loader:
        pad = b["ctx_pad"]
        slot = b["ctx_cat"][..., 0]
        if variant == "no_other_trains":
            pad = torch.ones_like(pad)
        elif variant == "no_rake_link":
            pad = pad | (slot == 6)
        elif variant == "no_route_traffic":
            pad = pad | (slot < 6)
        elif variant == "half_other_trains":
            pad = pad | torch.from_numpy(rng.random(pad.shape) < 0.5)
        b["ctx_pad"] = pad
        yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed")
    ap.add_argument("--ckpt", default=str(T.ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--max-samples", type=int, default=0, help="0 = whole test split")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--out", default=str(T.REPORTS / "ablation_test.json"))
    a = ap.parse_args()
    if a.packed:
        os.environ["PACKED_DIR"] = a.packed
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pk = T.Packed()
    idx = np.flatnonzero(pk.SM["split"] == {"val": 1, "test": 2}[a.split])
    if a.max_samples:
        idx = np.sort(np.random.default_rng(0).choice(idx, min(a.max_samples, len(idx)), replace=False))
    ck = torch.load(a.ckpt, map_location=dev)
    model = RouteTransformer(**ck["config"]).to(dev)
    model.load_state_dict(ck["model"])
    print(f"[ablation] {len(idx):,} {a.split} samples, checkpoint step {ck.get('step')}", flush=True)
    res = {}
    for v in VARIANTS:
        t0 = time.time()
        loader = DataLoader(T.Batches(idx, a.bs, False, 0.0, 0), batch_size=None, num_workers=3)
        res[v] = T.evaluate(model, masked(loader, v), dev, torch.bfloat16)
        print(f"[ablation] {v:<20} MAE {res[v]['mae_model']:.2f} min | pinball {res[v]['pinball_h']:.4f} | "
              f"10-90 coverage {res[v]['coverage_10_90']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    full = res["full"]["mae_model"]
    summary = {v: {"mae": res[v]["mae_model"], "mae_increase_vs_full_pct": 100 * (res[v]["mae_model"] / full - 1)}
               for v in VARIANTS}
    summary["baselines"] = {"persistence": res["full"]["mae_persistence"],
                            "persist_or_history": res["full"]["mae_persist_or_history"]}
    by_h = {h: {v: res[v]["by_horizon_min"].get(h, {}).get("mae_model") for v in VARIANTS}
            for h in res["full"]["by_horizon_min"]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "by_horizon_mae": by_h, "detail": res}, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()

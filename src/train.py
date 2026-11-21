"""Train the route transformer on the Apple GPU (MPS), evaluating against baselines.

  uv run python train.py                 # full run
  uv run python train.py --smoke         # few hundred steps, to check everything wires up
  uv run python train.py --resume       # continue from checkpoints/last.pt
  uv run python train.py --eval-only     # test-set report for checkpoints/best.pt

Split by journey date: train < 2025-09-01 <= val < 2026-03-01 <= test.
Writes logs/train_metrics.csv, checkpoints/{last,best}.pt, reports/eval_{val,test}.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from common import REPORTS, ROOT
from features import STOP_NUM, build_batch
from model import RouteTransformer, pinball
from pack import PACKED

CKPT = ROOT / "checkpoints"
LOGS = ROOT / "logs"
HORIZONS = [(-1e9, 0), (0, 30), (30, 60), (60, 120), (120, 240), (240, 480), (480, 960), (960, 1e9)]
JOURNEY_KEYS = ["train_id", "type_id", "pair_id", "day", "origin_sch_dep", "hol_is", "hol_to_next", "hol_since_prev"]


class Packed:
    """Read-only view of data/packed, opened lazily (safe to use in DataLoader workers)."""

    def __init__(self, root=None):
        root = Path(root or os.environ.get("PACKED_DIR", PACKED))
        self.meta = json.loads((root / "meta.json").read_text())
        self.S = np.load(root / "stops.npy", mmap_mode="r")
        self.C = np.load(root / "context.npy", mmap_mode="r")
        self.J = dict(np.load(root / "journeys.npz"))
        self.SM = dict(np.load(root / "samples.npz"))
        # stop-to-stop traffic (pack.py --segments-only); optional so older packs still load
        self.G = self.TN = None
        if (root / "segtok_num.npy").exists():
            self.G = np.load(root / "segments.npy", mmap_mode="r")
            self.GI = dict(np.load(root / "seg_index.npz"))
            self.TN = np.load(root / "segtok_num.npy", mmap_mode="r")
            self.TC = np.load(root / "segtok_cat.npy", mmap_mode="r")
            self.TI = dict(np.load(root / "segtok_index.npz"))

    def batch(self, idx: np.ndarray, rng=None, ctx_drop_max=0.0):
        sm = {k: v[idx] for k, v in self.SM.items()}
        j = sm["j"]
        so, sn = self.J["stop_off"][j], self.J["n_stops"][j]
        stops = [self.S[a:a + n] for a, n in zip(so, sn)]
        ctx = [self.C[a:a + n] for a, n in zip(sm["ctx_off"], sm["ctx_n"])]
        segs = toks = None
        if self.G is not None:
            segs = [self.G[a:a + n] for a, n in zip(self.GI["off"][idx], self.GI["n"][idx])]
            toks = [(self.TN[a:a + n], self.TC[a:a + n]) for a, n in zip(self.TI["off"][idx], self.TI["n"][idx])]
        info = {k: sm[k] for k in ("t", "cur_delay", "cur_known", "cur_age", "last_known_idx")}
        info |= {k: self.J[k][j] for k in JOURNEY_KEYS}
        info["n_stops"] = sn
        return build_batch(stops, ctx, info, rng=rng, ctx_drop_max=ctx_drop_max, segs=segs, toks=toks)


class Batches(IterableDataset):
    def __init__(self, indices: np.ndarray, batch_size: int, shuffle: bool, ctx_drop_max: float, seed: int,
                 epochs: int = 1):
        self.indices, self.bs, self.shuffle = indices, batch_size, shuffle
        self.ctx_drop_max, self.seed, self.epochs = ctx_drop_max, seed, epochs

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        pk = Packed()
        lens = np.minimum(pk.J["n_stops"][pk.SM["j"][self.indices]], 128)
        for ep in range(self.epochs):
            rng = np.random.default_rng(self.seed + 1000 * ep)
            if self.shuffle:
                # length bucketing: shuffle, sort chunks of 64 batches by journey length, cut into
                # batches, then shuffle batch order. Padding cost drops ~3x, randomness is kept.
                perm = rng.permutation(len(self.indices))
                batches = []
                chunk = 64 * self.bs
                for c in range(0, len(perm), chunk):
                    p = perm[c:c + chunk]
                    p = p[np.argsort(lens[p], kind="stable")]
                    batches += [self.indices[p[i:i + self.bs]] for i in range(0, len(p), self.bs)]
                batches = [batches[i] for i in rng.permutation(len(batches))]
            else:
                order = self.indices
                batches = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
            wrng = np.random.default_rng(self.seed + 1000 * ep + wid + 1)
            for bi in range(wid, len(batches), nw):
                # sorted indices -> more sequential reads from the memory-mapped arrays
                yield {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in
                       pk.batch(np.sort(batches[bi]), rng=wrng if self.shuffle else None,
                                ctx_drop_max=self.ctx_drop_max).items()}


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, dev, amp_dtype, max_batches=None) -> dict:
    model.eval()
    acc = {"n": 0, "loss": 0.0, "ae_model": 0.0, "ae_persist": 0.0, "ae_hist": 0.0, "ae_smart": 0.0,
           "cover": 0.0, "width": 0.0, "h": {i: [0, 0.0, 0.0, 0.0] for i in range(len(HORIZONS))}}
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        b = to_dev(b, dev)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(b)
        m = b["tmask"]
        y = b["target"][m]
        p = pred[m].float()
        n = int(m.sum())
        base = b["base"].unsqueeze(1).expand_as(b["target"])[m]
        hist = b["h_med_raw"][m]
        known = b["cur_known"].unsqueeze(1).expand_as(b["target"])[m]
        smart = torch.where(known, base, torch.where(torch.isnan(hist), base, hist))
        acc["n"] += n
        acc["loss"] += float(pinball(pred.float(), b["target"], m)) * n
        acc["ae_model"] += float((p[:, 1] - y).abs().sum())
        acc["ae_persist"] += float((base - y).abs().sum())
        acc["ae_hist"] += float((torch.where(torch.isnan(hist), base, hist) - y).abs().sum())
        acc["ae_smart"] += float((smart - y).abs().sum())
        acc["cover"] += float(((y >= p[:, 0]) & (y <= p[:, 2])).float().sum())
        acc["width"] += float((p[:, 2] - p[:, 0]).sum())
        h = b["horizon"][m]
        for k, (lo, hi) in enumerate(HORIZONS):
            sel = (h >= lo) & (h < hi)
            acc["h"][k][0] += int(sel.sum())
            acc["h"][k][1] += float((p[sel, 1] - y[sel]).abs().sum())
            acc["h"][k][2] += float((base[sel] - y[sel]).abs().sum())
            acc["h"][k][3] += float((smart[sel] - y[sel]).abs().sum())
    model.train()
    n = max(acc["n"], 1)
    out = {"stops": acc["n"], "pinball_h": acc["loss"] / n, "mae_model": acc["ae_model"] / n,
           "mae_persistence": acc["ae_persist"] / n, "mae_history": acc["ae_hist"] / n,
           "mae_persist_or_history": acc["ae_smart"] / n, "coverage_10_90": acc["cover"] / n,
           "mean_interval_width": acc["width"] / n, "by_horizon_min": {}}
    for k, (lo, hi) in enumerate(HORIZONS):
        c, am, ap, asm = acc["h"][k]
        if c:
            out["by_horizon_min"][f"{lo:g}..{hi:g}"] = {"stops": c, "mae_model": am / c, "mae_persistence": ap / c,
                                                        "mae_persist_or_history": asm / c}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=4e-4, help="peak learning rate")
    ap.add_argument("--schedule", choices=["cosine", "adaptive", "constant"], default="cosine",
                    help="adaptive: hold the peak rate after warm-up; whenever validation has not improved for "
                         "--patience checks, go back to the best weights and halve the rate; stop once the rate "
                         "has been halved --max-cuts times and validation still stalls. "
                         "constant: hold --lr after warm-up, never cut, never stop early")
    ap.add_argument("--reset-lr", action="store_true",
                    help="with --resume: forget the saved rate / adaptive state and ramp to --lr over 500 steps")
    ap.add_argument("--weights-from", help="with --resume: replace the resumed weights with these (e.g. best.pt)")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max-cuts", type=int, default=5)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ctx-layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cumulative", action="store_true",
                    help="predict the delay change per stretch and add them up along the route")
    ap.add_argument("--micro-bs", type=int, default=0,
                    help="split each batch into chunks of this size (gradient accumulation; same maths, less memory)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after this many steps but keep the full-run LR schedule (scaling probes)")
    ap.add_argument("--run", default="", help="run name: checkpoints_<run>/, train_metrics_<run>.csv, eval_*_<run>.json")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--val-batches", type=int, default=200)
    ap.add_argument("--ctx-drop", type=float, default=0.5)
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto",
                    help="cpu: for code checks while the GPU is busy with another run")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--packed", help="packed data dir (default data/packed)")
    ap.add_argument("--ckpt-dir", help="checkpoint dir (default checkpoints/, smoke: checkpoints_smoke/)")
    a = ap.parse_args()
    global CKPT
    sfx = "_smoke" if a.smoke else (f"_{a.run}" if a.run else "")
    CKPT = Path(a.ckpt_dir) if a.ckpt_dir else (ROOT / f"checkpoints{sfx}" if sfx else CKPT)
    micro = a.micro_bs or a.bs
    if a.packed:
        os.environ["PACKED_DIR"] = a.packed  # inherited by DataLoader workers

    CKPT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    dev = torch.device(a.device if a.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if dev.type == "cpu" and a.amp != "none":
        a.amp = "none"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    pk = Packed()
    meta = pk.meta
    split = pk.SM["split"]
    tr_idx, va_idx, te_idx = (np.flatnonzero(split == i) for i in range(3))
    if a.smoke:
        rng = np.random.default_rng(0)
        if len(tr_idx) == 0 or len(va_idx) == 0:  # tiny code-check packs may hold a single split
            tr_idx = va_idx = np.arange(len(split))
        tr_idx = rng.choice(tr_idx, min(20_000, len(tr_idx)), replace=False)
        va_idx = rng.choice(va_idx, min(5_000, len(va_idx)), replace=False)
        a.epochs, a.eval_every, a.val_batches = 1, 20, 5
    print(f"device={dev} amp={a.amp} train={len(tr_idx):,} val={len(va_idx):,} test={len(te_idx):,}", flush=True)

    cfg = dict(d=a.d, heads=a.heads, layers=a.layers, ctx_layers=a.ctx_layers, ff=a.ff, dropout=a.dropout,
               cumulative=a.cumulative, tokens=pk.G is not None)
    if a.eval_only:
        cfg = torch.load(CKPT / "best.pt", map_location="cpu")["config"]
        cfg = {k: v for k, v in cfg.items() if k not in ("n_station", "n_train", "n_type")}
    model = RouteTransformer(meta["n_station_ids"], meta["n_train_ids"], meta["n_type_ids"], **cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  config {cfg}  micro-batch {micro}  "
          f"stop features {len(STOP_NUM)}  stop-to-stop traffic {'yes' if pk.G is not None else 'NO'}", flush=True)
    dl_kw = dict(batch_size=None, num_workers=a.workers, persistent_workers=a.workers > 0,
                 prefetch_factor=4 if a.workers > 0 else None)
    # fixed validation subset: always val_batches x 512 samples, so runs of any batch size are comparable
    va_fixed = np.random.default_rng(1).choice(va_idx, min(len(va_idx), a.val_batches * 512), replace=False)
    val_loader = DataLoader(Batches(np.sort(va_fixed), micro, False, 0.0, 0), **{**dl_kw, "persistent_workers": False})

    if a.eval_only:
        ck = torch.load(CKPT / "best.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        for name, idx in (("val", va_idx), ("test", te_idx)):
            res = evaluate(model, DataLoader(Batches(idx, micro, False, 0.0, 0), **{**dl_kw, "persistent_workers": False}),
                           dev, amp_dtype)
            (REPORTS / f"eval_{name}{sfx}.json").write_text(json.dumps(res, indent=1))
            print(name, json.dumps({k: v for k, v in res.items() if k != "by_horizon_min"}), flush=True)
        return

    steps_per_epoch = math.ceil(len(tr_idx) / a.bs)
    total = steps_per_epoch * a.epochs
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "emb" in n_ or n_.startswith(("station", "pos", "slot", "dow")) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98))
    warm = max(1, min(2000, total // 10))
    ad = {"scale": 1.0, "bad": 0, "cuts": 0}  # adaptive-schedule state, saved in checkpoints
    rewarm = {"from": None}  # --reset-lr: step at which the ramp to the new rate starts

    def lr_factor(s: int) -> float:
        w = min(1.0, (s + 1) / warm)
        if rewarm["from"] is not None:
            w = min(1.0, (s - rewarm["from"] + 1) / 500)
        if a.schedule == "adaptive":
            return w * ad["scale"]
        if a.schedule == "constant":
            return w
        return w * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    step, best, start_epoch, resume_step = 0, float("inf"), 0, 0
    if a.resume and (CKPT / "last.pt").exists():
        ck = torch.load(CKPT / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]
        ad.update(ck.get("adaptive", {}))
        start_epoch, resume_step = step // steps_per_epoch, step
        if a.weights_from:
            model.load_state_dict(torch.load(a.weights_from, map_location=dev)["model"])
            print(f"weights replaced from {a.weights_from}", flush=True)
        if a.reset_lr:
            ad.update(scale=1.0, bad=0, cuts=0)
            rewarm["from"] = step
            sched.base_lrs = [a.lr for _ in opt.param_groups]
            for g in opt.param_groups:
                g["initial_lr"] = a.lr
                g["lr"] = a.lr * lr_factor(step)
            print(f"learning rate reset: ramping to {a.lr:g} over 500 steps ({a.schedule})", flush=True)
        elif a.schedule == "constant":
            # keep the rate the resumed run had reached, without further cuts
            sched.base_lrs = [a.lr for _ in opt.param_groups]
        print(f"resumed at step {step} (epoch {start_epoch}), best val pinball {best:.4f}, lr scale {ad['scale']:g}",
              flush=True)
    if a.resume and rewarm["from"] is None and (CKPT / "last.pt").exists() and ck.get("rewarm") is not None:
        rewarm["from"] = ck["rewarm"]  # a crash-resume inside a re-warm keeps the same ramp

    mpath = LOGS / f"train_metrics{sfx}.csv"
    metrics_f = open(mpath, "a")
    if os.path.getsize(mpath) == 0:
        metrics_f.write("step,epoch,train_pinball,val_pinball,val_mae_model,val_mae_persist_or_history,val_coverage,lr,elapsed_s\n")
    model.train()
    t0, run_loss, run_n, gn_sum, gn_clip = time.time(), 0.0, 0, 0.0, 0
    stop_early = False
    for ep in range(start_epoch, a.epochs):
        # when resuming mid-epoch, use a fresh shuffle (not the one already partly seen) and stop at the
        # epoch's step budget
        seed = ep if ep != start_epoch or resume_step == 0 else ep + 1_000_003 * resume_step
        loader = DataLoader(Batches(tr_idx, a.bs, True, a.ctx_drop, seed=seed % 2**32),
                            **{**dl_kw, "persistent_workers": False})
        for b in loader:
            if step >= (ep + 1) * steps_per_epoch or (a.max_steps and step >= a.max_steps):
                break
            b = to_dev(b, dev)
            opt.zero_grad(set_to_none=True)
            n_tot = b["tmask"].sum().clamp_min(1)
            loss, bad = 0.0, False
            for c in range(0, b["tmask"].shape[0], micro):
                mb = {k: v[c:c + micro] for k, v in b.items()}
                with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    pred = model(mb)
                # weight each chunk by its share of target stops -> identical to the full-batch mean
                lc = pinball(pred.float(), mb["target"], mb["tmask"]) * (mb["tmask"].sum() / n_tot)
                if not torch.isfinite(lc):
                    bad = True
                    break
                lc.backward()
                loss = loss + lc.detach()
            if bad:
                print(f"non-finite loss at step {step}, skipping batch", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            sched.step()
            step += 1
            run_loss += float(loss)
            run_n += 1
            gn_sum += gn
            gn_clip += gn > 1.0
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{total} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"grad {gn_sum / 100:.2f} (clipped {gn_clip}%) "
                      f"{(time.time() - t0) / 60:.1f}min ({100 * (step % steps_per_epoch) / steps_per_epoch:.1f}% of epoch)",
                      flush=True)
                gn_sum, gn_clip = 0.0, 0
            if step % a.eval_every == 0 or step == total or step == a.max_steps:
                v = evaluate(model, val_loader, dev, amp_dtype)
                tl = run_loss / max(run_n, 1)
                run_loss, run_n = 0.0, 0
                improved = v["pinball_h"] < best
                # adaptive schedule: an improvement must be worth at least 0.1% to reset the patience count
                if a.schedule == "adaptive" and step > warm:
                    if v["pinball_h"] < best * (1 - 1e-3):
                        ad["bad"] = 0
                    else:
                        ad["bad"] += 1
                best = min(best, v["pinball_h"])
                if improved:
                    torch.save({"model": model.state_dict(), "config": model.config, "step": step, "val": v},
                               CKPT / "best.pt")
                cut_msg = ""
                if a.schedule == "adaptive" and ad["bad"] >= a.patience:
                    if ad["cuts"] >= a.max_cuts:
                        stop_early = True
                        cut_msg = f" | stalled after {ad['cuts']} rate cuts -> stopping"
                    else:
                        # go back to the best weights and continue more carefully
                        model.load_state_dict(torch.load(CKPT / "best.pt", map_location=dev)["model"])
                        ad["scale"] *= 0.5
                        ad["cuts"] += 1
                        ad["bad"] = 0
                        sched.step(step)  # apply the new scale now (same step count)
                        cut_msg = f" | stalled -> back to best weights, lr x{ad['scale']:g}"
                state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "step": step, "best": best, "config": model.config, "args": vars(a), "adaptive": ad,
                         "rewarm": rewarm["from"]}
                torch.save(state, CKPT / "last.pt")
                metrics_f.write(f"{step},{ep},{tl:.5f},{v['pinball_h']:.5f},{v['mae_model']:.3f},"
                                f"{v['mae_persist_or_history']:.3f},{v['coverage_10_90']:.4f},"
                                f"{sched.get_last_lr()[0]:.3e},{time.time() - t0:.0f}\n")
                metrics_f.flush()
                print(f"== val @ {step}: pinball {v['pinball_h']:.4f} | MAE model {v['mae_model']:.2f} min vs "
                      f"persistence {v['mae_persistence']:.2f} / persist-or-history {v['mae_persist_or_history']:.2f} | "
                      f"10-90 coverage {v['coverage_10_90']:.3f} {'*best*' if improved else ''}{cut_msg}", flush=True)
                if stop_early:
                    break
        if stop_early:
            break
        if a.max_steps and step >= a.max_steps:
            print(f"probe done: stopped at step {step} (--max-steps); skipping the full val/test report", flush=True)
            return
    print("training done; evaluating best checkpoint on full val + test", flush=True)
    ck = torch.load(CKPT / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    for name, idx in (("val", va_idx), ("test", te_idx)):
        if a.smoke:
            idx = idx[:5000]
        res = evaluate(model, DataLoader(Batches(idx, micro, False, 0.0, 0), **{**dl_kw, "persistent_workers": False}),
                       dev, amp_dtype)
        (REPORTS / f"eval_{name}{sfx}.json").write_text(json.dumps(res, indent=1))
        print(name, json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()

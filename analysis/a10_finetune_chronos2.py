"""Step 10 - light fine-tuning (LoRA) of Chronos-2 on SAFEX-D reported trajectories, with a strict train / validation / TEST separation by traffic group.
Protocol fixed in advance (no search of any hyper-parameter, nothing selected on the test fold):
  train       episodes of folds 0,1,2   (stratified sample: mode x family)          -> the only data the model is adapted on
  validation  episodes of fold 3        (only used to print the validation error before / after, NOT to select anything)
  TEST        fold 4                    (never read here; evaluated by a5 --fold 4 --ckpt ... then a8 / a9)
  series      per UAV, REPORTED x, y, altitude at 5 s steps, forecast in INCREMENTS (the representation that passes the straight-line test); 120 s context (24 increments), 60 s horizon (12)
  fit         LoRA r=8 (library default), learning rate 1e-5, 1500 steps, batch 192, seed 0 - values written here before running, not tuned
  python a10_finetune_chronos2.py DATA_DIR --feats FEAT_DIR --splits OUT/splits.csv -o OUT/ft_chronos2 [--steps 1500] [--device cuda]
  (needs:  pip install peft)
After fitting it (1) reloads the saved checkpoint and checks that it reproduces the in-memory model, (2) prints the held-out validation error (m) of the pre-trained and the fine-tuned model."""
import argparse, json, os, numpy as np, pandas as pd
from a5_foundation_model import make_predictor, cv_predict, STEP, C, H

def series_of(path, rng, n_uav, min_len):
    """-> list of (3, L-1) float32 increment arrays (one per sampled UAV): longest finite run of the reported (x, y, alt) sampled every STEP s (random phase)."""
    d = pd.read_parquet(path, columns=["ep_t", "uav", "rep_x_m", "rep_y_m", "rep_alt_m"]); T = int(d.ep_t.max()) + 1; N = int(d.uav.max()) + 1
    A = np.full((T, N, 3), np.nan, np.float32); A[d.ep_t.values.astype(int), d.uav.values] = d[["rep_x_m", "rep_y_m", "rep_alt_m"]].values
    ok = np.where(np.isfinite(A[:, :, 0]).sum(0) >= min_len * STEP)[0]
    if len(ok) == 0: return []
    out = []
    for i in rng.choice(ok, min(n_uav, len(ok)), replace=False):
        s = A[rng.integers(0, STEP)::STEP, i]; f = np.isfinite(s[:, 0])
        for k in range(1, len(s)):                                   # forward-fill gaps of at most 2 steps (as a5 does for short gaps)
            if not f[k] and f[max(k - 3, 0):k].any() and np.isfinite(s[k - 1, 0]): s[k] = s[k - 1]; f[k] = True
        best, cur, st = (0, 0), 0, 0
        for k in range(len(s) + 1):
            if k < len(s) and f[k]: cur += 1
            else:
                if cur > best[1] - best[0]: best = (k - cur, k)
                cur = 0
        r = s[best[0]:best[1]]
        if len(r) >= min_len: out.append(np.diff(r, axis=0).T.astype(np.float32))
    return out

def pick(sp, folds, n, seed):
    sp = sp[sp.fold.isin(folds)]; per = max(1, n // max(1, sp.groupby(["mode", "family"]).ngroups))
    return sp.groupby(["mode", "family"], group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=seed))

def windows(ser, rng, n):
    """random (context 25 positions, true next H positions) windows from increment series, positions rebuilt by cumulative sum from an arbitrary origin"""
    X, Y = [], []
    cand = [s for s in ser if s.shape[1] >= C + H]
    for _ in range(n):
        s = cand[rng.integers(len(cand))]; e = rng.integers(C, s.shape[1] - H + 1); pos = np.concatenate([np.zeros((3, 1), np.float32), np.cumsum(s[:, e - C:e + H], axis=1)], 1).T
        X.append(pos[:C + 1]); Y.append(pos[C + 1:])
    return np.array(X), np.array(Y)

def ade(pred, y): return float(np.hypot(pred[..., 0] - y[..., 0], pred[..., 1] - y[..., 1]).mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--feats", required=True); ap.add_argument("--splits", required=True); ap.add_argument("-o", default="ft_chronos2")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--train-episodes", type=int, default=72); ap.add_argument("--val-episodes", type=int, default=12); ap.add_argument("--uavs-per-ep", type=int, default=150)
    ap.add_argument("--train-folds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--val-fold", type=int, default=3); a = ap.parse_args()
    rng = np.random.default_rng(0); sp = pd.read_csv(a.splits); sp = sp[sp.ep.map(lambda e: os.path.exists(os.path.join(a.dir, e + ".parquet")))]
    assert 4 not in a.train_folds and a.val_fold not in a.train_folds, "the test fold (4) and the validation fold must not be in the training folds"
    tr_ep = pick(sp, a.train_folds, a.train_episodes, 0); va_ep = pick(sp, [a.val_fold], a.val_episodes, 0)
    print(f"train: {len(tr_ep)} episodes ({sorted(tr_ep.fold.unique())}), validation: {len(va_ep)} episodes (fold {a.val_fold}); TEST fold 4 is not touched")
    tr = [s for e in tr_ep.ep for s in series_of(os.path.join(a.dir, e + ".parquet"), rng, a.uavs_per_ep, C + H + 1)]
    va = [s for e in va_ep.ep for s in series_of(os.path.join(a.dir, e + ".parquet"), rng, a.uavs_per_ep, C + H + 1)]
    print(f"{len(tr):,} training series (median length {int(np.median([s.shape[1] for s in tr]))} increments), {len(va):,} validation series")
    import torch
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=a.device)
    ft = pipe.fit(inputs=[{"target": s} for s in tr], prediction_length=H, validation_inputs=[{"target": s} for s in va[:2000]], finetune_mode="lora", learning_rate=a.lr, num_steps=a.steps,
                  batch_size=a.batch, context_length=C, min_past=C, output_dir=a.o, finetuned_ckpt_name="finetuned-ckpt")
    ck = os.path.join(a.o, "finetuned-ckpt"); print("checkpoint:", ck)
    # (1) the saved checkpoint must reproduce the in-memory fine-tuned model;  (2) held-out validation error before / after
    Xv, Yv = windows(va, np.random.default_rng(1), 3000); Xv = Xv.astype(np.float32); base = make_predictor("chronos2_diff", a.device, 256); tuned = make_predictor("chronos2_diff", a.device, 256, ck)
    pb, pt = base(Xv), tuned(Xv); pc = cv_predict(Xv)
    mid = int(np.argmin(np.abs(np.asarray(ft.quantiles, dtype=float) - 0.5))); inc = np.diff(Xv, axis=1)
    q = ft.predict([{"target": r.T.astype(np.float32)} for r in inc[:64]], prediction_length=H); pm = Xv[:64, -1:, :] + np.cumsum(np.stack([p[:, mid, :].float().cpu().numpy() for p in q]).transpose(0, 2, 1), axis=1)
    print(f"reload check (max |in-memory - reloaded| on 64 windows): {float(np.abs(pm - pt[:64]).max()):.3f} m   (should be ~0; if large, the saved LoRA weights were not applied)")
    res = {"validation_ADE_m": {"pretrained": ade(pb, Yv), "fine-tuned": ade(pt, Yv), "constant-velocity (15 s)": ade(pc, Yv)}, "n_windows": len(Xv), "steps": a.steps, "lr": a.lr, "train_series": len(tr), "train_episodes": tr_ep.ep.tolist()}
    print("validation ADE over 60 s (m):", {k: round(v, 2) for k, v in res["validation_ADE_m"].items()})
    json.dump(res, open(os.path.join(a.o, "finetune_summary.json"), "w"), indent=1)
    print("next: python a5_foundation_model.py DATA --feats FEATS --splits SPLITS --model chronos2_diff --device cuda --fold 4 --ckpt", ck, "-o OUT/fm_chronos2_ft_fold4   (and the same without --ckpt for the pre-trained model)")

"""Step 5 - time-series FOUNDATION MODELS as zero-shot trajectory predictors, followed by the separation rule.
For every sampled instant t the model predicts the next 60 s (5 s steps) of the REPORTED (x, y, altitude) of every visible aircraft from its last 120 s.
The conflict score of an aircraft = - (minimum predicted horizontal distance to a vertically compatible neighbour within h seconds), i.e. exactly the physics rule,
but fed with the model's trajectories instead of a constant-velocity extrapolation. Compared on the SAME rows and labels as the benchmark (a2/a3 outputs).
  pip install torch "chronos-forecasting>=2.0"                       # Chronos-2 / Chronos-Bolt (weights downloaded from Hugging Face)
  pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"   # TimesFM 2.5 (see its README)
  python a5_foundation_model.py DATA_DIR --feats FEAT_DIR --splits OUT/splits.csv --model cv        -o OUT/fm_cv          # sanity check, no network
  python a5_foundation_model.py DATA_DIR --feats FEAT_DIR --splits OUT/splits.csv --model chronos2  --device cuda -o OUT/fm_chronos2
  python a5_foundation_model.py DATA_DIR --feats FEAT_DIR --splits OUT/splits.csv --model timesfm25 --device cuda -o OUT/fm_timesfm25
The same episodes (fixed seed) are used for every model, so the results are directly comparable."""
import argparse, json, os, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score as AP, roc_auc_score
from a_common import PROFILES, parse_ep
from a3_benchmark import boot, metrics

STEP, C, H = 5, 24, 12                          # prediction step (s), context length (steps), horizon (steps)  -> 120 s context, 60 s horizon
HS = {10: 2, 30: 6, 60: 12}

def cv_predict(ctx):                            # ctx (n, C+1, 3) -> (n, H, 3): constant velocity from the last 15 s
    v = (ctx[:, -1] - ctx[:, -4]) / 3.0
    return ctx[:, -1][:, None, :] + v[:, None, :] * np.arange(1, H + 1)[None, :, None]

def wrap(core, diff):
    """core(series (n, L, 3)) -> median forecast (n, H, 3) of the SAME quantity. diff=False: the model forecasts the positions relative to the last one;
    diff=True: it forecasts the per-step increments (velocity x STEP), integrated from the last position. Non-finite outputs fall back to constant velocity."""
    def f(ctx):
        last = ctx[:, -1:, :]
        if diff: pr = last + np.cumsum(core(np.diff(ctx, axis=1)), axis=1)
        else: pr = core(ctx - last) + last
        bad = ~np.isfinite(pr).all(axis=(1, 2))
        if bad.any(): pr[bad] = cv_predict(ctx[bad])
        return pr
    return f

def make_predictor(name, device, batch, ckpt=None):
    """name = cv | chronos2 | chronos_bolt | timesfm25, optionally with the suffix _diff (forecast increments instead of positions)."""
    diff = name.endswith("_diff"); base = name[:-5] if diff else name
    if base == "cv": return cv_predict
    import torch
    if base == "chronos2":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained(ckpt or "amazon/chronos-2", device_map=device)      # ckpt = a fine-tuned checkpoint (a10)
        mid = int(np.argmin(np.abs(np.asarray(pipe.quantiles, dtype=float) - 0.5)))        # index of the median quantile, read from the model itself
        def core(sr):
            out = []
            for i in range(0, len(sr), batch):
                pr = pipe.predict([{"target": r.T.astype(np.float32)} for r in sr[i:i + batch]], prediction_length=H)           # list of (3, n_quantiles, H)
                out.append(np.stack([p[:, mid, :].float().cpu().numpy() for p in pr]).transpose(0, 2, 1))
            return np.concatenate(out)
        return wrap(core, diff)
    if base == "chronos_bolt":
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
        def core(sr):
            out = []
            for i in range(0, len(sr), batch):
                c = sr[i:i + batch]; n = len(c)
                q, _ = pipe.predict_quantiles(context=torch.tensor(c.transpose(0, 2, 1).reshape(n * 3, -1), dtype=torch.float32), prediction_length=H, quantile_levels=[0.5])
                out.append(q[..., 0].numpy().reshape(n, 3, H).transpose(0, 2, 1))
            return np.concatenate(out)
        return wrap(core, diff)
    if base == "timesfm25":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
        m.compile(timesfm.ForecastConfig(max_context=64, max_horizon=H, per_core_batch_size=256, normalize_inputs=True, use_continuous_quantile_head=False, force_flip_invariance=False, infer_is_positive=False, fix_quantile_crossing=True))
        def core(sr):
            out = []
            for i in range(0, len(sr), batch):
                c = sr[i:i + batch]; n = len(c)
                pt, _ = m.forecast(horizon=H, inputs=[c[j, :, d].astype(np.float32) for j in range(n) for d in range(3)])
                out.append(np.asarray(pt)[:, :H].reshape(n, 3, H).transpose(0, 2, 1))
            return np.concatenate(out)
        return wrap(core, diff)
    raise SystemExit("unknown model " + name)

def scores_from_traj(P0, Pk, prof):
    """P0 (n,2) current positions, Pk (n,H,3) predicted (x,y,alt) at 5 s steps -> {h: score (n,)}"""
    d_lowc, _, _, htol = PROFILES[prof]; n = len(P0); res = {h: np.full(n, -1e9) for h in HS}
    vmax = max(float(np.abs(Pk[:, 0, :2] - P0).max() / STEP), 10.0); pr = cKDTree(P0).query_pairs(d_lowc + 2 * vmax * 60, output_type="ndarray")
    if len(pr) == 0: return res
    for h, s in HS.items():
        mn = np.full(len(pr), 1e9)
        for k in range(s):
            dh = np.hypot(*(Pk[pr[:, 0], k, :2] - Pk[pr[:, 1], k, :2]).T); dz = np.abs(Pk[pr[:, 0], k, 2] - Pk[pr[:, 1], k, 2]); mn = np.minimum(mn, np.where(dz < htol, dh, dh + 1000.0))
        best = np.full(n, 1e9); np.minimum.at(best, pr[:, 0], mn); np.minimum.at(best, pr[:, 1], mn); res[h] = -best
    return res

def run_episode(path, F, predict, tstride):
    d = pd.read_parquet(path, columns=["ep_t", "uav", "rep_x_m", "rep_y_m", "rep_alt_m", "x_m", "y_m", "alt_m"]); T = int(d.ep_t.max()) + 1; N = int(d.uav.max()) + 1
    A = np.full((T, N, 3), np.nan, np.float32); A[d.ep_t.values.astype(int), d.uav.values] = d[["rep_x_m", "rep_y_m", "rep_alt_m"]].values
    TR = np.full((T, N, 3), np.nan, np.float32); TR[d.ep_t.values.astype(int), d.uav.values] = d[["x_m", "y_m", "alt_m"]].values      # TRUE state, used ONLY to score trajectories
    ts = sorted(F.t.unique())[::tstride]; jobs = []; ctxs = []
    for t in ts:
        a = np.where(np.isfinite(A[t, :, 0]))[0]
        if len(a) < 3: continue
        ctx = np.empty((len(a), C + 1, 3), np.float32); full = np.zeros(len(a), bool)
        for j, i in enumerate(a):
            idx = t - STEP * np.arange(C, -1, -1); c = A[np.clip(idx, 0, T - 1), i].copy(); idx_ok = (idx >= 0) & np.isfinite(c[:, 0]); first = int(np.argmax(idx_ok)) if idx_ok.any() else 0
            full[j] = bool(idx_ok.all()); c[:first] = c[first]
            for k in range(1, C + 1):                              # forward-fill short gaps
                if not np.isfinite(c[k, 0]): c[k] = c[k - 1]
            ctx[j] = c
        jobs.append((t, a, full, len(ctxs))); ctxs.append(ctx)
    if not jobs: return None
    allc = np.concatenate(ctxs); allf = np.concatenate([j[2] for j in jobs]); pk = cv_predict(allc)
    if allf.any(): pk[allf] = predict(allc[allf])
    out = []; pos = 0
    for (t, a, full, _), ctx in zip(jobs, ctxs):
        n = len(a); P0 = ctx[:, -1, :2]; r = {"t": np.full(n, t), "uav": a}; pkk, cvk = pk[pos:pos + n], cv_predict(ctx)
        tru = np.stack([TR[t + STEP * k, a] if t + STEP * k < T else np.full((n, 3), np.nan, np.float32) for k in range(1, H + 1)], 1)      # (n, H, 3) true future
        r["fm_used"] = full
        for nm, pr_ in (("model", pkk), ("cvh", cvk)):
            eh = np.hypot(pr_[..., 0] - tru[..., 0], pr_[..., 1] - tru[..., 1]); ez = np.abs(pr_[..., 2] - tru[..., 2]); okk = np.isfinite(eh).all(1)
            r[f"ade_{nm}"] = np.where(okk, eh.mean(1), np.nan); r[f"adez_{nm}"] = np.where(okk, ez.mean(1), np.nan)
            for h_, s_ in HS.items(): r[f"fde{h_}_{nm}"] = np.where(okk, eh[:, s_ - 1], np.nan)
        for prof in PROFILES:
            sm, sc = scores_from_traj(P0, pkk, prof), scores_from_traj(P0, cvk, prof)
            for h in HS: r[f"model_{prof}_h{h}"] = sm[h]; r[f"cvh_{prof}_h{h}"] = sc[h]
        out.append(pd.DataFrame(r)); pos += n
    S = pd.concat(out, ignore_index=True)
    return F.merge(S, on=["t", "uav"], how="inner")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--feats", required=True); ap.add_argument("--splits", required=True); ap.add_argument("--model", default="cv")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=512); ap.add_argument("--episodes", type=int, default=24); ap.add_argument("--tstride", type=int, default=3, help="use every N-th sampled instant (3 = every 30 s)")
    ap.add_argument("--targets", nargs="+", default=["paper_tac", "paper_imm", "half_tac"]); ap.add_argument("--boot", type=int, default=100); ap.add_argument("--ckpt", default=None, help="fine-tuned Chronos-2 checkpoint directory (a10)"); ap.add_argument("--fold", type=int, default=None, help="use only the episodes of this fold (held-out test of a fine-tuned model)")
    ap.add_argument("-o", default="fm_results"); a = ap.parse_args()
    sp = pd.read_csv(a.splits); sp = sp[sp.ep.map(lambda e: os.path.exists(os.path.join(a.feats, e + ".parquet")))]
    if a.fold is not None: sp = sp[sp.fold == a.fold]
    per = max(1, a.episodes // 6); sp = sp.groupby(["mode", "family"], group_keys=False).sample(n=per, random_state=0) if len(sp) > a.episodes else sp
    predict = make_predictor(a.model, a.device, a.batch, a.ckpt); parts = []
    for e, g in zip(sp.ep, sp.group):
        F = pd.read_parquet(os.path.join(a.feats, e + ".parquet")); m = run_episode(os.path.join(a.dir, e + ".parquet"), F, predict, a.tstride)
        if m is not None: m["group"] = g; m["cls"] = sp.set_index("ep").cls[e]; m["mode"] = sp.set_index("ep")["mode"][e]; parts.append(m); print(e, len(m), "rows", flush=True)
    D = pd.concat(parts, ignore_index=True); D["maneuver"] = (D.raw_alt_chg_K.abs() > 5) | (D.raw_hd_chg_K > 0.1); rng = np.random.default_rng(0); R = {}
    for tgt in a.targets:
        prof, lv = tgt.split("_")
        for h in HS:
            y = f"y_{prof}_{lv}_h{h}"; R.setdefault(tgt, {})[str(h)] = {}
            for mode in ["free", "mvp", "all"]:
                M = D if mode == "all" else D[D["mode"] == mode]; M = M[M[y].notna()]
                for sub, mk in [("all", np.ones(len(M), bool)), ("maneuvering", M.maneuver.values)]:
                    yy = M[y].values[mk]
                    if yy.sum() < 20 or (1 - yy).sum() < 20: continue
                    r = {}
                    for name, col in [("constant-velocity (a2 builder, 5 s window)", f"cv_{prof}_h{h}"), ("constant-velocity (this harness, 15 s window)", f"cvh_{prof}_h{h}"), (a.model, f"model_{prof}_h{h}")]:
                        b, lo, hi = boot(yy, M[col].values[mk], M.group.values[mk], a.boot, rng); r[name] = dict(auprc=b[0], auprc_lo=lo[0], auprc_hi=hi[0], auroc=b[1], rec5=b[2], rec5_lo=lo[2], rec5_hi=hi[2], n=int(len(yy)), pos=int(yy.sum()))
                    R[tgt][str(h)].setdefault(mode, {})[sub] = r
            r = R[tgt][str(h)].get("all", {}).get("all", {})
            if r: print(f"{y:18s} AUPRC " + " | ".join(f"{n[:22]}={v['auprc']:.3f}" for n, v in r.items()), flush=True)
    # trajectory error (ADE/FDE, metres, against the TRUE future position), rows where the model was actually applied; group-bootstrap 95% CI of the mean
    TJ = {}; U = D[D.fm_used & D.ade_model.notna()]
    def mean_ci(v, g, B=a.boot):
        v, g = np.asarray(v, float), np.asarray(g); ug, inv = np.unique(g, return_inverse=True); ix = [np.where(inv == i)[0] for i in range(len(ug))]
        bs = [v[np.concatenate([ix[i] for i in rng.integers(0, len(ug), len(ug))])].mean() for _ in range(B)]; return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    for mode in ["free", "mvp", "all"]:
        M = U if mode == "all" else U[U["mode"] == mode]
        for sub, mk in [("all", np.ones(len(M), bool)), ("maneuvering", M.maneuver.values)]:
            if mk.sum() < 50: continue
            TJ.setdefault(mode, {})[sub] = {nm: {m_: mean_ci(M[f"{m_}_{('cvh' if nm == 'constant-velocity' else 'model')}"].values[mk], M.group.values[mk]) for m_ in ["ade", "adez", "fde10", "fde30", "fde60"]} | {"n": int(mk.sum())} for nm in ["constant-velocity", a.model]}
    R["trajectory_error_m"] = TJ
    for mode in TJ:
        r_ = TJ[mode]["all"]; print(f"ADE/FDE [{mode}] " + " | ".join(f"{k}: ADE {v['ade'][0]:.1f} m, FDE30 {v['fde30'][0]:.1f} m, FDE60 {v['fde60'][0]:.1f} m" for k, v in r_.items() if k != "n"), flush=True)
    json.dump(R, open(a.o + ".json", "w"), indent=1); D.drop(columns=["raw_alt_chg_K", "raw_hd_chg_K"]).to_parquet(a.o + "_scores.parquet", index=False); print("->", a.o + ".json")

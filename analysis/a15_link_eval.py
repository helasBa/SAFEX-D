"""Step 15 - how much does forecasting skill depend on the quality of the surveillance link?  (features from a14)
For every link setting s and every (mode, target):
  matched      models trained AND tested on setting s (5-fold grouped CV, folds of splits.csv): what is achievable if the model is trained on the link it will see
  base-trained models trained on the ideal link (setting 'base') and tested on setting s (same test fold): what happens when the link degrades after deployment
Models: constant-velocity rule (score computed from the degraded stream), learned (kinematics+neighbours), hybrid (rule score + learned). HistGradientBoosting, 150 iterations,
balanced class weights, no tuning (identical to a3). 95% CIs: bootstrap over traffic groups. Nothing is selected on the test fold.
  python a15_link_eval.py C:\\these\\safex2_link --splits C:\\these\\safex2_out\\splits.csv -o C:\\these\\safex2_out\\a15_link_eval.csv [--targets paper_tac] [--horizons 10 30] [--modes free mvp]
Resumable: finished (mode, target, horizon) blocks are skipped when the output file exists. Output columns: setting, kind, mode, target, horizon, model, auprc, auprc_lo, auprc_hi, auroc, n, pos, prevalence, age_mean, age_p90."""
import argparse, glob, os, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

def load(root, setting, sp):
    fs = sorted(glob.glob(os.path.join(root, setting, "*.parquet")))
    if not fs: raise SystemExit(f"no feature files in {os.path.join(root, setting)}")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).merge(sp[["ep", "cls", "family", "zone", "mode", "group", "fold"]], on="ep")

def boot_ap(y, s, g, B, rng):
    ug, inv = np.unique(g, return_inverse=True); ix = [np.where(inv == i)[0] for i in range(len(ug))]; v = []
    for _ in range(B):
        sel = np.concatenate([ix[i] for i in rng.integers(0, len(ug), len(ug))]); yy = y[sel]
        if yy.sum() >= 5 and (1 - yy).sum() >= 5: v.append(AP(yy, s[sel]))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) > 20 else (np.nan, np.nan)

def fit(tr, y, cols, it, max_train):
    tr = tr[tr[y].notna()]
    if len(tr) > max_train: tr = tr.sample(max_train, random_state=0)
    return HGB(max_iter=it, class_weight="balanced", random_state=0).fit(tr[cols], tr[y]) if tr[y].nunique() > 1 else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--splits", required=True); ap.add_argument("-o", default="a15_link_eval.csv")
    ap.add_argument("--targets", nargs="+", default=["paper_tac"]); ap.add_argument("--horizons", type=int, nargs="+", default=[10, 30]); ap.add_argument("--modes", nargs="+", default=["free", "mvp"])
    ap.add_argument("--iters", type=int, default=150); ap.add_argument("--max-train-rows", type=int, default=150000); ap.add_argument("--boot", type=int, default=100); a = ap.parse_args()
    sp = pd.read_csv(a.splits); settings = sorted([os.path.basename(d) for d in glob.glob(os.path.join(a.root, "*")) if os.path.isdir(d)], key=lambda s: (s != "base", s))
    print("settings:", settings); D = {s: load(a.root, s, sp) for s in settings}; base = D["base"]; rng = np.random.default_rng(0); t0 = time.time()
    done = pd.read_csv(a.o) if os.path.exists(a.o) else pd.DataFrame(); out = [done] if len(done) else []
    for tgt in a.targets:
        prof, lv = tgt.split("_")
        for h in a.horizons:
            y = f"y_{prof}_{lv}_h{h}"; cv = f"cv_{prof}_h{h}"
            for mode in a.modes:
                if len(done) and ((done["mode"] == mode) & (done.target == tgt) & (done.horizon == h)).any(): print("skip", mode, tgt, h); continue
                B = base[base["mode"] == mode]; folds = sorted(B.fold.unique()); cols = [c for c in B.columns if c.startswith(("raw_", "rel_", "seq_"))]; rows = []
                M = {s: D[s][D[s]["mode"] == mode] for s in settings}
                # models trained on the ideal link, one per fold
                bm = {f: (fit(B[B.fold != f], y, cols, a.iters, a.max_train_rows), fit(B[B.fold != f], y, [c for c in cols if not c.startswith("cv_")] + [cv], a.iters, a.max_train_rows)) for f in folds}
                
                for s in settings:
                    Ms = M[s]; hyb_cols = [c for c in Ms.columns if c.startswith(("raw_", "rel_", "seq_"))] + [cv]; lea_cols = [c for c in Ms.columns if c.startswith(("raw_", "rel_", "seq_"))]
                    lea_cols = [c for c in lea_cols if not c.startswith("cv_")]; hyb_cols = lea_cols + [cv]
                    for kind in ("matched", "base-trained"):
                        parts = {"constant-velocity rule": [], "learned": [], "hybrid": []}; tes = []
                        for f in folds:
                            te = Ms[(Ms.fold == f) & Ms[y].notna()]
                            if len(te) == 0 or te[y].sum() < 3: continue
                            if kind == "matched":
                                ml = fit(Ms[Ms.fold != f], y, lea_cols, a.iters, a.max_train_rows); mh = fit(Ms[Ms.fold != f], y, hyb_cols, a.iters, a.max_train_rows)
                            else:
                                ml, mh = bm[f]
                                if s == "base": continue                 # identical to 'matched' for the ideal link
                            if ml is None or mh is None: continue
                            parts["constant-velocity rule"].append(te[cv].fillna(-1e9).values); parts["learned"].append(ml.predict_proba(te[lea_cols])[:, 1]); parts["hybrid"].append(mh.predict_proba(te[hyb_cols])[:, 1]); tes.append(te)
                        if not tes: continue
                        te = pd.concat(tes, ignore_index=True); yy = te[y].values.astype(int)
                        for m, v in parts.items():
                            sc = np.concatenate(v); lo, hi = boot_ap(yy, sc, te.group.values, a.boot, rng)
                            rows.append(dict(setting=s, kind=kind, mode=mode, target=tgt, horizon=h, model=m, auprc=AP(yy, sc), auprc_lo=lo, auprc_hi=hi, auroc=AUC(yy, sc), n=len(yy), pos=int(yy.sum()),
                                             prevalence=float(yy.mean()), age_mean=float(te.raw_age.mean()), age_p90=float(te.raw_age.quantile(0.9))))
                    last = [r for r in rows if r["setting"] == s]
                    print(f"  [{time.time() - t0:5.0f}s] {mode} {y} {s:14s} " + " | ".join(f"{r['kind'][:4]}/{r['model'][:6]}={r['auprc']:.3f}" for r in last), flush=True)
                out.append(pd.DataFrame(rows)); pd.concat(out, ignore_index=True).to_csv(a.o, index=False)
    print("->", a.o)

"""Step 13 - forecasting under degraded links, calibration and explainability, from the out-of-fold predictions of the full feature set.
  python a13_link_quality.py C:\\these\\safex2_feats --splits C:\\these\\safex2_out\\splits.csv -o C:\\these\\safex2_out\\a13 [--horizons 10 30 60] [--modes free mvp] [--t-event 1800]
Folds = the traffic-group folds of splits.csv. Target: paper thresholds, LoWC (y_paper_tac_h*). Models (as in a3, nothing tuned): constant-velocity rule | learned (kinematics+neighbours) | hybrid |
hybrid without the report-age input (to see whether age information matters when the link is degraded). Out-of-fold scores are used for everything below.
Writes (CSV, prefix = -o):
  _age_strata.csv       AUPRC / AUROC (95% CI over traffic groups) by age of the last report: fresh (<1 s), 1-10 s, 10-30 s, >=30 s (outage and frozen reports)
  _degraded_ops.csv     operating point set on NOMINAL rows of the other folds (5% false alarms): recall and false-alarm rate per scenario class, rows after the event, and for stale reports
  _calibration.csv      Brier score and expected calibration error (10 equal-count bins), raw probability and isotonic recalibration fitted on the other folds
  _reliability.csv      reliability-diagram points (recalibrated hybrid and learned, raw hybrid)
  _perm_importance.csv  AUPRC lost when one group of inputs is shuffled in the held-out fold (physics score, own kinematics, report age, neighbours now, history)
"""
import argparse, glob, os, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, brier_score_loss
from a15_link_eval import boot_ap

BINS = [(0, 1, "fresh (<1 s)"), (1, 10, "1-10 s"), (10, 30, "10-30 s"), (30, 1e9, ">=30 s")]
MODELS = ["constant-velocity rule", "learned", "hybrid", "hybrid, no report age"]

def fit(tr, y, cols, it, mt):
    tr = tr[tr[y].notna()]
    if len(tr) > mt: tr = tr.sample(mt, random_state=0)
    return HGB(max_iter=it, class_weight="balanced", random_state=0).fit(tr[cols], tr[y])

def ece(p, y, nb=10):
    o = np.argsort(p); b = np.array_split(o, nb); return float(sum(len(i) * abs(p[i].mean() - y[i].mean()) for i in b if len(i)) / len(p))

def main(a):
    sp = pd.read_csv(a.splits)
    fs = sorted(glob.glob(os.path.join(a.feats, "*.parquet")))
    D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).merge(sp[["ep", "cls", "family", "zone", "mode", "group", "fold"]], on="ep")
    print(f"{len(D):,} rows, {D.ep.nunique()} episodes, {D.group.nunique()} groups", flush=True)
    rng = np.random.default_rng(0); t0 = time.time(); ST, OPS, CAL, REL, PI = [], [], [], [], []
    base_cols = [c for c in D.columns if c.startswith(("raw_", "rel_", "seq_"))]
    for h in a.horizons:
        y = f"y_paper_tac_h{h}"; cv = f"cv_paper_h{h}"
        for mode in a.modes:
            M = D[(D["mode"] == mode) & D[y].notna()].reset_index(drop=True); folds = sorted(M.fold.unique())
            c_h = base_cols + [cv]; c_na = [c for c in c_h if c != "raw_age"]
            groups = {"physics score": [cv], "own kinematics": [c for c in base_cols if c.startswith("raw_") and c != "raw_age"], "report age": ["raw_age"],
                      "neighbours (now)": [c for c in base_cols if c.startswith("rel_")], "history (lags)": [c for c in base_cols if c.startswith("seq_")]}
            sc = {m: np.zeros(len(M)) for m in MODELS}; sc["constant-velocity rule"] = M[cv].fillna(-1e9).values
            for f in folds:
                tr, ti = M[M.fold != f], np.where(M.fold == f)[0]; te = M.iloc[ti]
                mh = fit(tr, y, c_h, a.iters, a.max_train); ml = fit(tr, y, base_cols, a.iters, a.max_train); mn = fit(tr, y, c_na, a.iters, a.max_train)
                sc["learned"][ti] = ml.predict_proba(te[base_cols])[:, 1]; sc["hybrid"][ti] = mh.predict_proba(te[c_h])[:, 1]; sc["hybrid, no report age"][ti] = mn.predict_proba(te[c_na])[:, 1]
                if h in a.perm_horizons:
                    sub = te.sample(min(len(te), 150000), random_state=0); yy = sub[y].values.astype(int); b0 = AP(yy, mh.predict_proba(sub[c_h])[:, 1])
                    for gname, gc in groups.items():
                        d = []
                        for r in range(a.perm_repeats):
                            X = sub[c_h].copy(); X[gc] = X[gc].values[rng.permutation(len(X))]; d.append(b0 - AP(yy, mh.predict_proba(X)[:, 1]))
                        PI.append(dict(horizon=h, mode=mode, fold=f, group=gname, auprc_drop=float(np.mean(d)), base_auprc=b0))
                print(f"  [{time.time() - t0:5.0f}s] h={h} {mode} fold {f} done", flush=True)
            yv = M[y].values.astype(int); age = M.raw_age.fillna(0).values; g = M.group.values
            # 1. strata by age of the last report
            for lo, hi, lab in BINS:
                m = (age >= lo) & (age < hi)
                if yv[m].sum() < 30 or (1 - yv[m]).sum() < 30: continue
                for mname in MODELS:
                    l, u = boot_ap(yv[m], sc[mname][m], g[m], a.boot, rng)
                    ST.append(dict(horizon=h, mode=mode, age_bin=lab, model=mname, n=int(m.sum()), pos=int(yv[m].sum()), prevalence=float(yv[m].mean()), auprc=AP(yv[m], sc[mname][m]), auprc_lo=l, auprc_hi=u, auroc=AUC(yv[m], sc[mname][m])))
            # 2. operating point set on nominal rows of the other folds (5% false-alarm rate)
            post = (M.t.values >= a.t_event); nom = (M.family.values == "NOM")
            for mname in MODELS:
                thr = np.zeros(len(M))
                for f in folds:
                    ref = (~(M.fold.values == f)) & nom & (yv == 0)
                    thr[M.fold.values == f] = np.quantile(sc[mname][ref], 0.95) if ref.sum() > 100 else np.nan
                hit = sc[mname] > thr
                def row(lab, m):
                    P, N = m & (yv == 1), m & (yv == 0)
                    if P.sum() >= 10 and N.sum() >= 10: OPS.append(dict(horizon=h, mode=mode, subset=lab, model=mname, pos=int(P.sum()), neg=int(N.sum()), recall=float(hit[P].mean()), false_alarm_rate=float(hit[N].mean())))
                row("nominal episodes (all times)", nom); row("event episodes, after event", (~nom) & post); row("stale report (age >= 10 s)", age >= 10)
                for c in sorted(M.cls.unique()):
                    if not c.startswith("NOM"): row(c + " (after event)", (M.cls.values == c) & post)
            # 3. calibration (isotonic fitted on the other folds)
            for mname in ["learned", "hybrid", "constant-velocity rule"]:
                s = sc[mname]; cal = np.zeros(len(M))
                for f in folds:
                    trm = np.where(M.fold.values != f)[0]; trm = rng.choice(trm, min(len(trm), 300000), replace=False)
                    cal[M.fold.values == f] = IsotonicRegression(out_of_bounds="clip").fit(s[trm], yv[trm]).predict(s[M.fold.values == f])
                variants = [("isotonic", cal)] + ([("raw", s)] if mname != "constant-velocity rule" else [])
                for var, p in variants:
                    CAL.append(dict(horizon=h, mode=mode, model=mname, variant=var, brier=brier_score_loss(yv, np.clip(p, 0, 1)), ece=ece(np.clip(p, 0, 1), yv), prevalence=float(yv.mean())))
                    if mname in ("hybrid", "learned") or var == "isotonic":
                        o = np.argsort(p);
                        for k, ix in enumerate(np.array_split(o, 10)): REL.append(dict(horizon=h, mode=mode, model=mname, variant=var, bin=k, mean_pred=float(np.clip(p[ix], 0, 1).mean()), obs_freq=float(yv[ix].mean()), n=len(ix)))
            for nm, L in [("age_strata", ST), ("degraded_ops", OPS), ("calibration", CAL), ("reliability", REL), ("perm_importance", PI)]:
                if L: pd.DataFrame(L).to_csv(f"{a.o}_{nm}.csv", index=False)
            print(f"  [{time.time() - t0:5.0f}s] h={h} {mode}: outputs written", flush=True)
    print("->", a.o + "_*.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("feats"); ap.add_argument("--splits", required=True); ap.add_argument("-o", default="a13")
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 60]); ap.add_argument("--modes", nargs="+", default=["free", "mvp"]); ap.add_argument("--t-event", type=float, default=1800)
    ap.add_argument("--iters", type=int, default=150); ap.add_argument("--max-train", type=int, default=250000); ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--perm-horizons", type=int, nargs="+", default=[10, 30]); ap.add_argument("--perm-repeats", type=int, default=2); main(ap.parse_args())

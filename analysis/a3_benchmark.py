"""Step 3 - conflict-onset forecasting benchmark (physics baseline vs learned models), four evaluation protocols.
  python a3_benchmark.py FEAT_DIR --splits OUT/splits.csv -o OUT/benchmark [--targets paper_tac paper_imm half_tac] [--experiments cv5 zone transfer modeshift]
Protocols (every one is grouped by TRAFFIC KEY, so no test episode shares traffic with a training episode):
  cv5       5-fold grouped cross-validation, pooled over all classes, separately for mode=free and mode=mvp
  zone      leave-one-zone-out (train on two zones, test on the third)
  transfer  train on NOMINAL episodes only, test on contingency / degradation episodes of held-out traffic groups
  modeshift train on free (ASAS off) episodes, test on mvp (ASAS on) episodes of held-out groups, and the reverse
Models:  prevalence | density prior (neighbours within 1 km) | constant-velocity physics | learned: own kinematics |
         learned: kinematics + neighbours | hybrid: physics score + kinematics + neighbours (learned residual).
Metrics: AUPRC, AUROC, recall at 5% false-alarm rate; 95% CI by bootstrap over TRAFFIC GROUPS; subsets 'all' and 'maneuvering'."""
import argparse, glob, json, os, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, roc_curve

def rec5(y, s):
    f, t, _ = roc_curve(y, s); return float(t[f <= 0.05].max())

def metrics(y, s):
    return np.array([AP(y, s), AUC(y, s), rec5(y, s)])

def boot(y, s, g, B, rng):
    """pooled metrics + 95% CI, resampling traffic groups"""
    base = metrics(y, s); ug, inv = np.unique(g, return_inverse=True); ix = [np.where(inv == i)[0] for i in range(len(ug))]; bs = []
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug)); sel = np.concatenate([ix[i] for i in pick]); yy = y[sel]
        if yy.sum() >= 5 and (1 - yy).sum() >= 5: bs.append(metrics(yy, s[sel]))
    bs = np.array(bs) if bs else np.full((1, 3), np.nan)
    return base, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)

def fit_predict(tr, te, y, prof, h, cols, max_train, it):
    tr = tr[tr[y].notna()]
    if len(tr) > max_train: tr = tr.sample(max_train, random_state=0)
    raw = [c for c in cols if c.startswith("raw_")]; seq = [c for c in cols if c.startswith("seq_")]; rel = [c for c in cols if c.startswith("rel_")]; cv = f"cv_{prof}_h{h}"
    sc = {"prevalence": np.full(len(te), float(tr[y].mean())), "density prior": te["rel_cnt1000"].fillna(0).values, "constant-velocity physics": te[cv].fillna(-1e9).values}
    sets = {"learned: own kinematics": raw + [c for c in seq if "nn1" not in c and "cnt" not in c], "learned: kinematics + neighbours": raw + rel + seq, "hybrid: physics + kinematics + neighbours": raw + rel + seq + [cv]}
    if tr[y].nunique() < 2: return sc
    for n, c in sets.items(): sc[n] = HGB(max_iter=it, class_weight="balanced", random_state=0).fit(tr[c], tr[y]).predict_proba(te[c])[:, 1]
    return sc

def summarise(te, y, sc, B, rng, by_class=False):
    out = {}
    for sub, m in [("all", np.ones(len(te), bool)), ("maneuvering", te.maneuver.values)]:
        yy = te[y].values[m]
        if yy.sum() < 20 or (1 - yy).sum() < 20: continue
        if len(yy) > 250000:                                  # cap for speed (random, seeded)
            ix = rng.choice(len(yy), 250000, replace=False)
        else: ix = np.arange(len(yy))
        g = te.group.values[m][ix]; r = {}
        for n, s in sc.items():
            ss = np.asarray(s)[m][ix]
            if n == "prevalence": r[n] = dict(auprc=float(yy.mean()), auroc=0.5, rec5=0.05, n=int(len(yy)), pos=int(yy.sum())); continue
            b, lo, hi = boot(yy[ix], ss, g, B, rng); r[n] = dict(auprc=b[0], auprc_lo=lo[0], auprc_hi=hi[0], auroc=b[1], auroc_lo=lo[1], auroc_hi=hi[1], rec5=b[2], rec5_lo=lo[2], rec5_hi=hi[2], n=int(len(yy)), pos=int(yy.sum()))
        out[sub] = r
    if by_class:
        pc = {}
        for c, g in te.groupby("cls"):
            yy = g[y].values
            if yy.sum() >= 20 and (1 - yy).sum() >= 20: pc[c] = {n: float(AP(yy, np.asarray(s)[g.index.values])) for n, s in sc.items() if n in ("constant-velocity physics", "hybrid: physics + kinematics + neighbours", "learned: kinematics + neighbours")}
        out["by_class_auprc"] = pc
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("feats"); ap.add_argument("--splits", required=True); ap.add_argument("-o", default="benchmark")
    ap.add_argument("--targets", nargs="+", default=["paper_tac", "paper_imm", "half_tac"]); ap.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 60])
    ap.add_argument("--experiments", nargs="+", default=["cv5", "zone", "transfer", "modeshift"]); ap.add_argument("--max-train-rows", type=int, default=250000); ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--boot", type=int, default=60); ap.add_argument("--modes", nargs="+", default=["free", "mvp"]); a = ap.parse_args()
    sp = pd.read_csv(a.splits).rename(columns={"ep": "ep"}); fs = sorted(glob.glob(os.path.join(a.feats, "*.parquet")))
    D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).merge(sp[["ep", "cls", "family", "zone", "mode", "group", "fold"]], on="ep")
    D["maneuver"] = (D.raw_alt_chg_K.abs() > 5) | (D.raw_hd_chg_K > 0.1); cols = [c for c in D.columns if c.startswith(("raw_", "rel_", "seq_"))]; rng = np.random.default_rng(0)
    folds = sorted(D.fold.unique()); R = json.load(open(a.o + ".json")) if os.path.exists(a.o + ".json") else {}; t0 = time.time()   # resumable: finished runs are kept
    print(f"{len(D):,} rows, {D.ep.nunique()} episodes, {D.group.nunique()} traffic groups, {len(folds)} folds, maneuvering share {D.maneuver.mean():.3f}")
    def run(exp, key, splits_iter, y, prof, h, by_class=False):
        if str(h) in R.get(exp, {}).get(key, {}).get(y, {}): return
        parts = {}
        for tr, te in splits_iter:
            te = te[te[y].notna()].reset_index(drop=True)
            if len(te) == 0 or te[y].sum() < 5: continue
            sc = fit_predict(tr, te, y, prof, h, cols, a.max_train_rows, a.iters)
            for n, s in sc.items(): parts.setdefault(n, []).append(s)
            parts.setdefault("_te", []).append(te)
        if "_te" not in parts: return
        te = pd.concat(parts.pop("_te"), ignore_index=True); sc = {n: np.concatenate(v) for n, v in parts.items()}
        R.setdefault(exp, {}).setdefault(key, {}).setdefault(y, {})[str(h)] = summarise(te, y, sc, a.boot, rng, by_class)
        json.dump(R, open(a.o + ".json", "w"), indent=1)
        r = R[exp][key][y][str(h)].get("all", {}); print(f"  [{time.time() - t0:5.0f}s] {exp:9s} {key:22s} {y:18s} AUPRC " + " | ".join(f"{n.split(':')[0][:12]}={v['auprc']:.3f}" for n, v in r.items() if n != "prevalence"), flush=True)
    for tgt in a.targets:
        prof, lv = tgt.split("_")
        for h in a.horizons:
            y = f"y_{prof}_{lv}_h{h}"
            for mode in a.modes:
                M = D[D["mode"] == mode]
                if "cv5" in a.experiments: run("cv5", mode, ((M[M.fold != f], M[M.fold == f]) for f in folds), y, prof, h)
                if "zone" in a.experiments: run("zone", mode, ((M[M.zone != z], M[M.zone == z]) for z in sorted(M.zone.unique())), y, prof, h)
                if "transfer" in a.experiments: run("transfer", mode, ((M[(M.fold != f) & (M.family == "NOM")], M[(M.fold == f) & (M.family != "NOM")]) for f in folds), y, prof, h, by_class=True)
            if "modeshift" in a.experiments and len(a.modes) == 2:
                for src, dst in [("free", "mvp"), ("mvp", "free")]:
                    run("modeshift", f"{src}->{dst}", ((D[(D["mode"] == src) & (D.fold != f)], D[(D["mode"] == dst) & (D.fold == f)]) for f in folds), y, prof, h)
    json.dump(R, open(a.o + ".json", "w"), indent=1); print("->", a.o + ".json")

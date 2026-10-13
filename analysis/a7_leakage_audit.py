"""Step 7 - leakage audit of the forecasting task (answers reviewer point 1: circular logic / label leakage).
  python a7_leakage_audit.py DATA_DIR FEAT_DIR --splits OUT/splits.csv -o OUT/a7_leakage [--episodes 3] [--target y_half_tac_h30]

Five independent checks. Nothing here is tuned; every check prints PASS / FLAG with the numbers behind it.
  A  Provenance   every model input column is listed with its definition; names that look like label-derived quantities are rejected.
  B  Causality    FUTURE-INVARIANCE test: the raw episode is rebuilt after replacing everything after time t0 (reported positions, severity,
                  phase) by random values. Every input at t <= t0 must be bit-identical, while the future targets must change (sanity).
  C  Single-feature scan   AUROC of each input column alone against the target. A near-perfect single column would reveal leakage.
  D  Negative / positive controls, grouped 5-fold CV with the benchmark's folds:
        full model | labels shuffled inside each episode (must collapse to the prevalence) | without phase/type | without report age |
        without the physics score | POSITIVE control with the true future label added as a feature (must reach ~1, proves the audit can see leakage).
  E  Split integrity   no traffic group in two folds; the free/mvp twins of one traffic key sit in the same fold.
"""
import argparse, glob, json, os, re, sys, tempfile, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_common import *                                                    # noqa: E402,F401
from a2_build_task import build                                          # noqa: E402

FORBIDDEN = re.compile(r"sever|conflict|lowc|nmac|min_sep|resolv|true_|urgen|cascade|trend|_label|^y_", re.I)
DEFS = {"raw_alt": "reported altitude (m)", "raw_vz": "vertical speed from reported positions over 5 s", "raw_speed": "ground speed from reported positions over 5 s",
        "raw_hd_sin": "sin of heading (reported)", "raw_hd_cos": "cos of heading (reported)", "raw_type": "aircraft type code (flight-plan information)",
        "raw_phase": "flight phase code (climb/cruise/descent/emergency/transit), telemetry-declared", "raw_age": "age of the last position report (s)",
        "raw_alt_chg_K": "altitude change over the last 30 s", "raw_speed_chg_K": "speed change over the last 30 s", "raw_hd_chg_K": "absolute heading change over the last 30 s"}
def define(c):
    if c in DEFS: return DEFS[c]
    if c.startswith("rel_nn"): return "nearest-neighbour geometry among visible aircraft, REPORTED positions (" + c + ")"
    if c.startswith("rel_cnt"): return "number of visible aircraft within a radius, REPORTED positions"
    if c.startswith("rel_") and c.endswith("_chg_K"): return "30 s change of a neighbour feature"
    if c.startswith("seq_"): return "value of a reported-state feature at t minus the lag (history)"
    if c.startswith("cv_"): return "constant-velocity extrapolation of REPORTED positions tested against the profile threshold (a-priori physical rule, not learned)"
    return "?"

def inputs(F): return [c for c in F.columns if c.startswith(("raw_", "rel_", "seq_", "cv_"))]

# ---------------------------------------------------------------- A
def check_A(cols):
    rows = [dict(column=c, family=c.split("_")[0], definition=define(c), forbidden_name=bool(FORBIDDEN.search(c))) for c in cols]
    bad = [r["column"] for r in rows if r["forbidden_name"] or r["definition"] == "?"]
    return rows, bad

# ---------------------------------------------------------------- B
def future_invariance(path, t0, keep=1.0, seed=0):
    d = pd.read_parquet(path); m = (d.ep_t > t0).values; rng = np.random.default_rng(seed); n = int(m.sum())
    for c in ["rep_x_m", "rep_y_m", "rep_alt_m"]: d.loc[m, c] = rng.uniform(float(d[c].min()), float(d[c].max()), n).astype(d[c].dtype)
    d.loc[m, "rep_age_s"] = rng.uniform(0, 30, n).astype(d.rep_age_s.dtype); d.loc[m, "phase"] = rng.integers(0, 5, n).astype(d.phase.dtype)
    d.loc[m, "severity"] = rng.integers(0, 4, n).astype(d.severity.dtype); d.loc[m, "severity_half"] = rng.integers(0, 4, n).astype(d.severity_half.dtype)
    tmp = tempfile.mktemp(suffix=".parquet"); d.to_parquet(tmp, index=False)
    try: F1 = build(tmp, keep=keep)
    finally: os.remove(tmp)
    F0 = build(path, keep=keep)
    cols = inputs(F0); A = F0[F0.t <= t0].set_index(["t", "uav"]).sort_index(); B = F1[F1.t <= t0].set_index(["t", "uav"]).sort_index()
    same_index = A.index.equals(B.index); res = dict(rows_compared=int(len(A)), same_rows=bool(same_index), max_abs_diff=0.0, columns_changed=[])
    if same_index:
        for c in cols:
            a, b = A[c].values.astype(float), B[c].values.astype(float)
            ok = (a == b) | (np.isnan(a) & np.isnan(b));
            if not ok.all(): res["columns_changed"].append(c); res["max_abs_diff"] = float(max(res["max_abs_diff"], np.nanmax(np.abs(a[~ok] - b[~ok]))))
    ys = [c for c in F0.columns if c.startswith("y_")]; Af = F0[F0.t > t0 - 60].set_index(["t", "uav"]).sort_index(); Bf = F1[F1.t > t0 - 60].set_index(["t", "uav"]).sort_index()
    idx = Af.index.intersection(Bf.index); chg = float(np.mean([(Af.loc[idx, c].fillna(-1).values != Bf.loc[idx, c].fillna(-1).values).mean() for c in ys])) if len(idx) else float("nan")
    res["targets_changed_share_near_t0"] = chg
    return res

# ---------------------------------------------------------------- C
def check_C(D, y, cols, n=300000):
    S = D[D[y].notna()]; S = S.sample(min(n, len(S)), random_state=0); out = []
    for c in cols:
        v = S[c].values.astype(float); ok = np.isfinite(v)
        if ok.sum() < 100 or S[y].values[ok].min() == S[y].values[ok].max(): continue
        a = AUC(S[y].values[ok], v[ok]); out.append(dict(column=c, auroc=float(a), strength=float(max(a, 1 - a))))
    return sorted(out, key=lambda r: -r["strength"])

# ---------------------------------------------------------------- D
def cv_auprc(M, y, cols, maxrows, iters):
    ps, ys = [], []
    for f in sorted(M.fold.unique()):
        tr, te = M[M.fold != f], M[M.fold == f]; tr, te = tr[tr[y].notna()], te[te[y].notna()]
        if len(te) == 0 or te[y].sum() < 5 or tr[y].nunique() < 2: continue
        if len(tr) > maxrows: tr = tr.sample(maxrows, random_state=0)
        ps.append(HGB(max_iter=iters, class_weight="balanced", random_state=0).fit(tr[cols], tr[y]).predict_proba(te[cols])[:, 1]); ys.append(te[y].values)
    if not ys: return dict(auprc=float("nan"), prevalence=float("nan"), n=0)
    y_, p_ = np.concatenate(ys), np.concatenate(ps); return dict(auprc=float(AP(y_, p_)), prevalence=float(y_.mean()), n=int(len(y_)))

def check_D(D, y, prof, h, maxrows, iters):
    cols = inputs(D); cv = f"cv_{prof}_h{h}"; other_cv = [c for c in cols if c.startswith("cv_") and c != cv]; base = [c for c in cols if c not in other_cv]
    res = {}
    for mode in ["free", "mvp"]:
        M = D[D["mode"] == mode].copy(); r = {}
        r["full (hybrid inputs)"] = cv_auprc(M, y, base, maxrows, iters)
        S = M.copy(); ok = S[y].notna(); S.loc[ok, y] = S[ok].groupby("ep")[y].transform(lambda s: s.sample(frac=1, random_state=0).values)
        r["NEG control: labels shuffled inside each episode"] = cv_auprc(S, y, base, maxrows, iters)
        r["ablation: without phase and type"] = cv_auprc(M, y, [c for c in base if c not in ("raw_phase", "raw_type")], maxrows, iters)
        r["ablation: without report age"] = cv_auprc(M, y, [c for c in base if c != "raw_age"], maxrows, iters)
        r["ablation: without physics score"] = cv_auprc(M, y, [c for c in base if c != cv], maxrows, iters)
        P = M.copy(); P["CANARY_true_future_label"] = P[y]; r["POS control: true future label added as a feature"] = cv_auprc(P, y, base + ["CANARY_true_future_label"], maxrows, iters)
        res[mode] = r
    return res

# ---------------------------------------------------------------- E
def check_E(sp):
    g = sp.groupby("group").fold.nunique(); multi = g[g > 1].index.tolist()
    return dict(groups=int(len(g)), groups_in_several_folds=multi, episodes=int(len(sp)), folds=sorted(map(int, sp.fold.unique())))

def verdict(ok): return "PASS" if ok else "FLAG"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("feats"); ap.add_argument("--splits", required=True); ap.add_argument("-o", default="a7_leakage")
    ap.add_argument("--episodes", type=int, default=3); ap.add_argument("--target", default="y_half_tac_h30"); ap.add_argument("--max-train-rows", type=int, default=150000); ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--t0", type=int, nargs="+", default=[1200, 2400]); a = ap.parse_args()
    t_start = time.time(); sp = pd.read_csv(a.splits); y = a.target; _, prof, lv, hh = y.split("_"); h = int(hh[1:])
    fs = sorted(glob.glob(os.path.join(a.feats, "*.parquet"))); D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).merge(sp[["ep", "cls", "family", "zone", "mode", "group", "fold"]], on="ep")
    cols = inputs(D); R = {"target": y, "rows": int(len(D)), "episodes": int(D.ep.nunique())}; print(f"{len(D):,} rows, {D.ep.nunique()} episodes, {len(cols)} input columns, target {y}")

    rowsA, bad = check_A(cols); R["A_provenance"] = rowsA; R["A_flagged"] = bad; pd.DataFrame(rowsA).to_csv(a.o + "_inputs.csv", index=False)
    print(f"A  provenance: {len(cols)} inputs listed -> {a.o}_inputs.csv ; names rejected: {bad if bad else 'none'}  [{verdict(not bad)}]")

    eps = sorted(D.ep.unique()); pick = [e for e in eps if "_free_" in e][:max(1, a.episodes // 2 + a.episodes % 2)] + [e for e in eps if "_mvp_" in e][:max(1, a.episodes // 2)]
    R["B_future_invariance"] = {}; allok = True
    for e in pick:
        for t0 in a.t0:
            r = future_invariance(os.path.join(a.data, e + ".parquet"), t0); R["B_future_invariance"][f"{e}@{t0}"] = r
            ok = r["same_rows"] and not r["columns_changed"]; allok &= ok
            print(f"B  {e} t0={t0}: {r['rows_compared']:,} rows compared, columns changed by future corruption: {r['columns_changed'] or 'none'}, targets changed {r['targets_changed_share_near_t0']:.2f} [{verdict(ok)}]")
    R["B_pass"] = bool(allok)

    c = check_C(D, y, [x for x in cols if not x.startswith("cv_")]); R["C_single_feature"] = c[:15]
    top = c[0] if c else None; print("C  strongest single input (excluding the physics score): " + (f"{top['column']} AUROC {top['auroc']:.3f}" if top else "n/a") + f"  [{verdict(not top or top['strength'] < 0.90)}]")
    for r in c[:5]: print(f"     {r['column']:22s} AUROC {r['auroc']:.3f}")

    Dres = check_D(D, y, prof, h, a.max_train_rows, a.iters); R["D_controls"] = Dres
    for mode, r in Dres.items():
        full, neg, pos = r["full (hybrid inputs)"], r["NEG control: labels shuffled inside each episode"], r["POS control: true future label added as a feature"]
        okn = neg["auprc"] <= 1.3 * neg["prevalence"] if neg["n"] else False; okp = pos["auprc"] >= 0.95 if pos["n"] else False
        print(f"D  [{mode}] prevalence {full['prevalence']:.3f} | full AUPRC {full['auprc']:.3f} | shuffled {neg['auprc']:.3f} [{verdict(okn)}] | positive control {pos['auprc']:.3f} [{verdict(okp)}]")
        for k in ("ablation: without phase and type", "ablation: without report age", "ablation: without physics score"): print(f"     {k:36s} AUPRC {r[k]['auprc']:.3f}  (delta {r[k]['auprc'] - full['auprc']:+.3f})")
    R["E_split_integrity"] = check_E(sp); e = R["E_split_integrity"]
    print(f"E  splits: {e['groups']} traffic groups, folds {e['folds']}, groups present in several folds: {e['groups_in_several_folds'] or 'none'}  [{verdict(not e['groups_in_several_folds'])}]")
    json.dump(R, open(a.o + ".json", "w"), indent=1, default=str); print(f"-> {a.o}.json, {a.o}_inputs.csv  ({time.time() - t_start:.0f}s)")

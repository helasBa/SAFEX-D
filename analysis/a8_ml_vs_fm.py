"""Step 8 - ML versus foundation models on EXACTLY THE SAME ROWS (paired comparison).
a3 scores every row of every episode; a5 scores only a sample (fixed seed, every N-th instant, UAVs whose full 120 s context is available), so their AUPRC values are not
directly comparable. This script re-scores the learned models on the a5 sample, with the SAME out-of-fold logic as a3 (trained only on episodes of other folds / the other mode):
  python a8_ml_vs_fm.py FEAT_DIR --splits OUT/splits.csv --fm OUT/fm_chronos2_diff_scores.parquet OUT/fm_timesfm25_diff_scores.parquet -o OUT/a8_ml_vs_fm
Also writes a8_ml_vs_fm_f1.csv (precision, recall, F1, F1-macro, threshold chosen on the other folds).
Protocols: cv5 (5 grouped folds, per mode) and modeshift (train free -> test mvp and reverse). Only rows where every foundation model was actually applied (fm_used) are compared.
Reported per target / horizon / mode: AUPRC (95% CI, group bootstrap over traffic groups) of prevalence, density prior, constant-velocity (a2 builder, 5 s window), constant-velocity
(harness, 15 s window), learned (kinematics + neighbours), hybrid, and each foundation model; plus PAIRED differences (same bootstrap draws) with 95% CI:
  hybrid - model, learned - model, model - constant-velocity (harness). A difference whose interval contains 0 is NOT a difference.
Nothing is tuned: the learned models use the same settings as a3."""
import argparse, glob, json, os, time, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
from a3_benchmark import fit_predict

def paired_boot(y, S, g, B, rng, pairs):
    """S: {name: scores}. Returns point AUPRC + CI for each name and for each (a, b) difference, using common group-bootstrap draws."""
    ug, inv = np.unique(g, return_inverse=True); ix = [np.where(inv == i)[0] for i in range(len(ug))]
    pt = {n: float(AP(y, s)) for n, s in S.items()}; ptd = {f"{a} - {b}": pt[a] - pt[b] for a, b in pairs}
    bs = {n: [] for n in S}; bd = {k: [] for k in ptd}
    for _ in range(B):
        sel = np.concatenate([ix[i] for i in rng.integers(0, len(ug), len(ug))]); yy = y[sel]
        if yy.sum() < 5 or (1 - yy).sum() < 5: continue
        v = {n: float(AP(yy, s[sel])) for n, s in S.items()}
        for n in S: bs[n].append(v[n])
        for a, b in pairs: bd[f"{a} - {b}"].append(v[a] - v[b])
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) > 10 else (np.nan, np.nan)
    out = {n: dict(auprc=pt[n], lo=ci(bs[n])[0], hi=ci(bs[n])[1]) for n in S}
    out["_diff"] = {k: dict(diff=ptd[k], lo=ci(bd[k])[0], hi=ci(bd[k])[1]) for k in ptd}
    return out

def f1_block(y, S, fold, g, B, rng):
    """Binary metrics at an operating point that is NEVER chosen on the rows it is applied to: for each fold, the threshold maximising F1 on the pooled scores of the OTHER folds
    (candidate thresholds = 200 quantiles). Reports precision, recall, F1 of the event class, F1-macro (mean of event-class and non-event-class F1, the v1 metric) and 95% CI by
    bootstrap over traffic groups (fixed predictions). 'prevalence' has no ranking and is skipped."""
    def f1(tp, fp, fn): return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    ug, inv = np.unique(g, return_inverse=True); out = {}
    for n, sc in S.items():
        if n == "prevalence": continue
        sc = np.asarray(sc, float); pred = np.zeros(len(y), bool); ok = True
        for k in np.unique(fold):
            m = fold == k
            if m.all() or y[~m].sum() < 5: ok = False; break
            cand = np.unique(np.quantile(sc[~m], np.linspace(0.01, 0.99, 200))); yy = y[~m].astype(bool); best, thr = -1, cand[0]
            for c in cand:
                pp = sc[~m] >= c; tp = (pp & yy).sum(); v = f1(tp, (pp & ~yy).sum(), (~pp & yy).sum())
                if v > best: best, thr = v, c
            pred[m] = sc[m] >= thr
        if not ok: continue
        yb = y.astype(bool); cnt = np.zeros((len(ug), 4))                      # per group: tp, fp, fn, tn
        for j, v_ in enumerate([pred & yb, pred & ~yb, ~pred & yb, ~pred & ~yb]): np.add.at(cnt[:, j], inv, v_.astype(float))
        def stats(c):
            tp, fp, fn, tn = c; pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); return np.array([pr, rc, f1(tp, fp, fn), 0.5 * (f1(tp, fp, fn) + f1(tn, fn, fp))])
        pt = stats(cnt.sum(0)); bs = np.array([stats(cnt[rng.integers(0, len(ug), len(ug))].sum(0)) for _ in range(B)])
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        out[n] = {k: dict(v=float(pt[i]), lo=float(lo[i]), hi=float(hi[i])) for i, k in enumerate(["precision", "recall", "f1", "f1_macro"])}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("feats"); ap.add_argument("--splits", required=True); ap.add_argument("--fm", nargs="+", required=True, help="a5 *_scores.parquet files")
    ap.add_argument("-o", default="a8_ml_vs_fm"); ap.add_argument("--targets", nargs="+", default=["paper_tac", "paper_imm", "half_tac"]); ap.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 60])
    ap.add_argument("--max-train-rows", type=int, default=250000); ap.add_argument("--iters", type=int, default=150); ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--protocols", nargs="+", default=["cv5", "modeshift"]); a = ap.parse_args(); rng = np.random.default_rng(0); t0 = time.time()
    sp = pd.read_csv(a.splits)[["ep", "cls", "family", "zone", "mode", "group", "fold"]]
    # --- the foundation-model sample: rows (ep, t, uav) present in EVERY scores file, and where the model was applied
    names, S = [], None
    for f in a.fm:
        s = pd.read_parquet(f); nm = os.path.basename(f).replace("_scores.parquet", ""); names.append(nm)
        mc = [c for c in s.columns if c.startswith("model_")]; keep = ["ep", "t", "uav", "fm_used"] + mc
        s = s[keep].rename(columns={c: f"{nm}::{c}" for c in mc}).rename(columns={"fm_used": f"{nm}::used"})
        if S is None: S = s; base = pd.read_parquet(f)
        else: S = S.merge(s, on=["ep", "t", "uav"], how="inner")
    used = np.all([S[f"{n}::used"].values for n in names], axis=0); S = S[used].drop(columns=[f"{n}::used" for n in names])
    keys = base[["ep", "t", "uav", "maneuver"] + [c for c in base.columns if c.startswith("cvh_")]]; S = S.merge(keys, on=["ep", "t", "uav"], how="left")
    print(f"foundation-model sample: {len(S):,} rows (all models applied), {S.ep.nunique()} episodes, models {names}")
    # --- all benchmark features (training side) and the sample's own features (test side)
    fs = sorted(glob.glob(os.path.join(a.feats, "*.parquet"))); D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).merge(sp, on="ep")
    cols = [c for c in D.columns if c.startswith(("raw_", "rel_", "seq_"))]
    T = D.merge(S, on=["ep", "t", "uav"], how="inner"); print(f"sample matched to features: {len(T):,} rows")
    D["maneuver"] = (D.raw_alt_chg_K.abs() > 5) | (D.raw_hd_chg_K > 0.1)
    R = {}
    def evaluate(proto, mode_key, pairs_iter, y, prof, h):
        parts = {}; tes = []
        for tr, te in pairs_iter:
            te = te[te[y].notna()].reset_index(drop=True)
            if len(te) == 0 or te[y].sum() < 5: continue
            sc = fit_predict(tr, te, y, prof, h, cols, a.max_train_rows, a.iters)
            for n, s in sc.items(): parts.setdefault(n, []).append(s)
            tes.append(te)
        if not tes: return
        te = pd.concat(tes, ignore_index=True); sc = {n: np.concatenate(v) for n, v in parts.items()}; yy = te[y].values.astype(int)
        if yy.sum() < 20 or (1 - yy).sum() < 20: return
        Sc = {"prevalence": sc["prevalence"], "density prior": sc["density prior"], "constant-velocity (a2, 5 s window)": sc["constant-velocity physics"], "constant-velocity (harness, 15 s window)": te[f"cvh_{prof}_h{h}"].values,
              "learned: kinematics + neighbours": sc["learned: kinematics + neighbours"], "hybrid: physics + kinematics + neighbours": sc["hybrid: physics + kinematics + neighbours"]}
        for n in names: Sc[n] = te[f"{n}::model_{prof}_h{h}"].values
        cv, hy, ln = "constant-velocity (harness, 15 s window)", "hybrid: physics + kinematics + neighbours", "learned: kinematics + neighbours"
        pairs = [(hy, n) for n in names] + [(ln, n) for n in names] + [(n, cv) for n in names] + [(hy, cv), (ln, cv)]
        res = {}
        for sub, mk in [("all", np.ones(len(te), bool)), ("maneuvering", te.maneuver.values.astype(bool))]:
            if mk.sum() < 40 or yy[mk].sum() < 20 or (1 - yy[mk]).sum() < 20: continue
            res[sub] = paired_boot(yy[mk], {k: np.asarray(v)[mk] for k, v in Sc.items()}, te.group.values[mk], a.boot, rng, pairs); res[sub]["n"] = int(mk.sum()); res[sub]["pos"] = int(yy[mk].sum())
            res[sub]["_f1"] = f1_block(yy[mk], {k: np.asarray(v)[mk] for k, v in Sc.items()}, te["fold"].values[mk], te.group.values[mk], a.boot, rng)
        R.setdefault(proto, {}).setdefault(mode_key, {}).setdefault(y, {})[str(h)] = res
        r = res.get("all", {}); print(f"  [{time.time() - t0:5.0f}s] {proto:9s} {mode_key:11s} {y:16s} " + " | ".join(f"{k.split(':')[0][:14]}={v['auprc']:.3f}" for k, v in r.items() if isinstance(v, dict) and "auprc" in v and k != "prevalence"), flush=True)
        fb = r.get("_f1", {}); print("      F1 / F1-macro: " + " | ".join(f"{k.split(':')[0][:14]}={v['f1']['v']:.3f}/{v['f1_macro']['v']:.3f}" for k, v in fb.items()), flush=True)
    folds = sorted(D.fold.unique())
    for tgt in a.targets:
        prof, lv = tgt.split("_")
        for h in a.horizons:
            y = f"y_{prof}_{lv}_h{h}"
            if "cv5" in a.protocols:
                for mode in ["free", "mvp"]:
                    M, TM = D[D["mode"] == mode], T[T["mode"] == mode]
                    evaluate("cv5", mode, ((M[M.fold != f], TM[TM.fold == f]) for f in folds), y, prof, h)
            if "modeshift" in a.protocols:
                for src, dst in [("free", "mvp"), ("mvp", "free")]:
                    evaluate("modeshift", f"{src}->{dst}", ((D[(D["mode"] == src) & (D.fold != f)], T[(T["mode"] == dst) & (T.fold == f)]) for f in folds), y, prof, h)
    R["_meta"] = dict(models=names, sample_rows=int(len(T)), sample_episodes=int(T.ep.nunique()), boot=a.boot)
    json.dump(R, open(a.o + ".json", "w"), indent=1); print("->", a.o + ".json")
    # --- compact CSV of the paired differences (point, 95% CI) for the paper
    rows = []
    for proto, d1 in R.items():
        if proto == "_meta": continue
        for mk, d2 in d1.items():
            for y, d3 in d2.items():
                for h, d4 in d3.items():
                    r = d4.get("all")
                    if not r: continue
                    for k, v in r["_diff"].items(): rows.append(dict(protocol=proto, mode=mk, target=y, horizon=h, comparison=k, diff=v["diff"], lo=v["lo"], hi=v["hi"], excludes_zero=bool(v["lo"] > 0 or v["hi"] < 0), n=r["n"], pos=r["pos"]))
    pd.DataFrame(rows).to_csv(a.o + "_paired.csv", index=False); print("->", a.o + "_paired.csv")
    rows = []
    for proto, d1 in R.items():
        if proto == "_meta": continue
        for mk, d2 in d1.items():
            for y, d3 in d2.items():
                for h, d4 in d3.items():
                    for sub, r in d4.items():
                        for m_, v in r.get("_f1", {}).items(): rows.append(dict(protocol=proto, mode=mk, target=y, horizon=h, subset=sub, method=m_, n=r["n"], pos=r["pos"], auprc=r[m_]["auprc"], **{f"{k}{suf}": v[k][x] for k in ("precision", "recall", "f1", "f1_macro") for suf, x in (("", "v"), ("_lo", "lo"), ("_hi", "hi"))}))
    pd.DataFrame(rows).to_csv(a.o + "_f1.csv", index=False); print("->", a.o + "_f1.csv")

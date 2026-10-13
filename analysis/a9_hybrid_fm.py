"""Step 9 - does a foundation-model score ADD information to the hybrid model?  (hybrid + FM)
Protocol fixed BEFORE looking at results (nothing is tuned, same learner and settings as a3):
  rows        = the rows scored by a5 (reported-state inputs only, UAV not yet in conflict, full 120 s context available)
  folds       = the traffic-group folds of splits.csv (a model never sees the traffic group it is tested on), separately for mode=free and mode=mvp
  models      constant-velocity (harness) | foundation model alone | hybrid (physics + kinematics + neighbours) | hybrid + FM (the same features + the FM score)
  learner     HistGradientBoosting, 150 iterations, balanced class weights (identical to a3), trained ONLY on the training folds' rows of the a5 sample
Reported: AUPRC with 95% CI (group bootstrap) and PAIRED differences (same draws): hybrid+FM - hybrid, hybrid+FM - FM, hybrid - FM, hybrid - constant-velocity.
The pairing is the point: a difference whose interval contains 0 is not a difference.  Also writes precision / recall / F1 / F1-macro (threshold chosen on the other folds).
  python a9_hybrid_fm.py --scores OUT/fm_chronos2_diff_big_scores.parquet --splits OUT/splits.csv -o OUT/a9_hybrid_fm [--name-fm fm_chronos2_diff_big]"""
import argparse, json, os, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from a8_ml_vs_fm import paired_boot, f1_block

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True, help="a5 *_scores.parquet"); ap.add_argument("--splits", required=True); ap.add_argument("-o", default="a9_hybrid_fm")
    ap.add_argument("--targets", nargs="+", default=["paper_tac", "paper_imm", "half_tac"]); ap.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 60])
    ap.add_argument("--iters", type=int, default=150); ap.add_argument("--boot", type=int, default=200); ap.add_argument("--min-train-pos", type=int, default=30); a = ap.parse_args()
    rng = np.random.default_rng(0); t0 = time.time()
    S = pd.read_parquet(a.scores); S = S[S.fm_used].reset_index(drop=True); sp = pd.read_csv(a.splits)[["ep", "fold", "family"]]; S = S.merge(sp, on="ep", how="left")
    cols = [c for c in S.columns if c.startswith(("raw_", "rel_", "seq_"))]; folds = sorted(S.fold.unique())
    print(f"{len(S):,} rows, {S.ep.nunique()} episodes, {S.group.nunique()} traffic groups, folds {folds}")
    R = {}; rows_diff = []; rows_f1 = []
    for tgt in a.targets:
        prof, lv = tgt.split("_")
        for h in a.horizons:
            y = f"y_{prof}_{lv}_h{h}"; cv, fm = f"cv_{prof}_h{h}", f"model_{prof}_h{h}"; cvh = f"cvh_{prof}_h{h}"
            for mode in ["free", "mvp"]:
                M = S[(S["mode"] == mode) & S[y].notna()].reset_index(drop=True); parts = {"hybrid": [], "hybrid + FM": []}; tes = []
                for f in folds:
                    tr, te = M[M.fold != f], M[M.fold == f]
                    if len(te) == 0 or te[y].sum() < 5 or tr[y].sum() < a.min_train_pos or tr[y].nunique() < 2: continue
                    for nm, c in (("hybrid", cols + [cv]), ("hybrid + FM", cols + [cv, fm])):
                        parts[nm].append(HGB(max_iter=a.iters, class_weight="balanced", random_state=0).fit(tr[c], tr[y]).predict_proba(te[c])[:, 1])
                    tes.append(te)
                if not tes: print(f"  skip {y} {mode}: not enough positives in the training folds"); continue
                te = pd.concat(tes, ignore_index=True); yy = te[y].values.astype(int)
                if yy.sum() < 20 or (1 - yy).sum() < 20: print(f"  skip {y} {mode}: fewer than 20 positives in the test folds"); continue
                Sc = {"constant-velocity (harness, 15 s window)": te[cvh].values, "constant-velocity (a2, 5 s window)": te[cv].values, "foundation model alone": te[fm].values,
                      "hybrid": np.concatenate(parts["hybrid"]), "hybrid + FM": np.concatenate(parts["hybrid + FM"])}
                pairs = [("hybrid + FM", "hybrid"), ("hybrid + FM", "foundation model alone"), ("hybrid", "foundation model alone"), ("hybrid", "constant-velocity (harness, 15 s window)"), ("foundation model alone", "constant-velocity (harness, 15 s window)")]
                res = {}
                for sub, mk in [("all", np.ones(len(te), bool)), ("maneuvering", te.maneuver.values.astype(bool))]:
                    if mk.sum() < 40 or yy[mk].sum() < 20 or (1 - yy[mk]).sum() < 20: continue
                    r = paired_boot(yy[mk], {k: v[mk] for k, v in Sc.items()}, te.group.values[mk], a.boot, rng, pairs); r["n"] = int(mk.sum()); r["pos"] = int(yy[mk].sum())
                    r["_f1"] = f1_block(yy[mk], {k: v[mk] for k, v in Sc.items()}, te.fold.values[mk], te.group.values[mk], a.boot, rng); res[sub] = r
                    for k, v in r["_diff"].items(): rows_diff.append(dict(target=y, horizon=h, mode=mode, subset=sub, comparison=k, diff=v["diff"], lo=v["lo"], hi=v["hi"], excludes_zero=bool(v["lo"] > 0 or v["hi"] < 0), n=r["n"], pos=r["pos"]))
                    for m_, v in r["_f1"].items(): rows_f1.append(dict(target=y, horizon=h, mode=mode, subset=sub, method=m_, auprc=r[m_]["auprc"], n=r["n"], pos=r["pos"], **{f"{k}{suf}": v[k][x] for k in ("precision", "recall", "f1", "f1_macro") for suf, x in (("", "v"), ("_lo", "lo"), ("_hi", "hi"))}))
                R.setdefault(y, {}).setdefault(str(h), {})[mode] = res
                r = res.get("all")
                if r: print(f"  [{time.time() - t0:5.0f}s] {y:16s} {mode:4s} " + " | ".join(f"{k.split('(')[0].strip()[:18]}={r[k]['auprc']:.3f}" for k in Sc) + f"  || hybrid+FM - hybrid = {r['_diff']['hybrid + FM - hybrid']['diff']:+.3f} [{r['_diff']['hybrid + FM - hybrid']['lo']:+.3f}, {r['_diff']['hybrid + FM - hybrid']['hi']:+.3f}]", flush=True)
    json.dump(R, open(a.o + ".json", "w"), indent=1); pd.DataFrame(rows_diff).to_csv(a.o + "_paired.csv", index=False); pd.DataFrame(rows_f1).to_csv(a.o + "_f1.csv", index=False); print("->", a.o + ".json, _paired.csv, _f1.csv")

"""Step 18 - LaTeX tables for the communications part (booktabs). Every number is read from the result files; nothing is typed by hand.
  python a18_comm_tables.py C:\\these\\safex2_out [-o tables_comm.tex] [--margins 0.05 0.10]
Reads in the folder (whatever exists): a15_link_eval.csv a13_degraded_ops.csv a13_calibration.csv a16_inference_cost.csv
Tables
  tab_link_<mode>     AUPRC under each emulated link: constant-velocity rule, hybrid trained on the ideal link, hybrid trained on that link (95% CI); h10 and h30
  tab_link_req        link requirements: largest tested update period / latency / loss for which the hybrid (trained on the link) stays within a margin of its own ideal-link AUPRC
  tab_ops             alert quality at a 5% false-alarm threshold set on nominal traffic: nominal / after event / stale reports (rows with age >= 10 s)
  tab_calib           calibration of the hybrid score (ECE and Brier, raw vs isotonic)
  tab_cost            inference cost per aircraft
Requirement rule: drop = AUPRC(ideal link) - AUPRC(setting), same mode, horizon and model; point estimates, the grid is the one that was run (it is not interpolated)."""
import argparse, os, numpy as np, pandas as pd

H = "hybrid"; CVR = "constant-velocity rule"
NAMES = {"base": "ideal (1 Hz)", "P2": "period 2 s", "P5": "period 5 s", "P10": "period 10 s", "P10+w20": "period 10 s, 20 s window",
         "L2": "latency 2 s", "L5": "latency 5 s", "L10": "latency 10 s", "loss10": "loss 10\\%", "loss30": "loss 30\\%",
         "loss30+w10": "loss 30\\%, 10 s window", "loss50": "loss 50\\%"}
ORDER = ["base", "P2", "P5", "P10", "P10+w20", "L2", "L5", "L10", "loss10", "loss30", "loss30+w10", "loss50"]
AXES = {"update period": ["P2", "P5", "P10"], "latency": ["L2", "L5", "L10"], "packet loss": ["loss10", "loss30", "loss50"]}
LVL = {"P2": "2 s", "P5": "5 s", "P10": "10 s", "L2": "2 s", "L5": "5 s", "L10": "10 s", "loss10": "10\\%", "loss30": "30\\%", "loss50": "50\\%"}
MODE = {"free": "ASAS off", "mvp": "ASAS on (MVP)"}

def tab(cap, label, head, rows, fmt, pos="t"):
    return "\\begin{table}[%s]\n\\centering\\small\n\\caption{%s}\\label{%s}\n\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n" % (
        pos, cap, label, fmt, " & ".join(head), " \\\\\n".join(" & ".join(map(str, r)) for r in rows))

def get(d, **k):
    x = d
    for a, b in k.items(): x = x[x[a] == b]
    return x.iloc[0] if len(x) else None

def link_tables(d):
    T = []
    for m in [m for m in ("free", "mvp") if m in set(d["mode"])]:
        rows = []
        for s in [s for s in ORDER if s in set(d.setting)]:
            r = [NAMES[s]]
            for h in (10, 30):
                cv = get(d, setting=s, kind="matched", mode=m, horizon=h, model=CVR); hb = get(d, setting=s, kind="base-trained", mode=m, horizon=h, model=H); hm = get(d, setting=s, kind="matched", mode=m, horizon=h, model=H)
                r += [f"{cv.auprc:.3f}" if cv is not None else "-", f"{hb.auprc:.3f}" if hb is not None else "-",
                      (f"{hm.auprc:.3f} [{hm.auprc_lo:.3f}, {hm.auprc_hi:.3f}]" if hm is not None else "-")]
            r.insert(1, f"{get(d, setting=s, kind='matched', mode=m, horizon=30, model=H).age_mean:.1f}")
            rows.append(r)
        T.append(tab(f"AUPRC for LoWC (paper thresholds) under an emulated surveillance link, {MODE[m]}. Rule: constant-velocity rule (5\\,s velocity window, no age compensation). Hybrid (ideal): trained on the ideal link, tested on the degraded one. Hybrid (link): trained and tested on the same link, 95\\% group-bootstrap interval. Age: mean age of the last report (s). Subset of 120 episodes, so intervals are wider than in the main benchmark.",
                     f"tab:link_{m}", ["Link", "Age", "Rule", "Hybrid (ideal)", "Hybrid (link)", "Rule", "Hybrid (ideal)", "Hybrid (link)"], rows, "lrcccccc").replace(
            "\\toprule\n", "\\toprule\n & & \\multicolumn{3}{c}{$h=10$\\,s} & \\multicolumn{3}{c}{$h=30$\\,s} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n", 1).replace("\\\\\n\\midrule", "\\\\\n\\midrule", 1))
    return T

def requirements(d, margins):
    rows = []
    for m in [m for m in ("free", "mvp") if m in set(d["mode"])]:
        for h in (10, 30):
            b = get(d, setting="base", kind="matched", mode=m, horizon=h, model=H)
            if b is None: continue
            r = [MODE[m], f"{h}\\,s", f"{b.auprc:.3f}"]
            for ax, ss in AXES.items():
                for mg in margins:
                    ok = [s for s in ss if get(d, setting=s, kind="matched", mode=m, horizon=h, model=H) is not None and b.auprc - get(d, setting=s, kind="matched", mode=m, horizon=h, model=H).auprc <= mg + 1e-12]
                    # the largest level that still passes, provided every smaller level passes too (monotone reading of the grid)
                    best = None
                    for s in ss:
                        if s in ok: best = s
                        else: break
                    r.append(LVL[best] if best else "none")
            rows.append(r)
    head = ["Mode", "$h$", "Ideal AUPRC"] + [f"{ax.split()[-1]} $\\leq${mg:.2f}" for ax in AXES for mg in margins]
    cap = ("Largest tested link degradation for which the hybrid trained on that link loses at most 0.05 or 0.10 AUPRC relative to the ideal 1\\,Hz link (same mode and horizon; point estimates; tested levels only, so a level is a lower bound on what the model tolerates; ``none'' means even the mildest tested level exceeds the margin). Update period 2/5/10\\,s, latency 2/5/10\\,s, independent packet loss 10/30/50\\%.")
    return tab(cap, "tab:link_req", head, rows, "llc" + "cc" * len(AXES), "t")

def ops(o):
    f = os.path.join(o, "a13_degraded_ops.csv")
    if not os.path.exists(f): return None
    d = pd.read_csv(f); sub = {"nominal episodes (all times)": "Nominal traffic", "event episodes, after event": "After an event", "stale report (age >= 10 s)": "Stale report ($\\geq$10\\,s)"}
    rows = []
    for m in [m for m in ("free", "mvp") if m in set(d["mode"])][:1]:
        for h in sorted(set(d.horizon)):
            for s, lab in sub.items():
                r = [f"{h}\\,s", lab]
                x = d[(d["mode"] == m) & (d.horizon == h) & (d.subset == s)]
                if x.empty: continue
                for mod in (CVR, H):
                    y = x[x.model == mod]
                    r += [f"{y.recall.iloc[0]:.3f}", f"{y.false_alarm_rate.iloc[0]:.3f}"] if len(y) else ["-", "-"]
                r.append(f"{int(x.pos.iloc[0]):,}".replace(",", "\\,") + "/" + f"{int(x.neg.iloc[0]):,}".replace(",", "\\,"))
                rows.append(r)
    cap = ("Alert quality with the alert threshold fixed so that 5\\% of nominal-traffic rows raise a false alarm (threshold chosen on the other folds), ASAS off. Recall = share of LoWC rows flagged, FAR = share of non-conflict rows flagged. Stale rows are rare, so their figures are indicative.")
    t = tab(cap, "tab:ops", ["$h$", "Subset", "Rule recall", "Rule FAR", "Hybrid recall", "Hybrid FAR", "Positives/negatives"], rows, "llccccr")
    return t

def calib(o):
    f = os.path.join(o, "a13_calibration.csv")
    if not os.path.exists(f): return None
    d = pd.read_csv(f); rows = []
    for m in [m for m in ("free", "mvp") if m in set(d["mode"])][:1]:
        for h in sorted(set(d.horizon)):
            for mod in (CVR, "learned", H):
                r = [f"{h}\\,s", mod.replace("constant-velocity rule", "Constant-velocity rule").replace("learned", "Learned").replace("hybrid", "Hybrid")]
                for v in ("raw", "isotonic"):
                    x = d[(d["mode"] == m) & (d.horizon == h) & (d.model == mod) & (d.variant == v)]
                    r += [f"{x.ece.iloc[0]:.4f}", f"{x.brier.iloc[0]:.4f}"] if len(x) else ["-", "-"]
                rows.append(r)
    cap = "Calibration of the alert scores, ASAS off, 5-fold grouped cross-validation. ECE: expected calibration error; Brier: mean squared error of the probability. Isotonic: recalibration fitted on the other folds."
    return tab(cap, "tab:calib", ["$h$", "Model", "ECE raw", "Brier raw", "ECE isotonic", "Brier isotonic"], rows, "llcccc")

def cost(o):
    f = os.path.join(o, "a16_inference_cost.csv")
    if not os.path.exists(f): return None
    d = pd.read_csv(f); lab = {"cv": "Constant-velocity rule", "chronos2_diff": "Chronos-2 (zero-shot)", "timesfm25_diff": "TimesFM 2.5 (zero-shot)"}
    rows = [[lab.get(r.model, r.model.replace("hybrid (GBM, 150 trees, one horizon)", "Hybrid (GBM, 150 trees, one horizon)")), r.device.upper(), f"{r.ms_per_aircraft:.4f}" if r.ms_per_aircraft < 0.1 else f"{r.ms_per_aircraft:.3f}", f"{r.aircraft_per_s:,.0f}".replace(",", "\\,")] for r in d.itertuples()]
    cap = "Inference cost of the predictors, model only (feature construction and the separation rule are not included), batch of 1000 aircraft. GPU: laptop RTX 3050 (6 GB); CPU: Intel (family 6, model 154)."
    return tab(cap, "tab:cost", ["Predictor", "Device", "ms per aircraft", "Aircraft per s"], rows, "llrr")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("-o", default=None); ap.add_argument("--margins", type=float, nargs="+", default=[0.05, 0.10]); a = ap.parse_args()
    T = []; f = os.path.join(a.out_dir, "a15_link_eval.csv")
    if os.path.exists(f):
        d = pd.read_csv(f); d = d[d.target == "paper_tac"]; T += link_tables(d); T.append(requirements(d, a.margins))
    for fn in (ops, calib, cost):
        t = fn(a.out_dir)
        if t: T.append(t)
    o = a.o or os.path.join(a.out_dir, "tables_comm.tex"); open(o, "w", encoding="utf-8").write("% generated by a18_comm_tables.py; needs \\usepackage{booktabs}\n" + "\n".join(T)); print(len(T), "tables ->", o)

if __name__ == "__main__": main()

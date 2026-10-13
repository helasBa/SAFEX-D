"""Step 6 - LaTeX tables (booktabs) from the JSON / CSV outputs of a1-a5.
  python a6_make_tables.py OUT_DIR            # reads OUT_DIR/{class_table.csv, event_effect_summary.csv, benchmark.json, fm_*.json} (whatever exists)
Writes OUT_DIR/tables_main.tex (compact, paper-ready) and OUT_DIR/tables_appendix.tex (exhaustive). Numbers are copied from the files, never typed by hand."""
import glob, json, os, sys, numpy as np, pandas as pd

def esc(s): return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
def tab(cap, label, head, rows, fmt=None):
    fmt = fmt or "l" + "r" * (len(head) - 1)
    return "\\begin{table}[t]\n\\centering\\small\n\\caption{%s}\\label{%s}\n\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n" % (
        cap, label, fmt, " & ".join(head), " \\\\\n".join(" & ".join(map(str, r)) for r in rows))
def ci(d, k, p=3): return f"{d[k]:.{p}f} [{d[k + '_lo']:.{p}f}, {d[k + '_hi']:.{p}f}]" if k + "_lo" in d and np.isfinite(d.get(k + "_lo", np.nan)) else f"{d[k]:.{p}f}"

SHORT = {"prevalence": "Prevalence", "density prior": "Density prior", "constant-velocity physics": "Constant-velocity rule", "learned: own kinematics": "Learned (own kinematics)",
         "learned: kinematics + neighbours": "Learned (kin.+neighbours)", "hybrid: physics + kinematics + neighbours": "Hybrid (rule+learned)"}
def pm(d, k="auprc"): return f"{d[k]:.3f}" + (f"$\\pm${(d[k + '_hi'] - d[k + '_lo']) / 2:.3f}" if k + "_lo" in d and np.isfinite(d[k + "_lo"]) else "")
def leaf(B, exp, key, tgt, h, sub="all"):
    try: return B[exp][key][f"y_{tgt}_h{h}"][str(h)][sub]
    except KeyError: return None

def main(o):
    """Compact, paper-ready tables (point estimate +/- half-width of the 95% CI over traffic groups). Every number is copied from the JSON files."""
    M = []; fb = os.path.join(o, "benchmark.json")
    if os.path.exists(fb):
        B = json.load(open(fb)); hs = (10, 30, 60)
        for tgt, ttl in [("paper_tac", "paper thresholds, LoWC or worse ($\\geq$ 496.8 m)"), ("paper_imm", "paper thresholds, NMAC or worse ($\\leq$ 152.4 m)"), ("half_tac", "half-scale thresholds, LoWC or worse")]:
            rows = []
            for m in SHORT:
                r = [SHORT[m]]
                for mode in ("free", "mvp"):
                    for h in hs:
                        a = leaf(B, "cv5", mode, tgt, h); r.append(pm(a[m]) if a and m in a else "-")
                rows.append(r)
            M.append(tab(f"Conflict-onset forecasting, 5-fold grouped cross-validation: AUPRC$\\pm$half-width of the 95\\% CI over traffic groups. Target: {ttl}. Left: ASAS off (free); right: ASAS on (MVP).", f"tab:main-{tgt}",
                         ["Model"] + [f"free {h} s" for h in hs] + [f"MVP {h} s" for h in hs], rows, "l" + "r" * 6))
        rows = []
        for exp, keys, lab in [("cv5", ("free", "mvp"), "5-fold CV"), ("zone", ("free", "mvp"), "leave-one-zone-out"), ("transfer", ("free", "mvp"), "nominal$\\to$events"), ("modeshift", ("free->mvp", "mvp->free"), "mode shift")]:
            for key in keys:
                r = [lab, esc(key).replace("->", "$\\to$")]
                for tgt in ("paper_tac", "paper_imm", "half_tac"):
                    a = leaf(B, exp, key, tgt, 30)
                    for m in ("constant-velocity physics", "hybrid: physics + kinematics + neighbours"): r.append(f"{a[m]['auprc']:.3f}" if a and m in a else "-")
                    r.append(f"{a['prevalence']['auprc']:.3f}" if a else "-")
                rows.append(r)
        M.append(tab("Robustness of the 30 s forecast across evaluation protocols (AUPRC). For each target: constant-velocity rule / hybrid / prevalence of positives.", "tab:main-robust",
                     ["Protocol", "Mode", "tac CV", "tac Hyb.", "prev.", "imm CV", "imm Hyb.", "prev.", "half CV", "half Hyb.", "prev."], rows, "ll" + "r" * 9))
        bc = []
        for key in ("free", "mvp"):
            d = B.get("transfer", {}).get(key, {}).get("y_paper_tac_h30", {}).get("30", {}).get("by_class_auprc", {})
            for c, v in d.items(): bc.append([esc(c), key] + [f"{v[m]:.3f}" if m in v else "-" for m in ("constant-velocity physics", "learned: kinematics + neighbours", "hybrid: physics + kinematics + neighbours")])
        if bc: M.append(tab("AUPRC per event class, models trained on nominal traffic only (target paper\\_tac, 30 s).", "tab:main-class", ["Event class", "Mode", "CV rule", "Learned", "Hybrid"], bc, "llrrr"))
    fms = {os.path.basename(f)[3:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(o, "fm_*.json")))}
    if fms:
        rows = []
        for tgt in ("paper_tac", "half_tac", "paper_imm"):
            for h in ("10", "30", "60"):
                for mode in ("free", "mvp"):
                    cells, ref = [], None
                    for nm, R in fms.items():
                        r = R.get(tgt, {}).get(h, {}).get(mode, {}).get("all")
                        if not r: continue
                        ks = list(r.keys()); ref = ref or r[ks[1]]; cells.append(pm(r[ks[2]]))
                    if ref and len(cells) == len(fms): rows.append([esc(tgt), h, mode, f"{ref['pos']}/{ref['n']}", f"{ref['auprc']:.3f}"] + cells)
        if rows: M.append(tab("Zero-shot time-series foundation models as trajectory predictors followed by the same separation rule, on a fixed sample of held-out episodes (AUPRC$\\pm$half-width of the 95\\% CI). CV rule = constant-velocity extrapolation of the last 15 s with the same rule. Rows with very few positives are not interpretable.",
                              "tab:main-fm", ["Target", "h (s)", "Mode", "Pos./rows", "CV rule"] + [esc(n) for n in fms], rows, "lllr" + "r" * (1 + len(fms))))

    for f in sorted(glob.glob(os.path.join(o, "fm_*.json"))):
        TJ = json.load(open(f)).get("trajectory_error_m")
        if not TJ: continue
        rows = []
        for mode in ["free", "mvp"]:
            for sub in ["all", "maneuvering"]:
                for m, v in TJ.get(mode, {}).get(sub, {}).items():
                    if m == "n": continue
                    rows.append([mode, sub, esc(m), TJ[mode][sub][m]["n"] if "n" in TJ[mode][sub][m] else "-"] + [f"{v[k][0]:.1f} [{v[k][1]:.1f}, {v[k][2]:.1f}]" for k in ("ade", "fde10", "fde30", "fde60")])
        M.append(tab(f"Trajectory error of the zero-shot predictor against the true future position (metres, 95\\% CI over traffic groups) ({esc(os.path.basename(f))}).", f"tab:traj-{os.path.basename(f)[:-5]}",
                     ["Mode", "Subset", "Predictor", "Rows", "ADE", "FDE 10 s", "FDE 30 s", "FDE 60 s"], rows, "lllrrrrr"))
    open(os.path.join(o, "tables_main.tex"), "w").write("\n".join(M)); print(len(M), "paper tables ->", os.path.join(o, "tables_main.tex"))

if __name__ == "__main__":
    o = sys.argv[1]; T = []
    f = os.path.join(o, "class_table.csv")
    if os.path.exists(f):
        c = pd.read_csv(f); rows = [[esc(r.cls), r["mode"], int(r.n), f"{r.conc_mean_mean:.0f}", f"{r.lowc_events_mean:.0f}", f"{r.nmac_events_mean:.0f}", f"{100 * r.paper_sev_ge1_share_mean:.1f}", f"{100 * r.paper_sev_ge2_share_mean:.2f}", f"{100 * r.half_sev_ge1_share_mean:.1f}"] for _, r in c.iterrows()]
        T.append(tab("Corpus summary per class and simulation mode (mean over episodes). Severity shares are percentages of aircraft-seconds.", "tab:corpus", ["Class", "Mode", "Ep.", "Conc. UAV", "LoWC ev.", "NMAC ev.", "Sev$\\geq$1 (\\%)", "Sev$\\geq$2 (\\%)", "Half sev$\\geq$1 (\\%)"], rows, "llrrrrrrr"))
    f = os.path.join(o, "event_effect_summary.csv")
    if os.path.exists(f):
        e = pd.read_csv(f); rows = [[esc(r.cls), r["mode"], int(r.n_pairs), f"{r.uavs_differ_mean:.0f}", f"{r.d_lowc_events:+.0f} [{r.d_lowc_events_lo:+.0f}, {r.d_lowc_events_hi:+.0f}]" if np.isfinite(r.d_lowc_events_lo) else f"{r.d_lowc_events:+.0f}",
                f"{r.d_half_ge1_pp:+.2f}", f"{r.d_sev3_rows:+.0f}", f"{r.ev_rep_err_xy_rms_m:.1f}"] for _, r in e.iterrows()]
        T.append(tab("Measured effect of each event against its nominal twin (same traffic, seed, mode): aircraft whose true trajectory changes, change in loss-of-well-clear events, change in the share of aircraft-seconds with half-profile severity$\\geq$1 (pp), change in collision-level aircraft-seconds, and measured RMS error of the reported horizontal position (m).",
                     "tab:events", ["Event", "Mode", "Pairs", "UAV changed", "$\\Delta$ LoWC events [95\\% CI]", "$\\Delta$ half sev$\\geq$1 (pp)", "$\\Delta$ sev 3 rows", "Rep. err (m)"], rows, "llrrrrrr"))
    f = os.path.join(o, "benchmark.json")
    if os.path.exists(f):
        B = json.load(open(f))
        for exp, ttl in [("cv5", "5-fold grouped cross-validation"), ("zone", "leave-one-zone-out"), ("transfer", "nominal $\\rightarrow$ events"), ("modeshift", "ASAS off $\\leftrightarrow$ ASAS on")]:
            for key, byt in B.get(exp, {}).items():
                for tgt, byh in byt.items():
                    hs = sorted(byh, key=int); models = list(byh[hs[0]].get("all", {}).keys()) if hs else []
                    if not models: continue
                    rows = [[esc(m)] + [ci(byh[h]["all"][m], "auprc") if m in byh[h].get("all", {}) else "-" for h in hs] + [ci(byh[h]["maneuvering"][m], "auprc") if "maneuvering" in byh[h] and m in byh[h]["maneuvering"] else "-" for h in hs] for m in models]
                    T.append(tab(f"AUPRC [95\\% CI over traffic groups], {ttl}, {esc(key)}, target {esc(tgt)}. Left: all rows; right: maneuvering aircraft.", f"tab:{exp}-{key}-{tgt}".replace(">", "").replace(" ", ""),
                                 ["Model"] + [f"all {h}s" for h in hs] + [f"man. {h}s" for h in hs], rows, "l" + "r" * (2 * len(hs))))
    for f in sorted(glob.glob(os.path.join(o, "fm_*.json"))):
        R = json.load(open(f))
        for tgt, byh in R.items():
            if tgt == "trajectory_error_m": continue
            hs = sorted(byh, key=int); r0 = byh[hs[0]].get("all", {}).get("all", {}); models = list(r0.keys())
            if not models: continue
            rows = [[esc(m)] + [ci(byh[h]["all"]["all"][m], "auprc") for h in hs] + [ci(byh[h]["all"]["maneuvering"][m], "auprc") if "maneuvering" in byh[h]["all"] else "-" for h in hs] for m in models]
            T.append(tab(f"Foundation-model trajectory prediction followed by the separation rule: AUPRC [95\\% CI], target {esc(tgt)} ({esc(os.path.basename(f))}).", f"tab:fm-{os.path.basename(f)[:-5]}-{tgt}", ["Predictor"] + [f"all {h}s" for h in hs] + [f"man. {h}s" for h in hs], rows, "l" + "r" * (2 * len(hs))))
    open(os.path.join(o, "tables_appendix.tex"), "w").write("\n".join(T)); print(len(T), "exhaustive tables ->", os.path.join(o, "tables_appendix.tex"))
    main(o)

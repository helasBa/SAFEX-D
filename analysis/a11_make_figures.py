"""Step 11 - figures for the paper and the presentation (PNG 300 dpi + vector PDF).  Every figure reads the result files already produced by a1-a10; a figure whose input is missing is skipped.
  python a11_make_figures.py C:\\these\\safex2_out [-o C:\\these\\safex2_out\\figures]
Inputs looked for in the folder (names as in the previous steps):
  benchmark.json (or benchmark_full.json)   event_effect_summary.csv   a7_leakage.json   fm_chronos2_diff_big.json
  a8_ml_vs_fm.json  a8_ml_vs_fm_f1.csv   a9_hybrid_fm_paired.csv   fm_chronos2_base_fold4.json  fm_chronos2_ft_fold4.json  a8_ft_fold4_paired.csv
Figures:
  fig01_degradation_error      reported-position error of every degradation class against the nominal level (reviewer point 3)
  fig02_leakage_controls       leakage audit: shuffled-label control, true-label control, ablations (reviewer point 1)
  fig03_benchmark_horizons     AUPRC against horizon, physics / learned / hybrid, mode free and mvp, three targets (95% CI)
  fig04_protocols              the same models under cross-validation, leave-one-zone-out, nominal->event and mode shift
  fig05_class_transfer         AUPRC per scenario class, trained on nominal episodes only
  fig06_fm_horizons            foundation models next to physics, learned and hybrid on the same rows
  fig07_fm_paired_forest       paired AUPRC differences with 95% CI (foundation model - physics, hybrid - foundation model, hybrid+FM - hybrid)
  fig08_trajectory_error       ADE / FDE: physics, pre-trained Chronos-2, fine-tuned Chronos-2 (held-out fold)
  fig09_finetune_gain          AUPRC gain over physics, pre-trained against fine-tuned (held-out fold)
  fig10_f1                     precision, recall, F1 and F1-macro at an operating point chosen on the other folds
Colours: fixed categorical order, one colour per entity in every figure (validated palette, adjacent-pair CVD check passed; the three light slots below 3:1 contrast are labelled directly and use distinct markers)."""
import argparse, json, os, glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
COL = {"cv": "#2a78d6", "learned": "#eb6834", "hybrid": "#1baf7a", "chronos": "#eda100", "timesfm": "#e87ba4", "hyb_fm": "#6250d6", "base": "#9a9994", "base2": "#c9c8c3"}
STY = {  # key -> (label, colour, marker, linestyle)
    "prevalence": ("prevalence", COL["base2"], "", ":"), "density prior": ("density prior", COL["base"], "", "--"),
    "constant-velocity physics": ("constant velocity (5 s)", COL["cv"], "o", "-"), "constant-velocity (a2, 5 s window)": ("constant velocity (5 s)", COL["cv"], "o", "-"),
    "constant-velocity (harness, 15 s window)": ("constant velocity (15 s)", COL["cv"], "s", "--"),
    "learned: own kinematics": ("learned, own kinematics", "#f3a883", "s", "-"), "learned: kinematics + neighbours": ("learned, kinematics + neighbours", COL["learned"], "s", "-"),
    "hybrid: physics + kinematics + neighbours": ("hybrid", COL["hybrid"], "^", "-"), "fm_chronos2_diff": ("Chronos-2 (zero-shot)", COL["chronos"], "D", "-"), "fm_timesfm25_diff": ("TimesFM 2.5 (zero-shot)", COL["timesfm"], "v", "-"),
    "hybrid + FM": ("hybrid + Chronos-2", COL["hyb_fm"], "P", "-"), "foundation model alone": ("Chronos-2 (zero-shot)", COL["chronos"], "D", "-")}
TGT = {"y_paper_tac": "paper thresholds, LoWC (497 m)", "y_paper_imm": "paper thresholds, NMAC (152 m)", "y_half_tac": "half thresholds, LoWC (248 m)"}
HS = (10, 30, 60)
SAVED = []

def style():
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "legend.frameon": False, "axes.titlesize": 9.5, "axes.titleweight": "regular", "axes.titlecolor": INK, "lines.linewidth": 1.6, "lines.markersize": 5.5})

def save(fig, outdir, name):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); SAVED.append(name); print("  ok ", name)

def find(d, *names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p): return p
    return None

def legend_for(fig, keys, ncol=4, y=-0.02):
    hs = [Line2D([0], [0], color=STY[k][1], marker=STY[k][2] or None, ls=STY[k][3], lw=1.6, ms=5.5) for k in keys]
    fig.legend(hs, [STY[k][0] for k in keys], loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), fontsize=8.5, handlelength=2.6)

def line_panel(ax, R, y, keys, sub="all", lo="auprc_lo", hi="auprc_hi"):
    for k in keys:
        pts = []
        for h in HS:
            r = R.get(y, {}).get(str(h), {}) if False else None
        # R is dict h -> {sub -> {method -> metrics}}
    return

def series(blob, y, keys, sub="all", lo="auprc_lo", hi="auprc_hi"):
    """blob[y][h][sub][method] -> {k: (vals, lows, highs)} over HS (nan if missing)"""
    out = {}
    for k in keys:
        v, l, u = [], [], []
        for h in HS:
            m = blob.get(y, {}).get(str(h), {}).get(sub, {}).get(k)
            if m is None: v.append(np.nan); l.append(np.nan); u.append(np.nan)
            else: v.append(m["auprc"]); l.append(m.get(lo, m["auprc"])); u.append(m.get(hi, m["auprc"]))
        out[k] = (np.array(v), np.array(l), np.array(u))
    return out

def draw_lines(ax, S, keys, label_end=None):
    for k in keys:
        v, l, u = S[k]; lab, c, mk, ls = STY[k]
        if np.all(np.isnan(v)): continue
        ax.plot(HS, v, color=c, marker=mk or None, ls=ls, label=lab, zorder=3)
        if np.isfinite(l).all() and np.nanmax(u - l) > 0 and k not in ("prevalence",): ax.fill_between(HS, l, u, color=c, alpha=0.15, lw=0, zorder=2)
    ax.set_xticks(HS); ax.set_xticklabels([f"{h} s" for h in HS]); ax.set_xlim(7, 63)

def fig_degradation(d, out):
    p = find(d, "event_effect_summary.csv")
    if not p: return
    e = pd.read_csv(p); e = e[e["mode"] == "free"].copy(); nom = float(e[e.cls == "CONT-INTRUDER"].nom_rep_err_xy_rms_m.iloc[0])
    e = e[e.cls.str.startswith("DEG")].sort_values("ev_rep_err_xy_rms_m"); fig, ax = plt.subplots(figsize=(6.4, 3.0))
    cols = [COL["cv"] if v > 3.5 else COL["base"] for v in e.ev_rep_err_xy_rms_m]
    ax.barh(e.cls.str.replace("DEG-", ""), e.ev_rep_err_xy_rms_m, color=cols, height=0.62, zorder=3)
    for y, v in enumerate(e.ev_rep_err_xy_rms_m): ax.text(v * 1.08, y, f"{v:.1f} m", va="center", fontsize=8.5, color=INK)
    ax.axvline(nom, color=INK2, lw=1, ls="--", zorder=4); ax.text(nom * 1.08, len(e) - 0.35, f"nominal level, {nom:.1f} m", fontsize=8, color=INK2, va="center")
    ax.set_xscale("log"); ax.set_xlim(2, 400); ax.set_xticks([3, 10, 30, 100, 300]); ax.set_xticklabels(["3", "10", "30", "100", "300"]); ax.minorticks_off(); ax.set_xlabel("RMS horizontal error of the reported position after the event (m, log scale)"); ax.grid(axis="y", visible=False)
    ax.set_title("Reported-position error by degradation class (SENSOR-FAULT acts on altitude only)", loc="left", fontsize=9, pad=8)
    save(fig, out, "fig01_degradation_error")

def fig_leak(d, out):
    p = find(d, "a7_leakage.json")
    if not p: return
    D = json.load(open(p))["D_controls"]; fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    order = [("NEG control: labels shuffled inside each episode", "labels shuffled", COL["base"]), ("ablation: without physics score", "without physics score", COL["learned"]), ("ablation: without report age", "without report age", COL["cv"]),
             ("ablation: without phase and type", "without phase and type", COL["cv"]), ("full (hybrid inputs)", "full inputs", COL["hybrid"]), ("POS control: true future label added as a feature", "true label added", COL["base2"])]
    for ax, mode in zip(axs, ("free", "mvp")):
        if mode not in D: continue
        rows = [(lab, D[mode][k]["auprc"], c) for k, lab, c in order if k in D[mode]]; prev = list(D[mode].values())[0]["prevalence"]
        ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], height=0.62, zorder=3)
        for i, r in enumerate(rows): ax.text(r[1] + 0.012, i, f"{r[1]:.3f}", va="center", fontsize=8.5)
        ax.axvline(prev, color=INK2, ls="--", lw=1, zorder=4); ax.set_xlim(0, 1.12); ax.set_title(f"mode {mode}", loc="left"); ax.set_xlabel("AUPRC (target half, LoWC, 30 s)"); ax.grid(axis="y", visible=False)
        ax.text(prev + 0.01, -0.62, f"prevalence {prev:.2f}", fontsize=8, color=INK2, va="center")
    fig.suptitle("Leakage audit: shuffled labels fall back to prevalence, the true label gives 1.0, and removing the physics score costs the most", x=0.01, ha="left", fontsize=9, y=1.02)
    save(fig, out, "fig02_leakage_controls")

def load_bench(d):
    p = find(d, "benchmark.json", "benchmark_full.json"); return json.load(open(p)) if p else None

def fig_bench_horizons(d, out):
    B = load_bench(d)
    if not B: return
    keys = ["prevalence", "density prior", "constant-velocity physics", "learned: kinematics + neighbours", "hybrid: physics + kinematics + neighbours"]; tg = list(TGT)
    fig, axs = plt.subplots(2, 3, figsize=(8.4, 5.0), sharex=True)
    for j, y in enumerate(tg):
        for i, mode in enumerate(("free", "mvp")):
            ax = axs[i, j]; blob = {f"{y}_h{h}": None for h in HS}; S = {}
            blob = {yy: B["cv5"][mode].get(yy, {}) for yy in [f"{y}_h{h}" for h in HS]}
            # blob[y_h][h] -> sub -> method; build per-h lookup
            for k in keys:
                v, l, u = [], [], []
                for h in HS:
                    m = blob[f"{y}_h{h}"].get(str(h), {}).get("all", {}).get(k)
                    v.append(m["auprc"] if m else np.nan); l.append(m.get("auprc_lo", m["auprc"]) if m else np.nan); u.append(m.get("auprc_hi", m["auprc"]) if m else np.nan)
                S[k] = (np.array(v), np.array(l), np.array(u))
            draw_lines(ax, S, keys); ax.set_ylim(0, 1.0)
            if i == 0: ax.set_title(TGT[y], loc="left")
            if j == 0: ax.set_ylabel(f"mode {mode}\nAUPRC")
            if i == 1: ax.set_xlabel("prediction horizon")
    legend_for(fig, keys, ncol=3, y=-0.07); fig.suptitle("Forecasting benchmark, five-fold cross-validation grouped by traffic group (full corpus, band = 95% CI over traffic groups)", x=0.01, ha="left", fontsize=9.5, y=0.99)
    save(fig, out, "fig03_benchmark_horizons")

def fig_protocols(d, out):
    B = load_bench(d)
    if not B: return
    prot = [("cv5", "free", "cross-val.\nfree"), ("cv5", "mvp", "cross-val.\nmvp"), ("zone", "free", "leave one\nzone out"), ("transfer", "free", "nominal ->\nevents"), ("modeshift", "free->mvp", "free ->\nmvp"), ("modeshift", "mvp->free", "mvp ->\nfree")]
    keys = ["constant-velocity physics", "learned: kinematics + neighbours", "hybrid: physics + kinematics + neighbours"]; fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.2), sharey=False)
    for ax, (y, h) in zip(axs, (("y_paper_tac", 30), ("y_paper_imm", 30), ("y_half_tac", 30))):
        w = 0.26
        for i, k in enumerate(keys):
            vals = [B[e][m].get(f"{y}_h{h}", {}).get(str(h), {}).get("all", {}).get(k, {}).get("auprc", np.nan) for e, m, _ in prot]
            ax.bar(np.arange(len(prot)) + (i - 1) * w, vals, w * 0.92, color=STY[k][1], label=STY[k][0], zorder=3)
        ax.set_xticks(range(len(prot))); ax.set_xticklabels([p[2] for p in prot], fontsize=7.2); ax.set_title(f"{TGT[y]}, 30 s", loc="left", fontsize=8.8); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("AUPRC"); legend_for(fig, keys, ncol=3, y=-0.08)
    fig.suptitle("Hybrid and physics hold up across protocols; learned models lose most of their skill when the resolution mode changes", x=0.01, ha="left", fontsize=9.5, y=1.02)
    save(fig, out, "fig04_protocols")

def fig_class(d, out):
    B = load_bench(d)
    if not B: return
    keys = ["constant-velocity physics", "learned: kinematics + neighbours", "hybrid: physics + kinematics + neighbours"]; fig, axs = plt.subplots(1, 2, figsize=(8.2, 5.4), sharey=True)
    for ax, mode in zip(axs, ("free", "mvp")):
        bc = B["transfer"][mode]["y_paper_tac_h30"]["30"].get("by_class_auprc", {}); cls = sorted(bc, key=lambda c: bc[c].get("hybrid: physics + kinematics + neighbours", 0))
        for k in keys: ax.scatter([bc[c].get(k, np.nan) for c in cls], range(len(cls)), color=STY[k][1], marker=STY[k][2], s=26, label=STY[k][0], zorder=3, edgecolors=SURF, linewidths=0.6)
        ax.set_yticks(range(len(cls))); ax.set_yticklabels(cls, fontsize=7.5); ax.set_xlim(0.45, 1.0); ax.set_title(f"mode {mode}", loc="left"); ax.set_xlabel("AUPRC (paper thresholds, LoWC, 30 s)"); ax.grid(axis="x")
    legend_for(fig, keys, ncol=3, y=-0.04); fig.suptitle("Trained on nominal episodes only, tested on contingency and degradation classes of unseen traffic groups", x=0.01, ha="left", fontsize=9.5, y=0.995)
    save(fig, out, "fig05_class_transfer")

def fig_fm_horizons(d, out):
    p = find(d, "a8_ml_vs_fm.json")
    if not p: return
    R = json.load(open(p)); keys = ["constant-velocity (a2, 5 s window)", "constant-velocity (harness, 15 s window)", "learned: kinematics + neighbours", "hybrid: physics + kinematics + neighbours", "fm_chronos2_diff", "fm_timesfm25_diff"]
    fig, axs = plt.subplots(2, 3, figsize=(8.4, 5.0), sharex=True)
    for j, y in enumerate(TGT):
        for i, mode in enumerate(("free", "mvp")):
            ax = axs[i, j]; blob = R.get("cv5", {}).get(mode, {}); S = {}
            for k in keys:
                v, l, u = [], [], []
                for h in HS:
                    m = blob.get(f"{y}_h{h}", {}).get(str(h), {}).get("all", {}).get(k)
                    v.append(m["auprc"] if m else np.nan); l.append(m["lo"] if m else np.nan); u.append(m["hi"] if m else np.nan)
                S[k] = (np.array(v), np.array(l), np.array(u))
            draw_lines(ax, S, keys); ax.set_ylim(0, 1.0)
            if i == 0: ax.set_title(TGT[y], loc="left")
            if j == 0: ax.set_ylabel(f"mode {mode}\nAUPRC")
            if i == 1: ax.set_xlabel("prediction horizon")
    legend_for(fig, keys, ncol=3, y=-0.08); fig.suptitle("Zero-shot foundation models against physics, learned and hybrid models on the same rows (24 episodes, cross-validation)", x=0.01, ha="left", fontsize=9.5, y=0.99)
    save(fig, out, "fig06_fm_horizons")

def forest(ax, df, comp, label, colour):
    s = df[(df.comparison == comp) & (df["subset"] == "all")].copy()
    s = s[s.pos >= 100]; s["lab"] = s.target.str.replace("y_", "").str.replace("_h", " ") + " s, " + s["mode"]; s = s.sort_values(["target", "mode", "horizon"], ascending=[False, True, False]).reset_index(drop=True)
    for i, r in s.iterrows():
        sig = r.lo > 0 or r.hi < 0
        ax.plot([r.lo, r.hi], [i, i], color=colour, lw=1.4, zorder=3, alpha=1 if sig else 0.45)
        ax.scatter([r["diff"]], [i], color=colour if sig else SURF, edgecolors=colour, s=22, zorder=4, linewidths=1.2)
    ax.axvline(0, color=INK2, lw=0.9, zorder=2); ax.set_yticks(range(len(s))); ax.set_yticklabels(s.lab, fontsize=7); ax.set_title(label, loc="left", fontsize=8.6); ax.grid(axis="y", visible=False)
    return len(s)

def fig_forest(d, out):
    p = find(d, "a9_hybrid_fm_paired.csv")
    if not p: return
    df = pd.read_csv(p); fig, axs = plt.subplots(1, 3, figsize=(9.4, 5.2), sharey=True)
    forest(axs[0], df, "foundation model alone - constant-velocity (harness, 15 s window)", "Chronos-2 minus physics (15 s)", COL["chronos"])
    forest(axs[1], df, "hybrid - foundation model alone", "hybrid minus Chronos-2", COL["hybrid"])
    forest(axs[2], df, "hybrid + FM - hybrid", "(hybrid + Chronos-2) minus hybrid", COL["hyb_fm"])
    for ax in axs: ax.set_xlabel("difference in AUPRC")
    fig.suptitle("Paired differences with 95% CI (120 episodes). Filled dot: interval excludes 0. Cases with fewer than 100 positives are left out", x=0.01, ha="left", fontsize=9.2, y=0.995)
    save(fig, out, "fig07_fm_paired_forest")

def fig_traj(d, out):
    pb, pf = find(d, "fm_chronos2_base_fold4.json"), find(d, "fm_chronos2_ft_fold4.json")
    if not (pb and pf): return
    b, f = json.load(open(pb))["trajectory_error_m"], json.load(open(pf))["trajectory_error_m"]; cells = [("free", "all", "free\nall UAVs"), ("free", "maneuvering", "free\nmanoeuvring"), ("mvp", "all", "mvp\nall UAVs"), ("mvp", "maneuvering", "mvp\nmanoeuvring")]
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2)); series_ = [("constant velocity (15 s)", COL["cv"], lambda m, s: b[m][s]["constant-velocity"]), ("Chronos-2 pre-trained", COL["chronos"], lambda m, s: b[m][s]["chronos2_diff"]), ("Chronos-2 fine-tuned", "#8a5f00", lambda m, s: f[m][s]["chronos2_diff"])]
    for ax, met, ttl in zip(axs, ("ade", "fde60"), ("mean error over the 60 s horizon (ADE)", "error at 60 s (FDE)")):
        w = 0.26
        for i, (nm, c, fn) in enumerate(series_):
            vals = np.array([fn(m, s)[met] for m, s, _ in cells]); ax.bar(np.arange(4) + (i - 1) * w, vals[:, 0], w * 0.92, color=c, label=nm, zorder=3, yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], error_kw=dict(ecolor=INK2, lw=0.9, capsize=2))
        ax.set_xticks(range(4)); ax.set_xticklabels([c[2] for c in cells], fontsize=8); ax.set_ylabel("metres"); ax.set_title(ttl, loc="left"); ax.grid(axis="x", visible=False)
    h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=8.5); fig.suptitle("Trajectory error against the true future position, held-out fold (95% CI over traffic groups)", x=0.01, ha="left", fontsize=9.5, y=1.02)
    save(fig, out, "fig08_trajectory_error")

def fig_ft(d, out):
    p = find(d, "a8_ft_fold4_paired.csv")
    if not p: return
    df = pd.read_csv(p); df = df[(df.pos >= 100)]; cvn = "constant-velocity (harness, 15 s window)"
    a = df[df.comparison == f"fm_chronos2_base_fold4 - {cvn}"].set_index(["target", "mode", "horizon"]); b = df[df.comparison == f"fm_chronos2_ft_fold4 - {cvn}"].set_index(["target", "mode", "horizon"])
    idx = [i for i in b.index if i in a.index]; idx = sorted(idx, key=lambda t: (t[0], t[1], -t[2])); fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for k, i in enumerate(idx):
        ax.plot([a.loc[i, "diff"], b.loc[i, "diff"]], [k, k], color=GRID, lw=2, zorder=2)
        for r, c, mk in ((a.loc[i], COL["chronos"], "D"), (b.loc[i], "#8a5f00", "o")):
            ax.plot([r.lo, r.hi], [k, k], color=c, lw=0.9, alpha=0.6, zorder=3); ax.scatter([r["diff"]], [k], color=c, marker=mk, s=26, zorder=4, edgecolors=SURF, linewidths=0.6)
    ax.axvline(0, color=INK2, lw=0.9); ax.set_yticks(range(len(idx))); ax.set_yticklabels([f"{t[0].replace('y_', '').rsplit('_h', 1)[0].replace('_', ' ')}, {t[2]} s, {t[1]}" for t in idx], fontsize=7.4); ax.set_xlabel("AUPRC gain over constant velocity (15 s), 95% CI"); ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([0], [0], color=COL["chronos"], marker="D", ls="", ms=5, label="Chronos-2 pre-trained"), Line2D([0], [0], color="#8a5f00", marker="o", ls="", ms=5, label="Chronos-2 fine-tuned (LoRA)")], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)
    ax.set_title("Light fine-tuning moves Chronos-2 from slightly below to slightly above physics (single held-out fold)", loc="left", fontsize=9)
    save(fig, out, "fig09_finetune_gain")

def fig_f1(d, out):
    p = find(d, "a8_ml_vs_fm_f1.csv")
    if not p: return
    f = pd.read_csv(p); f = f[(f.protocol == "cv5") & (f["mode"] == "free") & (f.target == "y_paper_tac") & (f["subset"] == "all")] if "y_paper_tac" in set(f.target) else f
    f = f[(f.protocol == "cv5") & (f["mode"] == "free") & (f.target.str.startswith("y_paper_tac")) & (f["subset"] == "all")]
    keys = ["constant-velocity (a2, 5 s window)", "constant-velocity (harness, 15 s window)", "learned: kinematics + neighbours", "hybrid: physics + kinematics + neighbours", "fm_chronos2_diff", "fm_timesfm25_diff"]
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.2), sharey=True)
    for ax, h in zip(axs, HS):
        g = f[f.horizon == h].set_index("method"); w = 0.38
        for i, (col, ttl) in enumerate((("f1", "F1 (event class)"), ("f1_macro", "F1-macro"))):
            vals = [g.loc[k, col] if k in g.index else np.nan for k in keys]; lo = [g.loc[k, col + "_lo"] if k in g.index else np.nan for k in keys]; hi = [g.loc[k, col + "_hi"] if k in g.index else np.nan for k in keys]
            ax.bar(np.arange(len(keys)) + (i - 0.5) * w, vals, w * 0.92, color=[STY[k][1] for k in keys], alpha=1 if i == 0 else 0.55, zorder=3, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], error_kw=dict(ecolor=INK2, lw=0.8, capsize=1.5))
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(["CV 5 s", "CV 15 s", "learned", "hybrid", "Chronos-2", "TimesFM"], fontsize=7.5, rotation=40, ha="right"); ax.set_title(f"horizon {h} s", loc="left"); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("score"); fig.suptitle("F1 (solid) and F1-macro (light) at a threshold chosen on the other folds. Paper thresholds, LoWC, mode free", x=0.01, ha="left", fontsize=9.3, y=1.02)
    save(fig, out, "fig10_f1")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("-o", default=None); a = ap.parse_args(); out = a.o or os.path.join(a.dir, "figures"); os.makedirs(out, exist_ok=True); style()
    for fn in (fig_degradation, fig_leak, fig_bench_horizons, fig_protocols, fig_class, fig_fm_horizons, fig_forest, fig_traj, fig_ft, fig_f1):
        try: fn(a.dir, out)
        except Exception as ex: print("  FAILED", fn.__name__, type(ex).__name__, ex)
    print(f"{len(SAVED)} figures in {out}")

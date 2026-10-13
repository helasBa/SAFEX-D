"""Step 17 - figures for the "communications and AI" part of the paper (PNG 300 dpi + PDF), same style and colours as a11.
  python a17_comm_figures.py C:\\these\\safex2_out [-o C:\\these\\safex2_out\\figures] [--horizon 30]
Inputs looked for in the folder (a figure whose input is missing is skipped):
  a13_age_strata.csv  a13_degraded_ops.csv  a13_calibration.csv  a13_reliability.csv  a13_perm_importance.csv  a15_link_eval.csv  a16_inference_cost.csv
  fig11_age_strata         AUPRC against the age of the last report (fresh, 1-10 s, 10-30 s, >=30 s), 95% CI
  fig12_degraded_ops_<m>   recall and false-alarm rate per degradation class after the event, alert threshold fixed on nominal traffic (5% false alarms)
  fig13_reliability        reliability diagrams (isotonic recalibration on the other folds; raw hybrid probabilities for comparison)
  fig14_perm_importance    AUPRC lost when one group of inputs is shuffled (hybrid), mean and spread over folds
  fig15_link_sweep         AUPRC against update period, latency and packet loss: trained on the same link / trained on the ideal link / physics rule
  fig16_inference_cost     milliseconds per aircraft for each predictor (log scale)"""
import argparse, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from a11_make_figures import style, save, COL, INK, INK2, GRID, SURF, find
M_COL = {"constant-velocity rule": COL["cv"], "learned": COL["learned"], "hybrid": COL["hybrid"], "hybrid, no report age": "#6250d6"}
M_MK = {"constant-velocity rule": "o", "learned": "s", "hybrid": "^", "hybrid, no report age": "D"}
MODE_T = {"free": "ASAS off (free)", "mvp": "ASAS on (MVP)"}

def fig11(R, o, hs):
    d = pd.read_csv(R)
    if d.empty: return
    modes = [m for m in ("free", "mvp") if m in d["mode"].unique()]; hs = [h for h in hs if h in d.horizon.unique()]
    fig, ax = plt.subplots(len(modes), len(hs), figsize=(3.4 * len(hs), 2.9 * len(modes)), squeeze=False, sharey=True)
    for i, m in enumerate(modes):
        for j, h in enumerate(hs):
            a = ax[i, j]; x = d[(d["mode"] == m) & (d.horizon == h)]; bins = [b for b in ["fresh (<1 s)", "1-10 s", "10-30 s", ">=30 s"] if b in set(x.age_bin)]
            for k, mod in enumerate(M_COL):
                z = x[x.model == mod].set_index("age_bin").reindex(bins)
                xs = np.arange(len(bins)) + (k - 1.5) * 0.07
                a.errorbar(xs, z.auprc, yerr=[z.auprc - z.auprc_lo, z.auprc_hi - z.auprc], color=M_COL[mod], marker=M_MK[mod], ms=5, lw=1.4, capsize=2, label=mod)
            pv = x[x.model == "hybrid"].set_index("age_bin").reindex(bins).prevalence
            a.plot(np.arange(len(bins)), pv, color=COL["base2"], ls=":", marker="x", ms=4, label="prevalence")
            a.set_xticks(range(len(bins))); a.set_xticklabels(bins, fontsize=7.5); a.set_title(f"{MODE_T[m]}, {h} s horizon", loc="left"); a.set_ylim(0, 1.02)
            if j == 0: a.set_ylabel("AUPRC")
            if i == len(modes) - 1: a.set_xlabel("age of the last report")
    h_, l_ = ax[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), fontsize=8.5); fig.tight_layout(rect=(0, 0.02, 1, 1)); save(fig, o, "fig11_age_strata")

def fig12(R, o, h):
    d = pd.read_csv(R); d = d[d.horizon == h]
    for m in [m for m in ("free", "mvp") if m in d["mode"].unique()]:
        x = d[(d["mode"] == m) & d.subset.str.endswith("(after event)")]
        if x.empty: continue
        cls = sorted(x.subset.unique(), key=lambda s: -x[(x.subset == s) & (x.model == "hybrid")].recall.mean()); cls = [c for c in cls]
        fig, ax = plt.subplots(1, 2, figsize=(8.6, 0.55 * len(cls) + 1.8), sharey=True); mods = ["constant-velocity rule", "learned", "hybrid"]
        for k, (col, ttl) in enumerate([("recall", "share of conflicts that raise an alert"), ("false_alarm_rate", "share of non-conflicts that raise an alert")]):
            for q, mod in enumerate(mods):
                v = [x[(x.subset == c) & (x.model == mod)][col].mean() for c in cls]; ax[k].barh(np.arange(len(cls)) + (q - 1) * 0.26, v, height=0.24, color=M_COL[mod], label=mod)
            ax[k].set_title(ttl, loc="left"); ax[k].set_yticks(range(len(cls))); ax[k].set_yticklabels([c.replace(" (after event)", "") for c in cls], fontsize=8); ax[k].invert_yaxis(); ax[k].set_xlim(0, 1)
        ax[1].axvline(0.05, color=INK2, lw=0.8, ls="--"); ax[1].text(0.06, -0.4, "5% on nominal traffic", fontsize=7.5, color=INK2, va="top")
        fig.suptitle(f"{MODE_T[m]}, {h} s horizon: alert threshold fixed on nominal traffic of the other folds", x=0.01, ha="left", fontsize=9.5)
        h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8.5); fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, o, f"fig12_degraded_ops_{m}")

def fig13(R, o, h):
    d = pd.read_csv(R); d = d[d.horizon == h]
    if d.empty: return
    modes = [m for m in ("free", "mvp") if m in d["mode"].unique()]; fig, ax = plt.subplots(1, len(modes), figsize=(3.9 * len(modes), 3.7), squeeze=False)
    for j, m in enumerate(modes):
        a = ax[0, j]; a.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        for (mod, var), sty in {("hybrid", "isotonic"): (COL["hybrid"], "^", "-", "hybrid, recalibrated"), ("learned", "isotonic"): (COL["learned"], "s", "-", "learned, recalibrated"),
                                ("constant-velocity rule", "isotonic"): (COL["cv"], "o", "-", "rule, recalibrated"), ("hybrid", "raw"): ("#9a9994", "^", ":", "hybrid, raw output")}.items():
            z = d[(d["mode"] == m) & (d.model == mod) & (d.variant == var)].sort_values("bin")
            if len(z): a.plot(z.mean_pred, z.obs_freq, color=sty[0], marker=sty[1], ls=sty[2], ms=4.5, label=sty[3])
        a.set_xlabel("predicted probability"); a.set_title(MODE_T[m], loc="left"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        if j == 0: a.set_ylabel("observed frequency")
    ax[0, 0].legend(fontsize=7.5, loc="upper left"); fig.tight_layout(); save(fig, o, "fig13_reliability")

def fig14(R, o):
    d = pd.read_csv(R)
    if d.empty: return
    hs = sorted(d.horizon.unique()); modes = [m for m in ("free", "mvp") if m in d["mode"].unique()]
    fig, ax = plt.subplots(len(modes), len(hs), figsize=(3.6 * len(hs), 2.5 * len(modes)), squeeze=False)
    order = ["physics score", "neighbours (now)", "history (lags)", "own kinematics", "report age"]
    for i, m in enumerate(modes):
        for j, h in enumerate(hs):
            x = d[(d["mode"] == m) & (d.horizon == h)].groupby("group").auprc_drop.agg(["mean", "std"]).reindex(order)
            ax[i, j].barh(range(len(order)), x["mean"], xerr=x["std"].fillna(0), color=COL["hybrid"], height=0.6, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
            ax[i, j].set_yticks(range(len(order))); ax[i, j].set_yticklabels(order if j == 0 else [], fontsize=8); ax[i, j].invert_yaxis(); ax[i, j].set_title(f"{MODE_T[m]}, {h} s", loc="left")
            if i == len(modes) - 1: ax[i, j].set_xlabel("AUPRC lost when the inputs are shuffled")
    fig.tight_layout(); save(fig, o, "fig14_perm_importance")

def fig15(R, o, h):
    d = pd.read_csv(R); d = d[d.horizon == h]
    if d.empty: return
    fam = {"update period": ["base", "P2", "P5", "P10"], "latency": ["base", "L2", "L5", "L10"], "packet loss": ["base", "loss10", "loss30", "loss50"]}
    lab = {"base": "ideal\n1 Hz", "P2": "2 s", "P5": "5 s", "P10": "10 s", "L2": "2 s", "L5": "5 s", "L10": "10 s", "loss10": "10%", "loss30": "30%", "loss50": "50%"}
    aware = {"update period": "P10+w20", "packet loss": "loss30+w10"}
    modes = [m for m in ("free", "mvp") if m in d["mode"].unique()]; fig, ax = plt.subplots(len(modes), 3, figsize=(10.2, 2.9 * len(modes)), squeeze=False, sharey=True)
    for i, m in enumerate(modes):
        for j, (fn, st) in enumerate(fam.items()):
            a = ax[i, j]; x = d[d["mode"] == m]; xs = np.arange(len(st))
            def ser(kind, mod, names=st):
                z = x[(x.kind == kind) & (x.model == mod)].set_index("setting")
                if kind == "base-trained" and "base" in names: z = pd.concat([z, x[(x.kind == "matched") & (x.model == mod) & (x.setting == "base")].set_index("setting")])
                return z.reindex(names)
            for mod, kind, ls, lb in [("hybrid", "matched", "-", "hybrid, trained on this link"), ("hybrid", "base-trained", "--", "hybrid, trained on the ideal link"), ("constant-velocity rule", "matched", "-", "constant-velocity rule (5 s window)")]:
                z = ser(kind, mod); a.errorbar(xs, z.auprc, yerr=[z.auprc - z.auprc_lo, z.auprc_hi - z.auprc] if "auprc_lo" in z else None, color=M_COL[mod], marker=M_MK[mod], ls=ls, ms=5, capsize=2, label=lb, alpha=1 if kind == "matched" else 0.75)
            if fn in aware and aware[fn] in set(x.setting):
                z = x[(x.setting == aware[fn]) & (x.model == "constant-velocity rule") & (x.kind == "matched")]
                if len(z): a.scatter([xs[-2] if fn == "packet loss" else xs[-1]], z.auprc, marker="*", s=70, color="#0b0b0b", zorder=5, label="rule with longer velocity window")
            a.set_xticks(xs); a.set_xticklabels([lab[s] for s in st], fontsize=8); a.set_title(f"{MODE_T[m]}: {fn}", loc="left")
            if j == 0: a.set_ylabel(f"AUPRC ({h} s horizon)")
    h_, l_ = ax[0, 0].get_legend_handles_labels(); seen = dict(zip(l_, h_))
    for a in ax.ravel():
        hh, ll = a.get_legend_handles_labels(); seen.update(dict(zip(ll, hh)))
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=8.5); fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, o, "fig15_link_sweep")

def fig16(R, o):
    d = pd.read_csv(R)
    if d.empty: return
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(d) + 1.4)); names = [f"{r.model} ({r.device})" for r in d.itertuples()]
    ax.barh(range(len(d)), d.ms_per_aircraft, color=COL["cv"], height=0.6); ax.set_xscale("log"); ax.set_yticks(range(len(d))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    for i, v in enumerate(d.ms_per_aircraft): ax.text(v * 1.15, i, f"{v:.3g} ms", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("milliseconds per aircraft and forecast (log scale), model only"); fig.tight_layout(); save(fig, o, "fig16_inference_cost")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("-o", default=None); ap.add_argument("--horizon", type=int, default=30); a = ap.parse_args()
    o = a.o or os.path.join(a.out, "figures"); os.makedirs(o, exist_ok=True); style()
    def need(n):
        p = find(a.out, n); print(("  --  missing " + n) if p is None else "  in  " + n); return p
    if (p := need("a13_age_strata.csv")): fig11(p, o, [10, 30, 60])
    if (p := need("a13_degraded_ops.csv")): fig12(p, o, a.horizon)
    if (p := need("a13_reliability.csv")): fig13(p, o, a.horizon)
    if (p := need("a13_perm_importance.csv")): fig14(p, o)
    if (p := need("a15_link_eval.csv")): fig15(p, o, a.horizon)
    if (p := need("a16_inference_cost.csv")): fig16(p, o)

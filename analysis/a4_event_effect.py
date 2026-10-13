"""Step 4 - measured effect of every contingency / degradation, as a PAIRED comparison.
Each event episode is compared with the NOMINAL episode of the same zone, density (NORMAL), seed and mode (identical traffic).
  python a4_event_effect.py DATA_DIR -o OUT_DIR [--workers 4]
Writes event_effect_pairs.csv (1 row / pair) and event_effect_summary.csv (mean and 95% CI over seeds x zones, per class and mode).
Window: from the first instant at which any aircraft differs from its nominal twin (the event onset; for COMM-PARTIAL and SENSOR-FAULT the first outage / fault) to the end.
Columns of the summary:
  uavs_differ         aircraft whose TRUE trajectory differs (> 1 cm) from the nominal twin
  d_lowc_events       change in the number of loss-of-well-clear events started in the window (event - nominal)
  d_*_pp              change in the share of aircraft-seconds under severity >= 1 / >= 2 (percentage points), paper and half profiles
  d_sev3_rows         change in the number of collision-level (severity 3) aircraft-seconds
  rep_err_xy_rms_m    root-mean-square horizontal error of the REPORTED position (measured, to compare with the configured sigma)"""
import argparse, json, os, numpy as np, pandas as pd
from multiprocessing import Pool
from a_common import *

COLS = ["ep_t", "uav", "x_m", "y_m", "alt_m", "rep_x_m", "rep_y_m", "severity", "severity_half", "resolving", "comm_out", "gnss_lost", "sensor_fault", "failed"]

def stats(D, C, w0):
    W = D[D.ep_t >= w0]; air = W[W.alt_m > 5]; e = np.hypot(air.rep_x_m - air.x_m, air.rep_y_m - air.y_m)
    return dict(paper_ge1=float((W.severity >= 1).mean() * 100), paper_ge2=float((W.severity >= 2).mean() * 100), half_ge1=float((W.severity_half >= 1).mean() * 100),
                half_ge2=float((W.severity_half >= 2).mean() * 100), sev3_rows=int((W.severity >= 3).sum()), lowc_events=int((C.t_start >= w0).sum()), resolving=float(W.resolving.mean() * 100),
                rep_err_xy_rms_m=float(np.sqrt(np.nanmean(e.values ** 2))) if len(e) else np.nan, comm_out_share=float(W.comm_out.mean()), gnss_lost_share=float(W.gnss_lost.mean()),
                sensor_fault_share=float(W.sensor_fault.mean()), failed_rows=int(W.failed.sum()))

def one(args):
    d, ep = args; r = parse_ep(ep); ref = f"NOM-{r['zone']}-{r['dens']}_{r['mode']}_{r['idx']:04d}"
    if not os.path.exists(os.path.join(d, ref + "_meta.json")): return None
    E = pd.read_parquet(os.path.join(d, ep + ".parquet"), columns=COLS); N = pd.read_parquet(os.path.join(d, ref + ".parquet"), columns=COLS)
    CE = pd.read_parquet(os.path.join(d, ep + "_conflicts.parquet")); CN = pd.read_parquet(os.path.join(d, ref + "_conflicts.parquet"))
    te = json.load(open(os.path.join(d, ep + "_meta.json")))["t_event"]
    m = E[["ep_t", "uav", "x_m", "y_m", "alt_m"]].merge(N[["ep_t", "uav", "x_m", "y_m", "alt_m"]], on=["ep_t", "uav"], suffixes=("", "_n"))
    dd = np.maximum.reduce([(m.x_m - m.x_m_n).abs().values, (m.y_m - m.y_m_n).abs().values, (m.alt_m - m.alt_m_n).abs().values]); mv = dd > 0.01; t = m.ep_t.values
    w0 = float(t[mv].min()) if mv.any() else float(te)
    r.update(ref=ref, t_event=te, window_start=w0, uavs_common=int(m.uav.nunique()), uavs_differ=int(m[mv].uav.nunique()), first_diff_s=w0 if mv.any() else np.nan,
             rows_differ_before_event=int((mv & (t < te)).sum()), median_pos_diff_m=float(np.median(np.hypot(m.x_m - m.x_m_n, m.y_m - m.y_m_n).values[mv])) if mv.any() else 0.0,
             airborne_at_event=int(((N.ep_t == int(te)) & (N.alt_m > 5)).sum()))
    se, sn = stats(E, CE, w0), stats(N, CN, w0)
    for k in se: r["ev_" + k] = se[k]; r["nom_" + k] = sn[k]
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("-o", default="analysis_out"); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    os.makedirs(a.o, exist_ok=True); eps = [e for e in list_episodes(a.dir) if not e.startswith("NOM")]; print(len(eps), "event episodes")
    with Pool(a.workers) as p: P = pd.DataFrame([x for x in p.imap_unordered(one, [(a.dir, e) for e in eps], chunksize=2) if x])
    for k in ["paper_ge1", "paper_ge2", "half_ge1", "half_ge2", "resolving"]: P["d_" + k + "_pp"] = P["ev_" + k] - P["nom_" + k]
    P["d_lowc_events"] = P.ev_lowc_events - P.nom_lowc_events; P["d_sev3_rows"] = P.ev_sev3_rows - P.nom_sev3_rows; P.to_csv(os.path.join(a.o, "event_effect_pairs.csv"), index=False)
    rows = []
    for (c, mo), g in P.groupby(["cls", "mode"]):
        r = dict(cls=c, mode=mo, n_pairs=len(g), uavs_differ_mean=g.uavs_differ.mean(), airborne_at_event=g.airborne_at_event.mean(), first_diff_median_s=g.first_diff_s.median(), rows_differ_before_event=g.rows_differ_before_event.max(),
                 median_pos_diff_m=g.median_pos_diff_m.median(), ev_rep_err_xy_rms_m=g.ev_rep_err_xy_rms_m.mean(), nom_rep_err_xy_rms_m=g.nom_rep_err_xy_rms_m.mean())
        for k in ["d_lowc_events", "d_paper_ge1_pp", "d_paper_ge2_pp", "d_half_ge1_pp", "d_half_ge2_pp", "d_sev3_rows", "d_resolving_pp"]:
            m, lo, hi = ci95(g[k]); r[k] = m; r[k + "_lo"] = lo; r[k + "_hi"] = hi
        rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(os.path.join(a.o, "event_effect_summary.csv"), index=False); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S[["cls", "mode", "n_pairs", "uavs_differ_mean", "first_diff_median_s", "rows_differ_before_event", "d_lowc_events", "d_paper_ge2_pp", "d_half_ge1_pp", "d_sev3_rows", "ev_rep_err_xy_rms_m"]].round(2).to_string(index=False))

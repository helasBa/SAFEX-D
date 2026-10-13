#!/usr/bin/env python3
"""Acceptance tests for SAFEX-D v2 episodes. An episode is kept only if it passes.
Usage: python safex2_verify.py DIR [--baseline-suffix] -o verify.csv
For each episode: ghost-free, 1 Hz continuity, concurrency, resolution activity, LoWC rate, reported-vs-true error vs configured sigma,
and (for event classes) counterfactual divergence against the NOM episode with the same traffic key and mode."""
import argparse, glob, json, os, re, numpy as np, pandas as pd

def load(d, ep):
    return pd.read_parquet(os.path.join(d, ep + ".parquet")), json.load(open(os.path.join(d, ep + "_meta.json")))

def summarize(d, ep, base=None):
    D, M = load(d, ep); r = dict(episode=ep, cls=M["cls"], mode=M["mode"], rows=len(D), uavs=int(D.uav.nunique()), conc_mean=M["concurrent_mean"], conc_max=M["concurrent_max"],
                                 lowc_events=M["n_lowc_events"], nan_deleted=M.get("n_nan_deleted", 0), spawned=M["n_spawned"], deleted=M.get("n_deleted", 0))
    g = D.sort_values(["uav", "ep_t"]); last = g.groupby("uav").tail(1)
    gr = g[(g.alt_m < 0.5) & (g.phase != 0)].groupby("uav").size(); r["ghost_rows_alt<0.5m"] = int(gr[gr > 2].sum())          # on the ground after having flown (should be ~0: deleted at touchdown)
    r["uavs_ending_on_ground_still_logged"] = int(((last.alt_m < 0.5) & (last.ep_t < D.ep_t.max() - 2)).sum())
    dt = g.groupby("uav").ep_t.diff().dropna(); r["gap_rows>1s"] = int((dt > 1.01).sum())
    r["resolving_share"] = float(D.resolving.mean()); r["severity_ge1_share"] = float((D.severity >= 1).mean()); r["severity_ge2_share"] = float((D.severity >= 2).mean()); r["severity_3_rows"] = int((D.severity == 3).sum())
    A = D[D.cooperative == 1].dropna(subset=["rep_x_m"]); fresh = A[A.rep_age_s < 0.5]
    if len(fresh): r["rep_err_xy_std_m"] = float(np.sqrt((((fresh.rep_x_m - fresh.x_m) ** 2 + (fresh.rep_y_m - fresh.y_m) ** 2) / 2).mean()))
    r["comm_out_share"] = float(D.comm_out.mean()); r["gnss_lost_share"] = float(D.gnss_lost.mean()); r["sensor_fault_share"] = float(D.sensor_fault.mean()); r["failed_rows"] = int(D.failed.sum())
    r["max_wind_ms"] = float(D.wind_ms.max()); r["noncoop_share"] = float((D.cooperative == 0).mean())
    if base is not None and not M["cls"].startswith("NOM"):
        B, BM = load(d, base); te = M["t_event"]; m = D.merge(B, on=["uav", "ep_t"], suffixes=("", "_b"))
        dpos = np.maximum.reduce([(m.x_m - m.x_m_b).abs().values, (m.y_m - m.y_m_b).abs().values, (m.alt_m - m.alt_m_b).abs().values]); mv = dpos > 0.5
        r["base"] = base; r["uavs_differ"] = int(m[mv].uav.nunique()); r["uavs_common"] = int(m.uav.nunique()); r["first_diff_s"] = float(m[mv].ep_t.min()) if mv.any() else np.nan
        r["diff_before_event_rows"] = int((mv & (m.ep_t.values < te)).sum()); post = m[(m.ep_t >= te + 30)]; pm = mv[(m.ep_t >= te + 30).values]
        r["median_pos_diff_after_m"] = float(np.median(dpos[(m.ep_t >= te + 30).values][pm])) if pm.any() else 0.0
        air = set(m[(m.ep_t.round() == round(te)) & (m.phase.isin([0, 1, 2]))].uav); r["share_airborne_at_event_that_differ"] = float(len(set(m[mv & (m.ep_t >= te)].uav) & air) / max(1, len(air)))
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("-o", default="verify.csv"); a = ap.parse_args()
    eps = sorted(os.path.basename(f)[:-10] for f in glob.glob(os.path.join(a.dir, "*_meta.json")))
    keyof = lambda e: (json.load(open(os.path.join(a.dir, e + "_meta.json")))["traffic_key"], json.load(open(os.path.join(a.dir, e + "_meta.json")))["mode"])
    nom = {keyof(e): e for e in eps if json.load(open(os.path.join(a.dir, e + "_meta.json")))["cls"].startswith("NOM")}
    rows = []
    for e in eps:
        M = json.load(open(os.path.join(a.dir, e + "_meta.json"))); r = summarize(a.dir, e, None if M["cls"].startswith("NOM") else nom.get(keyof(e))); rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(a.o, index=False); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(R.drop(columns=[c for c in ["episode"] if c in R]).round(3).T.to_string() if len(R) <= 2 else R.round(3).to_string())

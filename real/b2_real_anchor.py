"""Plausibility anchor: compare the kinematics of the SIMULATED corpus with REAL low-altitude ADS-B trajectories (e.g. helicopters from OpenSky).
Nothing is trained on the real data; it only shows whether simulated speeds, climb rates and turn rates fall inside the envelope seen in reality.

1) Export real state vectors (needs a free OpenSky account with Trino access, see pyopensky docs) to CSV or parquet with at least:
     time (unix s or datetime), icao24 (or any id), velocity (m/s), heading (deg), vertrate (m/s), geoaltitude or baroaltitude (m), onground (optional)
   keep low, slow traffic only, e.g. rotorcraft or velocity < 90 m/s and altitude < 1500 m. Any region with ADS-B coverage will do (the comparison is on kinematics, not on geography).
2) python b2_real_anchor.py DATA_DIR --real real_lowalt.parquet -o OUT_DIR [--episodes 12] [--max-speed 90 --max-alt 1500]
Writes anchor_table.csv (quantiles + Kolmogorov-Smirnov distance for each quantity and fleet group) and anchor_plot.png."""
import argparse, glob, json, os, numpy as np, pandas as pd
from scipy.stats import ks_2samp

QS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

def real_kin(path, max_speed, max_alt, min_speed, maxdt=15.0):
    d = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    d = d.rename(columns={"baroaltitude": "alt", "geoaltitude": "alt"}) if "geoaltitude" not in d else d.rename(columns={"geoaltitude": "alt"})
    if "alt" not in d and "baroaltitude" in d: d["alt"] = d.baroaltitude
    if "onground" in d: d = d[~d.onground.astype(bool)]
    idc = "icao24" if "icao24" in d else d.columns[0]
    t = pd.to_datetime(d["time"], unit="s") if np.issubdtype(d["time"].dtype, np.number) else pd.to_datetime(d["time"])
    d = d.assign(t=((t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).values).dropna(subset=["velocity", "heading", "vertrate", "alt"])
    d = d[(d.velocity <= max_speed) & (d.velocity >= min_speed) & (d.alt <= max_alt)].sort_values([idc, "t"])
    dt = d.groupby(idc).t.diff(); dh = d.groupby(idc).heading.diff(); dh = (dh + 180) % 360 - 180; ok = (dt > 0) & (dt <= maxdt)
    return dict(speed=d.velocity.values, vrate=d.vertrate.values, turn=np.abs((dh / dt)[ok]).values, n=len(d), n_ids=d[idc].nunique())

def sim_kin(data_dir, n_ep):
    eps = sorted(glob.glob(os.path.join(data_dir, "NOM-*_free_0001.parquet")))[:n_ep]; out = {"sUAS": [[], [], []], "eVTOL": [[], [], []]}
    for f in eps:
        d = pd.read_parquet(f, columns=["ep_t", "uav", "type", "gs_ms", "vs_ms", "trk_deg", "phase"]).sort_values(["uav", "ep_t"])
        dh = d.groupby("uav").trk_deg.diff(); dh = np.abs((dh + 180) % 360 - 180); ok = d.groupby("uav").ep_t.diff() == 1
        d = d.assign(turn=dh.where(ok)); d = d[d.gs_ms > 2]            # all airborne phases (climb, cruise, descent), like the real traffic
        for g, m in [("sUAS", d.type < 8), ("eVTOL", d.type.isin([8, 9]))]:
            x = d[m]
            if len(x): out[g][0].append(x.gs_ms.values); out[g][1].append(x.vs_ms.values); out[g][2].append(x.turn.dropna().values)
    return {g: dict(speed=np.concatenate(v[0]), vrate=np.concatenate(v[1]), turn=np.concatenate(v[2])) for g, v in out.items() if v[0]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--real", required=True); ap.add_argument("-o", default="anchor_out"); ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-speed", type=float, default=90); ap.add_argument("--max-alt", type=float, default=1500); ap.add_argument("--min-speed", type=float, default=2); a = ap.parse_args()
    os.makedirs(a.o, exist_ok=True); R = real_kin(a.real, a.max_speed, a.max_alt, a.min_speed); S = sim_kin(a.dir, a.episodes); print("real rows", R["n"], "aircraft", R["n_ids"]); rows = []
    for g, s in S.items():
        for q in ["speed", "vrate", "turn"]:
            r, x = np.abs(R[q]) if q == "vrate" else R[q], np.abs(s[q]) if q == "vrate" else s[q]
            rec = dict(group=g, quantity=q, real_n=len(r), sim_n=len(x), ks=float(ks_2samp(r[:200000], x[:200000]).statistic), sim_inside_real_p1_p99=float(((x >= np.quantile(r, .01)) & (x <= np.quantile(r, .99))).mean()))
            rec.update({f"real_q{int(p*100):02d}": float(np.quantile(r, p)) for p in QS}); rec.update({f"sim_q{int(p*100):02d}": float(np.quantile(x, p)) for p in QS}); rows.append(rec)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(a.o, "anchor_table.csv"), index=False); pd.set_option("display.width", 220)
    print(T[["group", "quantity", "ks", "sim_inside_real_p1_p99", "real_q50", "sim_q50", "real_q95", "sim_q95"]].round(3).to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, (q, lab) in enumerate([("speed", "ground speed (m/s)"), ("vrate", "|vertical rate| (m/s)"), ("turn", "|turn rate| (deg/s)")]):
        r = np.abs(R[q]) if q == "vrate" else R[q]; hi = np.quantile(r, .99); ax[i].hist(r, bins=60, range=(0, hi), density=True, alpha=.5, label="real ADS-B")
        for g, s in S.items(): x = np.abs(s[q]) if q == "vrate" else s[q]; ax[i].hist(x, bins=60, range=(0, hi), density=True, histtype="step", label="sim " + g)
        ax[i].set_xlabel(lab); ax[i].set_yscale("log")
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(a.o, "anchor_plot.png"), dpi=150); print("->", a.o)

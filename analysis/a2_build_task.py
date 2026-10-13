"""Step 2 - leakage-free conflict-ONSET forecasting task, built from the UTM-VISIBLE (reported) state only.
  python a2_build_task.py DATA_DIR -o FEAT_DIR [--keep 0.15] [--workers 4] [--max-episodes N]
Row = (episode, t, UAV) on a 10 s grid, UAV visible (reported position available for the last 35 s) and currently NOT in conflict.
Targets (future, never an input): y_{paper|half}_{tac|imm}_h{10|30|60} = severity >= 1 (tac) / >= 2 (imm) at any 1 Hz row in (t, t+h].
    NaN when the UAV is already in conflict under that profile, or when t+h is beyond the episode.
Inputs: own kinematics from REPORTED positions (finite differences over 5 s), neighbour geometry from REPORTED positions of visible aircraft,
        explicit history at t-10/-20/-30 s, flight phase, report age. No severity, min_sep, in_conflict_cd, resolving or true-state column is used.
Physics baseline: constant-velocity extrapolation of reported positions, min predicted horizontal distance among vertically compatible pairs
        (cv_{profile}_h{h}); the thresholds of the profile are only used here, as the physical rule being tested."""
import argparse, os, zlib, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.spatial import cKDTree
from a_common import *

S_GRID, LAGS, HS, VW = 10, (10, 20, 30), (10, 30, 60), 5     # grid step (s), history lags (s), horizons (s), velocity window (s)

def build(path, keep=0.15, seed=0, link=None, vw=None):   # link: optional emulated surveillance link applied to the reported stream (a14); None = unchanged behaviour
    vwin = VW if vw is None else int(vw)   # velocity window (s); default 5. a15 uses a longer window for link-aware physics when reports are sparse
    cols = ["ep_t", "uav", "type", "rep_x_m", "rep_y_m", "rep_alt_m", "rep_age_s", "phase", "severity", "severity_half"]
    d = pd.read_parquet(path, columns=cols); T = int(d.ep_t.max()) + 1; u = d.uav.values; N = int(u.max()) + 1; ti = d.ep_t.values.astype(int)
    def dense(c, fill=np.nan, dt=np.float32):
        A = np.full((T, N), fill, dt); A[ti, u] = d[c].values; return A
    RX, RY, RZ = dense("rep_x_m"), dense("rep_y_m"), dense("rep_alt_m"); AGE, PH, TY = dense("rep_age_s"), dense("phase"), dense("type")
    SV = {"paper": dense("severity", 0, np.float32), "half": dense("severity_half", 0, np.float32)}
    if link is not None: RX, RY, RZ, AGE = link(RX, RY, RZ, AGE, np.random.default_rng(zlib.crc32(os.path.basename(path).encode()) ^ 12345))
    rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()) ^ seed)
    def vel(t):
        vx, vy, vz = (RX[t] - RX[t - vwin]) / vwin, (RY[t] - RY[t - vwin]) / vwin, (RZ[t] - RZ[t - vwin]) / vwin; return vx, vy, vz
    def nbr(t):
        """neighbour geometry among visible aircraft at time t -> dict of (N,) arrays"""
        o = {k: np.full(N, np.nan, np.float32) for k in ["nn1_d", "nn1_dz", "nn1_close", "nn1_bear", "nn2_d", "nn3_d", "cnt500", "cnt1000", "cnt2000"]}
        vx, vy, vz = vel(t); a = np.where(np.isfinite(RX[t]) & np.isfinite(vx))[0]
        if len(a) < 3: return o
        P = np.c_[RX[t, a], RY[t, a]]; tr = cKDTree(P); dd, jj = tr.query(P, k=4); j = jj[:, 1]
        o["nn1_d"][a] = dd[:, 1]; o["nn2_d"][a] = dd[:, 2]; o["nn3_d"][a] = dd[:, 3]; o["nn1_dz"][a] = np.abs(RZ[t, a][j] - RZ[t, a])
        rp = P[j] - P; rv = np.c_[vx[a][j] - vx[a], vy[a][j] - vy[a]]; o["nn1_close"][a] = -(rp * rv).sum(1) / np.maximum(dd[:, 1], 1e-6)
        hd = np.arctan2(vx[a], vy[a]); o["nn1_bear"][a] = np.cos(np.arctan2(rp[:, 0], rp[:, 1]) - hd)
        for r, n in [(500, "cnt500"), (1000, "cnt1000"), (2000, "cnt2000")]: o[n][a] = [len(x) - 1 for x in tr.query_ball_point(P, r)]
        return o
    def cvscore(t):
        """constant-velocity physics score per profile/horizon: higher = more likely conflict"""
        vx, vy, vz = vel(t); a = np.where(np.isfinite(RX[t]) & np.isfinite(vx))[0]; out = {(p, h): np.full(N, np.nan, np.float32) for p in PROFILES for h in HS}
        if len(a) < 2: return out
        P = np.c_[RX[t, a], RY[t, a]]; V = np.c_[vx[a], vy[a]]; Z, Vz = RZ[t, a], vz[a]; vmax = float(np.nanmax(np.hypot(V[:, 0], V[:, 1])))
        pr = cKDTree(P).query_pairs(PROFILES["paper"][0] + 2 * max(vmax, 10) * max(HS), output_type="ndarray")
        best = {(p, h): np.full(len(a), 1e9) for p in PROFILES for h in HS}
        if len(pr):
            dP = P[pr[:, 0]] - P[pr[:, 1]]; dV = V[pr[:, 0]] - V[pr[:, 1]]; dz0 = Z[pr[:, 0]] - Z[pr[:, 1]]; dvz = Vz[pr[:, 0]] - Vz[pr[:, 1]]
            run = {p: np.full(len(pr), 1e9) for p in PROFILES}
            for tau in range(2, max(HS) + 1, 2):
                dh = np.hypot(*(dP + dV * tau).T); dz = np.abs(dz0 + dvz * tau)
                for p in PROFILES: run[p] = np.minimum(run[p], np.where(dz < PROFILES[p][3], dh, dh + 1000.0))
                if tau in HS:
                    for p in PROFILES:
                        for q in (0, 1): np.minimum.at(best[(p, tau)], pr[:, q], run[p])
        for k, v in best.items(): out[k][a] = -v
        return out
    grid = np.arange(2 * vwin + max(LAGS) + 5, T - min(HS), S_GRID); NB = {int(t): nbr(int(t)) for t in grid}; rows = []
    for t in grid:
        t = int(t)
        if t - max(LAGS) not in NB: continue
        offs = [0] + list(range(vwin, max(LAGS) + vwin + 1, vwin)); ok = np.isfinite(RX[t])
        for o in offs: ok &= np.isfinite(RX[t - o])
        idx = np.where(ok)[0]; idx = idx[rng.random(len(idx)) < keep]
        if len(idx) == 0: continue
        cv = cvscore(t); vx, vy, vz = vel(t); sp = np.hypot(vx, vy); hd = np.arctan2(vx, vy); n = len(idx)
        R = {"ep": np.full(n, os.path.basename(path)[:-8]), "t": np.full(n, t), "uav": idx, "raw_alt": RZ[t, idx], "raw_vz": vz[idx], "raw_speed": sp[idx], "raw_hd_sin": np.sin(hd[idx]), "raw_hd_cos": np.cos(hd[idx]),
             "raw_type": TY[t, idx], "raw_phase": PH[t, idx], "raw_age": AGE[t, idx], "raw_alt_chg_K": RZ[t, idx] - RZ[t - max(LAGS), idx]}
        vx0, vy0, vz0 = vel(t - max(LAGS)); sp0 = np.hypot(vx0, vy0); hd0 = np.arctan2(vx0, vy0)
        R["raw_speed_chg_K"] = sp[idx] - sp0[idx]; R["raw_hd_chg_K"] = np.abs(np.angle(np.exp(1j * (hd[idx] - hd0[idx]))))
        for k, v in NB[t].items(): R["rel_" + k] = v[idx]
        R["rel_nn1_d_chg_K"] = NB[t]["nn1_d"][idx] - NB[t - max(LAGS)]["nn1_d"][idx]; R["rel_cnt1000_chg_K"] = NB[t]["cnt1000"][idx] - NB[t - max(LAGS)]["cnt1000"][idx]
        for L in LAGS:
            vxl, vyl, vzl = vel(t - L); hl = np.arctan2(vxl, vyl)
            R[f"seq_speed_l{L}"] = np.hypot(vxl, vyl)[idx]; R[f"seq_alt_l{L}"] = RZ[t - L, idx] - RZ[t, idx]; R[f"seq_vz_l{L}"] = vzl[idx]
            R[f"seq_hd_l{L}"] = np.angle(np.exp(1j * (hl[idx] - hd[idx]))); R[f"seq_nn1d_l{L}"] = NB[t - L]["nn1_d"][idx]; R[f"seq_nn1c_l{L}"] = NB[t - L]["nn1_close"][idx]; R[f"seq_cnt1000_l{L}"] = NB[t - L]["cnt1000"][idx]
        for (p, h), v in cv.items(): R[f"cv_{p}_h{h}"] = v[idx]
        for p in PROFILES:
            cur = SV[p][t, idx]
            for h in HS:
                if t + h <= T - 1: fut = SV[p][t + 1:t + h + 1][:, idx].max(0)
                else: fut = np.full(n, np.nan)
                for lv, th in LEVELS.items(): R[f"y_{p}_{lv}_h{h}"] = np.where((cur >= 1) | np.isnan(fut), np.nan, (fut >= th).astype(np.float32))
        rows.append(pd.DataFrame(R))
    F = pd.concat(rows, ignore_index=True); fc = [c for c in F.columns if c not in ("ep", "t", "uav") and not c.startswith("y_")]
    F[fc] = F[fc].astype(np.float32); return F

def work(a_):
    src, dst, keep = a_
    if os.path.exists(dst): return dst
    build(src, keep).to_parquet(dst + ".tmp", index=False); os.replace(dst + ".tmp", dst); return dst

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("-o", default="feats"); ap.add_argument("--keep", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--max-episodes", type=int, default=0); a = ap.parse_args()
    os.makedirs(a.o, exist_ok=True); eps = list_episodes(a.dir); eps = eps[:a.max_episodes] if a.max_episodes else eps
    jobs = [(os.path.join(a.dir, e + ".parquet"), os.path.join(a.o, e + ".parquet"), a.keep) for e in eps]; print(len(jobs), "episodes ->", a.o)
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(work, jobs), 1):
            if i % 20 == 0 or i == len(jobs): print(f"[{i}/{len(jobs)}]", flush=True)

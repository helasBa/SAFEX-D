"""Step 14 - EMULATED SURVEILLANCE LINK: rebuild the forecasting features after degrading the REPORTED stream.
The reported stream of an episode (rep_x/y/alt, rep_age) is treated as what a traffic manager receives over a link. We apply, per aircraft,
  update period P (s)  : one report every P seconds (random phase per aircraft), the receiver holds the last report between updates
  latency L (s)        : the report delivered at time t describes the aircraft at t - L (its age grows by L)
  packet loss p        : each scheduled report is lost with probability p (independent), last report held
Labels are NOT touched (they come from the true state). The aircraft's own outage/GNSS degradations of the episode stay in the stream underneath.
Nothing is re-simulated; this is a post-processing of the reported columns, so it is cheap and exactly reproducible.
Settings are named:  base | P<s> (period) | L<s> (latency) | loss<percent> | w<s> (window of the velocity estimate, default 5 s), combined with '+' (P5+loss30, P10+w20).
'+w<s>' gives the physics rule a LINK-AWARE velocity estimate (a longer window when reports are sparse) so that the baseline is not handicapped by a naive 5 s window.
  python a14_link_build.py DATA_DIR --splits OUT/splits.csv -o C:\\these\\safex2_link --per-fold 12 --settings base,P2,P5,P10,P10+w20,L2,L5,L10,loss10,loss30,loss30+w10,loss50 --workers 4
Selection: per fold and per mode, `--per-fold` episodes taken round-robin over the scenario classes (nominal and event classes both present), identical for all settings.
Writes  <o>/<setting>/<episode>.parquet  (same columns as a2) and <o>/selected_episodes.csv."""
import argparse, os, re, sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
from a2_build_task import build

def parse_setting(name):
    P, L, p, vw = 1, 0, 0.0, None
    if name != "base":
        for part in name.split("+"):
            m = re.fullmatch(r"(P|L|loss|w)(\d+)", part)
            if not m: raise SystemExit("bad setting " + name)
            v = int(m.group(2))
            if m.group(1) == "P": P = v
            elif m.group(1) == "L": L = v
            elif m.group(1) == "w": vw = v
            else: p = v / 100.0
    return P, L, p, vw

def make_link(P=1, L=0, p=0.0):
    def link(RX, RY, RZ, AGE, rng):
        T, N = RX.shape
        if P == 1 and L == 0 and p == 0.0: return RX, RY, RZ, AGE
        off = rng.integers(0, P, N) if P > 1 else np.zeros(N, int)
        oX, oY, oZ = (np.full((T, N), np.nan, np.float32) for _ in range(3)); oA = np.full((T, N), np.nan, np.float32)
        hx, hy, hz, ha = (np.full(N, np.nan, np.float32) for _ in range(4))
        for t in range(T):
            ha = ha + 1.0                                             # held report gets older by one second (NaN stays NaN)
            s = max(t - L, 0)
            sched = ((t + off) % P == 0)
            if p > 0: sched = sched & (rng.random(N) >= p)
            new = sched & np.isfinite(RX[s])
            if new.any():
                hx[new], hy[new], hz[new] = RX[s][new], RY[s][new], RZ[s][new]
                ha[new] = np.nan_to_num(AGE[s][new], nan=0.0) + L
            oX[t], oY[t], oZ[t], oA[t] = hx, hy, hz, ha
        present = np.isfinite(RX)                                     # an aircraft that is not in the original stream at t stays absent (no ghosts after landing / before spawn)
        for A in (oX, oY, oZ, oA): A[~present] = np.nan
        return oX, oY, oZ, oA
    return link

def select(sp, per_fold):
    """Per (mode, fold): half of the episodes from degradation classes (all six when per_fold >= 12), the rest split between nominal and contingency classes.
    Classes are taken in a rotated order (start depends on the fold) so that different folds cover different nominal / contingency classes. Same episodes for every setting."""
    out = []
    nd = min(6, per_fold // 2); nn = (per_fold - nd) // 2; quota = {"DEG": nd, "NOM": nn, "CONT": per_fold - nd - nn}
    for (mode, fold), g in sp.groupby(["mode", "fold"]):
        for fam, q in quota.items():
            gf = g[g.family == fam]; cls = sorted(gf.cls.unique())
            if not cls: continue
            cls = cls[int(fold) % len(cls):] + cls[:int(fold) % len(cls)]; byc = {c: list(gf[gf.cls == c].ep.sort_values()) for c in cls}; k = 0
            while q > 0 and any(byc.values()):
                c = cls[k % len(cls)]; k += 1
                if byc[c]: out.append((mode, fold, byc[c].pop(0))); q -= 1
    return pd.DataFrame(out, columns=["mode", "fold", "ep"])

def work(a):
    src, dst, setting, keep = a
    if os.path.exists(dst): return dst
    P, L, p, vw = parse_setting(setting); t0 = time.time()
    build(src, keep, 0, link=make_link(P, L, p), vw=vw).to_parquet(dst + ".tmp", index=False); os.replace(dst + ".tmp", dst); return dst + f"  ({time.time() - t0:.0f} s)"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--splits", required=True); ap.add_argument("-o", required=True); ap.add_argument("--per-fold", type=int, default=12)
    ap.add_argument("--settings", default="base,P2,P5,P10,P10+w20,L2,L5,L10,loss10,loss30,loss30+w10,loss50"); ap.add_argument("--keep", type=float, default=0.15); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); sp = pd.read_csv(a.splits); sel = select(sp, a.per_fold); os.makedirs(a.o, exist_ok=True); sel.to_csv(os.path.join(a.o, "selected_episodes.csv"), index=False)
    sets = a.settings.split(","); jobs = []
    for s in sets:
        parse_setting(s); os.makedirs(os.path.join(a.o, s), exist_ok=True)
        jobs += [(os.path.join(a.dir, e + ".parquet"), os.path.join(a.o, s, e + ".parquet"), s, a.keep) for e in sel.ep]
    print(f"{len(sel)} episodes x {len(sets)} settings = {len(jobs)} builds -> {a.o}", flush=True)
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs), 1):
            if i % 10 == 0 or i == len(jobs) or i <= 2: print(f"[{i}/{len(jobs)}] {r}", flush=True)

#!/usr/bin/env python
"""SAFEX-D v2 campaign runner: builds the episode manifest and runs it in parallel (resumable).

Every scenario is generated in a PAIR of modes (free = ASAS off, mvp = ASAS on) and every event episode shares its
traffic key (zone-density-idx) with the NOM episode of the same idx, so counterfactual comparisons are exact.

  python safex2_campaign.py --preset pilot --out D:/safex2/pilot  --workers 4
  python safex2_campaign.py --preset core  --out D:/safex2/data   --workers 8
  python safex2_campaign.py --preset full  --out D:/safex2/data   --workers 8     # adds seeds on top of core (resumes)
  python safex2_campaign.py --preset core  --dry-run                              # only prints counts

Re-running the same command skips every episode whose *_meta.json already exists (written last).
"""
import argparse, os, sys, time, json, itertools
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from safex2_gen import NOMINAL, EVENTS, ZONES  # noqa: E402

MODES = ["free", "mvp"]


def manifest(preset):
    """Return list of (cls, zone, dens, idx, mode)."""
    jobs = []
    if preset == "pilot":      # ~20 episodes: one seed, A only, a few classes, both modes
        for cls in ["NOM-A-NORMAL", "NOM-A-BUSY"]:
            for m in MODES: jobs.append((cls, "A", cls.split("-")[2], 1, m))
        for ev in ["CONT-VEHICLE-FAILURE", "CONT-INTRUDER", "DEG-GNSS-L2", "DEG-COMM-FULL", "DEG-SENSOR-FAULT"]:
            for m in MODES: jobs.append((ev, "A", "NORMAL", 1, m))
        for z in "BC":
            for m in MODES: jobs.append((f"NOM-{z}-NORMAL", z, "NORMAL", 1, m))
        return jobs
    n_nom, n_ev, n_ev_busy = {"core": (20, 8, 0), "full": (30, 12, 5)}[preset]
    for z in "ABC":
        for d in ["LOW", "NORMAL", "BUSY"]:
            for i in range(1, n_nom + 1):
                for m in MODES: jobs.append((f"NOM-{z}-{d}", z, d, i, m))
    for z in "ABC":
        for ev in EVENTS:
            for i in range(1, n_ev + 1):
                for m in MODES: jobs.append((ev, z, "NORMAL", i, m))
    if n_ev_busy:
        for ev in EVENTS:
            for i in range(1, n_ev_busy + 1):
                for m in MODES: jobs.append((ev, "A", "BUSY", i, m))
    return jobs


def ep_id(cls, zone, dens, idx, mode):
    return f"{cls}_{mode}_{idx:04d}" if cls.startswith("NOM") else f"{cls}-{zone}-{dens}_{mode}_{idx:04d}"


def work(job):
    cls, zone, dens, idx, mode, out, wind_csv = job
    from safex2_gen import run_episode
    t0 = time.time()
    try:
        m = run_episode(cls, zone, dens, idx, mode, out, quiet=True, wind_csv=wind_csv if cls == "CONT-WEATHER" else None)
        return dict(ep=m["episode_id"], ok=1, wall_s=round(time.time() - t0, 1), rows=m["rows"], conc=round(m["concurrent_mean"], 1),
                    n_nan=m.get("n_nan_deleted", 0), err="")
    except Exception as e:  # keep the campaign running, record the failure
        return dict(ep=ep_id(cls, zone, dens, idx, mode), ok=0, wall_s=round(time.time() - t0, 1), rows=0, conc=0, n_nan=0, err=repr(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["pilot", "core", "full"], default="pilot")
    ap.add_argument("--out", default="safex2_data")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--wind-csv", default=None, help="real wind table from real/b1_fetch_real_wind.py; used by CONT-WEATHER")
    a = ap.parse_args()
    jobs = manifest(a.preset)
    todo = [j for j in jobs if not os.path.exists(os.path.join(a.out, ep_id(*j) + "_meta.json"))]
    # longest first (busy / dense) for better load balance
    rank = {"LOW": 0, "NORMAL": 1, "BUSY": 2, "DENSE": 3}
    todo.sort(key=lambda j: -rank[j[2]])
    print(f"preset={a.preset}: {len(jobs)} episodes, {len(jobs) - len(todo)} already done, {len(todo)} to run, workers={a.workers}")
    if a.dry_run: return
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "_campaign_log.csv")
    new = not os.path.exists(log)
    f = open(log, "a");
    if new: f.write("ep,ok,wall_s,rows,conc,n_nan,err\n")
    ctx = mp.get_context("spawn"); t0 = time.time(); done = 0
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(work, [(*j, a.out, a.wind_csv) for j in todo]):
            done += 1
            f.write(f'{r["ep"]},{r["ok"]},{r["wall_s"]},{r["rows"]},{r["conc"]},{r["n_nan"]},"{r["err"]}"\n'); f.flush()
            el = time.time() - t0
            print(f'[{done}/{len(todo)}] {r["ep"]} ok={r["ok"]} {r["wall_s"]}s  elapsed {el/60:.1f} min, ETA {(el/done)*(len(todo)-done)/60:.0f} min', flush=True)
    print("campaign finished. Next: python safex2_verify.py", a.out)


if __name__ == "__main__":
    main()

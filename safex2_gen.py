#!/usr/bin/env python3
"""SAFEX-D v2 episode generator (BlueSky driven from Python).

Design goals (each one answers a defect or reviewer remark on v1):
  * aircraft are DELETED on touchdown            -> no ghost records
  * true state AND reported (UTM-visible) state   -> degradations are measurable against their configured sigma
  * conflict ground truth computed INSIDE the simulator at 5 Hz (dt=0.2 s) from true positions, logged at 1 Hz
    (min separation / worst severity over each logging interval) + an event table of every loss of well-clear
  * every event is applied by code and has a logged ground-truth record (time, affected ids, parameters)
  * traffic depends only on (zone, density, seed index) -> episodes with the same traffic key are exact counterfactuals
  * two resolution modes: 'free' (no resolution) and 'mvp' (BlueSky ASAS, MVP) -> real resolution manoeuvres are logged
  * real eVTOL types are defined explicitly (BlueSky ships none: unknown names silently become a Boeing 747)

Usage:
  python safex2_gen.py --cls NOM --zone A --density NORMAL --idx 1 --mode mvp --out out_dir
  python safex2_gen.py --cls CONT-WEATHER --zone A --density NORMAL --idx 1 --mode mvp --out out_dir
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

# ------------------------------------------------------------------ configuration
ZONES = {  # centre (lat, lon), side length km, altitude cap m AGL, OD decay length km (smaller = more concentrated)
    "A": dict(c=(22.54, 114.06), side=15.1, cap=200, L=4.5),   # Futian-CBD-inspired
    "B": dict(c=(22.60, 113.88), side=11.7, cap=400, L=3.0),   # Bao'an-airport-inspired (taller envelope)
    "C": dict(c=(22.50, 114.03), side=11.7, cap=200, L=6.0)}   # Qianhai-FTZ-inspired
DENSITY = {"LOW": 520, "NORMAL": 1950, "BUSY": 3900, "DENSE": 6500}           # arrivals per hour (concurrency is measured, not assumed)
SUAS = ["M600", "Amzn", "Mnet", "Phan4", "M100", "M200", "Mavic", "Horsefly"]
CRUISE = dict(M600=14, Amzn=18, Mnet=9, Phan4=16, M100=17, M200=18, Mavic=14, Horsefly=17, EH216=28, AAVM=45, EC35=50)  # m/s
FLEET = {"A": dict(suas=0.97, ev=0.03), "B": dict(suas=0.70, ev=0.30), "C": dict(suas=0.99, ev=0.01)}  # ASSUMED mixes, documented as such
SUAS_LAYERS = [60, 80, 100, 120, 140, 160, 180]
D_LOWC, D_NMAC, D_MAC, H_TOL = 496.8, 152.4, 10.0, 30.48     # same separation thresholds as v1 (manned-aviation derived)
T_EP, WARM, T_EVENT, DT, LOG_DT = 3600.0, 600.0, 1800.0, 0.2, 1.0
M_PER_FT, KT = 0.3048, 1.943844
MIN_DEP_GAP = 20.0   # s between two departures from the same vertiport (slot spacing)
PHASE = dict(climb=0, cruise=1, descent=2, emergency=3, transit=4)

NOMINAL = [f"NOM-{z}-{d}" for z in "ABC" for d in DENSITY]
EVENTS = ["CONT-VEHICLE-FAILURE", "CONT-ALL-LAND", "CONT-VERTIPORT-CLOSURE", "CONT-WEATHER", "CONT-NON-COOPERATIVE", "CONT-INTRUDER",
          "CONT-PASSENGER-DISTRESS", "DEG-GNSS-L1", "DEG-GNSS-L2", "DEG-GNSS-LOSS", "DEG-COMM-PARTIAL", "DEG-COMM-FULL", "DEG-SENSOR-FAULT"]

def seed_of(*a): return int(hashlib.sha256("-".join(map(str, a)).encode()).hexdigest()[:8], 16)

# ------------------------------------------------------------------ geometry helpers
class Frame:
    def __init__(self, zone):
        z = ZONES[zone]; self.lat0, self.lon0 = z["c"]; self.half = z["side"] * 500.0
        self.kx = 111320.0 * math.cos(math.radians(self.lat0)); self.ky = 110574.0
    def xy(self, lat, lon): return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky
    def ll(self, x, y): return self.lat0 + np.asarray(y) / self.ky, self.lon0 + np.asarray(x) / self.kx

def vertiports(zone, n=12):
    """Fixed per zone (independent of episode seed): positions in metres + demand weights."""
    rng = np.random.default_rng(seed_of("vp", zone)); h = ZONES[zone]["side"] * 500.0 * 0.85; pts = []
    while len(pts) < n:
        p = rng.uniform(-h, h, 2)
        if all(np.hypot(*(p - q)) > 1500 for q in pts): pts.append(p)
    return np.array(pts), rng.lognormal(0, 0.6, n)

def make_schedule(zone, dens, idx, warm=WARM):
    """Traffic schedule: identical for every episode with the same (zone, density, idx), whatever the scenario class or mode."""
    rng = np.random.default_rng(seed_of("traffic", zone, dens, idx)); V, w = vertiports(zone); L = ZONES[zone]["L"] * 1000
    n = rng.poisson(DENSITY[dens] / 3600.0 * (T_EP + warm)); ts = np.sort(rng.uniform(-warm, T_EP, n)); out = []
    for t in ts:
        o = rng.choice(len(V), p=w / w.sum()); d = np.hypot(*(V - V[o]).T)
        pw = np.where((d > 1200) & (d < 12000), w * np.exp(-d / L), 0.0); pw[o] = 0
        j = rng.choice(len(V), p=pw / pw.sum())
        ev = rng.random() < FLEET[zone]["ev"]
        typ = (rng.choice(["EH216", "AAVM"]) if zone == "B" else "EH216") if ev else rng.choice(SUAS)
        cap = ZONES[zone]["cap"]
        layers = ([cap - 20 * k for k in range(0, 8) if cap - 20 * k >= 180] if ev else SUAS_LAYERS)
        out.append(dict(t=float(t), o=int(o), d=int(j), typ=str(typ), alt=float(rng.choice(layers))))
    return out, V

# ------------------------------------------------------------------ per-aircraft record
class U:
    __slots__ = ("acid", "n", "typ", "o", "d", "calt", "cspd", "phase", "t0", "hdg_cmd", "dest_xy", "coop", "failed", "nx", "ny", "sig", "comm", "gnss_loss",
                 "fault_t", "bias", "rx", "ry", "ralt", "age", "drift", "last_alt_cmd", "intr", "dest_id")
    def __init__(self, **k):
        self.coop = True; self.failed = False; self.nx = self.ny = 0.0; self.sig = 0.0; self.comm = []; self.gnss_loss = None; self.fault_t = None; self.bias = 0.0
        self.rx = self.ry = self.ralt = np.nan; self.age = 0.0; self.drift = 0.0; self.last_alt_cmd = None; self.intr = False; self.dest_id = -1
        for a, b in k.items(): setattr(self, a, b)

def bearing(dx, dy): return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

# ------------------------------------------------------------------ main episode runner
def pick_real_wind(path, zone, seed, thr=8.0):
    """Real wind for CONT-WEATHER: one hour drawn (seeded, independent of mode) from the ERA5 reanalysis at the zone centre (b1_fetch_real_wind.py).
    Rule, fixed in advance: hours whose 100 m wind speed >= thr m/s; if fewer than 30 such hours exist, the windiest 5 % of hours."""
    import pandas as _pd
    d = _pd.read_csv(path); d = d[d.zone == zone].reset_index(drop=True)
    if d.empty: raise SystemExit(f"no rows for zone {zone} in {path}")
    sel = d[d.wind_speed_100m_ms >= thr]
    if len(sel) < 30: sel = d[d.wind_speed_100m_ms >= d.wind_speed_100m_ms.quantile(0.95)]
    r = sel.iloc[int(np.random.default_rng(seed).integers(0, len(sel)))]
    return dict(source="ERA5 via Open-Meteo", time=str(r["time"]), speed_ms=float(r.wind_speed_100m_ms), from_deg=float(r.wind_direction_100m_deg),
                gust10_ms=float(r.wind_gusts_10m_ms), speed10_ms=float(r.wind_speed_10m_ms), n_candidate_hours=int(len(sel)), rule=f"100 m wind >= {thr} m/s, else top 5 %")

def run_episode(cls, zone, dens, idx, mode, out_dir, T=T_EP, warm=WARM, quiet=False, t_event=T_EVENT, wind_csv=None):
    import bluesky as bs
    from bluesky import stack
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from patch_types import ensure_types
    ensure_types()
    bs.init(mode="sim", detached=True)
    cmds = ["DT 0.2", "CDMETHOD STATEBASED", "DTLOOK 30", "ZONER 0.1", "ZONEDH 100"]
    if mode == "mvp": cmds += ["ASAS ON", "RESO MVP"]
    else: cmds += ["ASAS OFF"]
    for c in cmds: stack.stack(c)
    stack.process()
    F = Frame(zone); sched, V = make_schedule(zone, dens, idx, warm); cap = ZONES[zone]["cap"]
    ekey = seed_of("event", cls, zone, dens, idx); er = np.random.default_rng(ekey)
    base = cls.split("-", 1)[0]; is_nom = (base == "NOM"); name = cls if is_nom else cls
    ep_id = f"{cls}-{zone}-{dens}_{mode}_{idx:04d}" if not is_nom else f"{cls}_{mode}_{idx:04d}"
    meta = dict(episode_id=ep_id, cls=cls, zone=zone, density=dens, idx=idx, mode=mode, traffic_key=f"{zone}-{dens}-{idx:04d}", t_event=t_event, warmup_s=warm,
                duration_s=T, log_dt=LOG_DT, sim_dt=DT, event_log=[], thresholds=dict(lowc=D_LOWC, nmac=D_NMAC, mac=D_MAC, htol=H_TOL),
                vertiports_xy_m=V.tolist(), n_scheduled=len(sched))
    # persistent degradation parameters (assigned per aircraft at spawn from event RNG)
    gsig = {"DEG-GNSS-L1": 9.0, "DEG-GNSS-L2": 18.0}.get(cls, 0.0)
    comm_p = {"DEG-COMM-PARTIAL": 0.3}.get(cls, 0.0); sens_p = 0.2 if cls == "DEG-SENSOR-FAULT" else 0.0
    nav_p = 0.3 if cls == "DEG-GNSS-LOSS" else 0.0
    closed = set(); landing_all = False; wind_target = None; wind_dir = 0.0; wind_t0 = None
    UL, rows, conf_open, conf_done = [], [], {}, []
    ms = np.zeros(0); sv = np.zeros(0, dtype=np.int8); sv2 = np.zeros(0, dtype=np.int8)
    si = 0; nsp = 0; queue = []; vp_free = np.zeros(len(V));  tick = int(round(1.0 / DT)); logtick = int(round(LOG_DT / DT)); nsteps = int(round((T + warm) / DT)); closed_ev = None
    phi = math.exp(-1.0 / 30.0); t_wall = time.time(); conc_log = []

    def tp(typ_i): return {n: i for i, n in enumerate(SUAS + ["EH216", "AAVM", "EC35"])}[typ_i]
    def cmd(s): stack.stack(s)
    def nearest_open(x, y, excl=()):
        d = np.hypot(V[:, 0] - x, V[:, 1] - y);
        for j in np.argsort(d):
            if j not in closed and j not in excl: return int(j)
        return int(np.argmin(d))
    def spawn(item, simt):
        nonlocal nsp, ms, sv, sv2
        acid = f"U{nsp:05d}"; nsp += 1; o = V[item["o"]]; la, lo = F.ll(o[0], o[1]); d = V[item["d"]]
        h = bearing(d[0] - o[0], d[1] - o[1]); typ = item["typ"]; cs = CRUISE[typ]
        cmd(f"CRE {acid} {typ} {float(la):.6f} {float(lo):.6f} {h:.1f} 0 0"); cmd(f"ALT {acid} {item['alt'] / M_PER_FT:.1f}"); cmd(f"SPD {acid} {cs * KT:.1f}"); cmd(f"HDG {acid} {h:.1f}"); stack.process()
        u = U(acid=acid, n=nsp - 1, typ=typ, o=item["o"], d=item["d"], calt=item["alt"], cspd=cs, phase=PHASE["climb"], t0=simt - warm, hdg_cmd=h, dest_xy=d.copy(), dest_id=item["d"])
        u.sig = gsig
        if comm_p and er.random() < comm_p:   # partial C2 outages: 1-3 windows of 10-60 s anywhere in the episode
            for _ in range(er.integers(1, 4)): s0 = er.uniform(0, T - 60); u.comm.append((s0, s0 + er.uniform(10, 60)))
        if sens_p and er.random() < sens_p: u.fault_t = er.uniform(0.17 * T, 0.83 * T); u.bias = er.choice([-1, 1]) * er.uniform(20, 40)
        if nav_p and er.random() < nav_p: u.gnss_loss = (t_event + er.uniform(0, 600), 120.0)
        UL.append(u); ms = np.append(ms, np.inf); sv = np.append(sv, np.int8(0)); sv2 = np.append(sv2, np.int8(0))

    for k in range(nsteps):
        simt = k * DT; ept = simt - warm
        while si < len(sched) and sched[si]["t"] + warm <= simt: queue.append(sched[si]); si += 1
        if queue and k % 5 == 0:
            rest = []
            for it in queue:
                if landing_all: continue
                if it["o"] in closed: continue
                if it["d"] in closed:   # replan to an open destination
                    cand = [j for j in range(len(V)) if j not in closed and j != it["o"]]
                    if not cand: continue
                    it = dict(it); it["d"] = int(min(cand, key=lambda j: np.hypot(*(V[j] - V[it["o"]]))))
                if simt < vp_free[it["o"]]: rest.append(it); continue
                vp_free[it["o"]] = simt + MIN_DEP_GAP; spawn(it, simt)
            queue[:] = rest
        # ---------------- scenario events at t_event (applied by code, logged)
        if not is_nom and meta.get("event_applied") is None and ept >= t_event:
            meta["event_applied"] = ept; air = [i for i, u in enumerate(UL) if u.phase in (0, 1, 2)]
            ev = dict(t=ept, cls=cls, n_airborne=len(air))
            if cls == "CONT-VEHICLE-FAILURE":
                kf = max(3, int(round(0.05 * len(air)))); pick = er.choice(air, min(kf, len(air)), replace=False) if air else []
                for i in pick: u = UL[i]; u.failed = True; u.phase = PHASE["emergency"]; cmd(f"ALT {u.acid} 0"); cmd(f"SPD {u.acid} 0")
                ev["affected"] = [UL[i].acid for i in pick]
            elif cls == "CONT-ALL-LAND":
                landing_all = True; aff = []
                for i in air:
                    u = UL[i]
                    if u.phase == PHASE["descent"]: continue   # already landing: keep its vertiport and phase (a reset to cruise left it at alt 0, never deleted)
                    la, lo = bs.traf.lat[i], bs.traf.lon[i]; x, y = F.xy(la, lo); j = nearest_open(float(x), float(y)); u.dest_xy = V[j].copy(); u.dest_id = j
                    aff.append(u.acid)
                ev["affected"] = aff
            elif cls == "CONT-VERTIPORT-CLOSURE":
                cnt = np.bincount([UL[i].dest_id for i in air], minlength=len(V)) if air else np.zeros(len(V), int); jc = int(np.argmax(cnt)); closed.add(jc); aff = []
                for i in air:
                    u = UL[i]
                    if u.dest_id == jc:
                        la, lo = bs.traf.lat[i], bs.traf.lon[i]; x, y = F.xy(la, lo); j = nearest_open(float(x), float(y)); u.dest_xy = V[j].copy(); u.dest_id = j; aff.append(u.acid)
                ev.update(closed_vertiport=jc, affected=aff)
            elif cls == "CONT-WEATHER":
                wind_target = float(er.uniform(8, 14)); wind_dir = float(er.uniform(0, 360)); wind_t0 = ept   # (draws kept so the event RNG stream is unchanged)
                src = "synthetic"
                if wind_csv:
                    src = pick_real_wind(wind_csv, zone, seed_of("wind", zone, idx)); wind_target, wind_dir = src["speed_ms"], src["from_deg"]
                ev.update(wind_target_ms=wind_target, wind_from_deg=wind_dir, ramp_s=120.0, wind_source=src)
            elif cls == "CONT-NON-COOPERATIVE":
                frac = [0.2, 0.5, 1.0][idx % 3]; kf = int(round(frac * len(air))); pick = er.choice(air, kf, replace=False) if kf else []
                for i in pick: UL[i].coop = False; bs.traf.cr.resooffac[i] = True
                ev.update(fraction=frac, affected=[UL[i].acid for i in pick])
            elif cls == "CONT-INTRUDER":
                nin = int(er.integers(1, 4)); aff = []
                for q in range(nin):
                    ang = er.uniform(0, 2 * np.pi); r = F.half * 0.98; p0 = np.array([r * math.cos(ang), r * math.sin(ang)]) + er.normal(0, 800, 2); p1 = -p0 + er.normal(0, 1500, 2)
                    la, lo = F.ll(p0[0], p0[1]); h = bearing(*(p1 - p0)); acid = f"X{q:04d}"; alt = float(er.uniform(90, 150))
                    cmd(f"CRE {acid} EC35 {float(la):.6f} {float(lo):.6f} {h:.1f} {alt / M_PER_FT:.1f} {CRUISE['EC35'] * KT:.1f}")
                    u = U(acid=acid, n=-1 - q, typ="EC35", o=-1, d=-1, calt=alt, cspd=CRUISE["EC35"], phase=PHASE["transit"], t0=ept, hdg_cmd=h, dest_xy=p1, intr=True, coop=False)
                    UL.append(u); ms = np.append(ms, np.inf); sv = np.append(sv, np.int8(0)); sv2 = np.append(sv2, np.int8(0)); aff.append(acid)
                stack.process()
                for u in UL[-nin:]: i = len(UL) - nin + UL[-nin:].index(u); bs.traf.cr.resooffac[i] = True
                ev["affected"] = aff
            elif cls == "CONT-PASSENGER-DISTRESS":
                ev_air = [i for i in air if UL[i].typ in ("EH216", "AAVM")] or air; kf = min(len(ev_air), int(er.integers(1, 4))); pick = er.choice(ev_air, kf, replace=False) if kf else []
                for i in pick:
                    u = UL[i]; x, y = F.xy(bs.traf.lat[i], bs.traf.lon[i]); j = nearest_open(float(x), float(y)); u.dest_xy = V[j].copy(); u.dest_id = j; u.cspd = min(1.25 * u.cspd, 0.95 * 36 if u.typ == "EH216" else 1.25 * u.cspd)
                    cmd(f"SPD {u.acid} {u.cspd * KT:.1f}")
                ev["affected"] = [UL[i].acid for i in pick]
            elif cls == "DEG-COMM-FULL":
                dur = float(er.uniform(60, 120))
                for u in UL: u.comm.append((ept, ept + dur))
                ev.update(duration_s=dur, all_aircraft=True)
            meta["event_log"].append(ev)
        if not is_nom and cls == "DEG-COMM-FULL" and meta.get("event_applied") is not None:
            pass
        # ---------------- weather ramp (wind field replaced every 10 s during the ramp)
        if wind_target is not None and k % (10 * tick // 5 if False else 50) == 0:
            f = min(1.0, (ept - wind_t0) / 120.0); w = wind_target * f
            bs.traf.wind.clear(); stack.stack(f"WIND {F.lat0:.4f} {F.lon0:.4f} {wind_dir:.1f} {w * KT:.2f}")
        # ---------------- 1 Hz guidance / state layer
        if k % tick == 0 and UL:
            stack.process()
            n = bs.traf.ntraf
            if n != len(UL): raise RuntimeError(f"state/list misalignment: traf={n} list={len(UL)} at t={ept}")
            lat, lon, alt = bs.traf.lat, bs.traf.lon, bs.traf.alt; X, Y = F.xy(lat, lon); dele = []
            for i, u in enumerate(UL[:n]):
                x, y, z = float(X[i]), float(Y[i]), float(alt[i]); u.age += 1.0
                in_comm = any(a <= ept < b for a, b in u.comm); in_loss = u.gnss_loss is not None and u.gnss_loss[0] <= ept < u.gnss_loss[0] + u.gnss_loss[1]
                if u.sig > 0 and ept >= t_event:   # GNSS degradation starts at the event (paired design)
                    u.nx = phi * u.nx + math.sqrt(1 - phi * phi) * u.sig * er.standard_normal(); u.ny = phi * u.ny + math.sqrt(1 - phi * phi) * u.sig * er.standard_normal()
                if not (in_comm or in_loss):
                    u.rx, u.ry = x + u.nx, y + u.ny; u.ralt = z + (u.bias if (u.fault_t is not None and ept >= u.fault_t) else 0.0); u.age = 0.0
                if u.phase in (PHASE["descent"], PHASE["emergency"]) and z < 0.5: dele.append(i); continue
                if u.intr:
                    if abs(x) > F.half * 1.03 or abs(y) > F.half * 1.03: dele.append(i)
                    continue
                if u.phase == PHASE["emergency"]:
                    if z < 0.5: dele.append(i)
                    continue
                if in_loss:   # GNSS loss: dead reckoning, heading random walk, no re-aiming
                    u.drift += er.normal(0, 0.6); cmd(f"HDG {u.acid} {(u.hdg_cmd + u.drift) % 360:.1f}"); continue
                if in_comm: continue
                if u.fault_t is not None and ept >= u.fault_t and abs(u.bias) > 0 and u.phase in (0, 1):
                    tgt = u.calt - u.bias
                    if u.last_alt_cmd != round(tgt): cmd(f"ALT {u.acid} {tgt / M_PER_FT:.1f}"); u.last_alt_cmd = round(tgt)
                dx, dy = u.dest_xy[0] - u.rx, u.dest_xy[1] - u.ry; d = math.hypot(dx, dy); h = bearing(dx, dy)
                if u.phase == PHASE["climb"] and z > u.calt - 3: u.phase = PHASE["cruise"]
                if u.phase in (0, 1) and d < max(120.0, u.cspd * max(z, 10.0) / 3.0 + 60.0):
                    u.phase = PHASE["descent"]; cmd(f"ALT {u.acid} 0"); cmd(f"SPD {u.acid} {min(u.cspd, 6.0) * KT:.1f}")
                if u.phase == PHASE["descent"] and z < 0.5: dele.append(i); continue
                if u.phase == PHASE["descent"] and d < 40 and z < 4: dele.append(i); continue
                if abs(((h - u.hdg_cmd + 180) % 360) - 180) > 1.5: u.hdg_cmd = h; cmd(f"HDG {u.acid} {h:.1f}")
            if dele:
                stack.process(); bs.traf.delete(np.array(dele)); keep = np.ones(len(UL), bool); keep[dele] = False
                UL[:] = [u for u, kk in zip(UL, keep) if kk]; ms = ms[keep]; sv = sv[keep]; sv2 = sv2[keep]
                meta.setdefault("n_deleted", 0); meta["n_deleted"] += len(dele)
            stack.process()
        # ---------------- physics step
        bs.sim.step()
        n = bs.traf.ntraf
        if n:
            bad = np.where(~(np.isfinite(bs.traf.lat) & np.isfinite(bs.traf.lon) & np.isfinite(bs.traf.alt)))[0]
            if len(bad):
                bs.traf.delete(bad); keep = np.ones(len(UL), bool); keep[bad] = False; UL[:] = [u for u, kk in zip(UL, keep) if kk]; ms = ms[keep]; sv = sv[keep]; sv2 = sv2[keep]
                meta["n_nan_deleted"] = meta.get("n_nan_deleted", 0) + int(len(bad)); n = bs.traf.ntraf
        if n != len(UL): raise RuntimeError(f"state/list misalignment after step: traf={n} list={len(UL)} at t={ept}")
        if ept < 0 or n < 2:
            if ept >= 0 and k % logtick == 0 and n: pass
        # ---------------- 5 Hz separation ground truth (true positions)
        if ept >= 0 and n >= 2:
            X, Y = F.xy(bs.traf.lat, bs.traf.lon); Z = bs.traf.alt
            pr = cKDTree(np.c_[X, Y]).query_pairs(D_LOWC, output_type="ndarray")
            cur = {}
            if len(pr):
                dz = np.abs(Z[pr[:, 0]] - Z[pr[:, 1]]); m = dz < H_TOL; pr = pr[m]; dzf = dz[m]
                if len(pr):
                    dh = np.hypot(X[pr[:, 0]] - X[pr[:, 1]], Y[pr[:, 0]] - Y[pr[:, 1]]); s = np.where(dh < D_MAC, 3, np.where(dh < D_NMAC, 2, 1)).astype(np.int8)
                    np.minimum.at(ms, pr[:, 0], dh); np.minimum.at(ms, pr[:, 1], dh); np.maximum.at(sv, pr[:, 0], s); np.maximum.at(sv, pr[:, 1], s)
                    m2 = (dzf < 15.24) & (dh < 248.4); s2 = np.where(dh < D_MAC, 3, np.where(dh < 76.2, 2, 1)).astype(np.int8); np.maximum.at(sv2, pr[m2, 0], s2[m2]); np.maximum.at(sv2, pr[m2, 1], s2[m2])
                    for (a, b), dhh, ss in zip(pr, dh, s): cur[(UL[a].acid, UL[b].acid)] = (float(dhh), int(ss))
            for key, (dhh, ss) in cur.items():
                if key in conf_open: e = conf_open[key]; e[1] = min(e[1], dhh); e[2] = max(e[2], ss); e[3] = ept
                else: conf_open[key] = [ept, dhh, ss, ept]
            for key in [kk for kk in conf_open if kk not in cur]:
                e = conf_open.pop(key); conf_done.append(dict(a=key[0], b=key[1], t_start=e[0], t_end=e[3] + DT, min_dh_m=e[1], severity_max=e[2]))
        # ---------------- 1 Hz logging
        if ept >= 0 and k % logtick == 0 and n:
            X, Y = F.xy(bs.traf.lat, bs.traf.lon); wspd = 0.0 if wind_target is None else wind_target * min(1.0, max(0.0, (ept - wind_t0) / 120.0))
            try: res = np.asarray(bs.traf.cr.hdgactive | bs.traf.cr.vsactive | bs.traf.cr.tasactive, dtype=np.int8)
            except Exception: res = np.zeros(n, np.int8)
            try: inc = np.asarray(bs.traf.cd.inconf, dtype=np.int8)
            except Exception: inc = np.zeros(n, np.int8)
            ul = UL[:n]
            rows.append(pd.DataFrame(dict(
                ep_t=np.full(n, ept, np.float32), uav=np.array([u.n for u in ul], np.int32), type=np.array([tp(u.typ) for u in ul], np.int8),
                lat=bs.traf.lat.astype(np.float64), lon=bs.traf.lon.astype(np.float64), x_m=X.astype(np.float32), y_m=Y.astype(np.float32), alt_m=bs.traf.alt.astype(np.float32),
                gs_ms=bs.traf.gs.astype(np.float32), trk_deg=bs.traf.trk.astype(np.float32), vs_ms=bs.traf.vs.astype(np.float32),
                rep_x_m=np.array([u.rx if u.coop else np.nan for u in ul], np.float32), rep_y_m=np.array([u.ry if u.coop else np.nan for u in ul], np.float32),
                rep_alt_m=np.array([u.ralt if u.coop else np.nan for u in ul], np.float32), rep_age_s=np.array([u.age for u in ul], np.float32),
                phase=np.array([u.phase for u in ul], np.int8), cooperative=np.array([u.coop for u in ul], np.int8), failed=np.array([u.failed for u in ul], np.int8),
                comm_out=np.array([any(a <= ept < b for a, b in u.comm) for u in ul], np.int8),
                gnss_lost=np.array([u.gnss_loss is not None and u.gnss_loss[0] <= ept < u.gnss_loss[0] + u.gnss_loss[1] for u in ul], np.int8),
                sensor_fault=np.array([u.fault_t is not None and ept >= u.fault_t for u in ul], np.int8), resolving=res, in_conflict_cd=inc,
                min_sep_h_m=np.where(np.isfinite(ms[:n]), ms[:n], np.nan).astype(np.float32), severity=sv[:n].copy(), severity_half=sv2[:n].copy(), dist_vp_m=np.hypot(X[:, None] - V[None, :, 0], Y[:, None] - V[None, :, 1]).min(1).astype(np.float32), wind_ms=np.full(n, wspd, np.float32))))
            ms[:] = np.inf; sv[:] = 0; sv2[:] = 0; conc_log.append(n)
        if not quiet and k % 5000 == 0: print(f"  {ep_id} t={ept:7.1f}s n={bs.traf.ntraf} wall={time.time() - t_wall:5.0f}s", flush=True)
    # ------------------------------------------------------------ write
    os.makedirs(out_dir, exist_ok=True); D = pd.concat(rows, ignore_index=True)
    D.to_parquet(os.path.join(out_dir, ep_id + ".parquet"), index=False, compression="zstd")
    pd.DataFrame(conf_done + [dict(a=k[0], b=k[1], t_start=e[0], t_end=e[3] + DT, min_dh_m=e[1], severity_max=e[2]) for k, e in conf_open.items()]).to_parquet(os.path.join(out_dir, ep_id + "_conflicts.parquet"), index=False)
    meta.update(rows=int(len(D)), n_spawned=nsp, concurrent_mean=float(np.mean(conc_log)), concurrent_max=int(np.max(conc_log)), wall_s=time.time() - t_wall, n_lowc_events=len(conf_done) + len(conf_open),
                type_names=SUAS + ["EH216", "AAVM", "EC35"], phase_names=list(PHASE), wind_ramp=dict(t0=wind_t0, target=wind_target, from_deg=wind_dir))
    json.dump(meta, open(os.path.join(out_dir, ep_id + "_meta.json"), "w"), indent=1, default=str)
    return meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cls", default="NOM-A-NORMAL"); ap.add_argument("--zone", default=None); ap.add_argument("--density", default=None)
    ap.add_argument("--idx", type=int, default=1); ap.add_argument("--mode", default="mvp", choices=["free", "mvp"]); ap.add_argument("--out", default="out")
    ap.add_argument("--duration", type=float, default=T_EP); ap.add_argument("--t-event", type=float, default=T_EVENT); ap.add_argument("--warmup", type=float, default=WARM); ap.add_argument("--wind-csv", default=None); a = ap.parse_args()
    cls = a.cls; zone = a.zone or (cls.split("-")[1] if cls.startswith("NOM") else "A"); dens = a.density or (cls.split("-")[2] if cls.startswith("NOM") else "NORMAL")
    m = run_episode(cls, zone, dens, a.idx, a.mode, a.out, T=a.duration, warm=a.warmup, t_event=a.t_event, wind_csv=a.wind_csv)
    print({k: m[k] for k in ["episode_id", "rows", "n_spawned", "concurrent_mean", "concurrent_max", "n_lowc_events", "wall_s"]})

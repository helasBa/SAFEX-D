"""Shared helpers for the SAFEX-D analysis scripts (a1 ... a6)."""
import glob, os, re
import numpy as np, pandas as pd

PROFILES = {"paper": (496.8, 152.4, 10.0, 30.48), "half": (248.4, 76.2, 10.0, 15.24)}   # LoWC, NMAC, MAC (m), vertical tolerance (m)
SEV_COL = {"paper": "severity", "half": "severity_half"}
LEVELS = {"tac": 1, "imm": 2}                       # tactical-or-worse (>= LoWC) / imminent-or-worse (>= NMAC)

def parse_ep(stem):
    """'CONT-ALL-LAND-A-NORMAL_mvp_0001' -> dict(ep, cls, family, zone, dens, mode, idx, group, category)"""
    m = re.match(r"(.+)_(free|mvp)_(\d+)$", stem)
    if not m: raise ValueError(stem)
    base, mode, idx = m.group(1), m.group(2), int(m.group(3))
    if base.startswith("NOM"):
        _, zone, dens = base.split("-"); cls = base; fam = "NOM"
    else:
        cls, zone, dens = base.rsplit("-", 2); fam = cls.split("-")[0]
    cat = {"NOM": "nominal", "CONT": "contingency", "DEG": "degradation"}[fam]
    return dict(ep=stem, cls=cls, family=fam, category=cat, zone=zone, dens=dens, mode=mode, idx=idx, group=f"{zone}-{dens}-{idx:04d}")

def list_episodes(d):
    """Only episodes whose *_meta.json exists (written last by the generator)."""
    return sorted(os.path.basename(f)[:-10] for f in glob.glob(os.path.join(d, "*_meta.json")))

def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan)
    m = x.mean(); h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)); return float(m), float(m - h), float(m + h)

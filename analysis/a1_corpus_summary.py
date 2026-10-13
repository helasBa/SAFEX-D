"""Step 1 - corpus summary + traffic-group splits.
  python a1_corpus_summary.py DATA_DIR -o OUT_DIR [--folds 5] [--workers 4]
Writes OUT_DIR/corpus_summary.csv (1 row / episode), class_table.csv (mean +- std per class and mode),
splits.csv (episode -> traffic group -> fold) and fleet_by_zone.csv.
Splits: every episode that shares a traffic key (zone-density-idx) lives in the SAME fold (NOM and event episodes, free and mvp)."""
import argparse, json, os, numpy as np, pandas as pd
from multiprocessing import Pool
from a_common import *

def one(args):
    d, ep = args
    D = pd.read_parquet(os.path.join(d, ep + ".parquet"), columns=["uav", "type", "alt_m", "severity", "severity_half", "resolving", "ep_t"])
    M = json.load(open(os.path.join(d, ep + "_meta.json"))); C = pd.read_parquet(os.path.join(d, ep + "_conflicts.parquet"))
    r = parse_ep(ep); u = D.drop_duplicates("uav")
    r.update(rows=len(D), uavs=int(D.uav.nunique()), conc_mean=M["concurrent_mean"], conc_max=M["concurrent_max"], lowc_events=int(M["n_lowc_events"]),
             nmac_events=int((C.severity_max >= 2).sum()) if len(C) else 0, mac_events=int((C.severity_max >= 3).sum()) if len(C) else 0,
             evtol_share=float(u.type.isin([8, 9]).mean()), resolving_share=float(D.resolving.mean()), wall_s=M["wall_s"])
    for p, c in SEV_COL.items():
        for k, lv in [("ge1", 1), ("ge2", 2), ("ge3", 3)]: r[f"{p}_sev_{k}_share"] = float((D[c] >= lv).mean())
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("-o", default="analysis_out"); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    os.makedirs(a.o, exist_ok=True); eps = list_episodes(a.dir); print(len(eps), "episodes")
    with Pool(a.workers) as p: S = pd.DataFrame(p.map(one, [(a.dir, e) for e in eps], chunksize=4))
    S.to_csv(os.path.join(a.o, "corpus_summary.csv"), index=False)
    # --- splits (stratified by zone-density so that every fold sees every stratum)
    G = S.drop_duplicates("group")[["group", "zone", "dens"]].copy(); rng = np.random.default_rng(0); nf = min(a.folds, len(G)); fold = {}
    ev = set(S[S.family != "NOM"].group)                       # traffic groups that also carry event episodes (28 episodes each instead of 2)
    # 1) event-bearing groups first, shuffled and dealt round-robin so every fold gets the same number of the heavy groups
    eg = np.array(sorted(ev)); rng.shuffle(eg)
    for i, k in enumerate(eg): fold[k] = i % nf
    # 2) remaining nominal-only groups: round-robin inside each zone x density stratum, continuing the counter so strata stay balanced
    c = len(eg)
    for _, g in G[~G.group.isin(ev)].groupby(["zone", "dens"]):
        ks = np.array(g.group.tolist()); rng.shuffle(ks)
        for k in ks: fold[k] = c % nf; c += 1
    S["fold"] = S.group.map(fold); S[["ep", "cls", "family", "category", "zone", "dens", "mode", "idx", "group", "fold"]].to_csv(os.path.join(a.o, "splits.csv"), index=False)
    print(S.groupby("fold").group.nunique().rename("groups_per_fold").to_string())
    # --- class table
    cols = ["conc_mean", "lowc_events", "nmac_events", "mac_events", "paper_sev_ge1_share", "paper_sev_ge2_share", "half_sev_ge1_share", "half_sev_ge2_share", "resolving_share", "evtol_share"]
    T = S.groupby(["cls", "mode"]).agg(n=("ep", "size"), **{c + "_mean": (c, "mean") for c in cols}, **{c + "_std": (c, "std") for c in cols}).round(4)
    T.to_csv(os.path.join(a.o, "class_table.csv")); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T[["n", "conc_mean_mean", "lowc_events_mean", "nmac_events_mean", "paper_sev_ge1_share_mean", "paper_sev_ge2_share_mean", "half_sev_ge1_share_mean", "resolving_share_mean"]].to_string())
    S[S.family == "NOM"].groupby(["zone", "dens", "mode"])[["evtol_share", "conc_mean"]].mean().round(3).to_csv(os.path.join(a.o, "fleet_by_zone.csv"))
    print("->", a.o)

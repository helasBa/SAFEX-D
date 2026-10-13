"""Build the Zenodo release folder for SAFEX-D v2 and check that nothing is missing.
  python make_release.py --data C:\\these\\safex2_data --out C:\\these\\release --small C:\\these\\safex2_out --wind C:\\these\\real_wind.csv [--extra file ...] [--check-only]
What it does
  1. lists the episodes (those with a *_meta.json) and checks that each one has its three files: <ep>.parquet, <ep>_conflicts.parquet, <ep>_meta.json
  2. writes one uncompressed .tar per family x mode (NOM/CONT/DEG x free/mvp = 6 archives; parquet is already zstd-compressed)
  3. copies the small files (top-level *.csv and *.json of --small, the wind table, --extra files) into <out>\\metadata
  4. writes MANIFEST.csv (every file, its archive, size) and CHECKSUMS.sha256 (archives and metadata)
  5. refuses to finish if any file of the data folder is not in an archive, or if the episode count differs from splits.csv
Nothing in --data is modified."""
import argparse, glob, hashlib, os, re, shutil, sys, tarfile, csv

RX = re.compile(r"^(?P<ep>.+_(?P<mode>free|mvp)_\d+?)(?P<suf>_conflicts\.parquet|_meta\.json|\.parquet)$")

def sha256(p, blk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""): h.update(b)
    return h.hexdigest()

def fam(ep): return "NOM" if ep.startswith("NOM") else ep.split("-")[0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--small", help="folder with splits.csv, corpus_summary.csv, class_table.csv, benchmark*.json, a13-a16 csv ...")
    ap.add_argument("--wind", help="real_wind.csv (ERA5 table used for CONT-WEATHER)"); ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--check-only", action="store_true"); a = ap.parse_args()
    files = sorted(os.listdir(a.data)); eps = {}; other = []; campaign_log = None
    for f in files:
        if not os.path.isfile(os.path.join(a.data, f)): other.append(f + "/"); continue
        if f == "_campaign_log.csv": campaign_log = os.path.join(a.data, f); continue
        m = RX.match(f)
        if not m: other.append(f); continue
        eps.setdefault(m.group("ep"), {})[m.group("suf")] = f
    complete = {e: d for e, d in eps.items() if len(d) == 3 and "_meta.json" in d}
    bad = {e: sorted({"_conflicts.parquet", "_meta.json", ".parquet"} - set(d)) for e, d in eps.items() if e not in complete}
    print(f"{len(complete)} complete episodes, {len(bad)} incomplete, {len(other)} other entries in {a.data}")
    for e, miss in list(bad.items())[:10]: print("  INCOMPLETE", e, "missing", miss)
    for o in other[:15]: print("  NOT IN ARCHIVES (decide: delete or move out of the data folder):", o)
    groups = {}
    for e in complete: groups.setdefault(f"{fam(e)}_{complete[e] and re.search('_(free|mvp)_', e).group(1)}", []).append(e)
    for k, v in sorted(groups.items()): print(f"  {k:10s} {len(v):4d} episodes")
    if a.small and os.path.exists(os.path.join(a.small, "splits.csv")):
        n = sum(1 for _ in open(os.path.join(a.small, "splits.csv"), encoding="utf-8")) - 1
        print("splits.csv lists", n, "episodes ->", "OK" if n == len(complete) else "MISMATCH with the data folder")
        if n != len(complete): sys.exit(1)
    if bad or other:
        print("Fix the lines above first (an archive must contain exactly the episodes of the corpus)."); sys.exit(1)
    if a.check_only: return
    os.makedirs(os.path.join(a.out, "metadata"), exist_ok=True); man = []
    for k, v in sorted(groups.items()):
        tp = os.path.join(a.out, f"safex2_{k.lower()}.tar"); print("writing", tp, flush=True)
        with tarfile.open(tp, "w") as t:
            for e in sorted(v):
                for suf in (".parquet", "_conflicts.parquet", "_meta.json"):
                    f = complete[e][suf]; t.add(os.path.join(a.data, f), arcname=f"{k.lower()}/{f}")
                    man.append((f, os.path.basename(tp), os.path.getsize(os.path.join(a.data, f))))
    md = os.path.join(a.out, "metadata"); small = []
    if a.small:
        for f in sorted(glob.glob(os.path.join(a.small, "*.csv")) + glob.glob(os.path.join(a.small, "*.json"))):
            if os.path.getsize(f) > 50 << 20: print("skipped (>50 MB):", f); continue
            small.append(f)
    if a.wind: small.append(a.wind)
    small += list(a.extra)
    if campaign_log: small.append(campaign_log)
    for f in small:
        if not os.path.isfile(f): print("MISSING small file:", f); continue
        shutil.copy2(f, md)
    with open(os.path.join(a.out, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "archive", "bytes"]); w.writerows(man)
    with open(os.path.join(a.out, "CHECKSUMS.sha256"), "w", newline="\n") as f:
        for p in sorted(glob.glob(os.path.join(a.out, "*.tar")) + glob.glob(os.path.join(md, "*")) + [os.path.join(a.out, "MANIFEST.csv")]):
            f.write(f"{sha256(p)}  {os.path.relpath(p, a.out).replace(os.sep, '/')}\n")
    print("done:", len(man), "files archived,", len(os.listdir(md)), "metadata files. Add README.md, DATASHEET.md, LICENSE to", a.out)

if __name__ == "__main__": main()

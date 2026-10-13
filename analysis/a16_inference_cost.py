"""Step 16 - inference cost: what does it cost, per aircraft and per surveillance cycle, to produce the conflict forecast?  (edge / ground-station budget)
Timed on the machine you run it on (state the hardware in the paper: CPU model, GPU model). Median of --repeats runs after a warm-up, batch of --n aircraft.
  python a16_inference_cost.py --feats C:\\these\\safex2_feats --splits C:\\these\\safex2_out\\splits.csv -o C:\\these\\safex2_out\\a16_inference_cost.csv [--device cuda] [--models cv chronos2_diff timesfm25_diff] [--n 1000]
Rows: constant-velocity extrapolation | gradient-boosted hybrid (prediction only, features given) | Chronos-2 | TimesFM 2.5 (trajectory forecast only; the separation rule that follows is the same for all)
Reported: ms per batch, ms per aircraft, aircraft per second, parameters. Feature construction (neighbour search) is NOT included for the hybrid, and the separation rule is NOT included for any model:
the table compares the model-specific cost only; say so in the text."""
import argparse, glob, os, platform, time, numpy as np, pandas as pd

def timeit(fn, repeats, warm=1):
    for _ in range(warm): fn()
    v = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); v.append(time.perf_counter() - t)
    return float(np.median(v))

def sync(dev):
    if dev.startswith("cuda"):
        import torch; torch.cuda.synchronize()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--feats"); ap.add_argument("--splits"); ap.add_argument("-o", default="a16_inference_cost.csv"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", nargs="+", default=["cv", "chronos2_diff", "timesfm25_diff"]); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--no-gbm", action="store_true"); a = ap.parse_args()
    from a5_foundation_model import make_predictor
    rng = np.random.default_rng(0); ctx = np.cumsum(rng.normal(0, 1, (a.n, 25, 3)) + np.array([60.0, 40.0, 0.0]), axis=1).astype(np.float32)   # 25 positions = 24 increments (120 s at 5 s)
    rows = []; hw = dict(cpu=platform.processor() or platform.machine(), python=platform.python_version(), device=a.device)
    try:
        import torch
        if a.device.startswith("cuda"): hw["gpu"] = torch.cuda.get_device_name(0)
    except Exception: pass
    for m in a.models:
        try:
            f = make_predictor(m, a.device, a.batch); par = np.nan
            t = timeit(lambda: (f(ctx), sync(a.device)), a.repeats)
            rows.append(dict(model=m, device=("cpu" if m.startswith("cv") else a.device), n=a.n, ms_batch=1e3 * t, ms_per_aircraft=1e3 * t / a.n, aircraft_per_s=a.n / t))
        except Exception as e: print("skip", m, repr(e)[:200])
    if not a.no_gbm and a.feats:
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
        sp = pd.read_csv(a.splits); fs = sorted(glob.glob(os.path.join(a.feats, "*.parquet")))[:20]
        D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True); y = "y_paper_tac_h30"; D = D[D[y].notna()]
        cols = [c for c in D.columns if c.startswith(("raw_", "rel_", "seq_"))] + ["cv_paper_h30"]; mdl = HGB(max_iter=150, class_weight="balanced", random_state=0).fit(D[cols].iloc[:200000], D[y].iloc[:200000])
        X = D[cols].sample(a.n, replace=True, random_state=0)
        t = timeit(lambda: mdl.predict_proba(X), a.repeats); rows.append(dict(model="hybrid (GBM, 150 trees, one horizon)", device="cpu", n=a.n, ms_batch=1e3 * t, ms_per_aircraft=1e3 * t / a.n, aircraft_per_s=a.n / t))
    R = pd.DataFrame(rows); R.to_csv(a.o, index=False); open(os.path.splitext(a.o)[0] + "_hardware.txt", "w").write(str(hw)); print(R.round(3).to_string(index=False)); print(hw)

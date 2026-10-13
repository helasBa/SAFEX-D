"""Sanity check of the foundation-model adapters on SYNTHETIC straight-line trajectories (no dataset needed).
  python a5_sanity_lines.py --model chronos2 --device cuda
  python a5_sanity_lines.py --model timesfm25 --device cuda
  python a5_sanity_lines.py --model chronos2,chronos2_diff --device cuda     # positions vs increments (_diff) in ONE run
  python a5_sanity_lines.py --model cv --device cpu            # reference
A constant-velocity aircraft is the easiest possible case: a correct adapter must predict a straight line, with an error that grows smoothly with the
horizon and is of the order of the position noise. A model whose error is large at 10 s and smaller at 30 s (as in the first campaign) points to an
adapter problem (alignment, scaling, ordering of the variates) rather than to a property of the data. The same noise levels as SAFEX-D (9 m, 18 m) are used."""
import argparse, json, numpy as np
from a5_foundation_model import make_predictor, cv_predict, STEP, C, H

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="cv", help="one or several names separated by commas, e.g. chronos2,chronos2_diff"); ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--ckpt", default=None, help="fine-tuned Chronos-2 checkpoint")
    ap.add_argument("-o", default=None); a = ap.parse_args(); rng = np.random.default_rng(0); preds = {m: make_predictor(m, a.device, a.batch, a.ckpt) for m in a.model.split(",")}; out = {}
    n = a.n; sp = rng.uniform(5, 25, n); hd = rng.uniform(0, 2 * np.pi, n); v = np.c_[sp * np.sin(hd), sp * np.cos(hd), np.zeros(n)]; p0 = np.c_[rng.uniform(-3000, 3000, n), rng.uniform(-3000, 3000, n), rng.uniform(60, 200, n)]
    tt = (np.arange(-C, H + 1) * STEP)[None, :, None]                                 # time axis in seconds, 0 = now (index C)
    truth = p0[:, None, :] + v[:, None, :] * tt                                      # (n, C+1+H, 3)
    for sig in (0.0, 9.0, 18.0):
        ctx = (truth[:, :C + 1] + rng.normal(0, sig, (n, C + 1, 3)) * np.array([1, 1, 0.1])).astype(np.float32); fut = truth[:, C + 1:]
        res = {}
        for nm, f in [("constant-velocity", cv_predict)] + list(preds.items()):
            pr = f(ctx); e = np.hypot(pr[..., 0] - fut[..., 0], pr[..., 1] - fut[..., 1]); res[nm] = e.mean(0)
        out[str(sig)] = {k: v_.tolist() for k, v_ in res.items()}
        print(f"\nnoise sigma = {sig:.0f} m   mean horizontal error (m) at horizon:")
        print("  " + "".join(f"{(k + 1) * STEP:>8d}s" for k in range(H)))
        for nm, e in res.items(): print(f"{nm[:16]:16s}" + "".join(f"{x:9.1f}" for x in e))
    if a.o: json.dump(out, open(a.o, "w"), indent=1)

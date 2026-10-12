"""LP upper bound alpha*(eps) on every strategy-proof mechanism in the complementarity model,
for a grid of eps and grid sizes N.  Writes results/data/lp_eps_curve.csv."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core import lp
Ns = [int(s) for s in sys.argv[1:]] or [20, 40, 80]
EPS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.2]
out = ROOT / "results/data/lp_eps_curve.csv"
rows = []
for N in Ns:
    for e in EPS:
        t = time.time(); r = lp.solve_sp_lp(N, eps=e, symmetric=True)
        rows.append({"N": N, "eps": e, "alpha": r.alpha, "binding": int(r.binding.sum()), "seconds": time.time() - t})
        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(out, index=False)

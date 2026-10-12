"""How does the Theorem-1 LP bound depend on the grid size N?  Writes results/data/lp_convergence.csv.

Usage: python experiments/lp_convergence.py [eps] [N1 N2 ...]
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import lp  # noqa: E402

eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
Ns = [int(s) for s in sys.argv[2:]] or [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300]
out = Path(__file__).resolve().parents[1] / "results" / "data" / f"lp_convergence_eps{eps:g}.csv"
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
done = {int(r["N"]) for r in rows}
for N in Ns:
    if N in done:
        continue
    t = time.time()
    r = lp.solve_sp_lp(N, eps=eps, symmetric=True)
    rows.append({"N": N, "eps": eps, "alpha": r.alpha, "binding": int(r.binding.sum()), "seconds": time.time() - t})
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)

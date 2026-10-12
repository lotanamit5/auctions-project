"""Adaptive type grids for the Theorem-1 LP: refine the type set where the dual lives.
Writes results/data/lp_adaptive_types.csv (one row per iteration) and the final type set."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core import lp
eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
N0 = int(sys.argv[2]) if len(sys.argv) > 2 else 40
max_types = int(sys.argv[3]) if len(sys.argv) > 3 else 320
T = list(np.linspace(0, 1, N0 + 1))
rows = []
it = 0
while True:
    t0 = time.time(); r = lp.solve_sp_lp(0, eps=eps, types=T, symmetric=True)
    g = np.array(T)
    w = r.dual_welfare.sum(axis=1) + r.dual_welfare.sum(axis=0)          # dual mass per type
    rows.append({"iter": it, "n_types": len(T), "alpha": r.alpha, "seconds": time.time() - t0})
    print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(ROOT / f"results/data/lp_adaptive_types_eps{eps:g}.csv", index=False)
    np.savetxt(ROOT / f"results/data/lp_adaptive_types_eps{eps:g}_types.txt", g)
    if len(T) >= max_types:
        break
    # add midpoints on both sides of the heaviest types (up to 25% more types per round)
    k = max(4, len(T) // 4)
    heavy = np.argsort(-w)[:k]
    new = set()
    for i in heavy:
        if i > 0: new.add(0.5 * (g[i] + g[i - 1]))
        if i < len(g) - 1: new.add(0.5 * (g[i] + g[i + 1]))
    T = sorted(set(T) | new)
    it += 1

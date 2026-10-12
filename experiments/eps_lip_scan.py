"""eps model: competitive ratio of eps-aware LIP(a) over (a, eps); best a per eps; best mixture
with take-all.  Writes results/data/eps_lip_grid.csv and results/data/eps_lip_best.csv."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core import eps as E
NESTED = "--nested" in sys.argv
SUF = "_nested" if NESTED else ""

EPS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.2, 1.5, 2.0]
A = [0.5, 1, 1.5, 2, 2.1, 2.5, 3, 4, 5, 6, 8, 10, 15, 20]
g = np.linspace(0, 1, 401)
X, Y = np.meshgrid(g, g, indexing="ij"); tri = X >= Y
X, Y = X[tri], Y[tri]


def grid_min(a, e, lam=1.0):
    R = E.ratio_mixture(X, Y, a, e, lam, nested=NESTED)
    i = int(np.argmin(R)); return float(R[i]), float(X[i]), float(Y[i])


grid_out = ROOT / f"results/data/eps_lip_grid{SUF}.csv"
if not grid_out.exists():
    rows = []
    for e in EPS:
        for a in A:
            r, x, y = grid_min(a, e)
            rows.append({"eps": e, "a": a, "ratio_grid": r, "x": x, "y": y}); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(grid_out, index=False)

best_out = ROOT / f"results/data/eps_lip_best{SUF}.csv"
best = pd.read_csv(best_out).to_dict("records") if best_out.exists() else []
done = {float(r["eps"]) for r in best}
for e in EPS:
    if float(e) in done:
        continue
    t0 = time.time()
    cands = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 12, 20, 40]
    vals = [grid_min(a, e)[0] for a in cands]; i = int(np.argmax(vals))
    lo_a, hi_a = cands[max(i - 1, 0)], cands[min(i + 1, len(cands) - 1)]
    r = minimize_scalar(lambda a: -grid_min(a, e)[0], bounds=(lo_a, hi_a), method="bounded", options={"xatol": 1e-3})
    a_star = float(r.x); val, x, y = grid_min(a_star, e)
    lo, up, arg, _ = E.bracket_lip(a_star, e, tol=2e-4, max_cells=400_000, nested=NESTED)
    # best mixture lam * LIP(a*) + (1-lam) take-all (coarse lam grid, then certify)
    lams = np.linspace(0, 1, 41)
    vals = [grid_min(a_star, e, lam)[0] for lam in lams]
    lam_star = float(lams[int(np.argmax(vals))])
    mlo, mup, _, _ = E.bracket_mixture(a_star, e, lam_star, tol=2e-4, max_cells=400_000, nested=NESTED)
    best.append({"eps": e, "a_star": a_star, "ratio_grid": val, "x": x, "y": y, "lower": lo, "upper": up,
                 "takeall": (1 + e) / (1 + max(e, 1)), "equal": (1 + e / 2) / (1 + max(e, 1)),
                 "lam_star": lam_star, "mix_lower": mlo, "mix_upper": mup,
                 "lip2_grid": grid_min(2.0, e)[0], "seconds": time.time() - t0})
    print(best[-1], flush=True)
    pd.DataFrame(best).to_csv(best_out, index=False)

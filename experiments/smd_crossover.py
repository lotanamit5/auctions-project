"""Where SMD overtakes the fixed-menu mechanisms: certified ratio of nested LIP (best a) next to
the SMD closed form and rho_B, for eps around the crossover.  Writes results/data/smd_crossover.csv.
SMD beats every fixed menu once it exceeds rho_B, i.e. for eps > (sqrt(3)-1)/2."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core import eps as E, split

g = np.linspace(0, 1, 401); X, Y = np.meshgrid(g, g, indexing="ij"); tri = X >= Y; X, Y = X[tri], Y[tri]
grid_min = lambda a, e: float(E.ratio_mixture(X, Y, a, e, 1.0, nested=True).min())
rows = []
for e in [0.30, 0.32, 0.34, 0.345, 0.35, 0.36, 0.37, 0.40]:
    r = minimize_scalar(lambda a: -grid_min(a, e), bounds=(1.2, 2.4), method="bounded", options={"xatol": 1e-3})
    lo, up, _, _ = E.bracket_lip(float(r.x), e, tol=2e-4, max_cells=400_000, nested=True)
    rows.append({"eps": e, "a_best": float(r.x), "nested_lower": lo, "nested_upper": up,
                 "smd": float(split.ratio_closed_form(np.array([e]))[0]), "rho_B": (1 + e) / (1 + 2 * e - e * e)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/data/smd_crossover.csv", index=False)
print("SMD = rho_B at eps =", (np.sqrt(3) - 1) / 2)

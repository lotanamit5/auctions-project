"""Best swap-dictatorial (fixed-menu) mechanism on a grid, per eps (LP), next to the SP-LP.
Writes results/data/sd_lp_curve.csv and the optimal menus to results/data/sd_menu_eps*.csv."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core import lp
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
EPS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.2]
out = ROOT / "results/data/sd_lp_curve.csv"
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
done = {(int(r["N"]), float(r["eps"])) for r in rows}
for e in EPS:
    if (N, float(e)) in done:
        continue
    t = time.time(); r = lp.solve_sd_lp(N, eps=e)
    rows.append({"N": N, "eps": e, "alpha_sd": r.alpha, "seconds": time.time() - t})
    pd.DataFrame({"type": r.types, "m1": r.m1, "m2": r.m2, "m12": r.m12}).to_csv(ROOT / f"results/data/sd_menu_N{N}_eps{e:g}.csv", index=False)
    print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(out, index=False)

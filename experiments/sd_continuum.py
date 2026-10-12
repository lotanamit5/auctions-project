"""Grid-optimal swap-dictatorial menus (SD-LP) evaluated EXACTLY on the continuous type space.
Usage: python experiments/sd_continuum.py EPS N1 N2 ...   -> results/data/sd_continuum.csv (append)"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core import lp
from core.menu import FiniteMenuMechanism
eps = float(sys.argv[1]); Ns = [int(s) for s in sys.argv[2:]]
out = ROOT / "results/data/sd_continuum.csv"
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
for N in Ns:
    t = time.time(); r = lp.solve_sd_lp(N, eps=eps)
    items = np.unique(np.round(np.c_[r.m1, r.m2, r.m12], 10), axis=0)
    pd.DataFrame(items, columns=["m1", "m2", "m12"]).to_csv(ROOT / f"results/data/sd_menu_N{N}_eps{eps:g}.csv", index=False)
    val, pt = FiniteMenuMechanism(items, eps).exact_ratio()
    rows.append({"eps": eps, "N": N, "sd_lp_grid": r.alpha, "n_items": len(items), "exact_continuum": val, "x": pt[0], "y": pt[1], "seconds": time.time() - t})
    print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(out, index=False)

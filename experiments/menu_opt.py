"""Directly optimise a finite, item-symmetric menu of K lotteries for the continuous type space,
maximising the *exact* competitive ratio (FiniteMenuMechanism.exact_ratio).

Usage: python experiments/menu_opt.py EPS K [restarts]
Writes results/data/menu_opt_eps{EPS}_K{K}.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from core.menu import FiniteMenuMechanism
from core import lip

eps = float(sys.argv[1]); K = int(sys.argv[2]); restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 8
rng = np.random.default_rng(0)


def unpack(theta):
    """theta -> item-symmetric menu.  K 'half' items (m1 >= m2) plus their mirror images,
    nested lotteries (m12 = min).  Coordinates squashed into [0, 1] by a sigmoid."""
    th = 1 / (1 + np.exp(-np.asarray(theta)))
    pts = th.reshape(K, 2)
    items = []
    for a, b in pts:
        hi, lo = max(a, b), min(a, b)
        items.append((hi, lo, lo)); items.append((lo, hi, lo))
    return items


def objective(theta):
    try:
        return -FiniteMenuMechanism(unpack(theta), eps).exact_ratio()[0]
    except Exception:
        return 1.0


best = (-1, None)
t0 = time.time()
# seed: LIP(a*) frontier points as a warm start, plus random
seeds = []
g = np.linspace(0.5, 1, K)
q1, q2 = lip.bundle_additive(g, 2.1013)
seeds.append(np.log(np.clip(np.c_[q1, q2], 1e-3, 1 - 1e-3) / (1 - np.clip(np.c_[q1, q2], 1e-3, 1 - 1e-3))).ravel())
for _ in range(restarts - 1):
    seeds.append(rng.normal(0, 1.5, 2 * K))
for i, s in enumerate(seeds):
    r = minimize(objective, s, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8, "adaptive": True})
    val = -r.fun
    print(f"restart {i}: ratio {val:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    if val > best[0]:
        best = (val, r.x)
        out = {"eps": eps, "K": K, "ratio": val, "items": unpack(r.x)}
        (ROOT / f"results/data/menu_opt_eps{eps:g}_K{K}.json").write_text(json.dumps(out, indent=1))
print("best", best[0], unpack(best[1]))

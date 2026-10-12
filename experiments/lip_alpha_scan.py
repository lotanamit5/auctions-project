"""alpha(a) for LIP(a) with additive valuations: certified bracket on a grid of a, the
decomposition alpha(a) = min(theta(a), beta(a)) and the optimal a.  Writes
results/data/lip_alpha_scan.csv and results/data/lip_alpha_opt.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import lip  # noqa: E402

rows = []
for a in np.round(np.arange(0.5, 6.01, 0.1), 2):
    lo, up, _ = lip.sandwich_adaptive(a, tol=1e-6)
    _, _, _, (xu, yu) = lip.sandwich(a, 1000)
    th_lo, th_up = lip.theta_bounds(a, 100_000)
    rows.append({"a": a, "alpha_lower": lo, "alpha_upper": up, "x_worst": xu, "y_worst": yu,
                 "theta_lower": th_lo, "theta_upper": th_up})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/data/lip_alpha_scan.csv", index=False)

f = lambda a: -lip.sandwich_adaptive(a, tol=1e-6)[1]
r = minimize_scalar(f, bounds=(2.0, 2.2), method="bounded", options={"xatol": 1e-6})
a_star = float(r.x)
lo, up, _ = lip.sandwich_adaptive(a_star, tol=1e-6)
opt = {"a_star": a_star, "alpha_lower": lo, "alpha_upper": up,
       "theta_at_a_star": lip.theta_bounds(a_star, 100_000),
       "alpha_2_lower": lip.sandwich_adaptive(2.0, tol=1e-6)[0],
       "alpha_2_upper": lip.sandwich_adaptive(2.0, tol=1e-6)[1]}
(ROOT / "results/data/lip_alpha_opt.json").write_text(json.dumps(opt, indent=2))
print(opt)

"""Builds notebooks/06_substitutes.ipynb (side exploration)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb import NB

nb = NB("06_substitutes")

nb.md(r"""
# 06 — Side note: substitutes ($\varepsilon<0$)

The proposal fixes $\varepsilon\ge0$. Out of curiosity we let $\varepsilon<0$ (the goods are
substitutes: holding both is worth less than the sum). **Caveat:** for $\varepsilon<-v$ the
marginal value of item 1 becomes negative when the agent already holds item 2, and we do *not*
model free disposal, so the exercise is only meaningful for mild substitutability, say
$\varepsilon\in[-\tfrac12,0)$.

* First best: pooling now gives $1+\varepsilon<1$, so $\mathrm{OPT}=1+|x-y|$ — the additive split.
* The McCormick LP needs no change (utilities are still bilinear).
* The fixed-menu LIP mechanism is still strategy-proof; the dictator's objective is no longer
  guaranteed concave, so we use the robust grid+refine solver of `core.menu`.
""")

nb.code(r"""
from core import lp
from core.menu import LIPMenu, MenuMechanism
rows = []
for e in (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25):
    r = lp.solve_sp_lp(30, eps=e, symmetric=True)
    best = max(((MenuMechanism(LIPMenu(a), e, nested=False, grid=801).grid_min(150)[0], a) for a in (1.0, 1.5, 2.0, 2.1, 2.5, 3.0, 4.0)))
    rows.append({"eps": e, "SP upper bound (LP, N=30)": r.alpha, "best eps-aware LIP (a in coarse set)": best[0], "a": best[1],
                 "equal split": (1 + e / 2) / 2, "take-all": (1 + e) / 2})
pd.DataFrame(rows).round(4)
""")

nb.md(r"""
With substitutes the picture is the mirror image of complementarities: the first best never
pools, the LP bound *falls* below the additive 0.84 (the designer cannot exploit any
"agreement" between agents, and the dictator's complementarity-driven over-buying now hurts),
and the deterministic $\varepsilon$-aware LIP keeps most of its performance — the menu class is
less handicapped because splitting is what menus do well. We leave a proper treatment (with
free disposal) as future work.
""")

nb.save(execute=True)

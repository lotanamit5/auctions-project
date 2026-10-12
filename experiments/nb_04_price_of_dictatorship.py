"""Builds notebooks/04_price_of_dictatorship.ipynb."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb import NB

nb = NB("04_price_of_dictatorship")

nb.md(r"""
# 04 — The price of dictatorship: fixed-menu mechanisms vs all strategy-proof mechanisms

The paper's nested families are $\mathrm{LIP}\subsetneq\mathrm{IP}\subsetneq\mathrm{SD}\subsetneq\mathrm{SP}$
(linear increasing-price, increasing-price, swap-dictatorial, strategy-proof). It bounds
LIP from below (0.828) and SP from above (0.841), and says nothing about where SD sits. Here:

1. **the best swap-dictatorial mechanism on a type grid is itself an LP** (menu items are the
   variables; incentive and welfare constraints are linear in them) — `core.lp.solve_sd_lp`;
2. in the additive model its value is **exactly 5/6**, independently of the grid, certified by
   the three types $\{0,\tfrac12,1\}$ — and the dual uses *no* incentive constraint, which led
   to a four-line proof (`docs/theory/03_price_of_dictatorship.md`, Theorem A);
3. with complementarities the gap between fixed-menu and general strategy-proof mechanisms
   becomes large.
""")

nb.md(r"""
## 1. Additive model: SD $\le 5/6$

Theorem A (paraphrased): if $M(x,y)=\lambda A(x)+(1-\lambda)B(y)$ with $A$ depending only on
agent 1's report and $B$ only on agent 2's, then summing welfare over the four profiles
$(\tfrac12,0),(1,\tfrac12),(0,\tfrac12),(\tfrac12,1)$ (first best $\tfrac32$ each) gives
$\sum W\le 5$: the middle type's allocation cancels and the extreme types contribute at most
$\pm1$. So $\min W\le\tfrac54=\tfrac56\cdot\tfrac32$.

Numerically, the SD-LP returns $5/6$ for every grid we tried, and already for 3 types:
""")

nb.code(r"""
from fractions import Fraction
from core import lp, lip
from core.menu import FiniteMenuMechanism
import json
rows = []
for name, T in (("{0, 1/2, 1}", [0, .5, 1]), ("{0, 1/4, ..., 1}", np.linspace(0, 1, 5)), ("{j/20}", np.linspace(0, 1, 21)), ("{j/100}", np.linspace(0, 1, 101)), ("{j/200}", np.linspace(0, 1, 201))):
    r = lp.solve_sd_lp(0, types=T)
    rows.append({"type set": name, "SD-LP value": r.alpha, "as fraction": str(Fraction(r.alpha).limit_denominator(1000))})
pd.DataFrame(rows)
""")

nb.code(r"""
r3 = lp.solve_sd_lp(0, types=[0, .5, 1])
print("menu (m1, m2) chosen by types 0, 1/2, 1:\n", np.c_[r3.types, r3.m1, r3.m2].round(3))
print("dual weights on competitiveness rows (j >= k):\n", r3.dual_welfare.round(4))
print("dual weights on IC rows:", r3.dual_ic.round(6).max(), "(all zero: incentive constraints are not needed for the bound)")
print("ratio per profile:\n", r3.ratio.round(4))
""")

nb.md(r"""
The LP-optimal mechanism on $\{0,\tfrac12,1\}$ is the obvious one — each type takes its own item,
the middle type splits — and it is exactly $5/6$-competitive there: the profiles $(1,\tfrac12)$
and $(\tfrac12,0)$ both give welfare $1.25$ against $1.5$.

## 2. Where does LIP sit?  Can any menu beat it on the continuum?

$\mathrm{LIP}(a^\star)$ achieves $0.82939$; the SD bound is $0.83333$; the SP bound (LP, large $N$) is
about $0.835$–$0.837$. So **LIP is within $0.004$ of the best possible menu mechanism**, and the
best non-dictatorial mechanism can gain at most $\approx0.007$ more.

Using a grid-optimal SD menu for *all* types does not transfer well (it exploits the grid):
""")

nb.code(r"""
r = lp.solve_sd_lp(100, eps=0.0)
items = np.unique(np.round(np.c_[r.m1, r.m2, r.m12], 9), axis=0)
f = FiniteMenuMechanism(items, 0.0)
val, pt = f.exact_ratio()
print(f"N=100 SD-LP menu: {len(items)} distinct items; value on the grid {r.alpha:.4f}; exact value on the continuum {val:.4f} at {pt}")
opt = json.loads((DATA / "lip_alpha_opt.json").read_text())
print(f"LIP(a*) on the continuum: {opt['alpha_lower']:.5f}")
res = sorted(DATA.glob("menu_opt_eps0_K*.json"))
for p in res:
    d = json.loads(p.read_text()); print(f"direct finite-menu optimiser ({p.name}): exact ratio {d['ratio']:.5f} with {len(d['items'])} items")
""")

nb.code(r"""
# the LIP(a*) frontier against the grid-optimal SD menu points
fig, ax = plt.subplots(figsize=(5.5, 5))
z = np.linspace(0, 1, 300)
ax.plot(z, lip.Q(1 - lip.cost(z, opt["a_star"]), opt["a_star"]), label=f"LIP(a*={opt['a_star']:.3f}) frontier")
ax.scatter(r.m1, r.m2, s=12, c="C3", label="SD-LP optimal menu items (N=100)")
for p in res[-1:]:
    d = json.loads(p.read_text()); it = np.array(d["items"]); ax.scatter(it[:, 0], it[:, 1], s=40, marker="x", c="k", label=f"finite-menu optimiser ({d['ratio']:.4f})")
ax.set(xlabel="$m_1$ (share / prob. of item 1)", ylabel="$m_2$", title="Menus in the additive model", aspect="equal"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "04_menus_eps0.png"); plt.show()
""")

nb.md(r"""
## 3. Complementarities: dictatorship becomes expensive

With $\varepsilon>0$ the SD-LP values are again rational and grid-independent; and the optimal
menus are tiny. At $\varepsilon=\tfrac12$ the menu is $\{(\tfrac37,0),(0,\tfrac37),\text{balanced pooling lottery}\}$
and its *exact* ratio on the continuum is $6/7$: the dictator never takes more than $3/7$ of
anything — the mechanism is closer to equal split than to a dictatorship.
""")

nb.code(r"""
sd100 = pd.read_csv(DATA / "sd_lp_curve.csv")
sd = sd100[sd100.N == sd100.N.max()].sort_values("eps")
lpc = pd.read_csv(DATA / "lp_eps_curve.csv"); lpN = lpc[lpc.N == lpc.N.max()].sort_values("eps")
det = pd.read_csv(DATA / "eps_lip_best.csv").drop_duplicates("eps", keep="last").sort_values("eps"); nes = pd.read_csv(DATA / "eps_lip_best_nested.csv").drop_duplicates("eps", keep="last").sort_values("eps")
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(lpN.eps, lpN.alpha, "k-", lw=2, label=f"all SP mechanisms (LP, N={lpc.N.max()})")
ax.plot(sd.eps, sd.alpha_sd, "r--", lw=2, label=f"fixed-menu mechanisms (SD-LP, N={sd100.N.max()})")
ax.plot(det.eps, det.lower, "o-", ms=4, label="eps-aware LIP(a*)"); ax.plot(nes.eps, nes.lower, "s-", ms=4, label="nested LIP(a*)")
ax.fill_between(sd.eps, sd.alpha_sd, np.interp(sd.eps, lpN.eps, lpN.alpha), color="gray", alpha=0.2, label="price of dictatorship")
ax.set(xlabel=r"$\varepsilon$", ylabel="competitive ratio", xlim=(0, 1.2), ylim=(0.8, 1.01)); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "04_price_of_dictatorship.png"); plt.show()
tab = sd[["eps", "alpha_sd"]].copy(); tab["fraction"] = [str(Fraction(v).limit_denominator(2000)) for v in tab.alpha_sd]
tab = tab.merge(lpN[["eps", "alpha"]].rename(columns={"alpha": "SP-LP"}), on="eps", how="left"); tab.round(5)
""")

nb.code(r"""
rows = []
for e in (0.25, 0.5, 0.75):
    r = lp.solve_sd_lp(100, eps=e)
    items = np.unique(np.round(np.c_[r.m1, r.m2, r.m12], 6), axis=0)
    f = FiniteMenuMechanism(items, e); val, pt = f.exact_ratio()
    rows.append({"eps": e, "SD-LP (grid)": r.alpha, "distinct menu items": len(items), "exact ratio of that menu on the continuum": val, "worst profile": pt,
                 "menu": [tuple(np.round(i, 3)) for i in items]})
pd.DataFrame(rows)
""")

nb.md(r"""
For $\varepsilon=\tfrac12$ (and $\tfrac34$) the grid-optimal finite menu is optimal on the
continuum as well, so the SD value is *exact*: $6/7$ and $28/31$. Compare with the SP-LP
($\approx0.95$ and $\approx0.98$): the price of dictatorship is about $0.1$ at
$\varepsilon=\tfrac12$. The reason is structural — with complementarities the efficient rule
pools when agents agree and splits when they disagree, which a rule that mixes two one-sided
choices cannot do (Theorem A's argument). The LP-optimal mechanisms do exactly that (notebook
02-style heatmaps for $\varepsilon=0.5$ below): nearly take-all when both prefer the same item,
a nested split otherwise.
""")

nb.code(r"""
N = 40; e = 0.5
r = lp.solve_sp_lp(N, eps=e, symmetric=True); g = np.arange(N + 1) / N; X, Y = np.meshgrid(g, g, indexing="ij")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for k, (M, name) in enumerate(((r.m1, "$m_1$ = P(agent 1 gets item 1)"), (r.m12, "$m_{12}$ = P(agent 1 gets both)"), (1 - r.m1 - r.m2 + r.m12, "P(agent 2 gets both)"))):
    im = ax[k].pcolormesh(X, Y, M, cmap="viridis", vmin=0, vmax=1, shading="nearest"); plt.colorbar(im, ax=ax[k]); ax[k].set(title=name, xlabel="x", ylabel="y")
plt.suptitle(f"LP-optimal strategy-proof mechanism at eps={e} (value {r.alpha:.4f}, N={N})"); plt.tight_layout(); plt.savefig(FIG / "04_lp_mechanism_eps05.png"); plt.show()
""")

nb.md(r"""
## Summary

* New, simple theorem: no swap-dictatorial mechanism (nor any mixture of two one-sided rules)
  is better than $5/6$-competitive for two items; the LIP mechanism is within $0.004$ of this.
* The best fixed-menu mechanism for any $\varepsilon$ is computable by an LP; for
  $\varepsilon=\tfrac12$ it is a 3-item menu with exact ratio $6/7$.
* Under complementarities the right design is *not* dictatorial; the LP-optimal mechanism
  conditions pooling on agreement between the reports. Designing a clean closed-form
  mechanism of that type is the natural next step (open).
""")

nb.save(execute=True)

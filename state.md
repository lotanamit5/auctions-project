# Project state (short)

Last updated: 2026-10-12 (end of session 1)

**Goal.** Course project on Guo & Conitzer (2010), *Strategy-proof allocation of multiple items
between two agents without payments or priors*. Part A (reading report) is `proposal/main.tex`.
Part B replicates the paper and extends the model with a complementarity term `eps z1 z2`.
Plan/todo: `docs/plan.md`. Decisions, mistakes, dead ends: `docs/log.md`. Proofs: `docs/theory/`.
Start reading at `notebooks/00_overview.ipynb`; all headline numbers with their sources are in `docs/results_summary.md`.

## Results

Additive model (eps = 0)
- Replication: LP N=50 → 0.841225 (paper 0.841); LIP(2) ∈ [0.8281462, 0.8281472]; Figure 1.
- The LP bound decreases with N (0.8368 at N=150, →≈0.8346): 0.841 is an N=50 artefact.
- a = 2 is not optimal: a* = 2.1013, alpha* = 0.82939; alpha(a) = min(theta(a), beta(a)),
  beta in closed form (theory/01).
- **Theorem A**: any fixed-menu (swap-dictatorial, incl. all increasing-price) mechanism — in
  fact any mixture of two one-sided rules — is at most 5/6-competitive (four-profile proof).
- **Theorem C**: 5/6 is attained by an explicit menu (steepest-chain construction, closed
  form; discretised menus reach 0.83328). So: LIP 0.8294 < menus = 0.8333 < all SP ≲ 0.835.

Complementarity model (eps > 0), theory/02–04
- First best = 1 + max(eps, |x−y|); take-all is first best for eps ≥ 1.
- eps-aware LIP (fixed menu) is strategy-proof; equals the chatbot draft's "strategic LIP".
  Certified brackets (P3–P4). Nested (comonotone) lotteries: free improvement, ratio 1 at eps ≥ 1.
- Upper bound on all SP mechanisms: McCormick-moment LP (P7): 0.84 (eps=0) → 0.95 (0.5) → 1.
- **Theorem B**: menus ≤ min{(5+2eps)/6, (1+eps)/(1+2eps−eps²)}, exact for eps ≥ 0.3,
  attained for eps ≥ ½ by the "modest dictator" (share s* = (1−eps²)/(1+2eps−eps²)).
  Nested LIP(a*) essentially attains it for eps ≥ 0.5.
- **Propositions D/E**: "split if strongly opposed (x ≥ (1+eps)/2, y ≤ (1−eps)/2), else
  modest dictator" is strategy-proof with ratio 2(1+eps)²/(3+3eps+2eps²) (eps ≤ ½) and
  (1+eps)²/(1+3eps) (eps ≥ ½): beats every menu for eps > 0.366 (0.900 vs 6/7 at eps = ½).
  LP-optimised fallback menus (`lp.solve_split_menu_lp`) add ~0.02–0.05 for eps in [0.25, 0.45].
- Naive additive rule under eps > 0: not SP, gains ≤ eps Q(½)²/2, strategic play = eps-aware LIP.
- Substitutes (eps < 0): sketch only (notebook 06).

## Open
1. Closed-form SP mechanism approaching the LP bound for 0 < eps < 1 (gap 0.90 vs 0.95 at ½).
2. Analytic proof that LIP's worst case lies on the edge or anti-diagonal.
3. Limit of the Theorem-1 LP as N → ∞ (≈ 0.8346) in closed form.

## Code map
`core/lip.py` (LIP, brackets), `core/lp.py` (SP-LP, SD-LP, split-menu LP), `core/eps.py`
(eps model, nested), `core/menu.py` (general menus, exact evaluation, Theorem C menu),
`core/split.py` (SMD), `core/bracket.py`. Tests: `.venv/bin/pytest` (25+ tests).
Notebooks are built by `experiments/nb_*.py`; scans by the other `experiments/*.py`.

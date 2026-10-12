# Project state (short)

Last updated: 2026-10-12 (session 1, mid-way)

**Goal.** Course project on Guo & Conitzer (2010). Part B extends the model with a
complementarity term `eps z1 z2` (common knowledge eps >= 0). See `docs/plan.md` for the todo
list and `docs/log.md` for decisions, mistakes and dead ends.

**Results so far** (all reproducible from `notebooks/` and `tests/`):

Additive model (replication + sharpening)
- LP of Theorem 1 reproduced (0.841225 at N=50) but the bound *decreases* with N:
  0.8368 at N=150, extrapolating to ~0.8346. The paper's 0.841 is an N=50 artefact.
- LIP(2) certified in [0.8281462, 0.8281472] (adaptive refinement, 20 ms).
- a = 2 is not optimal: a* = 2.1013, alpha* = 0.82939; alpha(a) = min(theta(a), beta(a)) with
  theta = the paper's intrinsic value and beta a closed-form edge branch (docs/theory/01).
- NEW: the best *swap-dictatorial* (fixed-menu) mechanism is at most 5/6 = 0.8333, certified
  by an LP with only three types {0, 1/2, 1}. So LIP(a*) is within 0.004 of the best possible
  menu mechanism; the remaining gap to all SP mechanisms is <= 0.0075.

Complementarity model
- First best = 1 + max(eps, |x-y|); take-all is optimal (ratio 1) for eps >= 1.
- eps-aware LIP(a) (dictator picks her favourite menu point) is strategy-proof; its ratio
  equals the chatbot draft's "strategic LIP" (same reachable set). Certified brackets.
- Nested (comonotone) lotteries dominate deterministic bundles; nested LIP reaches ratio 1
  for eps >= 1.
- LP upper bound over all SP mechanisms via McCormick moments: 0.843 (eps=0) -> ~0.95 (0.5) -> 1.
- Best swap-dictatorial mechanism is an LP too: 20/23 (eps=.25), 6/7 (.5), 28/31 (.75), with
  tiny menus (3 items at eps=.5, exact on the continuum). Dictatorship costs ~0.1 vs SP bound.

**Running / next.** Background scans: LP convergence N=200, LP eps curve N=80, eps LIP scans
(det + nested), SD curve; finite-menu optimiser for eps=0. Notebooks 01-02 done; 03 (eps
model), 04 (menus & price of dictatorship), 05 (naive rule), 00 (overview) to write.

**How to run.** `.venv/bin/pytest`; `.venv/bin/python experiments/nb_01_replication.py` etc.

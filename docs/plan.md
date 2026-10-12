# Research plan

Project: *Strategy-proof allocation of two items between two agents without payments or
priors* (Guo & Conitzer, AAMAS 2010), extended to **complementarities** (Part B of the
course project, see `proposal/main.tex`).

Model for Part B. Two agents, two divisible goods. Agent with type `(v, 1-v)` gets utility
`u(z1, z2) = v z1 + (1-v) z2 + eps z1 z2` from bundle `(z1, z2)`; `v` is private, `eps >= 0`
is common knowledge.

## Guiding questions

1. **Replication.** Do the paper's numbers hold up? (0.841 LP upper bound for N=50; LIP(2)
   sandwiched in [0.828, 0.829]; Figure 1 for m >= 3 items.) Is a = 2 actually the best LIP
   parameter? How does the LP bound move with the grid size N?
2. **Structure of the upper bound.** The paper gives no dual certificate ("it does not seem
   to shed much light"). Which type profiles are binding? Is there a *small* restricted type
   space that already gives a bound close to 0.841 — i.e. a human-checkable proof of an upper
   bound?
3. **Complementarities — mechanism.** With a fixed menu (the LIP budget set) the dictator
   simply picks her favourite bundle; that is strategy-proof for *any* preferences. So
   "eps-aware LIP(a)" is strategy-proof. Characterise the dictator's choice (unique, monotone in
   v?), extend the grid sandwich of Theorem 2, compute alpha(a, eps), and optimise a per eps.
4. **Complementarities — upper bound.** Lotteries matter now (utility is not linear in the
   expected bundle). Key trick: for bilinear utility a lottery is summarised by the moments
   (E z1, E z2, E z1 z2), whose feasible set is exactly the McCormick hull. This gives a finite
   LP that bounds *every* strategy-proof (randomised) mechanism for the eps model.
5. **Simple competitors.** "Give everything to a random agent" has ratio 1 once eps >= 1 and
   (1+eps)/2 below; equal split has (1 + eps/2)/(1 + max(eps,|x-y|)). Mixtures of
   strategy-proof mechanisms are strategy-proof: optimise the mixture.
6. **The naive rule.** Keep the additive report->bundle rule when eps > 0: it is no longer
   strategy-proof. Quantify manipulation gains and show that strategic play against the naive
   rule is outcome-equivalent to the eps-aware mechanism (same menu).
7. **Stretch.** Substitutes (eps < 0), three-or-more items with eps, a closed form or
   analytic upper bound in eps, dual certificates.

## Work packages / todo

- [x] WP0  Repo skeleton, venv, plan/log/state files.                       (this commit)
- [x] WP1  `core/lip.py`: LIP primitives (prices, Q, menu, additive dictator rule,
          p1/p2, welfare, ratio, grid sandwich). Tests against paper numbers.
- [x] WP2  `core/lp.py`: Theorem-1 LP (full and reduced local-IC form), N-convergence,
          duals. Tests: N=50 gives 0.841.
- [x] WP3  Notebook 01 — replication (Theorems 1, 2, Figure 1; best a).
- [x] WP4  LP structure: binding profiles, small-support certificate, notebook 02.
- [x] WP5  `core/eps.py`: eps model (first best, eps-aware dictator problem, monotonicity,
          sandwich bounds, competitors, mixtures). Notebook 03.
- [x] WP6  McCormick LP upper bound for the eps model. Notebook 04. Gap analysis vs LIP.
- [x] WP7  Naive-rule manipulation analysis. Notebook 05.
- [ ] WP8  Theory write-up `docs/theory/*.md` (claims, proofs, what is proved vs numerical).
- [ ] WP9  Stretch items as time allows.
- [ ] WP10 Final pass: README, state.md, consistency check of all numbers.

## Conventions

- Every numerical claim in the write-up must be reproducible from a notebook or test.
- Rigorous bounds (sandwich / LP) are kept separate from "sampled minima".
- Decisions and dead ends go to `docs/log.md`; the current snapshot lives in `state.md`.

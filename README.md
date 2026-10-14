# Strategy-proof allocation of two items without money — replication and complementarities

Course project (Auction Theory) on

> M. Guo and V. Conitzer, *Strategy-proof Allocation of Multiple Items between Two Agents
> without Payments or Priors*, AAMAS 2010. (`reference/paper.pdf`)

Part A (reading report) lives in `proposal/`. Part B — this repository — replicates the paper's
numbers (to higher precision) and extends the model to utilities `v z1 + (1-v) z2 + eps z1 z2` with a
common complementarity parameter `eps`.

**Start with `notebooks/00_overview.ipynb`**, then 01–06. A one-page status is in `state.md`.

Headline results (details and proofs in `docs/theory/`):

* the paper's 0.841 upper bound is a grid artefact — the same LP gives ≤ 0.8363 at N = 200;
  LIP(2) is certified in [0.8281462, 0.8281472]; the best LIP parameter is a* ≈ 2.1013 (0.82939);
* **every fixed-menu (swap-dictatorial / increasing-price) mechanism is at most 5/6-competitive,
  and 5/6 is attained** by an explicit menu (Theorems A, C);
* with complementarities: eps-aware and *nested* LIP mechanisms, an LP upper bound on all
  strategy-proof mechanisms, closed-form bounds for menus (Theorem B), and a simple
  non-dictatorial mechanism — "split if strongly opposed, else modest dictator" — that beats every
  menu for eps > 0.366 (Propositions D, E).

```
core/          LIP primitives, LPs (SP / swap-dictatorial / split-or-menu), eps model, menus
tests/         pytest suite (paper numbers and theorems are regression targets)
experiments/   scripts that build + execute the notebooks and run the scans into results/
notebooks/     00 overview · 01 replication · 02 LP structure · 03 eps model ·
               04 price of dictatorship · 05 naive rule · 06 substitutes (sketch)
docs/          plan.md (todo), log.md (decisions and dead ends), theory/ (write-ups)
results/       figures and csv produced by experiments/
```

Setup: `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`,
then `.venv/bin/pytest` and e.g. `.venv/bin/python experiments/nb_01_replication.py`.

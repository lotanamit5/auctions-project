# LIP mechanisms with complementarities

Course project (Auction Theory) on

> M. Guo and V. Conitzer, *Strategy-proof Allocation of Multiple Items between Two Agents
> without Payments or Priors*, AAMAS 2010. (`reference/paper.pdf`)

Part A (reading report) lives in `proposal/`. Part B — this repository — replicates the paper's
numerical results and extends the model to two agents whose utility has a complementarity term
`eps * z1 * z2` between the two goods.

```
core/          Python package: LIP primitives, LP bounds, eps model
tests/         pytest suite (paper numbers are regression targets)
experiments/   scripts that build + execute the notebooks and write results/
notebooks/     readable walk-throughs of every result (start with 00_overview)
docs/          plan.md (todo), log.md (decisions and dead ends), theory/ (write-ups)
results/       figures and data produced by experiments/
state.md       one-page snapshot of where the project stands
```

Setup: `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`,
then `.venv/bin/pytest` and `.venv/bin/python experiments/build_notebooks.py`.

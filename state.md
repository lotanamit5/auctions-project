# Project state (short)

Last updated: 2026-10-12 (session 1, start)

**Goal.** Course project on Guo & Conitzer (2010), "Strategy-proof allocation of multiple
items between two agents without payments or priors". Part A (proposal, done by the team) is in
`proposal/main.tex`. Part B extends the model with a complementarity term `eps z1 z2`.

**Where things are.**
- `docs/plan.md` — research plan and todo list. `docs/log.md` — decisions / attempts.
- `core/` — Python package (LIP primitives, LPs, eps model). `tests/` — pytest.
- `experiments/` — scripts that build and execute the notebooks in `notebooks/`.
- `results/` — figures and csv produced by the experiments.

**Status.** Skeleton only; no results yet.

**How to run.** `source .venv/bin/activate` (or use `.venv/bin/python`), `pytest`,
`python experiments/build_notebooks.py`.

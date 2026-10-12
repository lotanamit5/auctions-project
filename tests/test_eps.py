import numpy as np

from core import eps as E, lip


def test_reduces_to_claim5_at_eps0():
    g = np.linspace(0, 1, 101)
    t_eps = E.spend_eps(g, 2.0, 0.0)
    t_add = lip.spend_additive(g, 2.0)
    assert np.max(np.abs(t_eps - t_add)) < 1e-9


def test_dictator_optimality_and_monotonicity():
    a, eps = 2.0, 0.4
    g = np.linspace(0, 1, 201)
    t = E.spend_eps(g, a, eps)
    assert np.all(np.diff(t) >= -1e-9)                       # t*(v) non-decreasing
    ts = np.linspace(0, 1, 2001)
    for v in (0.0, 0.2, 0.5, 0.9):
        F = E.utility(v, lip.Q(ts, a), lip.Q(1 - ts, a), eps)
        tv = float(E.spend_eps(v, a, eps))
        Fv = float(E.utility(v, lip.Q(tv, a), lip.Q(1 - tv, a), eps))
        assert Fv >= F.max() - 1e-9                          # chosen point is the max


def test_first_best_formula():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y, eps = rng.random(), rng.random(), rng.random() * 2
        z = np.linspace(0, 1, 41)
        Z1, Z2 = np.meshgrid(z, z, indexing="ij")
        W = E.utility(x, Z1, Z2, eps) + E.utility(y, 1 - Z1, 1 - Z2, eps)
        assert abs(W.max() - E.first_best(x, y, eps)) < 1e-9


def test_bracket_matches_draft_table():
    # 'strategic LIP' column of the chatbot draft in proposal/main.tex
    for e, val in ((0.5, 0.81002), (1.0, 0.76515), (2.0, 0.71528)):
        lo, up, _, _ = E.bracket_lip(2.0, e, tol=1e-4)
        assert lo <= val + 1e-5 and up >= val - 1e-5 and up - lo <= 1e-4


def test_takeall_is_first_best_for_large_eps():
    assert np.all(E.welfare_takeall(0.3, 0.9, 1.5) == E.first_best(0.3, 0.9, 1.5))

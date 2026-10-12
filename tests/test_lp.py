import numpy as np
import pytest

from core import lp


def test_theorem1_value_N50():
    r = lp.solve_sp_lp(50)
    assert abs(r.alpha - 0.841) < 5e-4            # the paper reports 0.841
    assert abs(np.sum(r.dual_welfare * r.opt) - 1) < 1e-6


def test_local_ic_equals_full_ic():
    for eps in (0.0, 0.3):
        a_full = lp.solve_sp_lp(8, eps=eps, full_ic=True).alpha
        a_loc = lp.solve_sp_lp(8, eps=eps).alpha
        assert abs(a_full - a_loc) < 1e-7


def test_symmetry_does_not_change_value():
    assert abs(lp.solve_sp_lp(12, symmetric=True).alpha - lp.solve_sp_lp(12).alpha) < 1e-7
    assert abs(lp.solve_sp_lp(8, eps=0.4, symmetric=True).alpha - lp.solve_sp_lp(8, eps=0.4).alpha) < 1e-7


def test_solution_is_strategy_proof():
    """Re-check every pairwise deviation on the returned mechanism (eps model included)."""
    N, eps = 10, 0.25
    r = lp.solve_sp_lp(N, eps=eps)
    g = np.arange(N + 1) / N
    U1 = g[:, None, None] * r.m1[None] + (1 - g[:, None, None]) * r.m2[None] + eps * r.m12[None]
    # U1[j_true, j_report, k]; truthful = diagonal in the first two axes
    truthful = np.einsum("jjk->jk", U1)
    assert np.all(truthful[:, None, :] >= U1 - 1e-7)
    U2 = (g[None, None, :] * (1 - r.m1[:, :, None]) + (1 - g[None, None, :]) * (1 - r.m2[:, :, None])
          + eps * (1 - r.m1 - r.m2 + r.m12)[:, :, None])
    # U2[j, k_report, k_true]; truthful = k_report == k_true
    truthful2 = np.einsum("jkk->jk", U2)
    assert np.all(truthful2[:, None, :] >= U2 - 1e-7)
    assert np.all(r.m12 <= r.m1 + 1e-9) and np.all(r.m12 <= r.m2 + 1e-9)
    assert np.all(r.m12 >= r.m1 + r.m2 - 1 - 1e-9)


def test_eps_large_gives_ratio_one():
    # for eps >= 1 'everything to a random agent' is first best, so the LP value is 1
    assert abs(lp.solve_sp_lp(10, eps=1.0).alpha - 1.0) < 1e-7
    assert lp.solve_sp_lp(10, eps=0.5).alpha < 1.0

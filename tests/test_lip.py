"""Regression tests against the numbers printed in Guo & Conitzer (2010)."""
import math

import numpy as np
import pytest

from core import lip


def test_normalisation():
    # int_0^1 dt/(a t + b) = 1  <=>  Q(1) = 1
    for a in (0.1, 1.0, 2.0, 5.0):
        assert abs(lip.Q(1.0, a) - 1.0) < 1e-12
        assert abs(lip.cost(lip.Q(0.37, a), a) - 0.37) < 1e-12


def test_example_2():
    # type (0.5, 0.5) under LIP(2) buys 0.717 of both items
    q1, q2 = lip.bundle_additive(0.5, 2.0)
    assert abs(q1 - 0.717) < 5e-4 and abs(q2 - 0.717) < 5e-4
    # type (1, 0) takes item 1 entirely
    q1, q2 = lip.bundle_additive(1.0, 2.0)
    assert q1 == 1.0 and q2 == 0.0


def test_claim5_thresholds_and_lemma2():
    a = 2.0
    lo, hi = lip.regime_thresholds(a)
    assert abs(lo - 0.119) < 5e-4 and abs(hi - 0.881) < 5e-4
    # Lemma 2: for interior types, final prices are proportional to the type
    b = lip.b_of(a)
    x = 0.6
    t = float(lip.spend_additive(x, a))
    assert abs((a * t + b) / (a + 2 * b) - x) < 1e-12


def test_monotonicity_of_shares():
    g = np.linspace(0, 1, 201)
    q1, q2 = lip.bundle_additive(g, 2.0)
    assert np.all(np.diff(q1) >= -1e-12) and np.all(np.diff(q2) <= 1e-12)


def test_theorem2_bracket():
    lo, up, _ = lip.sandwich_adaptive(2.0, tol=1e-5)
    assert 0.828 <= lo <= up <= 0.829            # the paper's statement
    assert up - lo <= 1e-5
    lo_u, up_u, _, _ = lip.sandwich(2.0, 400)
    assert lo_u <= lo and up_u >= up - 1e-12      # the uniform grid is looser


def test_theorem3_bounds_monotone_in_m():
    lo3, up3 = lip.theorem3_bounds(2.0, 3)
    lo100, up100 = lip.theorem3_bounds(2.0, 100)
    assert up3 == pytest.approx(up100)            # theta(a) does not depend on m
    assert lo100 < lo3 < up3 < 1
    # theta bounds differ by at most h/2
    lo, up = lip.theta_bounds(2.0, 1000)
    assert 0 <= up - lo <= 2.0 / (1000 * (2.0 + lip.b_of(2.0))) / 2 + 1e-12

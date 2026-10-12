import numpy as np
import pytest

from core import lp
from core.menu import FiniteMenuMechanism, LIPMenu, MenuMechanism


def test_sd_lp_additive_is_five_sixths():
    for T in ([0, .5, 1], np.linspace(0, 1, 11), np.linspace(0, 1, 41)):
        assert abs(lp.solve_sd_lp(0, types=T).alpha - 5 / 6) < 1e-7


def test_sd_lp_formula_B():
    for e in (0.3, 0.5, 0.75):
        rho = (1 + e) / (1 + 2 * e - e * e)
        assert abs(lp.solve_sd_lp(0, eps=e, types=np.linspace(0, 1, 21)).alpha - rho) < 1e-7


def test_sd_lp_formula_A_small_eps_three_types():
    for e in (0.02, 0.05, 0.1):
        assert abs(lp.solve_sd_lp(0, eps=e, types=[0, .5, 1]).alpha - (5 + 2 * e) / 6) < 1e-7


def test_modest_dictator_menu_attains_rho_B_on_continuum():
    for e in (0.5, 0.6, 0.75, 0.9):
        s = (1 - e * e) / (1 + 2 * e - e * e)
        f = FiniteMenuMechanism([(s, 0, 0), (0, s, 0)], e)
        val, _ = f.exact_ratio()
        assert abs(val - (1 + e) / (1 + 2 * e - e * e)) < 1e-9


def test_finite_menu_exact_matches_grid():
    rng = np.random.default_rng(1)
    items = [(a, b, min(a, b)) for a, b in rng.random((6, 2))]
    f = FiniteMenuMechanism(items, 0.3)
    val, _ = f.exact_ratio()
    g = np.linspace(0, 1, 401); X, Y = np.meshgrid(g, g, indexing="ij"); m = X >= Y
    grid = f.ratio(X[m], Y[m]).min()
    assert val <= grid + 1e-9 and grid - val < 5e-3


def test_menu_mechanism_reproduces_lip():
    m = MenuMechanism(LIPMenu(2.0), 0.0, nested=False, grid=1001)
    val, x, y = m.grid_min(200)
    assert abs(val - 0.8281) < 2e-3


def test_optimal_sd_menu_approaches_five_sixths():
    from core.menu import optimal_sd_menu, optimal_sd_choice, steepest_chain_D
    assert abs(steepest_chain_D(1.0) - 2.0) < 1e-12
    x = np.linspace(0, 1, 2001)
    m1, m2 = optimal_sd_choice(x)
    d = m1 - m2
    assert np.all(np.diff(d) >= -1e-9) and abs(d[0] + 1) < 1e-6 and abs(d[-1] - 1) < 1e-6
    # increment condition on a subsample
    sub = x[::50]; dd = d[::50]
    S = sub[:, None] - sub[None, :]; mask = S > 0.2
    assert np.all((dd[:, None] - dd[None, :])[mask] >= (5 / 3 - 1 / (3 * S[mask])) - 1e-6)
    vals = [FiniteMenuMechanism(optimal_sd_menu(n), 0.0).exact_ratio()[0] for n in (501, 2001)]
    assert vals[0] < vals[1] < 5 / 6 + 1e-9 and 5 / 6 - vals[1] < 3e-4

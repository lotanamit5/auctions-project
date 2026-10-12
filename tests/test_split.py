import numpy as np

from core import split


def test_closed_form_matches_grid_and_is_ic():
    g = np.linspace(0, 1, 401); X, Y = np.meshgrid(g, g, indexing="ij"); m = X >= Y
    for e in (0.1, 0.3, 0.5, 0.7, 0.9):
        cf = float(split.ratio_closed_form(e))
        grid = split.ratio(X[m], Y[m], e).min()
        assert abs(cf - grid) < 2e-3 and grid >= cf - 1e-9
        assert split.ic_violation(e, n=81) < 1e-12


def test_smd_beats_menus_for_large_eps():
    for e in (0.4, 0.5, 0.75, 0.9):
        menus = min((5 + 2 * e) / 6, (1 + e) / (1 + 2 * e - e * e))
        assert float(split.ratio_closed_form(e)) > menus


def test_values():
    assert abs(float(split.ratio_closed_form(0.5)) - 0.9) < 1e-12
    assert abs(float(split.ratio_closed_form(0.5, s=0.0)) - 2 * 1.5 / 3.5) < 1e-12

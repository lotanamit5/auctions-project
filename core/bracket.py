"""Generic adaptive cell refinement for  min_{1 >= x >= y >= 0} ratio(x, y).

Used by both the additive model (``core.lip``) and the complementarity model (``core.eps``).
The caller supplies

    cell_lower(x0, y0, h) -> array    a *rigorous* lower bound of the ratio on the closed cell
                                       [x0, x0+h] x [y0, y0+h] (vectorised over cells),
    point_ratio(x, y)     -> array    the exact ratio at points of the triangle.

A cell is split 2 x 2 while its lower bound is below (best exact value seen) - tol.  Every point
of the triangle is covered by some leaf cell at all times, so the returned lower bound is a
valid bound over the whole triangle; the upper bound is attained at a concrete point.
"""
from __future__ import annotations

import numpy as np

__all__ = ["adaptive_bracket"]


def adaptive_bracket(cell_lower, point_ratio, tol=1e-4, N0=64, max_depth=20, max_cells=2_000_000):
    """Returns (lower, upper, (x_upper, y_upper), n_leaf_cells_at_last_level)."""
    h = 1.0 / N0
    I, J = np.meshgrid(np.arange(N0), np.arange(N0), indexing="ij")
    keep = I >= J
    x0, y0 = I[keep] * h, J[keep] * h
    pruned_min, upper, arg = np.inf, np.inf, (np.nan, np.nan)
    for _ in range(max_depth + 1):
        L = cell_lower(x0, y0, h)
        # exact ratios at the two cell corners that always lie in the triangle (x >= y)
        for xx, yy in ((x0, y0), (x0 + h, y0)):
            R = point_ratio(xx, yy)
            i = int(np.argmin(R))
            if R[i] < upper:
                upper, arg = float(R[i]), (float(xx[i]), float(yy[i]))
        need = L < upper - tol
        if not need.any() or 4 * need.sum() > max_cells:
            return float(min(pruned_min, L.min())), upper, arg, int(L.size)
        if (~need).any():
            pruned_min = min(pruned_min, float(L[~need].min()))
        h /= 2
        bx, by = x0[need], y0[need]
        xs = np.concatenate([bx, bx + h, bx, bx + h])
        ys = np.concatenate([by, by, by + h, by + h])
        ok = xs + h > ys + 1e-15                 # sub-cell touches the triangle x >= y
        x0, y0 = xs[ok], ys[ok]
    return float(min(pruned_min, L.min())), upper, arg, int(L.size)

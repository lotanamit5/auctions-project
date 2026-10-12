"""Linear increasing-price (LIP) mechanisms of Guo & Conitzer (2010), two agents.

Notation (paper, Section 4-5).  Price schedule f(t) = a t + b with the normalisation
int_0^1 dt / f(t) = 1, i.e. b = a / (e^a - 1).  Spending t units of artificial currency on an
item buys the proportion

    Q(t) = int_0^t ds / (a s + b) = ln(1 + a t / b) / a,           Q(0) = 0, Q(1) = 1.

Two items: agent 1 has type (x, 1-x), agent 2 has (y, 1-y).  With probability 1/2 agent i is
the dictator, spends her unit of currency and the other agent gets the remainder.

Everything here is for additive valuations (eps = 0); the complementarity model lives in
``core/eps.py``.  Functions are vectorised over numpy arrays.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "b_of",
    "Q",
    "cost",
    "spend_additive",
    "bundle_additive",
    "regime_thresholds",
    "expected_shares",
    "welfare",
    "first_best",
    "ratio",
    "sandwich",
    "theta_bounds",
    "theorem3_bounds",
]


def b_of(a: float) -> float:
    """Intercept b fixed by the normalisation int_0^1 dt/(a t + b) = 1."""
    return a / math.expm1(a)


def Q(t, a: float):
    """Proportion of an item obtained by spending ``t`` units of currency on it."""
    t = np.asarray(t, dtype=float)
    return np.log1p(a * t / b_of(a)) / a


def cost(q, a: float):
    """Inverse of Q: currency needed to buy proportion ``q`` of one item."""
    q = np.asarray(q, dtype=float)
    return np.expm1(a * q) / np.expm1(a)


def regime_thresholds(a: float) -> tuple[float, float]:
    """Claim 5: a dictator with value x for item 1 is all-in on item 2 iff x <= lo, all-in on
    item 1 iff x >= hi, and buys both items strictly in between."""
    b = b_of(a)
    return b / (a + 2 * b), (a + b) / (a + 2 * b)


def spend_additive(x, a: float):
    """Claim 5: currency a dictator of type (x, 1-x) spends on item 1 (additive utility).

    Interior case t = (x (a + 2b) - b) / a, clipped to [0, 1] which reproduces the two
    all-in regimes."""
    x = np.asarray(x, dtype=float)
    b = b_of(a)
    return np.clip((x * (a + 2 * b) - b) / a, 0.0, 1.0)


def bundle_additive(x, a: float):
    """(q1, q2): proportions of items 1 and 2 bought by a dictator of type x."""
    t = spend_additive(x, a)
    return Q(t, a), Q(1 - t, a)


def expected_shares(x, y, a: float, bundle=bundle_additive):
    """p_j(x, y): agent 1's expected share of item j (over the coin flip).

    ``bundle`` is the dictator rule (type -> (q1, q2)); defaults to the additive Claim-5 rule
    so the same function serves the eps model."""
    q1x, q2x = bundle(x, a)
    q1y, q2y = bundle(y, a)
    return 0.5 * q1x + 0.5 * (1 - q1y), 0.5 * q2x + 0.5 * (1 - q2y)


def welfare(x, y, a: float, bundle=bundle_additive):
    """S(x, y): expected social welfare under LIP(a) with additive valuations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p1, p2 = expected_shares(x, y, a, bundle)
    return x * p1 + (1 - x) * p2 + y * (1 - p1) + (1 - y) * (1 - p2)


def first_best(x, y):
    """First-best welfare with additive valuations: each item to whoever values it more."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 1 + np.abs(x - y)


def ratio(x, y, a: float, bundle=bundle_additive):
    """Achieved / first-best welfare at the profile (x, y)."""
    return welfare(x, y, a, bundle) / first_best(x, y)


def sandwich(a: float, N: int, bundle=bundle_additive) -> tuple[float, float, tuple, tuple]:
    """Theorem 2 bracket of alpha(a) = min_{1>=x>=y>=0} S(x,y)/(x+1-y) on a grid of step 1/N.

    Returns (lower, upper, argmin_cell_lower, argmin_point_upper).

    Lower bound: on each cell [x_i, x_i+h] x [y_j, y_j+h] with i >= j, each of the four
    non-negative terms of S is bounded below at its worst corner, using that p1 is increasing
    in x and decreasing in y while p2 is the reverse (true for any dictator rule whose q1 is
    increasing in the type and q2 decreasing).  The denominator is replaced by its cell
    maximum.  Upper bound: exact ratio at the grid points (feasible points of the triangle).
    """
    h = 1.0 / N
    g = np.arange(N + 1) * h
    q1, q2 = bundle(g, a)                      # dictator purchases at the grid types
    # p1[i, j] = agent 1's expected share of item 1 at (x_i, y_j)
    p1 = 0.5 * q1[:, None] + 0.5 * (1 - q1[None, :])
    p2 = 0.5 * q2[:, None] + 0.5 * (1 - q2[None, :])

    # --- upper bound: grid points with i >= j
    X, Y = g[:, None], g[None, :]
    S = X * p1 + (1 - X) * p2 + Y * (1 - p1) + (1 - Y) * (1 - p2)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = S / (X + 1 - Y)                    # (0, 1) is outside the triangle: masked below
    mask = np.tril(np.ones((N + 1, N + 1), dtype=bool))      # i >= j
    R_up = np.where(mask, R, np.inf)
    iu, ju = np.unravel_index(np.argmin(R_up), R.shape)
    upper = float(R_up[iu, ju])

    # --- lower bound: cells 0 <= j <= i < N
    xi, yj = g[:-1, None], g[None, :-1]
    num = (
        xi * p1[:-1, 1:]                       # x p1          at (x_i,   y_j+h)
        + (1 - xi - h) * p2[1:, :-1]           # (1-x) p2      at (x_i+h, y_j)
        + yj * (1 - p1[1:, :-1])               # y (1-p1)      at (x_i+h, y_j)
        + (1 - yj - h) * (1 - p2[:-1, 1:])     # (1-y) (1-p2)  at (x_i,   y_j+h)
    )
    den = xi + h + 1 - yj
    L = num / den
    maskc = np.tril(np.ones((N, N), dtype=bool))
    L = np.where(maskc, L, np.inf)
    il, jl = np.unravel_index(np.argmin(L), L.shape)
    lower = float(L[il, jl])
    return lower, upper, (g[il], g[jl]), (g[iu], g[ju])


def sandwich_adaptive(a: float, tol: float = 1e-4, N0: int = 64, max_depth: int = 20,
                      bundle=bundle_additive) -> tuple[float, float, int]:
    """Same bracket as :func:`sandwich` but with adaptive cell refinement.

    A cell is subdivided (2 x 2) only while its lower bound is below ``upper - tol``, where
    ``upper`` is the best exact ratio seen so far at any visited corner.  Returns
    (lower, upper, number_of_leaf_cells); on success ``upper - lower <= tol``.
    The lower bound is a rigorous bound over the whole triangle, exactly as in Theorem 2:
    every point of the triangle lies in some leaf cell."""
    h = 1.0 / N0
    I, J = np.meshgrid(np.arange(N0), np.arange(N0), indexing="ij")
    keep = I >= J
    x0, y0 = I[keep] * h, J[keep] * h
    pruned_min = np.inf
    upper = np.inf
    for _ in range(max_depth + 1):
        qx0 = bundle(x0, a); qxh = bundle(x0 + h, a)
        qy0 = bundle(y0, a); qyh = bundle(y0 + h, a)

        def p(qx, qy):      # (p1, p2) at a point given both dictators' purchases
            return 0.5 * qx[0] + 0.5 * (1 - qy[0]), 0.5 * qx[1] + 0.5 * (1 - qy[1])

        p1_a, p2_a = p(qx0, qyh)        # (x0,   y0+h)
        p1_b, p2_b = p(qxh, qy0)        # (x0+h, y0)
        num = (x0 * p1_a + (1 - x0 - h) * p2_b + y0 * (1 - p1_b) + (1 - y0 - h) * (1 - p2_a))
        L = num / (x0 + h + 1 - y0)
        # exact ratios at the two corners that lie in the triangle for every kept cell
        for (xx, yy, (pp1, pp2)) in ((x0, y0, p(qx0, qy0)), (x0 + h, y0, (p1_b, p2_b))):
            S = xx * pp1 + (1 - xx) * pp2 + yy * (1 - pp1) + (1 - yy) * (1 - pp2)
            upper = min(upper, float(np.min(S / (xx + 1 - yy))))
        need = L < upper - tol
        if not need.any():
            return float(min(pruned_min, L.min())), upper, int(L.size)
        pruned_min = min(pruned_min, float(L[~need].min())) if (~need).any() else pruned_min
        # refine
        h /= 2
        bx, by = x0[need], y0[need]
        xs = np.concatenate([bx, bx + h, bx, bx + h])
        ys = np.concatenate([by, by, by + h, by + h])
        ok = xs + h > ys + 1e-15                 # subcell touches the triangle x >= y
        x0, y0 = xs[ok], ys[ok]
    return float(min(pruned_min, L.min())), upper, int(L.size)


# ---------------------------------------------------------------------------------------
# m >= 3 items: Section 5.2 of the paper (intrinsic value theta(a) and Theorem 3)
# ---------------------------------------------------------------------------------------

def theta_bounds(a: float, N: int) -> tuple[float, float]:
    """Lower and upper bounds on the intrinsic value theta(a) (Claims 7-8).

    theta(a) = min_{b/(a+b) <= u <= 1} (u - 1) ln(u) / (2a) + 1/2 + u/2 .
    Grid step h = a / (N (a + b)); lower bound uses (u_i + h - 1) ln(u_i + h)/(2a) + 1/2 + u_i/2,
    upper bound evaluates the function at the grid points (which exceeds the lower bound by at
    most h/2)."""
    b = b_of(a)
    h = a / (N * (a + b))
    u = b / (a + b) + np.arange(N + 1) * h
    lower = np.min((u[:-1] + h - 1) * np.log(u[:-1] + h) / (2 * a) + 0.5 + u[:-1] / 2)
    upper = np.min((u[1:] - 1) * np.log(u[1:]) / (2 * a) + 0.5 + u[1:] / 2)
    return float(lower), float(upper)


def theorem3_bounds(a: float, m: int, N: int = 10_000) -> tuple[float, float]:
    """Theorem 3: LIP(a) with m items is at least (a+b)/(a+2mb) theta(a)-competitive and at
    most theta(a)-competitive.  Returns (lower, upper)."""
    b = b_of(a)
    lo, up = theta_bounds(a, N)
    return (a + b) / (a + 2 * m * b) * lo, up

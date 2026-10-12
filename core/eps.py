"""Part B: two items with a complementarity term.

Agent with type (v, 1-v) gets  u_v(z1, z2) = v z1 + (1-v) z2 + eps z1 z2  from bundle (z1, z2).
``eps >= 0`` is common knowledge; v is private.

Mechanism "eps-aware LIP(a)": the menu is the LIP budget set
    B(a) = {(z1, z2) : cost(z1) + cost(z2) <= 1},   cost(q) = (e^{aq} - 1)/(e^a - 1),
a fair coin picks the dictator, who receives her favourite bundle in B(a) (computed by the
mechanism from her report and the public eps); the other agent gets the remainder.  Because the
menu is fixed and the dictator gets her favourite menu item, truthful reporting is dominant for
*every* utility function, in particular for every eps.  With eps = 0 this is exactly LIP(a).

The dictator's problem.  Utility is increasing in both goods, so the optimum lies on the
budget frontier t -> (Q(t), Q(1-t)), t in [0, 1] being the currency spent on item 1:

    F_v(t) = v Q(t) + (1-v) Q(1-t) + eps Q(t) Q(1-t).

Q is strictly concave and P(t) = Q(t) Q(1-t) is strictly concave (P'' = Q''(t)Q(1-t)
- 2Q'(t)Q'(1-t) + Q(t)Q''(1-t) < 0), so F_v is strictly concave: unique maximiser t*(v).
Moreover d/dv F_v'(t) = Q'(t) + Q'(1-t) > 0 (strictly increasing differences), so t*(v) is
non-decreasing in v: q1 = Q(t*) is non-decreasing and q2 = Q(1-t*) non-increasing in the type.
"""
from __future__ import annotations

import numpy as np

from core import lip
from core.bracket import adaptive_bracket

__all__ = [
    "utility", "first_best", "Fprime", "spend_eps", "bundle_eps", "welfare_lip",
    "ratio_lip", "bracket_lip", "welfare_takeall", "welfare_equal", "ratio_mixture",
    "bracket_mixture", "limit_ratio_eps_inf",
]


def utility(v, z1, z2, eps):
    v = np.asarray(v, dtype=float)
    return v * z1 + (1 - v) * z2 + eps * z1 * z2


def first_best(x, y, eps):
    """1 + max(eps, |x - y|): welfare is bilinear in agent 1's bundle, hence maximised at a
    corner of [0,1]^2 -- everything to one agent (1 + eps) or the additive split (1 + |x-y|)."""
    return 1 + np.maximum(eps, np.abs(np.asarray(x, float) - np.asarray(y, float)))


def Fprime(t, v, a, eps):
    """Derivative of the dictator's objective along the budget frontier."""
    b = lip.b_of(a)
    Qp = lambda s: 1.0 / (a * s + b)               # Q'(s)
    return (v * Qp(t) - (1 - v) * Qp(1 - t)
            + eps * (Qp(t) * lip.Q(1 - t, a) - lip.Q(t, a) * Qp(1 - t)))


def spend_eps(v, a, eps, iters=80):
    """t*(v): currency the dictator spends on item 1 (vectorised bisection on F' which is
    strictly decreasing).  Reduces to Claim 5 when eps = 0."""
    v = np.asarray(v, dtype=float)
    lo = np.zeros_like(v)
    hi = np.ones_like(v)
    at0 = Fprime(0.0, v, a, eps) <= 0             # corner: all-in on item 2
    at1 = Fprime(1.0, v, a, eps) >= 0             # corner: all-in on item 1
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = Fprime(mid, v, a, eps) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    t = 0.5 * (lo + hi)
    return np.where(at0, 0.0, np.where(at1, 1.0, t))


def bundle_eps(v, a, eps):
    t = spend_eps(v, a, eps)
    return lip.Q(t, a), lip.Q(1 - t, a)


def make_bundle(eps):
    """Dictator rule with eps baked in, in the (type, a) -> (q1, q2) signature of core.lip."""
    return lambda v, a: bundle_eps(v, a, eps)


# ---------------------------------------------------------------------------------------
# welfare of eps-aware LIP(a) and competitors
# ---------------------------------------------------------------------------------------

def welfare_lip(x, y, a, eps):
    """Expected welfare: average over the two dictator branches of realised utilities
    (the eps term is evaluated on realised bundles, not on expected shares)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    q1x, q2x = bundle_eps(x, a, eps)
    q1y, q2y = bundle_eps(y, a, eps)
    br1 = utility(x, q1x, q2x, eps) + utility(y, 1 - q1x, 1 - q2x, eps)      # agent 1 dictator
    br2 = utility(y, q1y, q2y, eps) + utility(x, 1 - q1y, 1 - q2y, eps)      # agent 2 dictator
    return 0.5 * (br1 + br2)


def ratio_lip(x, y, a, eps):
    return welfare_lip(x, y, a, eps) / first_best(x, y, eps)


def _cell_lower_lip(a, eps):
    """Rigorous lower bound of welfare_lip / first_best on a cell [x0,x0+h] x [y0,y0+h].

    Branch 'agent 1 dictator':  u_x(q(x)) >= u_x(q(x0)) (dictator optimality) and the right
    side is affine in x, so it is >= min over the two x-endpoints; u_y(1 - q(x)) uses the
    monotonicity of q(x) (q1 up, q2 down in x) and y, 1-y, eps >= 0.  Symmetric for the other
    branch.  Denominator: 1 + max(eps, x0 + h - y0) is the cell maximum of the first best."""
    def f(x0, y0, h):
        x1, y1 = x0 + h, y0 + h
        q1x0, q2x0 = bundle_eps(x0, a, eps); q1x1, q2x1 = bundle_eps(x1, a, eps)
        q1y0, q2y0 = bundle_eps(y0, a, eps); q1y1, q2y1 = bundle_eps(y1, a, eps)
        # agent-1-dictator branch
        A = np.minimum(utility(x0, q1x0, q2x0, eps), utility(x1, q1x0, q2x0, eps))
        B = y0 * (1 - q1x1) + (1 - y1) * (1 - q2x0) + eps * (1 - q1x1) * (1 - q2x0)
        # agent-2-dictator branch: dictator y's own utility >= u_y(q(y0)), affine in y
        C = np.minimum(utility(y0, q1y0, q2y0, eps), utility(y1, q1y0, q2y0, eps))
        D = x0 * (1 - q1y1) + (1 - x1) * (1 - q2y0) + eps * (1 - q1y1) * (1 - q2y0)
        return 0.5 * (A + B + C + D) / (1 + np.maximum(eps, x1 - y0))
    return f


def bracket_lip(a, eps, tol=1e-4, **kw):
    """Certified bracket of the competitive ratio of eps-aware LIP(a):
    (lower, upper, argmin point of the upper bound, cells)."""
    return adaptive_bracket(_cell_lower_lip(a, eps), lambda x, y: ratio_lip(x, y, a, eps), tol=tol, **kw)


def welfare_takeall(x, y, eps):
    """Everything to a fair-coin dictator: welfare 1 + eps regardless of types."""
    return np.full(np.broadcast(np.asarray(x), np.asarray(y)).shape, 1.0 + eps)


def welfare_equal(x, y, eps):
    """Equal split of both items: each agent gets (1/2, 1/2): welfare 1 + eps/2."""
    return np.full(np.broadcast(np.asarray(x), np.asarray(y)).shape, 1.0 + eps / 2)


def ratio_mixture(x, y, a, eps, lam):
    """Mixture: with prob. lam run eps-aware LIP(a), else take-all.  Strategy-proof because
    each component is and utilities are linear in the lottery."""
    W = lam * welfare_lip(x, y, a, eps) + (1 - lam) * welfare_takeall(x, y, eps)
    return W / first_best(x, y, eps)


def bracket_mixture(a, eps, lam, tol=1e-4, **kw):
    cl = _cell_lower_lip(a, eps)

    def cell_lower(x0, y0, h):
        # lam * (LIP welfare lower bound) + (1-lam)(1+eps), over the same cell denominator
        den = 1 + np.maximum(eps, x0 + h - y0)
        return lam * cl(x0, y0, h) + (1 - lam) * (1 + eps) / den

    return adaptive_bracket(cell_lower, lambda x, y: ratio_mixture(x, y, a, eps, lam), tol=tol, **kw)


def limit_ratio_eps_inf(a):
    """eps -> infinity with a fixed: every dictator buys the balanced bundle (Q(1/2), Q(1/2));
    welfare ~ eps (qs^2 + (1-qs)^2), first best ~ eps."""
    qs = float(lip.Q(0.5, a))
    return qs * qs + (1 - qs) ** 2

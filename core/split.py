"""Beyond dictatorship: the "split if strongly opposed, else modest dictator" mechanism (SMD).

Reports x (agent 1), y (agent 2); tau = (1 + eps)/2.
  * If x >= tau and y <= 1 - tau (agent 1 strongly prefers item 1, agent 2 strongly prefers
    item 2) -- or the mirror image -- the items are SPLIT: each agent gets her preferred item.
  * Otherwise a fair coin picks a dictator who takes a share s of her preferred item (the
    "modest dictator" menu {(s, 0), (0, s)}); the other agent gets the rest.

Strategy-proofness (sketch; brute-force checked in tests).  Against a strongly opposed
opponent, a type x >= 1/2 compares the split (utility x) with the fallback
(utility x s + (1 - s)(1 + eps)/2, affine in x with slope s < 1); the two cross exactly at
x = tau, so strong types prefer the split and moderate types the fallback, whatever s.  Against
any other opponent the outcome is the fixed-menu fallback, which is strategy-proof.

With s = 0 this is "pool unless strongly opposed" (take-all outside the split region).
"""
from __future__ import annotations

import numpy as np

from core.eps import first_best

__all__ = ["tau", "s_star", "ratio_closed_form", "welfare", "ratio", "ic_violation"]


def tau(eps):
    return (1 + eps) / 2


def s_star(eps):
    """Share that equalises the two binding profiles (closed form)."""
    e = np.asarray(eps, float)
    return np.where(e >= 0.5, (1 - e * e) / (1 + 3 * e), e * (1 + e) / (1.5 + 1.5 * e + e * e))


def ratio_closed_form(eps, s=None):
    """Competitive ratio of SMD(s); with s = None the optimal s is used.

    Binding profiles: (i) identical types (1 + eps - eps s)/(1 + max(eps, 1/2)) -- same-side
    gaps up to 1/2 exceed eps when eps < 1/2; (ii) the boundary of the split region x = 1,
    y = (1-eps)/2: [2(1+eps) + (1-eps) s]/(3 + eps)."""
    e = np.asarray(eps, float)
    s = s_star(e) if s is None else np.asarray(s, float)
    same = (1 + e - e * s) / (1 + np.maximum(e, 0.5))
    opp = (2 * (1 + e) + (1 - e) * s) / (3 + e)
    return np.minimum(same, opp)


def welfare(x, y, eps, s=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    s = float(s_star(eps)) if s is None else s
    t = tau(eps)
    split = ((x >= t) & (y <= 1 - t)) | ((y >= t) & (x <= 1 - t))
    # fallback: dictator of type v takes (s, 0) if v >= 1/2 else (0, s); welfare of one branch
    # with dictator type d and other type o: (1+eps) + (d - o - eps) m1 + (o - d - eps) m2
    def branch(d, o):
        m1 = np.where(d >= 0.5, s, 0.0); m2 = np.where(d >= 0.5, 0.0, s)
        return (1 + eps) + (d - o - eps) * m1 + (o - d - eps) * m2
    W_fb = 0.5 * (branch(x, y) + branch(y, x))
    return np.where(split, 1 + np.abs(x - y), W_fb)


def ratio(x, y, eps, s=None):
    return welfare(x, y, eps, s) / first_best(x, y, eps)


def ic_violation(eps, s=None, n=101):
    """Max gain of agent 1 from any misreport (grid of true types x report x opponent)."""
    s = float(s_star(eps)) if s is None else s
    t = np.linspace(0, 1, n); tt = tau(eps)
    worst = 0.0
    for y in t:
        R, Xt = np.meshgrid(t, t, indexing="ij")        # report (rows) x true type (cols)
        split = ((R >= tt) & (y <= 1 - tt)) | ((y >= tt) & (R <= 1 - tt))
        u_split = np.where(R >= 0.5, Xt, 1 - Xt)
        # fallback: dictator's item is chosen from the *report*; other agent's item from y
        m1r = np.where(R >= 0.5, s, 0.0); m2r = np.where(R >= 0.5, 0.0, s)
        m1y = s if y >= 0.5 else 0.0; m2y = 0.0 if y >= 0.5 else s
        u_d = Xt * m1r + (1 - Xt) * m2r
        u_o = Xt * (1 - m1y) + (1 - Xt) * (1 - m2y) + eps * (1 - m1y) * (1 - m2y)
        U = np.where(split, u_split, 0.5 * u_d + 0.5 * u_o)
        worst = max(worst, float((U - np.diag(U)[None, :]).max()))
    return worst

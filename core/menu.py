"""Swap-dictatorial mechanisms with a general menu, in the complementarity model.

A *menu* is a set of lotteries the dictator may choose from, described by the marginal shares
(m1, m2) she receives of items 1 and 2.  We only need its frontier m2 = phi(m1), phi decreasing
and concave on [0, m1max], with phi(0) = m2max (the set is downward closed, and utility is
increasing, so the dictator picks a frontier point).  Lotteries are comonotone ("nested"):
E[z1 z2] = min(m1, m2); the other agent gets the complementary realisation, with
E[(1-z1)(1-z2)] = 1 - max(m1, m2).  (With deterministic bundles replace min by the product.)

The dictator of type v maximises  G_v(m1) = v m1 + (1-v) phi(m1) + eps min(m1, phi(m1)),
which is concave in m1 (phi concave, min of concave functions).  G_v has increasing differences
in (m1, v) (d/dv = m1 - phi(m1) is increasing), so the maximiser m1*(v) is non-decreasing in v
and m2*(v) = phi(m1*(v)) non-increasing: the cell bounds of ``core.eps`` apply verbatim.

Families implemented:
    LIPMenu(a)            the LIP budget frontier  m2 = Q(1 - c(m1))
    PNormMenu(p, c)       (m1/c)^p + (m2/c)^p <= 1, p >= 1, 0 < c <= 1
    KnotMenu(points)      concave piecewise-linear frontier through given points
"""
from __future__ import annotations

import numpy as np

from core import lip
from core.bracket import adaptive_bracket
from core.eps import first_best, m12_of, welfare_takeall

__all__ = ["Menu", "LIPMenu", "PNormMenu", "KnotMenu", "MenuMechanism"]


class Menu:
    m1max: float = 1.0

    def phi(self, m1):            # frontier
        raise NotImplementedError


class LIPMenu(Menu):
    def __init__(self, a: float):
        self.a = a

    def phi(self, m1):
        return lip.Q(1 - lip.cost(np.clip(m1, 0, 1), self.a), self.a)

    def __repr__(self):
        return f"LIPMenu(a={self.a:.4g})"


class PNormMenu(Menu):
    def __init__(self, p: float, c: float = 1.0):
        self.p, self.c, self.m1max = p, c, c

    def phi(self, m1):
        r = np.clip(np.asarray(m1, float) / self.c, 0, 1)
        return self.c * np.maximum(1 - r ** self.p, 0) ** (1 / self.p)

    def __repr__(self):
        return f"PNormMenu(p={self.p:.4g}, c={self.c:.4g})"


class KnotMenu(Menu):
    """Frontier through points (m1_i, m2_i); the convex hull's upper-right boundary is used so
    the menu is concave regardless of the raw points."""

    def __init__(self, points):
        pts = np.array(sorted(set((float(a), float(b)) for a, b in points)))
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
        if not (pts[:, 0] == 0).any():
            pts = np.vstack([[0.0, pts[:, 1].max()], pts])
        # upper concave hull by x (monotone chain)
        hull = []
        for q in pts[np.argsort(pts[:, 0])]:
            while len(hull) >= 2:
                (x1, y1), (x2, y2) = hull[-2], hull[-1]
                if (x2 - x1) * (q[1] - y1) - (y2 - y1) * (q[0] - x1) >= 0:   # not a right turn
                    hull.pop()
                else:
                    break
            hull.append(tuple(q))
        self.xs = np.array([h[0] for h in hull]); self.ys = np.array([h[1] for h in hull])
        # keep only the decreasing part (downward closed menu)
        keep = np.concatenate([[True], np.diff(self.ys) <= 1e-12])
        self.xs, self.ys = self.xs[keep], self.ys[keep]
        self.m1max = float(self.xs[-1])

    def phi(self, m1):
        return np.interp(np.clip(m1, 0, self.m1max), self.xs, self.ys)

    def __repr__(self):
        return "KnotMenu(" + ", ".join(f"({x:.3f},{y:.3f})" for x, y in zip(self.xs, self.ys)) + ")"


class MenuMechanism:
    """Fair-coin dictator chooses her favourite lottery from the menu; nested or deterministic."""

    def __init__(self, menu: Menu, eps: float, nested: bool = True, grid: int = 2001):
        self.menu, self.eps, self.nested = menu, eps, nested
        self._g = np.linspace(0, menu.m1max, grid)
        self._phi_g = menu.phi(self._g)

    # ---- dictator's choice -------------------------------------------------------------
    def G(self, v, m1):
        m2 = self.menu.phi(m1)
        return v * m1 + (1 - v) * m2 + self.eps * m12_of(m1, m2, self.nested)

    def choose(self, v, refine: int = 40, chunk: int = 2048):
        """Marginals (m1, m2) chosen by a dictator of type v (vectorised).  Dense grid on the
        frontier, then golden-section refinement around the best grid point (G_v is concave).
        The grid search is chunked to keep memory bounded."""
        v = np.asarray(v, float)
        shape = v.shape
        vf = v.ravel()
        i = np.empty(vf.size, dtype=int)
        base = self.eps * m12_of(self._g, self._phi_g, self.nested)
        for s in range(0, vf.size, chunk):
            vv = vf[s:s + chunk, None]
            i[s:s + chunk] = np.argmax(vv * self._g + (1 - vv) * self._phi_g + base, axis=-1)
        i = i.reshape(shape)
        step = self._g[1] - self._g[0]
        lo = np.clip(self._g[i] - step, 0, self.menu.m1max)
        hi = np.clip(self._g[i] + step, 0, self.menu.m1max)
        gr = (np.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
        Gc, Gd = self.G(v, c), self.G(v, d)
        for _ in range(refine):
            left = Gc > Gd                      # maximum is in [lo, d]
            hi = np.where(left, d, hi); lo = np.where(left, lo, c)
            d = np.where(left, c, lo + gr * (hi - lo)); c = np.where(left, hi - gr * (hi - lo), d)
            Gc, Gd = self.G(v, c), self.G(v, d)
        m1 = 0.5 * (lo + hi)
        return m1, self.menu.phi(m1)

    # ---- welfare -----------------------------------------------------------------------
    def _u_dict(self, v, m1, m2):
        return v * m1 + (1 - v) * m2 + self.eps * m12_of(m1, m2, self.nested)

    def _u_other(self, w, m1, m2):
        return w * (1 - m1) + (1 - w) * (1 - m2) + self.eps * (1 - m1 - m2 + m12_of(m1, m2, self.nested))

    def welfare(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m1x, m2x = self.choose(x); m1y, m2y = self.choose(y)
        return 0.5 * (self._u_dict(x, m1x, m2x) + self._u_other(y, m1x, m2x)
                      + self._u_dict(y, m1y, m2y) + self._u_other(x, m1y, m2y))

    def ratio(self, x, y, lam: float = 1.0):
        """Ratio of the mixture lam * menu mechanism + (1 - lam) * take-all."""
        W = lam * self.welfare(x, y) + (1 - lam) * welfare_takeall(x, y, self.eps)
        return W / first_best(x, y, self.eps)

    def grid_min(self, N: int = 400, lam: float = 1.0):
        g = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(g, g, indexing="ij"); m = X >= Y
        R = self.ratio(X[m], Y[m], lam); i = int(np.argmin(R))
        return float(R[i]), float(X[m][i]), float(Y[m][i])

    # ---- certified bracket ---------------------------------------------------------------
    def _cell_lower(self, lam):
        eps, nested = self.eps, self.nested

        def other_lower(w0, w1, m1_hi, m2_hi):
            prod = 1 - np.maximum(m1_hi, m2_hi) if nested else (1 - m1_hi) * (1 - m2_hi)
            return w0 * (1 - m1_hi) + (1 - w1) * (1 - m2_hi) + eps * prod

        def f(x0, y0, h):
            x1, y1 = x0 + h, y0 + h
            m1x0, m2x0 = self.choose(x0); m1x1, m2x1 = self.choose(x1)
            m1y0, m2y0 = self.choose(y0); m1y1, m2y1 = self.choose(y1)
            A = np.minimum(self._u_dict(x0, m1x0, m2x0), self._u_dict(x1, m1x0, m2x0))
            B = other_lower(y0, y1, m1x1, m2x0)
            C = np.minimum(self._u_dict(y0, m1y0, m2y0), self._u_dict(y1, m1y0, m2y0))
            D = other_lower(x0, x1, m1y1, m2y0)
            den = 1 + np.maximum(eps, x1 - y0)
            return (lam * 0.5 * (A + B + C + D) + (1 - lam) * (1 + eps)) / den
        return f

    def bracket(self, tol=1e-4, lam: float = 1.0, **kw):
        return adaptive_bracket(self._cell_lower(lam), lambda x, y: self.ratio(x, y, lam), tol=tol, **kw)


# ---------------------------------------------------------------------------------------
# Finite menus: exact competitive ratio on the continuous type space
# ---------------------------------------------------------------------------------------

class FiniteMenuMechanism:
    """Fair-coin dictator chooses her favourite of finitely many lotteries (m1, m2, m12).

    Exact evaluation: type x's utility from item i is affine in x, so the type line splits
    into intervals on which the choice is constant (upper envelope of lines).  On a rectangle
    I_j x I_k of such intervals the welfare is affine in (x, y) and OPT = 1 + max(eps, |x-y|)
    is piecewise affine, so the ratio is linear-fractional on each piece and its minimum over
    the piece is attained at a vertex.  We enumerate the vertices of all pieces.
    """

    def __init__(self, items, eps: float, split_tau: float | None = None):
        """split_tau: if given, profiles with (x >= tau and y <= 1 - tau) or the mirror image
        are split (each agent gets her preferred item) instead of using the menu."""
        it = np.array([(float(a), float(b), float(c)) for a, b, c in items])
        self.items = it
        self.eps = eps
        self.split_tau = split_tau
        self.breaks, self.choice = self._envelope()

    def _utility_lines(self):
        m1, m2, m12 = self.items.T
        # u_x(item) = x m1 + (1-x) m2 + eps m12 = (m2 + eps m12) + x (m1 - m2)
        return m2 + self.eps * m12, m1 - m2

    def _envelope(self, tol=1e-12):
        """Upper envelope of the K utility lines on [0, 1]: at most K segments.
        Walk from x = 0: the current best line is overtaken first by the line (with larger
        slope) whose crossing point is smallest."""
        c, s = self._utility_lines()
        x = 0.0
        cur = int(np.argmax(c + 0.0 * s + 1e-15 * s))      # best at 0 (ties -> larger slope below)
        best0 = c + 0.0 * s
        cand = np.flatnonzero(best0 >= best0.max() - tol)
        cur = int(cand[np.argmax(s[cand])])
        xs, choice = [0.0], []
        while True:
            choice.append(cur)
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = (c[cur] - c) / (s - s[cur])            # where line i overtakes cur
            mask = (s > s[cur] + tol) & (xc > x + tol) & (xc < 1 - tol)
            if not mask.any():
                xs.append(1.0); break
            i = int(np.flatnonzero(mask)[np.argmin(xc[mask])])
            x = float(xc[i]); xs.append(x)
            # among lines crossing at (almost) the same point take the largest slope
            near = np.flatnonzero(mask & (np.abs(xc - x) < 1e-9))
            cur = int(near[np.argmax(s[near])])
        return np.array(xs), np.array(choice)

    def chosen(self, x):
        x = np.asarray(x, float)
        idx = np.clip(np.searchsorted(self.breaks, x, side="right") - 1, 0, len(self.choice) - 1)
        return self.items[self.choice[idx]]

    def welfare(self, x, y):
        eps = self.eps
        x = np.asarray(x, float); y = np.asarray(y, float)
        mx, my = self.chosen(x), self.chosen(y)

        def B(xd, yo, m):
            return (1 + eps) + (xd - yo - eps) * m[..., 0] + (yo - xd - eps) * m[..., 1] + 2 * eps * m[..., 2]
        W = 0.5 * (B(x, y, mx) + B(y, x, my))
        if self.split_tau is not None:
            t = self.split_tau
            sp = ((x >= t) & (y <= 1 - t)) | ((y >= t) & (x <= 1 - t))
            W = np.where(sp, 1 + np.abs(x - y), W)
        return W

    def ratio(self, x, y):
        return self.welfare(x, y) / first_best(x, y, self.eps)

    def exact_ratio(self, chunk_rows: int = 256):
        """Exact infimum of the ratio over the triangle x >= y, and the point attaining it.

        Cells are I_j x I_k of the choice intervals, further cut by the OPT kinks x - y = +-eps;
        candidate points are the cell corners and the crossings with x - y = eps, evaluated just
        inside the cell (the ratio is linear-fractional on each piece, so the infimum is at a
        vertex).  Processed in row chunks to keep memory O(chunk * cuts)."""
        eps = self.eps
        br = self.breaks
        extra = [0.0, 1.0] if self.split_tau is None else [0.0, 1.0, self.split_tau, 1 - self.split_tau]
        cuts = np.unique(np.clip(np.concatenate([br, br + eps, br - eps, extra]), 0, 1))
        xa_all, xb_all = cuts[:-1], cuts[1:]
        best = (np.inf, (np.nan, np.nan))
        for s0 in range(0, len(xa_all), chunk_rows):
            xa, xb = xa_all[s0:s0 + chunk_rows], xb_all[s0:s0 + chunk_rows]
            XA, YA = np.meshgrid(xa, xa_all, indexing="ij"); XB, YB = np.meshgrid(xb, xb_all, indexing="ij")
            keep = (XB > YA + 1e-15).ravel()
            if not keep.any():
                continue
            XA, XB, YA, YB = XA.ravel()[keep], XB.ravel()[keep], YA.ravel()[keep], YB.ravel()[keep]
            P, Q = [], []
            for (p, q) in ((XA, YA), (XB, YA), (XA, YB), (XB, YB)):
                P += [p, np.clip(q + eps, XA, XB), p]; Q += [q, q, np.clip(p - eps, YA, YB)]
            P = np.concatenate(P); Q = np.concatenate(Q)
            XA6 = np.tile(XA, 12); XB6 = np.tile(XB, 12); YA6 = np.tile(YA, 12); YB6 = np.tile(YB, 12)
            pp = np.clip(P, XA6 + 1e-12, XB6 - 1e-12); qq = np.clip(Q, YA6 + 1e-12, YB6 - 1e-12)
            ok = pp >= qq
            if not ok.any():
                continue
            R = self.ratio(pp[ok], qq[ok]); i = int(np.argmin(R))
            if R[i] < best[0]:
                best = (float(R[i]), (float(P[ok][i]), float(Q[ok][i])))
        return best


    def ic_violation(self, n: int = 401):
        """Max gain of agent 1 from any misreport on an n-grid of (true type, opponent); the
        report ranges over all menu items (and, with a split region, the split)."""
        eps, t = self.eps, self.split_tau
        xs = np.linspace(0, 1, n)
        m = self.items                                               # all items she could get
        u_items = xs[:, None] * m[None, :, 0] + (1 - xs[:, None]) * m[None, :, 1] + eps * m[None, :, 2]
        best_item = u_items.max(axis=1)                              # (n,)  best menu item
        worst = 0.0
        for y in xs:
            my = self.chosen(np.array([y]))[0]
            u_o = xs * (1 - my[0]) + (1 - xs) * (1 - my[1]) + eps * (1 - my[0] - my[1] + my[2])
            truthful_fb = 0.5 * (xs * self.chosen(xs)[:, 0] + (1 - xs) * self.chosen(xs)[:, 1] + eps * self.chosen(xs)[:, 2]) + 0.5 * u_o
            dev_fb = 0.5 * best_item + 0.5 * u_o
            if t is None:
                worst = max(worst, float((dev_fb - truthful_fb).max()))
                continue
            split_avail1 = y <= 1 - t          # reporting >= t yields split with item 1
            split_avail2 = y >= t              # reporting <= 1 - t yields split with item 2
            truthful = np.where((xs >= t) & split_avail1, xs, np.where((xs <= 1 - t) & split_avail2, 1 - xs, truthful_fb))
            options = [dev_fb]
            if split_avail1: options.append(xs)
            if split_avail2: options.append(1 - xs)
            # the fallback is reachable only by a report outside the split-triggering range;
            # with a menu item for every such report it is always reachable
            worst = max(worst, float((np.max(options, axis=0) - truthful).max()))
        return worst


# ---------------------------------------------------------------------------------------
# The optimal swap-dictatorial mechanism for additive valuations (ratio exactly 5/6)
# ---------------------------------------------------------------------------------------

def steepest_chain_D(x):
    """D(x) = max_k [5k/3 - k^2/(3x)] over k with x/k >= 1/5: the smallest non-decreasing
    function with increments D(x) - D(y) >= h(x - y) := 5/3 - 1/(3(x-y)) and D(0) = 0.
    Piecewise: 0 on [0, 1/5], 5/3 - 1/(3x) on [1/5, 3/5], 10/3 - 4/(3x) on [3/5, 1]; D(1) = 2."""
    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        k1 = np.where(x >= 0.2, 5 / 3 - 1 / (3 * np.maximum(x, 1e-300)), 0.0)
        k2 = np.where(x >= 0.4, 10 / 3 - 4 / (3 * np.maximum(x, 1e-300)), 0.0)
        k3 = np.where(x >= 0.6, 5 - 3 / np.maximum(x, 1e-300), 0.0)
    return np.maximum.reduce([np.zeros_like(x), k1, k2, k3])


def optimal_sd_choice(x):
    """Choice function (m1(x), m2(x)) of the 5/6-competitive menu mechanism.

    d(x) = m1 - m2 = (D(x) - D(1-x))/2 is antisymmetric, non-decreasing, d(0) = -1, d(1) = 1,
    and satisfies d(x) - d(y) >= 5/3 - 1/(3(x-y)) for all x > y, which is exactly the
    condition ratio(x, y) >= 5/6 (Theorem C).  Shares come from the envelope formula
    U(x) = 1 + int_0^x d, m2 = U - x d, m1 = U + (1-x) d (truthful choice from a fixed menu)."""
    x = np.asarray(x, float)
    grid = np.linspace(0, 1, 20001)
    d = 0.5 * (steepest_chain_D(grid) - steepest_chain_D(1 - grid))
    U = 1 + np.concatenate([[0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(grid))])
    dx = np.interp(x, grid, d); Ux = np.interp(x, grid, U)
    m2 = Ux - x * dx; m1 = Ux + (1 - x) * dx
    return np.clip(m1, 0, 1), np.clip(m2, 0, 1)


def optimal_sd_menu(n_items: int = 2001):
    """Finite menu approximating the optimal mechanism (nested lotteries irrelevant at eps=0)."""
    xs = np.linspace(0, 1, n_items)
    m1, m2 = optimal_sd_choice(xs)
    return np.unique(np.round(np.c_[m1, m2, np.minimum(m1, m2)], 10), axis=0)


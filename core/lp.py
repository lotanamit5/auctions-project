"""Linear-programming upper bounds on the competitive ratio of strategy-proof mechanisms.

Theorem 1 of Guo & Conitzer restricts the type space to the grid x_j = j/N, j = 0..N, and
solves for the best strategy-proof mechanism on that grid.  Restricting the type space only
relaxes the constraints, so the LP value is an upper bound on the competitive ratio of every
strategy-proof mechanism on the full type space.

This module generalises that LP to the complementarity model (``eps``) of Part B.

Variables, per report profile (j, k):
    m1  = E[z1], m2 = E[z2]      expected shares of items 1, 2 going to agent 1,
    m12 = E[z1 z2]                expected product.
Agent 1 with true type x: utility  x m1 + (1-x) m2 + eps m12 .
Agent 2 with true type y: utility  y (1-m1) + (1-y) (1-m2) + eps (1 - m1 - m2 + m12) .
Because utilities are bilinear in the bundle, a lottery over bundles is summarised exactly by
(m1, m2, m12), and the set of achievable triples is the convex hull of
{(z1, z2, z1 z2) : z in [0,1]^2}, i.e. the McCormick envelope

    m12 >= 0,   m12 >= m1 + m2 - 1,   m12 <= m1,   m12 <= m2 .

So the LP ranges over *all* randomised mechanisms on the grid; with eps = 0 the m12 variables
drop out and it is exactly the LP of Theorem 1.

Strategy-proofness.  Utilities are affine in the one-dimensional type, so full incentive
compatibility is equivalent to the adjacent-type ("local") constraints (which imply
monotonicity of m1 - m2 in the own report and hence all non-adjacent constraints).  ``full_ic``
builds every pairwise constraint instead; both are kept so they can be cross-checked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

__all__ = ["LPResult", "solve_sp_lp", "first_best_eps"]


def first_best_eps(x, y, eps: float):
    """First-best welfare in the eps model: 1 + max(eps, |x - y|) (corner of the bilinear
    welfare: either everything to one agent, or split along valuations)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 1 + np.maximum(eps, np.abs(x - y))


@dataclass
class LPResult:
    N: int
    eps: float
    alpha: float
    m1: np.ndarray            # (N+1, N+1): agent 1 report j (rows) x agent 2 report k (cols)
    m2: np.ndarray
    m12: np.ndarray
    welfare: np.ndarray       # achieved welfare at each profile
    opt: np.ndarray           # first-best welfare at each profile
    dual_welfare: np.ndarray  # multipliers lambda of the competitiveness rows; sum lambda*opt = 1
    status: str

    @property
    def ratio(self) -> np.ndarray:
        return self.welfare / self.opt

    @property
    def binding(self) -> np.ndarray:
        """Boolean mask of profiles whose competitiveness constraint carries dual weight."""
        return self.dual_welfare > 1e-9


class _Rows:
    """Accumulates sparse rows  sum coef * var <= rhs."""

    def __init__(self):
        self.r, self.c, self.v, self.rhs = [], [], [], []
        self.n = 0

    def add(self, cols, coefs, rhs=0.0):
        cols = np.asarray(cols).ravel()
        coefs = np.asarray(coefs, dtype=float).ravel()
        self.r.append(np.full(cols.size, self.n))
        self.c.append(cols)
        self.v.append(coefs)
        self.rhs.append(rhs)
        self.n += 1

    def add_block(self, cols, coefs, rhs):
        """Many rows at once: cols/coefs have shape (rows, nnz_per_row)."""
        cols = np.asarray(cols)
        coefs = np.asarray(coefs, dtype=float)
        rows, nnz = cols.shape
        self.r.append(np.repeat(np.arange(self.n, self.n + rows), nnz))
        self.c.append(cols.ravel())
        self.v.append(coefs.ravel())
        self.rhs.extend(np.asarray(rhs, dtype=float).ravel().tolist())
        self.n += rows

    def matrix(self, ncols):
        r = np.concatenate(self.r) if self.r else np.zeros(0, int)
        c = np.concatenate(self.c) if self.c else np.zeros(0, int)
        v = np.concatenate(self.v) if self.v else np.zeros(0)
        return sp.csr_matrix((v, (r, c)), shape=(self.n, ncols)), np.asarray(self.rhs)


def solve_sp_lp(N: int, eps: float = 0.0, full_ic: bool = False, symmetric: bool = False,
                with_product: bool | None = None, types=None) -> LPResult:
    """Max alpha s.t. some strategy-proof (randomised) mechanism on the grid {j/N} is
    alpha-competitive against the first best, in the model with complementarity ``eps``.

    symmetric: additionally impose symmetry over agents (m(j,k) = 1 - m(k,j) componentwise,
    m12(j,k) = 1 - m1(k,j) - m2(k,j) + m12(k,j)); by Claim 1 this does not change the optimum
    but produces a more structured solution.
    with_product: include the m12 variables (default: only if eps != 0).
    types: optional sorted array of types in [0, 1] replacing the uniform grid {j/N} (N is then
        ignored); both agents draw from the same set.  Local IC remains valid for any sorted
        one-dimensional type set.
    """
    if with_product is None:
        with_product = eps != 0.0
    if types is not None:
        g = np.sort(np.asarray(types, dtype=float)); N = len(g) - 1
    else:
        g = np.arange(N + 1) / N
    T = N + 1
    P = T * T                                   # profiles
    nvar_per = 3 if with_product else 2
    nv = nvar_per * P + 1                       # + alpha
    A = nv - 1                                  # index of alpha
    jj, kk = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")

    def idx(var, j, k):
        return var * P + j * T + k

    rows = _Rows()

    # ---- agent 1 IC:  U1(j; j, k) >= U1(j; j', k)
    #      U1(j; j', k) = x_j m1[j',k] + (1-x_j) m2[j',k] + eps m12[j',k]
    def ic1(j, jp, k):
        """rows for 'type j prefers report j over jp against k' (arrays of equal shape)."""
        x = g[j]
        cols = [idx(0, jp, k), idx(1, jp, k), idx(0, j, k), idx(1, j, k)]
        coefs = [x, 1 - x, -x, -(1 - x)]
        if with_product:
            cols += [idx(2, jp, k), idx(2, j, k)]
            coefs += [np.full_like(x, eps), np.full_like(x, -eps)]
        rows.add_block(np.stack(cols, -1), np.stack(np.broadcast_arrays(*coefs), -1), np.zeros(x.shape))

    # ---- agent 2 IC:  U2(k; j, k) >= U2(k; j, k')
    #      U2(k; j, k') = y_k (1 - m1[j,k']) + (1-y_k)(1 - m2[j,k']) + eps (1 - m1 - m2 + m12)[j,k']
    #      => depends on vars through  -(y_k + eps) m1 - (1 - y_k + eps) m2 + eps m12
    def ic2(k, kp, j):
        y = g[k]
        cols = [idx(0, j, kp), idx(1, j, kp), idx(0, j, k), idx(1, j, k)]
        coefs = [-(y + eps), -(1 - y + eps), (y + eps), (1 - y + eps)]
        if with_product:
            cols += [idx(2, j, kp), idx(2, j, k)]
            coefs += [np.full_like(y, eps), np.full_like(y, -eps)]
        rows.add_block(np.stack(cols, -1), np.stack(np.broadcast_arrays(*coefs), -1), np.zeros(y.shape))

    if full_ic:
        J, JP, K = np.meshgrid(np.arange(T), np.arange(T), np.arange(T), indexing="ij")
        m = (J != JP).ravel()
        ic1(J.ravel()[m], JP.ravel()[m], K.ravel()[m])
        ic2(J.ravel()[m], JP.ravel()[m], K.ravel()[m])
    else:
        J, K = np.meshgrid(np.arange(N), np.arange(T), indexing="ij")
        J, K = J.ravel(), K.ravel()
        ic1(J, J + 1, K)        # type j does not want to report j+1
        ic1(J + 1, J, K)        # type j+1 does not want to report j
        ic2(J, J + 1, K)
        ic2(J + 1, J, K)

    # ---- competitiveness:  alpha * OPT - W <= 0 with
    #      W = 1 + (x_j - y_k)(m1 - m2) + eps (1 - m1 - m2 + 2 m12)
    d = (g[jj] - g[kk]).ravel()
    opt = first_best_eps(g[jj], g[kk], eps)
    cols = [idx(0, jj, kk).ravel(), idx(1, jj, kk).ravel(), np.full(P, A)]
    coefs = [-(d - eps), -(-d - eps), opt.ravel()]
    if with_product:
        cols.append(idx(2, jj, kk).ravel())
        coefs.append(np.full(P, -2 * eps))
    # alpha*OPT - (d-eps) m1 - (-d-eps) m2 - 2 eps m12 <= 1 + eps
    rows.add_block(np.stack(cols, -1), np.stack(coefs, -1), np.full(P, 1 + eps))
    n_welfare_rows = P
    welfare_row0 = rows.n - P

    # ---- McCormick envelope
    if with_product:
        c0, c1, c2 = idx(0, jj, kk).ravel(), idx(1, jj, kk).ravel(), idx(2, jj, kk).ravel()
        one = np.ones(P)
        rows.add_block(np.stack([c2, c0], -1), np.stack([one, -one], -1), np.zeros(P))        # m12 <= m1
        rows.add_block(np.stack([c2, c1], -1), np.stack([one, -one], -1), np.zeros(P))        # m12 <= m2
        rows.add_block(np.stack([c0, c1, c2], -1), np.stack([one, one, -one], -1), np.ones(P))  # m1+m2-m12 <= 1

    A_ub, b_ub = rows.matrix(nv)

    # ---- optional symmetry (equalities)
    A_eq = b_eq = None
    if symmetric:
        eq = _Rows()
        for j in range(T):
            for k in range(j, T):
                eq.add([idx(0, j, k), idx(0, k, j)], [1, 1], 1.0)
                eq.add([idx(1, j, k), idx(1, k, j)], [1, 1], 1.0)
                if with_product:
                    eq.add([idx(2, j, k), idx(0, k, j), idx(1, k, j), idx(2, k, j)], [1, 1, 1, -1], 1.0)
        A_eq, b_eq = eq.matrix(nv)

    c = np.zeros(nv)
    c[A] = -1.0
    bounds = [(0, 1)] * (nv - 1) + [(0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    z = res.x
    m1 = z[idx(0, jj, kk)]
    m2 = z[idx(1, jj, kk)]
    m12 = z[idx(2, jj, kk)] if with_product else np.zeros_like(m1)
    W = 1 + (g[jj] - g[kk]) * (m1 - m2) + eps * (1 - m1 - m2 + 2 * m12)
    duals = -np.asarray(res.ineqlin.marginals)[welfare_row0:welfare_row0 + n_welfare_rows]
    return LPResult(N, eps, float(z[A]), m1, m2, m12, W, opt, duals.reshape(T, T), res.message)


# ---------------------------------------------------------------------------------------
# Best swap-dictatorial (fixed-menu) mechanism on a grid: also an LP
# ---------------------------------------------------------------------------------------

@dataclass
class SDResult:
    N: int
    eps: float
    alpha: float
    types: np.ndarray
    m1: np.ndarray            # menu item chosen by each type (dictator's marginals)
    m2: np.ndarray
    m12: np.ndarray
    welfare: np.ndarray       # (N+1, N+1) welfare at each profile
    opt: np.ndarray
    status: str
    dual_welfare: np.ndarray = None   # (N+1, N+1) multipliers of the competitiveness rows (j >= k)
    dual_ic: np.ndarray = None        # (N+1, N+1) multipliers of IC rows: [true type, deviation]

    @property
    def ratio(self):
        return self.welfare / self.opt


def solve_sd_lp(N: int, eps: float = 0.0, types=None, item_symmetric: bool = False) -> SDResult:
    """Best swap-dictatorial mechanism on the type grid, as an LP.

    A swap-dictatorial mechanism is a fixed menu; type j's choice is a lottery with moments
    m_j = (m1_j, m2_j, m12_j) in the McCormick hull.  Strategy-proofness = every type prefers
    its own item to every other type's item (full pairwise IC, O(N^2) rows).  With a fair
    coin, welfare at profile (j, k) is

        W(j,k) = 1/2 [ B(x_j, x_k, m_j) + B(x_k, x_j, m_k) ],
        B(x, y, m) = (1+eps) + (x - y - eps) m1 + (y - x - eps) m2 + 2 eps m12,

    linear in the menu.  So max alpha s.t. W >= alpha OPT is an LP.  Its value upper-bounds
    every swap-dictatorial mechanism (types restricted to the grid), and its menu is a concrete
    strategy-proof mechanism for all types (dictator picks her favourite of finitely many
    lotteries).  item_symmetric imposes m(1 - x) = swap(m(x))."""
    if types is not None:
        g = np.sort(np.asarray(types, dtype=float)); N = len(g) - 1
    else:
        g = np.arange(N + 1) / N
    T = N + 1
    nv = 3 * T + 1
    A = nv - 1

    def idx(var, j):
        return var * T + j

    rows = _Rows()
    # IC: u_j(m_k) - u_j(m_j) <= 0 for all j != k, u_j(m) = x_j m1 + (1-x_j) m2 + eps m12
    J, K = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    msk = J != K
    J, K = J[msk], K[msk]
    J_ic, K_ic = J, K
    x = g[J]
    cols = np.stack([idx(0, K), idx(1, K), idx(2, K), idx(0, J), idx(1, J), idx(2, J)], -1)
    coefs = np.stack([x, 1 - x, np.full_like(x, eps), -x, -(1 - x), np.full_like(x, -eps)], -1)
    rows.add_block(cols, coefs, np.zeros(J.size))
    # competitiveness: alpha OPT(j,k) - W(j,k) <= 0 for j >= k
    J, K = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    msk = J >= K
    J, K = J[msk], K[msk]
    xj, xk = g[J], g[K]
    opt = first_best_eps(xj, xk, eps)
    cols = np.stack([idx(0, J), idx(1, J), idx(2, J), idx(0, K), idx(1, K), idx(2, K), np.full(J.size, A)], -1)
    coefs = np.stack([-0.5 * (xj - xk - eps), -0.5 * (xk - xj - eps), np.full(J.size, -eps),
                      -0.5 * (xk - xj - eps), -0.5 * (xj - xk - eps), np.full(J.size, -eps), opt], -1)
    rows.add_block(cols, coefs, np.full(J.size, 1 + eps))
    welfare_row0, n_w = rows.n - J.size, J.size
    # McCormick
    j = np.arange(T); one = np.ones(T)
    rows.add_block(np.stack([idx(2, j), idx(0, j)], -1), np.stack([one, -one], -1), np.zeros(T))
    rows.add_block(np.stack([idx(2, j), idx(1, j)], -1), np.stack([one, -one], -1), np.zeros(T))
    rows.add_block(np.stack([idx(0, j), idx(1, j), idx(2, j)], -1), np.stack([one, one, -one], -1), np.ones(T))
    A_ub, b_ub = rows.matrix(nv)
    A_eq = b_eq = None
    if item_symmetric:
        eq = _Rows()
        for jj in range(T):
            kk = N - jj
            if kk < jj:
                break
            eq.add([idx(0, jj), idx(1, kk)], [1, -1], 0.0)
            eq.add([idx(1, jj), idx(0, kk)], [1, -1], 0.0)
            eq.add([idx(2, jj), idx(2, kk)], [1, -1], 0.0)
        A_eq, b_eq = eq.matrix(nv)
    c = np.zeros(nv); c[A] = -1.0
    bounds = [(0, 1)] * (nv - 1) + [(0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"SD LP failed: {res.message}")
    z = res.x
    m1, m2, m12 = z[idx(0, j)], z[idx(1, j)], z[idx(2, j)]
    X, Y = g[:, None], g[None, :]

    def B(xd, yo, a1, a2, a12):
        return (1 + eps) + (xd - yo - eps) * a1 + (yo - xd - eps) * a2 + 2 * eps * a12

    W = 0.5 * (B(X, Y, m1[:, None], m2[:, None], m12[:, None]) + B(Y, X, m1[None, :], m2[None, :], m12[None, :]))
    marg = -np.asarray(res.ineqlin.marginals)
    dual_w = np.zeros((T, T)); dual_w[J, K] = marg[welfare_row0:welfare_row0 + n_w]
    dual_ic = np.zeros((T, T)); dual_ic[J_ic, K_ic] = marg[:J_ic.size]
    return SDResult(N, eps, float(z[A]), g, m1, m2, m12, W, first_best_eps(X, Y, eps), res.message, dual_w, dual_ic)


# ---------------------------------------------------------------------------------------
# Split-or-menu mechanisms: best fallback menu given the split region (LP)
# ---------------------------------------------------------------------------------------

def solve_split_menu_lp(N: int, eps: float, tau: float | None = None) -> SDResult:
    """Mechanism: if (x >= tau and y <= 1 - tau) or the mirror image, split (each agent gets
    her preferred item); otherwise a fair-coin dictator picks from a fixed menu.  The menu is
    optimised by an LP: variables m_j (moments of type j's menu item), constraints

      * menu IC:            u_j(m_j) >= u_j(m_r)                           for all j, r;
      * split IC (strong):  x_j >= 1/2 u_j(m_j) + 1/2 u_j(1 - m_k)         for x_j >= tau,
                            x_k <= 1 - tau  (a strong type must not prefer the fallback; by
                            menu IC her best fallback item is m_j);
      * split IC (moderate): 1/2 u_j(m_j) + 1/2 u_j(1 - m_k) >= x_j         for 1/2 <= x_j < tau,
                            x_k <= 1 - tau  (claiming to be strong would yield item 1);
      * the mirror constraints for agent 1 preferring item 2 against x_k >= tau;
      * competitiveness only on non-split profiles.
    Default tau = (1 + eps)/2."""
    if tau is None:
        tau = (1 + eps) / 2
    g = np.arange(N + 1) / N
    T = N + 1
    nv = 3 * T + 1
    A = nv - 1

    def idx(var, j):
        return var * T + j

    rows = _Rows()

    def u_coefs(x):               # coefficients of u_x(m) on (m1, m2, m12)
        return x, 1 - x, eps

    # menu IC
    J, R = np.meshgrid(np.arange(T), np.arange(T), indexing="ij"); msk = J != R; J, R = J[msk], R[msk]
    x = g[J]
    rows.add_block(np.stack([idx(0, R), idx(1, R), idx(2, R), idx(0, J), idx(1, J), idx(2, J)], -1),
                   np.stack([x, 1 - x, np.full_like(x, eps), -x, -(1 - x), np.full_like(x, -eps)], -1), np.zeros(J.size))
    # split IC.  u_j(1 - m_k) = x(1-m1) + (1-x)(1-m2) + eps(1 - m1 - m2 + m12)
    #   = (1+eps) - (x+eps) m1 - (1-x+eps) m2 + eps m12
    strong1 = np.flatnonzero(g >= tau); strong2 = np.flatnonzero(g <= 1 - tau)
    mod1 = np.flatnonzero((g >= 0.5) & (g < tau)); mod2 = np.flatnonzero((g <= 0.5) & (g > 1 - tau))

    def fallback_rows(js, ks, sign, split_util):
        """sign=+1: fallback <= split_util(x_j)  (strong);  sign=-1: fallback >= split_util."""
        Jj, Kk = np.meshgrid(js, ks, indexing="ij"); Jj, Kk = Jj.ravel(), Kk.ravel()
        if Jj.size == 0:
            return
        x = g[Jj]
        cols = np.stack([idx(0, Jj), idx(1, Jj), idx(2, Jj), idx(0, Kk), idx(1, Kk), idx(2, Kk)], -1)
        coefs = np.stack([0.5 * x, 0.5 * (1 - x), np.full_like(x, 0.5 * eps),
                          -0.5 * (x + eps), -0.5 * (1 - x + eps), np.full_like(x, 0.5 * eps)], -1)
        rhs = split_util(x) - 0.5 * (1 + eps)
        rows.add_block(sign * cols * 0 + cols, sign * coefs, sign * rhs)

    fallback_rows(strong1, strong2, +1, lambda x: x)            # strong-1 vs strong-2: prefer split (item 1)
    fallback_rows(mod1, strong2, -1, lambda x: x)               # moderate-1 vs strong-2: prefer fallback
    fallback_rows(strong2, strong1, +1, lambda x: 1 - x)        # strong-2 vs strong-1: prefer split (item 2)
    fallback_rows(mod2, strong1, -1, lambda x: 1 - x)           # moderate-2 vs strong-1: prefer fallback
    # competitiveness on non-split profiles (j >= k)
    J, K = np.meshgrid(np.arange(T), np.arange(T), indexing="ij"); msk = J >= K; J, K = J[msk], K[msk]
    xj, xk = g[J], g[K]
    split = ((xj >= tau) & (xk <= 1 - tau)) | ((xk >= tau) & (xj <= 1 - tau))
    J, K, xj, xk = J[~split], K[~split], xj[~split], xk[~split]
    opt = first_best_eps(xj, xk, eps)
    cols = np.stack([idx(0, J), idx(1, J), idx(2, J), idx(0, K), idx(1, K), idx(2, K), np.full(J.size, A)], -1)
    coefs = np.stack([-0.5 * (xj - xk - eps), -0.5 * (xk - xj - eps), np.full(J.size, -eps),
                      -0.5 * (xk - xj - eps), -0.5 * (xj - xk - eps), np.full(J.size, -eps), opt], -1)
    rows.add_block(cols, coefs, np.full(J.size, 1 + eps))
    # McCormick
    j = np.arange(T); one = np.ones(T)
    rows.add_block(np.stack([idx(2, j), idx(0, j)], -1), np.stack([one, -one], -1), np.zeros(T))
    rows.add_block(np.stack([idx(2, j), idx(1, j)], -1), np.stack([one, -one], -1), np.zeros(T))
    rows.add_block(np.stack([idx(0, j), idx(1, j), idx(2, j)], -1), np.stack([one, one, -one], -1), np.ones(T))
    A_ub, b_ub = rows.matrix(nv)
    c = np.zeros(nv); c[A] = -1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * (nv - 1) + [(0, None)], method="highs")
    if res.status != 0:
        raise RuntimeError(f"split-menu LP failed: {res.message}")
    z = res.x
    m1, m2, m12 = z[idx(0, j)], z[idx(1, j)], z[idx(2, j)]
    X, Y = g[:, None], g[None, :]

    def B(xd, yo, a1, a2, a12):
        return (1 + eps) + (xd - yo - eps) * a1 + (yo - xd - eps) * a2 + 2 * eps * a12

    W = 0.5 * (B(X, Y, m1[:, None], m2[:, None], m12[:, None]) + B(Y, X, m1[None, :], m2[None, :], m12[None, :]))
    sp = ((X >= tau) & (Y <= 1 - tau)) | ((Y >= tau) & (X <= 1 - tau))
    W = np.where(sp, 1 + np.abs(X - Y), W)
    return SDResult(N, eps, float(z[A]), g, m1, m2, m12, W, first_best_eps(X, Y, eps), res.message)

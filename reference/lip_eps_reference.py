"""
Reference numerics for Part B: LIP mechanism with complementarity term eps*x1*x2.
Model: two agents, two divisible goods, types (x, 1-x) and (y, 1-y), common known eps >= 0.
Utility of agent with type (v, 1-v) from bundle (q1, q2):  v*q1 + (1-v)*q2 + eps*q1*q2.
LIP(a): coin flip -> dictator buys any affordable bundle, other agent gets the remainder.
Prices f(t) = a*t + b with b = a/(e^a - 1)  (normalization: 1 unit of currency buys exactly one full item).
Buying fraction q of one item costs c(q) = (e^{a q} - 1)/(e^a - 1).   Budget set: c(q1) + c(q2) <= 1.
Run:  python3 lip_eps_reference.py
"""
import math
import numpy as np

A = 2.0                      # price slope used in Guo & Conitzer's Theorem 2
GRID_Q = np.linspace(0, 1, 801)


def cost(q, a=A):
    return (math.exp(a * q) - 1) / (math.exp(a) - 1)


def frontier(a=A):
    """For each q1 in GRID_Q, the largest affordable q2 (budget exhausted)."""
    c1 = np.array([cost(q, a) for q in GRID_Q])
    q2 = np.log(1 + (1 - c1) * (math.exp(a) - 1)) / a
    return q2


def dictator_bundle(v, eps, q2max):
    """eps-optimal affordable bundle for dictator of type (v, 1-v). Utility is increasing in both
    goods, so the optimum lies on the budget frontier."""
    u = v * GRID_Q + (1 - v) * q2max + eps * GRID_Q * q2max
    i = int(np.argmax(u))
    return GRID_Q[i], q2max[i]


def claim5_bundle(x, a=A):
    """The paper's additive rule (Claim 5): report x -> purchase. Used only for the Step-1 example."""
    b = a / (math.exp(a) - 1)
    lo, hi = b / (a + 2 * b), (a + b) / (a + 2 * b)
    if x >= hi:
        return 1.0, 0.0
    if x <= lo:
        return 0.0, 1.0
    t = (x * (a + 2 * b) - b) / a
    q = lambda s: (math.log(a * s + b) - math.log(b)) / a
    return q(t), q(1 - t)


def welfare(v, w, eps, x1, x2):
    """Agent 1 (type v) gets (x1, x2); agent 2 (type w) gets the rest."""
    return (v * x1 + (1 - v) * x2 + eps * x1 * x2
            + w * (1 - x1) + (1 - w) * (1 - x2) + eps * (1 - x1) * (1 - x2))


def first_best(v, w, eps):
    # bilinear in (x1, x2) -> maximised at a corner of the unit square
    return max(welfare(v, w, eps, x1, x2) for x1 in (0, 1) for x2 in (0, 1))


def ratio_at(v, w, eps, q2max):
    d1 = dictator_bundle(v, eps, q2max)          # agent 1 dictator
    d2 = dictator_bundle(w, eps, q2max)          # agent 2 dictator -> agent 1 gets complement
    W = 0.5 * welfare(v, w, eps, *d1) + 0.5 * welfare(v, w, eps, 1 - d2[0], 1 - d2[1])
    return W / first_best(v, w, eps)


def competitive_ratio(eps, a=A, N=200):
    """Worst case over the triangle 1 >= x >= y >= 0 (grid points only -> this is an UPPER bound
    on the true ratio; the lower bound needs the cell-corner sandwich of Theorem 2)."""
    q2max = frontier(a)
    best, arg = 9.0, None
    for i in range(N + 1):
        for j in range(i + 1):
            r = ratio_at(i / N, j / N, eps, q2max)
            if r < best:
                best, arg = r, (i / N, j / N)
    return best, arg


def branch_B(eps, a=A):
    """Ratio at identical extreme types (1,0),(1,0): first-best = 1+eps, LIP = 1 + eps*s."""
    q = dictator_bundle(1.0, eps, frontier(a))
    s = q[0] * q[1] + (1 - q[0]) * (1 - q[1])
    return (1 + eps * s) / (1 + eps)


def limit_B(a=A):
    qs = math.log((math.exp(a) + 1) / 2) / a
    return qs * qs + (1 - qs) ** 2


if __name__ == "__main__":
    print("Sanity: eps=0 should give ~0.828 at (0.41, 0):", competitive_ratio(0.0, N=200))

    print("\nStep 1 example (a=2, eps=1, true x=0.8) under the paper's Claim-5 rule:")
    for rep in (0.80, 0.59):
        q = claim5_bundle(rep)
        print(f"  report {rep:.2f} -> bundle ({q[0]:.3f},{q[1]:.3f}), true utility {0.8*q[0]+0.2*q[1]+q[0]*q[1]:.3f}")

    print("\nalpha(2, eps) vs branch B(eps):")
    for eps in (0, 0.1, 0.25, 0.3, 0.5, 1, 2, 4, 8, 20):
        r, arg = competitive_ratio(eps, N=120)
        print(f"  eps={eps:<5} alpha={r:.4f} at {tuple(round(t,3) for t in arg)}   B={branch_B(eps):.4f}")
    print(f"\nlim_{{eps->inf}} B = {limit_B():.4f}")

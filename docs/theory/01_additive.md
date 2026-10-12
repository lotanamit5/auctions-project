# Additive valuations: structure of α(a) for LIP(a)

Notation as in the paper: f(t) = at + b, b = a/(e^a − 1), Q(t) = ln(1 + at/b)/a, agent 1 has
type (x, 1−x), agent 2 has (y, 1−y), 1 ≥ x ≥ y ≥ 0 w.l.o.g., S(x, y) = expected welfare of
LIP(a), first best = x + 1 − y, α(a) = min S/(x + 1 − y).

## Two one-dimensional restrictions give rigorous upper bounds

Restricting the minimisation to a subset of the triangle can only increase the minimum, so
each of the following is an upper bound on α(a).

**Anti-diagonal (x + y = 1).** This is the family of profiles of the paper's Claim 8
(types (u/(u+1), 1/(u+1)) and (1/(u+1), u/(u+1))). The paper shows that along it the ratio is
`(u−1) ln u/(2a) + 1/2 + u/2`, with u = y/x ∈ [b/(a+b), 1], and calls its minimum the
*intrinsic value* θ(a). Hence α(a) ≤ θ(a) for every a (this is the paper's Claim 8).

**Edge (y = 0).** Agent 2 only values item 2. As dictator she buys (0, 1). For agent 1 in the
interior regime of Claim 5, Lemma 2 gives the final prices at + b = x(a+2b) and
a(1−t) + b = (1−x)(a+2b), so

    Q(t) − Q(1−t) = (1/a) ln( (at+b) / (a(1−t)+b) ) = (1/a) ln( x / (1−x) ).

Welfare: branch "agent 1 dictator" = x Q(t) + (1−x) Q(1−t) + (1 − Q(1−t)); branch "agent 2
dictator" = 1 + x. Averaging,

    S(x, 0) = 1 + (x/2) (1 + (1/a) ln(x/(1−x)) ),

and the ratio is S(x,0)/(1+x). Define

    β(a) = min_{b/(a+2b) ≤ x ≤ (a+b)/(a+2b)}  [1 + (x/2)(1 + ln(x/(1−x))/a)] / (1 + x)

(outside the interior regime the ratio is 1 on this edge: both dictators produce the first-best
allocation). Hence α(a) ≤ β(a). Check at a = 2: minimiser x ≈ 0.4074, β(2) ≈ 0.82815, which is
exactly the certified value of α(2).

## Numerical theorem

    α(a) = min{ θ(a), β(a) }     (verified for a ∈ [0.5, 6] to 1e-6 by certified brackets)

- β binds for a < a*, θ binds for a > a*;
- the optimal LIP parameter solves θ(a) = β(a): a* ≈ 2.1013, α* = α(a*) ≈ 0.82939
  (vs 0.82815 for the paper's a = 2).

Both branches have a transparent interpretation. On the edge, agent 1 is *mildly* in favour of
item 2 (x ≈ 0.41) while agent 2 wants only item 2: as dictator, agent 1 buys mostly item 2,
taking it from the agent who wants it most. On the anti-diagonal the agents have mirror-image
preferences and each dictator takes a large share of the item the other agent prefers.
Steeper prices (larger a) hurt on the anti-diagonal (dictators buy more balanced bundles) and
help on the edge (a mildly-item-2 dictator buys less of it); the optimum equalises the two.

What is *proved*: α(a) ≤ min{θ(a), β(a)} for all a, and the certified lower bounds of
`lip.sandwich_adaptive` match to 1e-6. What is *not* proved: that the minimum over the whole
triangle is always attained on one of the two lines (an analytic argument would need the sign
of ∂S/∂y away from the lines; plausible but not done).

## The LP upper bound depends on the grid

The LP of Theorem 1 restricts types to {j/N}. Its value is an upper bound for every N and
decreases with N (0.8429 at N = 40, 0.8412 at 50, 0.8380 at 100, 0.8374 at 120, ...). The
paper's 0.841 is therefore just the N = 50 instance; see notebook 01 for the extrapolation.

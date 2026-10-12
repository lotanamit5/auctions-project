# The price of dictatorship: no fixed-menu mechanism beats 5/6

Setting: two items, additive normalised valuations (ε = 0), agent 1 of type (x, 1−x), agent 2
of type (y, 1−y). The paper's hierarchy is LIP ⊊ IP ⊊ SD ⊊ SP, with LIP(2) ≥ 0.828 and
SP ≤ 0.841 (LP, N = 50). We locate SD in between.

## Theorem A

Let M be any mechanism of the form
    M(x, y) = λ A(x) + (1 − λ) B(y),   λ ∈ [0, 1],
where A(x) is a (possibly random) allocation that depends only on agent 1's report and B(y)
one that depends only on agent 2's report (this covers every swap-dictatorial mechanism, with
arbitrary and even different menus for the two agents, with or without lotteries, and with no
incentive constraint used at all). Then M is at most 5/6-competitive.

*Proof.* Write a(x) = E[A1(x)] − E[A2(x)] (agent 1's expected share of item 1 minus item 2
under A), and similarly b(y) for B. With additive valuations the welfare at (x, y) of an
allocation with expected shares (m1, m2) for agent 1 is 1 + (x − y)(m1 − m2). Hence

    W(x, y) = 1 + (x − y) [ λ a(x) + (1 − λ) b(y) ].

Consider the four profiles (½, 0), (1, ½), (0, ½), (½, 1), each with first-best welfare 3/2.
Summing:

    Σ W = 4 + λ [ ½ a(½) + ½ a(1) − ½ a(0) − ½ a(½) ]
            + (1 − λ) [ −½ b(0) − ½ b(½) + ½ b(½) + ½ b(1) ]
        = 4 + ½ λ [ a(1) − a(0) ] + ½ (1 − λ) [ b(1) − b(0) ]
        ≤ 4 + ½ λ · 2 + ½ (1 − λ) · 2 = 5,

because a, b ∈ [−1, 1]. The contributions of the middle type cancel identically. So the
smallest of the four welfares is at most 5/4 = (5/6) · (3/2). ∎

Equality requires a(1) = b(1) = 1 and a(0) = b(0) = −1 (extreme types take their item in
full) and all four profiles tight: the three-type LP in notebook 04 shows the bound is attained
on the type set {0, ½, 1}, e.g. by the menu {(1,0), (½,½), (0,1)}.

## Consequences

- Every increasing-price mechanism (linear or not), every swap-dictatorial mechanism and every
  mixture of two one-sided rules is at most 5/6 ≈ 0.8333-competitive.
- LIP(a*) achieves 0.82939 (certified). Hence the best menu mechanism lies in
  [0.8294, 0.8333]: LIP is within 0.004 of the best possible dictatorial rule.
- The LP over *all* strategy-proof mechanisms gives ≤ 0.8368 (N = 150; extrapolating to
  ≈ 0.8346), strictly above 5/6. So the LP-optimal mechanisms must genuinely use both reports
  jointly; any further improvement over LIP of more than 0.004 has to come from non-dictatorial
  designs.
- 5/6 *is* attained on the continuous type space: see Theorem C below (closed-form menu). The
  grid-optimal menus of the SD-LP approach it from below (0.8197, 0.8264, 0.8299, 0.8310,
  0.8316, 0.8322 for N = 50 … 600) and beat LIP from N = 200 on.

## With complementarities

The same bookkeeping with B(x, y, m) = (1+ε) + (x−y−ε) m1 + (y−x−ε) m2 + 2ε m12 gives
ε-dependent bounds; the SD-LP on the grid {0, ¼, ½, ¾, 1} already returns the full-grid values
20/23 (ε = ¼), 6/7 (ε = ½), 28/31 (ε = ¾), and the corresponding menus have 3 items. At ε = ½
the 3-item menu {(3/7, 0), (0, 3/7), balanced pooling lottery} is *exactly* 6/7-competitive on
the continuum (exact evaluation, `core.menu.FiniteMenuMechanism`). The gap to the SP-LP
(≈ 0.95 at ε = ½) shows that with complementarities dictatorship is far more costly than in
the additive model: pooling when agents agree and splitting when they disagree needs both
reports.

## Theorem B (complementarities): closed-form bounds on fixed-menu mechanisms

Branch welfare when the dictator of type x picks the lottery with moments m = (m1, m2, m12)
and the other agent has type y:

    B(x, y, m) = (1+ε) + (x − y − ε) m1 + (y − x − ε) m2 + 2ε m12,   m12 ≤ min(m1, m2).

**(i) Extreme profiles.** For any menus, using only the profiles (1,1), (0,0) (first best
1+ε) and (1,0) (first best 2): write m(1) = (a1, a2, a12), m(0) = (c1, c2, c12).
W(1,1) = (1+ε) − ε(a1 + a2) + 2ε a12 ≤ (1+ε) − ε|a1 − a2|, similarly W(0,0), and
W(1,0) = ½[B(1,0,m(1)) + B(0,1,m(0))] = (1+ε) + ½(1−ε)(a1 + c2) − ½(1+ε)(a2 + c1) + ε(a12 + c12).
With s := the dictator's net share of her preferred item, the best trade-off between
W(1,1) = (1+ε) − εs and W(1,0) = (1+ε) + (1−ε)s is at

    s* = (1 − ε²)/(1 + 2ε − ε²),      ratio  ρ_B(ε) = (1+ε)/(1 + 2ε − ε²).

(Dual certificate: weights λ on (0,0) and (1,1) and μ = 2ελ/(1−ε) on (1,0) make the
coefficient of s vanish, and 2λ(1+ε) + 2μ = 1 normalises; e.g. λ = 1/7, μ = 2/7 at ε = ½. No
incentive constraint is used; `lp.solve_sd_lp(types=[0,1/2,1])` returns exactly ρ_B for
ε ≥ 0.3.)  Values: 20/23 (ε = ¼), 6/7 (½), 28/31 (¾).

**(ii) Middle type.** The four-profile argument of Theorem A with the ε terms: the middle
type's two branches sum to 2(1+ε) − 2ε(r1 + r2) + 4ε r12 ≤ 2(1+ε), and each extreme type's
cross branch is at most (1+ε) + max(0, ½ − ε). With first best 3/2 on all four profiles
(ε ≤ ½):

    ρ_A(ε) = (5 + 2ε)/6.

**Theorem B.** Every swap-dictatorial mechanism (any menus, lotteries allowed, no IC needed)
is at most min{ρ_A(ε), ρ_B(ε)}-competitive. ρ_A binds for ε ≲ 0.17, ρ_B for ε ≳ 0.17.

**Tightness.** The SD-LP on grids up to N = 200 equals ρ_B(ε) for all ε ≥ 0.3. For ε ≥ ½ the
two-item menu {(s*, 0), (0, s*)} ("the dictator takes s* of her preferred item, the rest goes
to the other agent") attains ρ_B on the whole continuum: same-side profiles have |x−y| ≤ ½ ≤ ε
so the first best is 1+ε there, and on opposite sides the ratio decreases in x − y. For
ε < ½ that menu fails near (1, ½⁺) (first best 3/2 but both take item 1), and the grid-optimal
menus add a balanced pooling lottery for middle types. Their exact continuum values: at
ε = 0.4 the N = 200 grid menu (25 items) attains ρ_B = 35/41 exactly; at ε = ¼ the N = 200
menu gives 0.8661 against ρ_B = 20/23 = 0.8696 (notebook 04, `results/data/sd_continuum.csv`). Between ε = 0.02 and 0.2 the grid value lies slightly below
min{ρ_A, ρ_B} (e.g. 0.86545 vs 0.86667 at ε = 0.1), so finer type sets matter there.

**Interpretation.** With complementarities the optimal dictator is *modest*: she takes only a
fraction s* < 1 of her favourite item (3/7 at ε = ½, 7/31 at ε = ¾), because the other agent's
complementarity gain from keeping both items is worth more than her own extra share. As ε → 1,
s* → 0 and the mechanism becomes equal split... yet take-all is first best at ε = 1: the
fixed-menu class is discontinuous at ε = 1 (ρ_B(1) = 1 is attained by s* = 0 only as a limit;
at ε = 1 the balanced pooling lottery, which is in the menu class, gives ratio 1 directly).

## Theorem C: 5/6 is attained — the optimal menu mechanism in closed form (ε = 0)

Reduce a symmetric fixed-menu mechanism to one function. If the dictator of type x chooses
(m1(x), m2(x)), write d(x) = m1(x) − m2(x) ∈ [−1, 1]. With additive valuations the welfare at
(x, y), x > y, is

    W(x, y) = 1 + ½ (x − y) [ d(x) − d(y) ],      OPT = 1 + (x − y),

so W/OPT ≥ 5/6 for all x > y  ⟺  d(x) − d(y) ≥ h(x − y) := 5/3 − 1/(3(x − y)).
(The constraint is vacuous for x − y ≤ 1/5.)  Incentive compatibility of a choice function
with utilities m2 + x d is equivalent to d non-decreasing, and the shares are then forced by
the envelope formula U(x) = U(0) + ∫₀ˣ d, m2 = U − x d, m1 = U + (1 − x) d.

*Existence.* Let D(x) = sup over chains 0 = x0 < x1 < … < xk = x of Σ h⁺(xi − xi−1). Since h is
concave, equal gaps are optimal for a given k, so D(x) = max_k [k h(x/k)]⁺ = max_k [5k/3 − k²/(3x)]:

    D(x) = 0 on [0, 1/5],   5/3 − 1/(3x) on [1/5, 3/5],   10/3 − 4/(3x) on [3/5, 1],   D(1) = 2.

(Two or three equal jumps give exactly 2 at x = 1; by AM–HM no partition gives more.)  By
construction D(x) − D(y) ≥ h(x − y). Hence d_min = D − 1 is feasible with d_min(0) = −1,
d_min(1) = 1. The constraints are linear and invariant under x ↦ 1 − x, d ↦ −d, so the
antisymmetric average d(x) = ½[D(x) − D(1 − x)] is feasible too, and with U(0) = 1 (type 0 takes
item 2) its envelope shares satisfy 0 ≤ m1, m2 ≤ 1 (checked numerically; the un-symmetrised
d_min violates m2 ≥ 0 slightly).

**Theorem C.** The mechanism "fair coin; the dictator of type x receives (m1(x), m2(x)) from the
menu generated by d above; the other agent gets the rest" is strategy-proof and exactly
5/6-competitive. Together with Theorem A: the best fixed-menu (swap-dictatorial, in particular
increasing-price) mechanism for two items is exactly 5/6-competitive. ∎

Numerically (`core.menu.optimal_sd_menu`, exact evaluation of the discretised menu): 501 /
2001 / 8001 items give 0.83258 / 0.83315 / 0.83328, converging to 5/6 = 0.83333 (the loss is
discretisation only: off-grid types pick a neighbouring item). The worst case sits at
(x, y) = (2/3, 1/3), a gap of 1/3 where h(1/3) = 2/3 is tight.

Interpretation: the optimal dictator is *less greedy than LIP* at the balanced type (she takes
(0.70, 0.70) instead of LIP(a*)'s (0.72, 0.72)) and her net share d moves at exactly the rate
needed so that two types s apart differ in net share by 5/3 − 1/(3s). The additive price
mechanism LIP(a*) is within 0.004 of this optimum.

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
- Whether 5/6 is attained on the continuous type space is open; the grid-optimal menus of the
  SD-LP only achieve 0.826 when used for all types, and a direct optimiser over finite menus
  (`experiments/menu_opt.py`) is used to search for better continuum menus.

## With complementarities

The same bookkeeping with B(x, y, m) = (1+ε) + (x−y−ε) m1 + (y−x−ε) m2 + 2ε m12 gives
ε-dependent bounds; the SD-LP on the grid {0, ¼, ½, ¾, 1} already returns the full-grid values
20/23 (ε = ¼), 6/7 (ε = ½), 28/31 (ε = ¾), and the corresponding menus have 3 items. At ε = ½
the 3-item menu {(3/7, 0), (0, 3/7), balanced pooling lottery} is *exactly* 6/7-competitive on
the continuum (exact evaluation, `core.menu.FiniteMenuMechanism`). The gap to the SP-LP
(≈ 0.95 at ε = ½) shows that with complementarities dictatorship is far more costly than in
the additive model: pooling when agents agree and splitting when they disagree needs both
reports.

# The complementarity model: what is proved, what is numerical

Two agents, two divisible items. Agent with private type (v, 1−v) has utility
u_v(z1, z2) = v z1 + (1−v) z2 + ε z1 z2 from the (possibly random) bundle (z1, z2); ε ≥ 0 is
common knowledge. Agents are expected-utility maximisers.

## P1. First best

OPT(x, y) = 1 + max{ε, |x − y|}.

*Proof.* Total welfare of giving agent 1 the bundle (z1, z2) is
1 + (x − y)(z1 − z2) + ε[z1 z2 + (1−z1)(1−z2)], bilinear in (z1, z2), hence maximised at a
corner of [0,1]²: both items to one agent (1 + ε) or the valuation-based split (1 + |x−y|).
Randomisation cannot help (an average never exceeds the best realisation). ∎

Consequence: for ε ≥ 1 the mechanism "fair coin, winner takes everything" (**take-all**) is
first best at every profile, hence strategy-proof with competitive ratio 1. For ε < 1 its ratio
is (1+ε)/2 (worst case: opposed extreme types). Equal split has (1 + ε/2)/(1 + max(ε, |x−y|)).

## P2. Fixed-menu dictatorships are strategy-proof for every ε

Let M be any set of lotteries over bundles. Mechanism: fair coin picks the dictator; she
receives her favourite element of M (computed from her report and the public ε); the other
agent receives the complementary realisation. Truthful reporting is dominant: the report
affects only the dictator branch, where any misreport can only replace the favourite element of
M by another element of M. The other agent's report does not affect the menu. ∎

"ε-aware LIP(a)" is this mechanism with M = the LIP budget set (deterministic bundles). With
ε = 0 it is exactly LIP(a).

## P3. The dictator's problem under ε-aware LIP(a)

Along the frontier t ↦ (Q(t), Q(1−t)) the objective is
F_v(t) = v Q(t) + (1−v) Q(1−t) + ε P(t), P(t) = Q(t) Q(1−t).
Q'' < 0 and P'' = Q''(t)Q(1−t) − 2Q'(t)Q'(1−t) + Q(t)Q''(1−t) < 0, so F_v is strictly concave:
the optimal spend t*(v) is unique. ∂²F/∂t∂v = Q'(t) + Q'(1−t) > 0, so t*(v) is non-decreasing
(Topkis). Hence the dictator's share of item 1 is non-decreasing in her type and her share of
item 2 non-increasing — the monotonicity the paper's grid sandwich needs. ∎

Regimes: the dictator is all-in on item 2 iff F'_v(0) ≤ 0 ⟺ (v + ε)/b ≤ (1 − v)/(a + b).
For ε ≥ b/(a+b) (≈ 0.135 at a = 2) no type is ever all-in: everybody buys both items.

## P4. Certified brackets in the ε model

For a cell [x0, x0+h] × [y0, y0+h] the welfare of the dictator branch "agent 1 dictates" is
bounded below by
 (i) u_x(q(x)) ≥ u_x(q(x0)) (q(x0) is on the menu), and the right side is affine in x, so it is
     ≥ min{u_{x0}(q(x0)), u_{x0+h}(q(x0))};
 (ii) u_y(1 − q(x)) ≥ y0(1 − q1(x0+h)) + (1 − y0 − h)(1 − q2(x0)) + ε(1 − q1(x0+h))(1 − q2(x0)),
     by monotonicity of q and non-negativity of the coefficients.
Symmetrically for the other branch; the denominator is at most 1 + max(ε, x0 + h − y0). The
point values at cell corners are exact ratios, so the adaptive refinement of `core.bracket`
yields a rigorous bracket. (Implemented in `core.eps.bracket_lip`.)

## P5. Strategic play against the *naive* rule = the ε-aware mechanism

If the mechanism keeps the additive report→bundle rule of Claim 5 while ε > 0, a dictator of
type x who reports r receives the frontier point t0(r) ∈ [0, 1]; r ↦ t0(r) is onto [0, 1]. Her
report is irrelevant in the other branch. Hence her dominant strategy is to report any r with
t0(r) = t*(x), and the realised allocation coincides with the ε-aware mechanism's. So the
"strategic LIP" column of the chatbot draft is the competitive ratio of ε-aware LIP(2)
(verified numerically to 5 digits: 0.83082, 0.84028, 0.81002, 0.76515, 0.71528 at
ε = 0.01, 0.1, 0.5, 1, 2). The naive rule is not strategy-proof for any ε > 0, and the maximal
gain from misreporting is ≤ ε·max_t P(t)/2 = ε Q(1/2)²/2 (draft's bound; re-derived in
notebook 05).

## P6. Comonotone ("nested") lotteries dominate deterministic bundles

Replace the dictator's deterministic bundle (m1, m2) by the lottery that gives her item 1 with
probability m1 and item 2 with probability m2 using a *single* uniform draw U (item j iff
U < m_j). Marginals are unchanged, but E[z1 z2] rises from m1 m2 to min(m1, m2), and for the
other agent E[(1−z1)(1−z2)] rises from (1−m1)(1−m2) to 1 − max(m1, m2). Both agents' ε-terms
weakly increase (difference 2 m_min (1 − m_max) ≥ 0 in total). The menu is still fixed, so P2
applies: the **nested LIP(a)** mechanism is strategy-proof. The dictator's objective becomes
v m1 + (1−v) φ(m1) + ε min(m1, φ(m1)), still concave with increasing differences, so P3–P4 carry
over (with 1 − max in place of the product in (ii)).

As ε → ∞ every dictator picks the balanced lottery (Q(½), Q(½)): she gets *everything* with
probability Q(½) and the other agent gets everything otherwise — welfare 1 + ε = OPT. In fact
for ε ≥ 1 the nested LIP(a) has competitive ratio 1 for every a tested, and the LP shows that
is the best possible.

Caveat: the dictator's *choice* changes between the two variants, so pointwise dominance of
welfare does not follow for the equilibrium outcome; numerically nested loses at some
intermediate (ε, a) pairs and wins elsewhere — see notebook 03 for the optimised comparison.

## P7. Upper bound on all strategy-proof mechanisms (McCormick LP)

A randomised mechanism gives agent 1 a lottery over bundles; because utilities are bilinear,
each agent's utility depends on the lottery only through (m1, m2, m12) = (E z1, E z2, E z1z2),
and the attainable triples are exactly the convex hull of {(z1, z2, z1 z2): z ∈ [0,1]²}, i.e.
the McCormick envelope m12 ≥ 0, m12 ≥ m1 + m2 − 1, m12 ≤ m1, m12 ≤ m2. Welfare is
1 + (x − y)(m1 − m2) + ε(1 − m1 − m2 + 2 m12). On a finite type grid the best strategy-proof
mechanism is therefore an LP (`core.lp.solve_sp_lp(N, eps)`), whose value upper-bounds every
strategy-proof mechanism on the continuous type space (restricting types only relaxes
constraints). Because utilities are affine in the own type, adjacent-type IC constraints suffice
(they imply monotonicity of m1 − m2 and hence all other constraints).

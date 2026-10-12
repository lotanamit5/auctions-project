# Beyond dictatorship: a strategy-proof mechanism that conditions on agreement

With complementarities the first best pools when the agents agree and splits when they
disagree; fixed-menu mechanisms cannot do both (Theorems A/B). This note gives a simple
mechanism that can, with a closed-form competitive ratio.

## The mechanism SMD(s) — "split if strongly opposed, else modest dictator"

Let τ = (1+ε)/2 and s ∈ [0, 1]. Reports x, y.

1. If x ≥ τ and y ≤ 1 − τ (agent 1 strongly prefers item 1, agent 2 strongly prefers item 2),
   or the mirror image: **split** — each agent receives her preferred item in full.
2. Otherwise: fair coin; the dictator receives s of her preferred item (menu {(s,0), (0,s)});
   the other agent receives the rest.

With s = 0 the fallback is take-all ("pool unless strongly opposed").

## Proposition D (strategy-proofness)

SMD(s) is strategy-proof for every s ∈ [0, 1] and ε ∈ [0, 1].

*Proof.* Fix agent 1 with type x (agent 2 symmetric). If agent 2's report y lies in
(1 − τ, τ), the outcome is the fallback whatever agent 1 reports, and the fallback is a
fixed-menu dictatorship, hence truthful (P2). Suppose y ≤ 1 − τ (agent 2 strongly prefers item 2;
she takes (0, s) as dictator). Agent 1's options are: report r ≥ τ → split, utility x; report
r < τ → fallback, utility ½·max{x s, (1−x) s} + ½[x·1 + (1−x)(1−s) + ε(1−s)]. For x ≥ ½ the
fallback utility is F(x) = x s + ½(1−s)(1+ε), affine with slope s < 1, and F(τ) = τ. Hence
x ≥ F(x) iff x ≥ τ: strong types (x ≥ τ) prefer the split, moderate types (½ ≤ x < τ) prefer the
fallback, which is what truthful reporting gives them. For x < ½ the split would hand her item 1,
worth x < ½ < F(x), so she prefers the fallback, again the truthful outcome. The case y ≥ τ is
the mirror image. ∎ (Brute-force verification: `core.split.ic_violation` returns 0.)

## Proposition E (competitive ratio)

In the split region the ratio is 1 (there x − y ≥ ε, so the first best is the split). Outside it
the welfare is (1+ε) + (x−y−ε) s on opposite sides and (1+ε) − ε s on the same side. The binding
profiles are

- identical types: (1 + ε − ε s)/(1 + max{ε, ½}) (for ε < ½ the same-side gap can reach ½ > ε);
- the edge of the split region, x = 1, y = (1−ε)/2: [2(1+ε) + (1−ε) s]/(3 + ε).

Equalising gives the optimal share and ratio

    ε ≥ ½:  s* = (1 − ε²)/(1 + 3ε),            ρ_SMD(ε) = (1 + ε)² / (1 + 3ε);
    ε ≤ ½:  s* = ε(1+ε)/(3/2 + 3ε/2 + ε²),     ρ_SMD(ε) = 2(1 + ε)² / (3 + 3ε + 2ε²).

Values: 0.729 (ε = 0.1), 0.806 (0.25), 0.867 (0.4), **0.900 (0.5)**, 0.914 (0.6), 0.942 (0.75),
0.976 (0.9). The grid minima agree to 4 decimals (notebook 04). With s = 0 the ratio is
2(1+ε)/(3+ε).

## Consequences

- ρ_SMD(ε) exceeds the fixed-menu bound min{ρ_A, ρ_B} for ε > 0.366: a genuinely
  non-dictatorial strategy-proof mechanism beats every menu mechanism there (0.900 vs 6/7 at
  ε = ½; 0.942 vs 28/31 at ε = ¾).
- The remaining gap to the LP upper bound (≈ 0.951 at ε = ½) is the price of this mechanism's
  coarseness: it uses only one threshold. A multi-threshold version with the modest-dictator
  fallback cannot help: against a strongly opposed opponent a type's utility is affine in the
  split probability with slope (1−s)(x − τ), so every type wants an extreme probability and only
  the two-tier structure is incentive compatible.
- The fallback menu can be optimised by an LP given the split region
  (`lp.solve_split_menu_lp`). Evaluated exactly on the continuum, this helps only for
  ε ≲ 0.45 (0.857 at ε = ¼, 0.885 at 0.4, N = 80); for ε ≥ ½ the closed-form SMD is at least
  as good, and at ε = 0.6 it is optimal within the split-or-menu class (LP value = ρ_SMD).
- For ε < 0.366 fixed menus (nested LIP / the SD-LP menus) remain the best known mechanisms.

## Why SMD is rigid (negative results)

- *Probabilistic split for moderately opposed profiles.* Let the split happen with probability
  p when the agents are on opposite sides but not both strong. Against a strongly opposed
  opponent a type x = ½⁺ gets p·x + (1−p)·[x s + ½(1−s)(1+ε)] from truth-telling, but
  ½[1 + ε(1−s)] from claiming the opponent's side (which switches the outcome to the pure
  fallback); the latter is larger for every p > 0. So p = 0 is forced: moderate types must be
  spared the split entirely.
- *Larger thresholds.* τ > (1+ε)/2 makes types in [(1+ε)/2, τ) prefer a split they cannot
  trigger, breaking IC, and lowers the ratio (scan in docs/log.md).
- *Richer fallbacks.* Nested LIP as fallback violates IC at the threshold (its fallback utility
  at τ exceeds τ). LP-optimised fallbacks help only for ε ≲ 0.45.

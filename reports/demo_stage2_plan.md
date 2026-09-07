# Stage 2 analysis plan (demonstration scale)

This file mirrors, at demonstration scale, the analysis plan that was fixed before the full-scale Stage 2 runs. The Stage 2 script requires a plan file so that the analysis specification always travels with the results.

## Specification

1. **Gap statistic.** For each scoring regime, gap = mean(metric | low-credibility) − mean(metric | high-credibility), computed for cascade size, audience reach, and time to peak.
2. **Cross-regime contrast.** gap(regime) − gap(additive), paired across replicates.
3. **Regimes.** additive (baseline), ablated (slow-gates-fast multiplicative), additive_retuned (reply weight halved), ratio_correction, reflective_floor.
4. **Scale.** This demonstration uses 5 replicates, 250 seeds per label, 2,000 users, and a 200-iteration cascade-level bootstrap. The full-scale plan uses 100 replicates, 2,500 seeds per label, 50,000 users, and a 1,000-iteration bootstrap.
5. **Decision rule (full scale).** On the bootstrap CI of the cascade-size contrast: ablated CI entirely below zero with the retuned CI overlapping zero supports the claim, an ablated CI overlapping zero is a null result, both CIs below zero calls for diagnosis, and an ablated CI entirely above zero is the opposite of the prediction.

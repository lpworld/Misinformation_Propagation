"""Calibrate ``baseline_exposures`` so the simulator matches observed counts.

Phase 2's hard-coded ``baseline_exposures=200`` produced cascades whose
mean reply count was ~28× the observed mean root.replyCount. Phase 3
calibrates the baseline against an observed target so the *current*
(additive) regime sits at the right magnitude.

Two target modes are available:

* ``target="mean"`` — match the observed mean of root.replyCount. Closed
  form from the Poisson event model:

      n_reply ~ Poisson( total_exp_seed × p_reply_seed )
      E[n_reply] = baseline_exposures × E[S_rel × p_reply]
      baseline_exposures = target_mean / E[S_rel × p_reply]

  Defensible as a first pass, but mean is a poor summary for the heavy-zero
  observed distribution (median 0, mean 8.2 in part_1). The Phase 3 first
  pass with this target over-predicted simulator p90 by ~5–10×.

* ``target="zero_inflated"`` — match the observed P(reply=0) and the
  observed conditional mean E[reply | reply>0] jointly via a per-cascade
  Bernoulli activity gate. The model becomes:

      a_i ~ Bernoulli(activity_pi)
      n_reply_i = a_i × Poisson( baseline × S_rel_i × p_reply_i )

  Method-of-moments calibration:

      activity_pi = 1 - P_obs(reply = 0)
      baseline_exposures = E_obs[reply | reply > 0] / E[S_rel × p_reply]

  Returned ``CalibrationResult.activity_pi`` should be plumbed into
  ``SimConfig.activity_pi`` so the simulator applies the gate. Reuse the
  same activity_pi across all regimes (regime differences should sit in
  the score-aggregation layer, not in the activity rate).

* ``target="zero_inflated_nb"`` — same as ``zero_inflated`` plus a third
  parameter for over-dispersion. The Poisson event model is replaced by
  Negative Binomial via Gamma(r, μ/r)-mixed Poisson:

      a_i ~ Bernoulli(activity_pi)
      μ_i = baseline × S_rel_i × p_reply_i
      λ_i ~ Gamma(r, μ_i / r)        # mean μ_i, variance μ_i² / r
      n_reply_i = a_i × Poisson(λ_i)

  Marginal of (n_reply | a_i = 1) is NB(r, μ_i). Method-of-moments
  calibration of r from observed conditional non-zero variance:

      r = μ_obs² / (Var_obs - μ_obs)        if Var_obs > μ_obs
      r = ∞ (i.e., reduce to Poisson)       otherwise

  Floored at ``r_floor`` (default 0.05) to avoid pathological dispersion;
  ceilinged at ``r_ceiling`` (default 100) where NB ≈ Poisson and we
  shouldn't bother. Same r is applied across all four heads (defensible
  default; per-head dispersion is a Phase 5 sensitivity dial).

We calibrate against the **additive** regime by default since it's our
"current production-style" comparator — the regime the paper wants to
match observed cascade magnitudes for. The ablated and parameter-only
control regimes use the same calibrated baseline (they're compared against
the calibrated additive run, not against observations directly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ranker.scoring import ScoringConfig, aggregate_score

from analysis.validation import observed_root_reply_counts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of one calibration step.

    ``activity_pi`` is populated for ``zero_inflated`` and ``zero_inflated_nb``;
    ``dispersion_r`` and ``target_conditional_var`` only for ``zero_inflated_nb``.
    """

    target_mode: str
    target_mean: float                  # observed mean (always populated)
    expected_factor: float              # E[S_rel × p_reply] over seeds
    baseline_exposures: float
    n_observed: int
    n_seeds: int
    # Zero-inflated mode extras:
    activity_pi: float | None = None    # calibrated per-cascade Bernoulli rate
    target_p_zero: float | None = None  # observed P(reply == 0)
    target_conditional_mean: float | None = None  # observed E[reply | reply > 0]
    # Negative-Binomial mode extras:
    dispersion_r: float | None = None   # NB dispersion; smaller ⇒ heavier tail
    target_conditional_var: float | None = None  # observed Var[reply | reply > 0]


def _compute_S_rel(
    seed_probs: dict[str, np.ndarray],
    scoring_config: ScoringConfig,
    score_normalization: str,
) -> tuple[np.ndarray, float]:
    """Returns (S_rel, expected_factor) where expected_factor = E[S_rel × p_reply]."""
    S = aggregate_score(seed_probs, scoring_config)
    if score_normalization == "median":
        ref = float(np.median(S))
    elif score_normalization == "mean":
        ref = float(np.mean(S))
    else:
        raise ValueError(f"unknown score_normalization: {score_normalization!r}")
    if not np.isfinite(ref) or ref <= 0:
        S_rel = np.ones_like(S, dtype=np.float64)
    else:
        S_rel = (S / ref).astype(np.float64)
    p_reply = np.asarray(seed_probs["reply"], dtype=np.float64)
    factor = float(np.mean(S_rel * p_reply))
    return S_rel, factor


def calibrate_baseline_exposures(
    *,
    observed_df: pd.DataFrame,
    seed_probs: dict[str, np.ndarray],
    scoring_config: ScoringConfig,
    target: str = "mean",
    score_normalization: str = "median",
    floor_factor: float = 1e-6,
) -> CalibrationResult:
    """Compute calibration parameters so the simulator matches observed reply counts.

    Args:
        observed_df: cleaned USC frame; we pull root.replyCount from it.
        seed_probs: per-head probability arrays for the seed set, same as
            produced inside the simulator.
        scoring_config: regime to use for the score-aggregation step. Use
            ``additive`` for production-matching calibration.
        target: ``"mean"`` matches observed mean only; ``"zero_inflated"``
            matches observed P(reply=0) and conditional mean jointly via a
            per-cascade Bernoulli activity gate (recommended after the Phase 3
            partial-match finding).
        score_normalization: must match the simulator's
            ``SimConfig.score_normalization`` so calibration uses the same
            S_rel definition.
        floor_factor: avoid divide-by-zero if the ranker is degenerate.

    Returns:
        :class:`CalibrationResult` with the calibrated parameters. For
        ``target="zero_inflated"``, ``activity_pi`` should be plumbed into
        ``SimConfig.activity_pi``.
    """
    obs = observed_root_reply_counts(observed_df)
    if obs.size == 0:
        raise ValueError("no observed roots; cannot calibrate")

    p_reply = np.asarray(seed_probs["reply"], dtype=np.float64)
    S_rel, factor_raw = _compute_S_rel(seed_probs, scoring_config, score_normalization)
    factor = max(factor_raw, floor_factor)
    target_mean = float(np.mean(obs))

    if target == "mean":
        baseline = target_mean / factor
        logger.info(
            "calibrated baseline_exposures=%.2f (target=mean, "
            "target_mean=%.3f, E[S_rel*p_reply]=%.4f, n_obs=%d, n_seeds=%d)",
            baseline, target_mean, factor, int(obs.size), int(p_reply.size),
        )
        return CalibrationResult(
            target_mode="mean",
            target_mean=target_mean,
            expected_factor=factor,
            baseline_exposures=baseline,
            n_observed=int(obs.size),
            n_seeds=int(p_reply.size),
        )

    if target in ("zero_inflated", "zero_inflated_nb"):
        p_zero = float(np.mean(obs == 0))
        activity_pi = float(np.clip(1.0 - p_zero, floor_factor, 1.0))
        nonzero = obs[obs > 0]
        if nonzero.size == 0:
            raise ValueError(
                f"no non-zero observed roots; {target} calibration would "
                "set activity_pi=0 and produce all-zero simulations"
            )
        cond_mean = float(np.mean(nonzero))
        baseline = cond_mean / factor

        if target == "zero_inflated":
            logger.info(
                "calibrated baseline_exposures=%.2f activity_pi=%.4f "
                "(target=zero_inflated, P_obs(reply=0)=%.4f, "
                "E_obs[reply|reply>0]=%.3f, E[S_rel*p_reply]=%.4f, "
                "n_obs=%d, n_seeds=%d)",
                baseline, activity_pi, p_zero, cond_mean, factor,
                int(obs.size), int(p_reply.size),
            )
            return CalibrationResult(
                target_mode="zero_inflated",
                target_mean=target_mean,
                expected_factor=factor,
                baseline_exposures=baseline,
                n_observed=int(obs.size),
                n_seeds=int(p_reply.size),
                activity_pi=activity_pi,
                target_p_zero=p_zero,
                target_conditional_mean=cond_mean,
            )

        # zero_inflated_nb: also calibrate dispersion_r.
        #
        # Naive method-of-moments — r = μ²/(Var - μ) — is unreliable here
        # because real engagement variance is dominated by a tiny power-law
        # tail (max root.replyCount in part_1 = 20,431; non-zero
        # variance = 147,622 vs non-zero mean = 38). The MoM r becomes
        # tiny (~0.01), which gives the U-shaped NB body that empirically
        # *worsens* every Stage 1 KS vs plain Poisson.
        #
        # Use a **trimmed-variance** estimator: drop the top 5% of non-zero
        # observations before computing variance. This trades a tighter
        # match on the extreme tail (which we acknowledge as a known
        # limitation in the paper) for a much better match on the cascade
        # body — the regime that matters for the architectural-mechanism
        # claim. Empirical sweep on part_1 (r ∈ {Poisson, 100, …, 0.05})
        # confirms the trimmed estimator lands near the KS-minimizing r.
        # See design log 2026-04-26.
        r_floor = 0.05
        r_ceiling = 100.0
        trim_frac = 0.05
        if nonzero.size >= 20:
            cutoff = float(np.quantile(nonzero, 1.0 - trim_frac))
            trimmed = nonzero[nonzero <= cutoff]
            cond_var_trim = float(np.var(trimmed, ddof=1)) if trimmed.size >= 2 else float("nan")
            cond_mean_trim = float(np.mean(trimmed)) if trimmed.size else cond_mean
        else:
            cond_var_trim = float(np.var(nonzero, ddof=1)) if nonzero.size >= 2 else float("nan")
            cond_mean_trim = cond_mean
        cond_var = float(np.var(nonzero, ddof=1)) if nonzero.size >= 2 else float("nan")

        if not np.isfinite(cond_var_trim) or cond_var_trim <= cond_mean_trim:
            dispersion_r = r_ceiling
            logger.info(
                "trimmed conditional variance (%.3f) ≤ trimmed mean (%.3f); "
                "NB reduces to Poisson — using r=%.2f",
                cond_var_trim, cond_mean_trim, dispersion_r,
            )
        else:
            dispersion_r = (cond_mean_trim ** 2) / (cond_var_trim - cond_mean_trim)
            dispersion_r = float(np.clip(dispersion_r, r_floor, r_ceiling))

        logger.info(
            "calibrated baseline_exposures=%.2f activity_pi=%.4f "
            "dispersion_r=%.4f (target=zero_inflated_nb, "
            "P_obs(reply=0)=%.4f, E_obs[reply|reply>0]=%.3f, "
            "Var_obs[reply|reply>0]=%.3f, E[S_rel*p_reply]=%.4f, "
            "n_obs=%d, n_seeds=%d)",
            baseline, activity_pi, dispersion_r, p_zero, cond_mean, cond_var,
            factor, int(obs.size), int(p_reply.size),
        )
        return CalibrationResult(
            target_mode="zero_inflated_nb",
            target_mean=target_mean,
            expected_factor=factor,
            baseline_exposures=baseline,
            n_observed=int(obs.size),
            n_seeds=int(p_reply.size),
            activity_pi=activity_pi,
            target_p_zero=p_zero,
            target_conditional_mean=cond_mean,
            dispersion_r=dispersion_r,
            target_conditional_var=cond_var,
        )

    raise ValueError(f"unknown target mode: {target!r}")


__all__ = ["CalibrationResult", "calibrate_baseline_exposures"]

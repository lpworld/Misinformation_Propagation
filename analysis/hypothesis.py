"""Stage 2 hypothesis test: differential propagation of low- vs high-credibility content.

Per the design specification §Two-stage analysis, the **primary metric** is the
**cascade-size gap**: mean(size | low_credibility) − mean(size | high_credibility),
computed *per regime*, then compared across regimes. Secondary metrics:
audience-reach gap, time-to-peak gap. All three are gap-of-means contrasts.

The architectural-mechanism claim predicts:

    gap_ablated  <  gap_additive            (the ablation closes the gap)
    gap_additive_retuned  ≈  gap_additive   (parameter tuning alone doesn't)

Pre-registered. See ``paper/phase4_preregistration.md`` for the locked-in
analysis plan.

Definitions (cf. the design specification):

* **size**: per cascade, ``n_reply + n_retweet + n_deep`` — the count-based
  primitive (deep = quote proxy in our schema). likeCount is excluded
  per the formula in the design specification.
* **audience-reach**: simulator-side proxy = ``total_exposures``. The
  observed-side definition (Σ unique-author follower counts) is not
  directly recoverable in the simulator since we don't track which users
  from the pool engage; total_exposures is the algorithmic-fan-out analog
  and the right contrast for the architectural claim. Documented as a
  proxy in the paper.
* **time-to-peak**: per cascade, the t_hour with maximum n_reply (same
  function used in Stage 1).

Uncertainty: replicate-level CIs from running the simulator with multiple
RNG seeds (each replicate produces one gap per regime; we take the mean
and percentile CI across replicates). This captures Monte-Carlo error
from the simulator's stochasticity. Cascade-level bootstrap is *not*
layered on top — replicates already vary the per-cascade event sampling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOW = "low_credibility"
HIGH = "high_credibility"


# ---- per-cascade metrics ------------------------------------------------

def cascade_size(per_cascade: pd.DataFrame) -> np.ndarray:
    """``n_reply + n_retweet + n_deep`` per cascade — the count-based size primitive."""
    return (
        per_cascade["n_reply"]
        + per_cascade["n_retweet"]
        + per_cascade["n_deep"]
    ).to_numpy(dtype=np.float64)


def audience_reach(per_cascade: pd.DataFrame) -> np.ndarray:
    """Algorithmic-fan-out proxy for audience reach: ``total_exposures``."""
    return per_cascade["total_exposures"].to_numpy(dtype=np.float64)


def time_to_peak_per_cascade(per_bin: pd.DataFrame) -> pd.DataFrame:
    """Per-cascade peak hour; columns ``[regime, replicate, cascade_id, t_peak]``."""
    if per_bin.empty:
        return pd.DataFrame(columns=["regime", "replicate", "cascade_id", "t_peak"])
    grouped = per_bin.loc[
        per_bin.groupby(
            ["regime", "replicate", "cascade_id"], sort=False
        )["n_reply"].idxmax(),
        ["regime", "replicate", "cascade_id", "t_hour"],
    ].rename(columns={"t_hour": "t_peak"})
    return grouped.reset_index(drop=True)


# ---- gap computation ----------------------------------------------------

@dataclass(frozen=True)
class RegimeGap:
    """Replicate-level gap (low − high) under one regime, one metric."""

    regime: str
    metric: str
    n_low_per_replicate: int
    n_high_per_replicate: int
    n_replicates: int
    per_replicate_low_mean: np.ndarray   # length n_replicates
    per_replicate_high_mean: np.ndarray
    per_replicate_gap: np.ndarray        # low_mean - high_mean per replicate
    gap_mean: float
    gap_ci_lo: float                     # 2.5th percentile
    gap_ci_hi: float                     # 97.5th percentile


@dataclass(frozen=True)
class CrossRegimeContrast:
    """Difference-of-gaps between two regimes."""

    metric: str
    regime_a: str                        # typically the baseline (additive)
    regime_b: str                        # typically the comparator (ablated / retuned)
    per_replicate_diff: np.ndarray       # gap_b - gap_a per replicate
    diff_mean: float
    diff_ci_lo: float
    diff_ci_hi: float


@dataclass(frozen=True)
class BootstrapGap:
    """Cascade-level percentile-bootstrap CI on the gap (low − high) per regime.

    Uncertainty source: which cascades happened to be sampled. Per-cascade
    outcome is the mean across replicates (averages out RNG noise), so the
    bootstrap captures *seed-sampling* uncertainty rather than RNG noise.
    For deterministic quantities (audience_reach), this is the only
    meaningful CI; for stochastic ones (cascade_size, time_to_peak), it
    complements the replicate-level CI.
    """

    metric: str
    regime: str
    n_low: int
    n_high: int
    n_bootstrap: int
    gap_point: float                     # mean over actual cascades (no resampling)
    gap_ci_lo: float
    gap_ci_hi: float


@dataclass(frozen=True)
class BootstrapContrast:
    """Paired cascade-level bootstrap CI on (gap_b − gap_a).

    Pairing: the same bootstrap resample of cascade_ids is applied to both
    regimes (the harness uses the same seeds across regimes, so each
    cascade_id has well-defined outcomes under both). This gives the
    cleanest comparison of what the architectural change does to the same
    underlying content.
    """

    metric: str
    regime_a: str
    regime_b: str
    n_low: int
    n_high: int
    n_bootstrap: int
    diff_point: float                    # contrast on actual cascades
    diff_ci_lo: float
    diff_ci_hi: float


def _percentile_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.percentile(x, 100 * alpha / 2))
    hi = float(np.percentile(x, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_regime_gap(
    per_cascade: pd.DataFrame,
    *,
    regime: str,
    metric_name: str,
    metric_values: np.ndarray,
) -> RegimeGap:
    """Compute the low-minus-high gap, broken out by replicate.

    ``per_cascade`` must contain ``regime``, ``replicate``, and a
    ``credibility_label`` column with values ``LOW`` / ``HIGH`` (others
    are ignored).
    """
    df = per_cascade.copy()
    df["_metric"] = metric_values
    df = df[df["regime"] == regime]
    df = df[df["credibility_label"].isin([LOW, HIGH])]

    per_rep = df.groupby(["replicate", "credibility_label"])["_metric"].mean().unstack()
    per_rep = per_rep.reindex(columns=[LOW, HIGH])
    if per_rep.isna().any().any():
        missing = per_rep.isna().sum().to_dict()
        raise ValueError(
            f"regime={regime!r} metric={metric_name!r}: a replicate is missing "
            f"low or high cascades — counts {missing}. Check stratified seed sampling."
        )

    rep_low = per_rep[LOW].to_numpy(dtype=np.float64)
    rep_high = per_rep[HIGH].to_numpy(dtype=np.float64)
    rep_gap = rep_low - rep_high
    ci_lo, ci_hi = _percentile_ci(rep_gap)

    n_low = int(df[df["credibility_label"] == LOW].groupby("replicate").size().median())
    n_high = int(df[df["credibility_label"] == HIGH].groupby("replicate").size().median())

    return RegimeGap(
        regime=regime,
        metric=metric_name,
        n_low_per_replicate=n_low,
        n_high_per_replicate=n_high,
        n_replicates=int(rep_gap.size),
        per_replicate_low_mean=rep_low,
        per_replicate_high_mean=rep_high,
        per_replicate_gap=rep_gap,
        gap_mean=float(np.mean(rep_gap)),
        gap_ci_lo=ci_lo,
        gap_ci_hi=ci_hi,
    )


def cross_regime_contrast(
    gap_a: RegimeGap,
    gap_b: RegimeGap,
) -> CrossRegimeContrast:
    """Paired difference of regime gaps, replicate by replicate.

    Replicates use the same seeds in the harness (just different RNG for
    event sampling), so the difference is paired.
    """
    if gap_a.metric != gap_b.metric:
        raise ValueError(f"metric mismatch: {gap_a.metric} vs {gap_b.metric}")
    if gap_a.n_replicates != gap_b.n_replicates:
        raise ValueError(
            f"replicate-count mismatch: {gap_a.n_replicates} vs {gap_b.n_replicates}"
        )
    diff = gap_b.per_replicate_gap - gap_a.per_replicate_gap
    ci_lo, ci_hi = _percentile_ci(diff)
    return CrossRegimeContrast(
        metric=gap_a.metric,
        regime_a=gap_a.regime,
        regime_b=gap_b.regime,
        per_replicate_diff=diff,
        diff_mean=float(np.mean(diff)),
        diff_ci_lo=ci_lo,
        diff_ci_hi=ci_hi,
    )


# ---- cascade-level bootstrap --------------------------------------------

def _replicate_mean_per_cascade(
    per_cascade: pd.DataFrame,
    metric_values: np.ndarray,
) -> pd.DataFrame:
    """Average ``metric_values`` across replicates for each (regime, cascade_id).

    Returns a long DataFrame with columns
    ``regime, cascade_id, credibility_label, value`` where ``value`` is
    the per-cascade mean across replicates.
    """
    work = per_cascade[["regime", "replicate", "cascade_id", "credibility_label"]].copy()
    work["_v"] = metric_values
    out = (
        work.groupby(["regime", "cascade_id", "credibility_label"], sort=False)["_v"]
        .mean()
        .reset_index()
        .rename(columns={"_v": "value"})
    )
    return out


def _wide_per_regime(
    repl_mean: pd.DataFrame, regimes: tuple[str, ...]
) -> pd.DataFrame:
    """Pivot to one row per cascade_id with one column per regime.

    Drops cascades that don't have an entry under every regime (shouldn't
    happen given the harness, but defensive).
    """
    wide = repl_mean.pivot_table(
        index=["cascade_id", "credibility_label"],
        columns="regime",
        values="value",
    )
    wide = wide.dropna(subset=list(regimes))
    return wide.reset_index()


def cascade_bootstrap_gap(
    repl_mean_wide: pd.DataFrame,
    *,
    regime: str,
    metric_name: str,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> BootstrapGap:
    """Percentile bootstrap CI on (mean_low − mean_high) for one regime."""
    rng = rng or np.random.default_rng(1337)
    low = repl_mean_wide[repl_mean_wide["credibility_label"] == LOW][regime].to_numpy(dtype=np.float64)
    high = repl_mean_wide[repl_mean_wide["credibility_label"] == HIGH][regime].to_numpy(dtype=np.float64)
    if low.size == 0 or high.size == 0:
        raise ValueError(f"empty {regime}/{metric_name} pool: low={low.size} high={high.size}")

    point = float(low.mean() - high.mean())
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        s_low = rng.choice(low, size=low.size, replace=True)
        s_high = rng.choice(high, size=high.size, replace=True)
        boots[b] = s_low.mean() - s_high.mean()
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    return BootstrapGap(
        metric=metric_name, regime=regime,
        n_low=int(low.size), n_high=int(high.size),
        n_bootstrap=n_bootstrap,
        gap_point=point, gap_ci_lo=ci_lo, gap_ci_hi=ci_hi,
    )


def cascade_bootstrap_contrast(
    repl_mean_wide: pd.DataFrame,
    *,
    regime_a: str,
    regime_b: str,
    metric_name: str,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> BootstrapContrast:
    """Paired cascade-level bootstrap CI on (gap_b − gap_a).

    The same resample of cascade_ids is applied to both regimes — this
    is the right design because the harness uses the same seeds across
    regimes, so each cascade has a well-defined paired outcome.
    """
    rng = rng or np.random.default_rng(1337)
    low_mask = repl_mean_wide["credibility_label"] == LOW
    high_mask = repl_mean_wide["credibility_label"] == HIGH
    a_low = repl_mean_wide.loc[low_mask, regime_a].to_numpy(dtype=np.float64)
    a_high = repl_mean_wide.loc[high_mask, regime_a].to_numpy(dtype=np.float64)
    b_low = repl_mean_wide.loc[low_mask, regime_b].to_numpy(dtype=np.float64)
    b_high = repl_mean_wide.loc[high_mask, regime_b].to_numpy(dtype=np.float64)
    if a_low.size != b_low.size or a_high.size != b_high.size:
        raise ValueError("paired bootstrap requires same cascade_ids per regime")

    gap_a_point = float(a_low.mean() - a_high.mean())
    gap_b_point = float(b_low.mean() - b_high.mean())
    point = gap_b_point - gap_a_point

    n_low = a_low.size
    n_high = a_high.size
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx_low = rng.integers(0, n_low, size=n_low)
        idx_high = rng.integers(0, n_high, size=n_high)
        gap_a = a_low[idx_low].mean() - a_high[idx_high].mean()
        gap_b = b_low[idx_low].mean() - b_high[idx_high].mean()
        boots[b] = gap_b - gap_a
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    return BootstrapContrast(
        metric=metric_name, regime_a=regime_a, regime_b=regime_b,
        n_low=int(n_low), n_high=int(n_high),
        n_bootstrap=n_bootstrap,
        diff_point=point, diff_ci_lo=ci_lo, diff_ci_hi=ci_hi,
    )


# ---- one-call orchestrator ----------------------------------------------

def run_stage2(
    per_cascade: pd.DataFrame,
    per_bin: pd.DataFrame,
    *,
    regimes: tuple[str, ...],
    baseline_regime: str = "additive",
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 1337,
) -> dict:
    """Compute the full Stage 2 table: gap per (metric, regime) plus contrasts.

    Two uncertainty layers:

    * **Replicate-level CI** (per pre-registration): variance across
      simulator RNG seeds with fixed cascade pool. Captures Monte-Carlo
      error from event sampling.
    * **Cascade-level bootstrap CI** (methodology amendment 2026-04-26):
      replicate-mean per cascade, then bootstrap over cascade_ids with
      paired resampling across regimes. Captures seed-sampling
      uncertainty. For deterministic metrics like ``audience_reach``,
      this is the only meaningful CI source.

    Returns:
        dict with keys:
            "gaps":               {metric: {regime: RegimeGap}}
            "contrasts":          {metric: {regime: CrossRegimeContrast}}
            "bootstrap_gaps":     {metric: {regime: BootstrapGap}}
            "bootstrap_contrasts":{metric: {regime: BootstrapContrast}}
    """
    if baseline_regime not in regimes:
        raise ValueError(
            f"baseline_regime {baseline_regime!r} not in regimes {regimes!r}"
        )

    # per-cascade metrics (vectorized once, then sliced per regime)
    size = cascade_size(per_cascade)
    reach = audience_reach(per_cascade)

    # time-to-peak comes from per_bin; left-join onto per_cascade
    t2p_df = time_to_peak_per_cascade(per_bin)
    pc = per_cascade.merge(
        t2p_df, on=["regime", "replicate", "cascade_id"], how="left"
    )
    t2p = pc["t_peak"].fillna(0.0).to_numpy(dtype=np.float64)

    metric_arrays = {
        "cascade_size": size,
        "audience_reach": reach,
        "time_to_peak": t2p,
    }

    gaps: dict[str, dict[str, RegimeGap]] = {}
    for metric_name, vals in metric_arrays.items():
        gaps[metric_name] = {}
        for regime in regimes:
            gaps[metric_name][regime] = compute_regime_gap(
                per_cascade, regime=regime,
                metric_name=metric_name, metric_values=vals,
            )

    contrasts: dict[str, dict[str, CrossRegimeContrast]] = {}
    for metric_name in metric_arrays:
        contrasts[metric_name] = {}
        for regime in regimes:
            if regime == baseline_regime:
                continue
            contrasts[metric_name][regime] = cross_regime_contrast(
                gaps[metric_name][baseline_regime],
                gaps[metric_name][regime],
            )

    # Cascade-level bootstrap: pool replicates, bootstrap over cascade_ids
    # with paired resampling for cross-regime contrasts.
    rng_b = np.random.default_rng(bootstrap_seed)
    bs_gaps: dict[str, dict[str, BootstrapGap]] = {}
    bs_contrasts: dict[str, dict[str, BootstrapContrast]] = {}
    for metric_name, vals in metric_arrays.items():
        repl_mean_long = _replicate_mean_per_cascade(pc, vals)
        wide = _wide_per_regime(repl_mean_long, regimes=regimes)
        bs_gaps[metric_name] = {}
        for regime in regimes:
            bs_gaps[metric_name][regime] = cascade_bootstrap_gap(
                wide, regime=regime, metric_name=metric_name,
                n_bootstrap=n_bootstrap, rng=rng_b,
            )
        bs_contrasts[metric_name] = {}
        for regime in regimes:
            if regime == baseline_regime:
                continue
            bs_contrasts[metric_name][regime] = cascade_bootstrap_contrast(
                wide, regime_a=baseline_regime, regime_b=regime,
                metric_name=metric_name, n_bootstrap=n_bootstrap, rng=rng_b,
            )

    return {
        "gaps": gaps,
        "contrasts": contrasts,
        "bootstrap_gaps": bs_gaps,
        "bootstrap_contrasts": bs_contrasts,
    }


__all__ = [
    "HIGH",
    "LOW",
    "BootstrapContrast",
    "BootstrapGap",
    "CrossRegimeContrast",
    "RegimeGap",
    "audience_reach",
    "cascade_bootstrap_contrast",
    "cascade_bootstrap_gap",
    "cascade_size",
    "compute_regime_gap",
    "cross_regime_contrast",
    "run_stage2",
    "time_to_peak_per_cascade",
]

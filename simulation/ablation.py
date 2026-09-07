"""Multi-regime × multi-replicate simulation harness.

Phase 4 needs to run the simulator multiple times — once per (scoring_regime,
replicate_seed) — to produce CIs over the simulator's stochasticity. This
module is a thin wrapper that loops over a regime grid and a replicate
grid, calling :func:`simulation.cascade.simulate_cascades` and concatenating
the per-cascade outputs with regime + replicate id columns.

Output is a long-form DataFrame with one row per (regime × replicate ×
cascade), suitable for pandas groupby aggregation in
:mod:`analysis.hypothesis`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch import nn

from ranker.features import FeatureScaler
from ranker.scoring import ScoringConfig
from simulation.cascade import SimConfig, simulate_cascades

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Combined per-cascade outputs across all (regime × replicate) runs."""
    per_cascade: pd.DataFrame  # one row per (regime, replicate, cascade)
    per_bin: pd.DataFrame      # one row per (regime, replicate, cascade, t_hour)


def run_ablation(
    *,
    seeds: pd.DataFrame,
    user_pool: pd.DataFrame,
    ranker: nn.Module,
    scaler: FeatureScaler,
    scoring_configs: dict[str, ScoringConfig],
    sim_config: SimConfig,
    n_replicates: int,
    diurnal_weights: np.ndarray | None = None,
    seed_offsets: tuple[int, ...] | None = None,
    score_multipliers: dict[str, np.ndarray] | None = None,
) -> AblationResult:
    """Run simulate_cascades across (regime × replicate) and concatenate outputs.

    Args:
        seeds: cascade-seed DataFrame; reused across all runs.
        user_pool: user-pool DataFrame; reused across all runs.
        ranker, scaler: trained ranker artifacts.
        scoring_configs: dict of ``regime_name -> ScoringConfig``; one
            simulator pass per regime.
        sim_config: base SimConfig; ``seed`` is overridden per replicate.
        n_replicates: number of independent simulator runs per regime
            (different RNG per replicate, same data).
        diurnal_weights: optional length-24 hour weights, passed through to
            the simulator unchanged.
        seed_offsets: optional explicit replicate seeds (length n_replicates).
            If None, uses ``sim_config.seed + i`` for i in [0, n_replicates).
        score_multipliers: optional ``regime_name -> per-seed multiplier``
            arrays, passed through to
            :func:`simulation.cascade.simulate_cascades` as
            ``score_multiplier`` for that regime only. Regimes absent from
            the dict run unmodified. Used by the composed content-gate
            experiments.

    Returns:
        :class:`AblationResult` with columns ``regime`` and ``replicate``
        prepended to the per_cascade and per_bin frames.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")

    if seed_offsets is None:
        seed_offsets = tuple(sim_config.seed + i for i in range(n_replicates))
    elif len(seed_offsets) != n_replicates:
        raise ValueError(
            f"seed_offsets has length {len(seed_offsets)}, expected {n_replicates}"
        )

    per_cascade_chunks: list[pd.DataFrame] = []
    per_bin_chunks: list[pd.DataFrame] = []

    for regime_name, sc in scoring_configs.items():
        for rep_idx, rep_seed in enumerate(seed_offsets):
            cfg_rep = SimConfig(
                n_time_bins=sim_config.n_time_bins,
                decay_tau_hours=sim_config.decay_tau_hours,
                baseline_exposures=sim_config.baseline_exposures,
                exposure_min=sim_config.exposure_min,
                exposure_max=sim_config.exposure_max,
                score_normalization=sim_config.score_normalization,
                use_circadian=sim_config.use_circadian,
                activity_pi=sim_config.activity_pi,
                dispersion_r=sim_config.dispersion_r,
                seed=int(rep_seed),
            )
            r = simulate_cascades(
                seeds=seeds,
                user_pool=user_pool,
                ranker=ranker,
                scaler=scaler,
                scoring_config=sc,
                sim_config=cfg_rep,
                diurnal_weights=diurnal_weights,
                score_multiplier=(
                    None if score_multipliers is None
                    else score_multipliers.get(regime_name)
                ),
            )
            pc = r.per_cascade.copy()
            pc.insert(0, "replicate", rep_idx)
            pc.insert(0, "regime", regime_name)
            per_cascade_chunks.append(pc)

            pb = r.per_bin.copy()
            pb.insert(0, "replicate", rep_idx)
            pb.insert(0, "regime", regime_name)
            per_bin_chunks.append(pb)

            logger.info(
                "ablation: regime=%s replicate=%d/%d (seed=%d) — n_cascades=%d",
                regime_name, rep_idx + 1, n_replicates, rep_seed, len(r.per_cascade),
            )

    per_cascade = pd.concat(per_cascade_chunks, ignore_index=True)
    per_bin = pd.concat(per_bin_chunks, ignore_index=True)
    return AblationResult(per_cascade=per_cascade, per_bin=per_bin)


__all__ = ["AblationResult", "run_ablation"]

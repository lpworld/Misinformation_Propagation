"""Cascade simulator (Phase 3).

Phase 3 changes vs Phase 2 MVP:

- **Generic ranker.** Accepts any ``nn.Module`` whose ``predict_proba``
  returns a ``{head: tensor}`` dict (works for both ``HeavyRankerStub`` and
  ``MaskNetRanker``). Inference runs on whatever device the model is on.
- **Circadian exposure.** Hourly weights modulate the exposure schedule;
  cascades posted at low-activity hours get less prompt fan-out. Diurnal
  weights are fit from the observed corpus by
  :func:`fit_diurnal_weights`.
- **Vectorized cascade generation.** Builds the full
  ``(n_seeds, n_bins, n_heads)`` Poisson rate tensor in one numpy op
  instead of a Python for-loop over seeds. ~50× speedup at 5K cascades.
- **Calibrated baseline_exposures.** Phase 3 driver feeds in a calibrated
  ``baseline_exposures`` from :mod:`simulation.calibration`; it's no longer
  hard-coded.

Output is still per-cascade engagement-count time series — count-based
primitives, not reply trees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from ranker.architecture import HEAD_NAMES
from ranker.features import FeatureScaler, extract_features
from ranker.scoring import ScoringConfig, aggregate_score

logger = logging.getLogger(__name__)


# ---- diurnal weight fitting ---------------------------------------------

def fit_diurnal_weights(df: pd.DataFrame, *, normalize: bool = True) -> np.ndarray:
    """Empirical hour-of-day activity weights from observed tweet timestamps.

    Returns a length-24 array of weights, one per UTC hour bucket. By default
    normalized so weights average to 1.0 (so multiplying by these weights
    leaves the daily-mean exposure rate unchanged).

    The weights are based on tweet *posting* times — a coarse proxy for when
    the audience is active, but the right shape for a Phase 3 stand-in.
    Phase 5 sensitivity work could refine this with reply-time bucketing
    instead.
    """
    if "epoch" not in df.columns:
        raise ValueError("df missing 'epoch' column")
    epoch = pd.to_numeric(df["epoch"], errors="coerce").dropna().to_numpy()
    if epoch.size == 0:
        return np.ones(24, dtype=np.float64)
    hours = ((epoch / 3600.0) % 24.0).astype(int)
    counts = np.bincount(hours, minlength=24).astype(np.float64)
    if counts.sum() <= 0:
        return np.ones(24, dtype=np.float64)
    if normalize:
        counts = counts * (24.0 / counts.sum())  # mean = 1.0
    return counts


def uniform_diurnal_weights() -> np.ndarray:
    """Constant weights — useful when comparing against a no-circadian baseline."""
    return np.ones(24, dtype=np.float64)


# ---- config ------------------------------------------------------------

@dataclass(frozen=True)
class SimConfig:
    """Knobs for the simulator."""

    n_time_bins: int = 24                 # hourly bins
    decay_tau_hours: float = 6.0          # exposure decay time-constant
    baseline_exposures: float = 200.0     # mean exposures per cascade at S_rel=1
    exposure_min: float = 0.0             # floor on per-cascade total exposures
    exposure_max: float = 50_000.0        # ceiling
    score_normalization: str = "median"   # {"median", "mean"} for S_rel
    use_circadian: bool = True            # multiply schedule by diurnal weights
    # Per-cascade activity gate: a_i ~ Bernoulli(activity_pi). If None, every
    # cascade is active (Phase 2 behavior). When set, inactive cascades have
    # all engagement counts zeroed before output — addresses the heavy-zero
    # observed distribution that pure Poisson with a calibrated mean can't
    # reproduce. See design log 2026-04-26.
    activity_pi: float | None = None
    # Dispersion parameter for the Negative-Binomial event model. None →
    # per-bin Poisson (Phase 2 behavior). Float r > 0 → cascade-total drawn
    # from NB(r, μ) via Gamma(r, μ/r)-mixed Poisson, then split across bins
    # in proportion to the schedule. Lower r ⇒ heavier tail; r → ∞ recovers
    # Poisson. Calibrate from observed Var/Mean of the conditional non-zero
    # reply distribution.
    dispersion_r: float | None = None
    seed: int = 1337


def _normalize_scores(S: np.ndarray, mode: str) -> np.ndarray:
    """Compute relative scores S_rel for exposure scaling."""
    if mode == "median":
        ref = float(np.median(S))
    elif mode == "mean":
        ref = float(np.mean(S))
    else:
        raise ValueError(f"unknown score_normalization mode: {mode!r}")
    if not np.isfinite(ref) or ref <= 0:
        return np.ones_like(S, dtype=np.float64)
    return S / ref


def _build_schedule(
    total_exp: np.ndarray,        # (n_seeds,)
    seed_hours: np.ndarray,       # (n_seeds,) UTC hour-of-day at posting time
    n_bins: int,
    tau: float,
    diurnal_weights: np.ndarray,  # (24,) mean-1 weights
    use_circadian: bool,
) -> np.ndarray:
    """Build the (n_seeds, n_bins) exposure schedule matrix.

    For each seed, time-bin t corresponds to UTC hour ``(seed_hour + t) % 24``.
    Weights are decay × diurnal[that hour]; normalized per-seed to sum to 1
    before being scaled by ``total_exp``.
    """
    t = np.arange(n_bins, dtype=np.float64)
    decay = np.exp(-t / max(tau, 1e-6))                          # (n_bins,)

    if use_circadian:
        # (n_seeds, n_bins) hour grid
        seed_h = seed_hours.astype(int)[:, None]                  # (n_seeds, 1)
        bin_t = t.astype(int)[None, :]                            # (1, n_bins)
        hour_grid = (seed_h + bin_t) % 24                         # (n_seeds, n_bins)
        diurnal = diurnal_weights[hour_grid]                      # (n_seeds, n_bins)
        weights = decay[None, :] * diurnal
    else:
        weights = np.broadcast_to(decay[None, :], (total_exp.shape[0], n_bins)).copy()

    weights /= weights.sum(axis=1, keepdims=True)                  # per-seed normalize
    return weights * total_exp[:, None]                            # (n_seeds, n_bins)


# ---- result ------------------------------------------------------------

@dataclass
class SimResult:
    """Output of one simulator run.

    Attributes:
        per_bin: long-form (cascade_id × t_hour × head) DataFrame.
        per_cascade: one row per cascade with totals + the seed's score,
                 head probabilities, and exposure plan.
        regime: name of the ScoringConfig.regime used.
    """
    per_bin: pd.DataFrame
    per_cascade: pd.DataFrame
    regime: str


# ---- ranker inference --------------------------------------------------

def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _seed_probs(
    seeds: pd.DataFrame,
    ranker: nn.Module,
    scaler: FeatureScaler,
    *,
    batch_size: int = 16_384,
) -> dict[str, np.ndarray]:
    """Run the ranker over the seeds, batched, and return a probabilities dict."""
    X_raw = extract_features(seeds)
    X = scaler.transform(X_raw)
    device = _model_device(ranker)

    out: dict[str, list[np.ndarray]] = {k: [] for k in HEAD_NAMES}
    with torch.no_grad():
        ranker.eval()
        for start in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).float().to(device)
            probs = ranker.predict_proba(xb)
            for k in HEAD_NAMES:
                out[k].append(probs[k].cpu().numpy().astype(np.float64))
    return {k: np.concatenate(v) for k, v in out.items()}


# ---- main entry point --------------------------------------------------

def simulate_cascades(
    seeds: pd.DataFrame,
    user_pool: pd.DataFrame,  # noqa: ARG001 (still not consumed; see Phase 3 limitations)
    ranker: nn.Module,
    scaler: FeatureScaler,
    scoring_config: ScoringConfig,
    sim_config: SimConfig | None = None,
    *,
    diurnal_weights: np.ndarray | None = None,
    score_multiplier: np.ndarray | None = None,
) -> SimResult:
    """Run the cascade simulator over a batch of seeds under one scoring regime.

    ``diurnal_weights`` is a length-24 array (UTC hour → relative weight,
    mean ≈ 1). If None and ``sim_config.use_circadian`` is True, falls back
    to uniform weights.

    ``score_multiplier`` is an optional per-seed multiplicative factor applied
    to the aggregate score *before* normalization (S ← S × m). Used by the
    composed content-gate experiments (S = S_form · g^η), where the gate g is
    computed from seed text outside the engagement-scoring layer. None (the
    default) leaves all regimes byte-identical to prior behavior.
    """
    sim_config = sim_config or SimConfig()
    rng = np.random.default_rng(sim_config.seed)

    n_seeds = len(seeds)
    if n_seeds == 0:
        raise ValueError("seeds is empty")

    probs = _seed_probs(seeds, ranker, scaler)
    S = aggregate_score(probs, scoring_config)
    if score_multiplier is not None:
        m = np.asarray(score_multiplier, dtype=np.float64).ravel()
        if m.shape[0] != n_seeds:
            raise ValueError(
                f"score_multiplier has length {m.shape[0]}, expected {n_seeds}"
            )
        S = S * m
    S_rel = _normalize_scores(S, sim_config.score_normalization)

    total_exp = np.clip(
        sim_config.baseline_exposures * S_rel,
        sim_config.exposure_min,
        sim_config.exposure_max,
    ).astype(np.float64)

    seed_epoch = pd.to_numeric(seeds["epoch"], errors="coerce").fillna(0.0).to_numpy()
    seed_hours = (seed_epoch / 3600.0) % 24.0

    if diurnal_weights is None:
        diurnal_weights = uniform_diurnal_weights()
    elif diurnal_weights.shape != (24,):
        raise ValueError("diurnal_weights must be length 24")

    schedule = _build_schedule(
        total_exp=total_exp,
        seed_hours=seed_hours,
        n_bins=sim_config.n_time_bins,
        tau=sim_config.decay_tau_hours,
        diurnal_weights=np.asarray(diurnal_weights, dtype=np.float64),
        use_circadian=sim_config.use_circadian,
    )  # (n_seeds, n_bins)

    # Per-head event sampling. Two paths:
    #
    # (a) dispersion_r is None → per-bin Poisson with rate = schedule × p_head.
    #     Cascade-total = sum of per-bin Poissons = Poisson(total_exp × p_head).
    #     This is the Phase 2 / first-pass behavior — thin tails.
    #
    # (b) dispersion_r is a float > 0 → per-cascade total drawn from
    #     NB(r, μ) via Gamma(r, μ/r)-mixed Poisson, then split across bins
    #     in proportion to the schedule. Marginal cascade total is exactly
    #     NB(r, μ); per-bin counts are conditionally Poisson at proportional
    #     rates. Heavy tail when r is small.
    n_bins = sim_config.n_time_bins
    counts: dict[str, np.ndarray] = {}

    use_nb = sim_config.dispersion_r is not None
    if use_nb:
        r = float(sim_config.dispersion_r)
        if r <= 0:
            raise ValueError(f"dispersion_r must be positive, got {r}")
        # schedule rows sum to total_exp by construction (see _build_schedule).
        # Guard against zero-exposure rows when normalizing for the multinomial-
        # equivalent split.
        denom = total_exp.copy()
        denom[denom <= 0] = 1.0  # avoid /0 — those rows contribute zero counts anyway
        schedule_norm = schedule / denom[:, None]                 # (n_seeds, n_bins), rows sum to 1 (or 0 for inactive)

    for k in HEAD_NAMES:
        if not use_nb:
            lam = schedule * probs[k][:, None]                    # (n_seeds, n_bins)
            counts[k] = rng.poisson(lam).astype(np.int64)
        else:
            mu_total = total_exp * probs[k]                       # (n_seeds,)
            lambda_total = np.zeros(n_seeds, dtype=np.float64)
            mask = mu_total > 0
            if mask.any():
                # Gamma(shape=r, scale=μ/r): mean μ, variance μ²/r.
                lambda_total[mask] = rng.gamma(
                    shape=r, scale=mu_total[mask] / r,
                )
            lambda_per_bin = lambda_total[:, None] * schedule_norm
            counts[k] = rng.poisson(lambda_per_bin).astype(np.int64)

    # Per-cascade activity gate. Inactive cascades are zeroed across all bins
    # and all heads, producing structural zeros rather than Poisson zeros.
    if sim_config.activity_pi is not None:
        if not 0.0 <= sim_config.activity_pi <= 1.0:
            raise ValueError(
                f"activity_pi must be in [0, 1], got {sim_config.activity_pi}"
            )
        is_active = (rng.random(n_seeds) < sim_config.activity_pi).astype(np.int64)
        for k in HEAD_NAMES:
            counts[k] = counts[k] * is_active[:, None]
    else:
        is_active = np.ones(n_seeds, dtype=np.int64)

    # Build long-form per_bin.
    cascade_axis = np.repeat(np.arange(n_seeds, dtype=np.int64), n_bins)
    bin_axis_full = np.tile(np.arange(n_bins, dtype=np.int64), n_seeds)
    per_bin = pd.DataFrame({
        "cascade_id": cascade_axis,
        "t_hour": bin_axis_full,
        **{f"n_{k}": counts[k].reshape(-1) for k in HEAD_NAMES},
    })

    # Per-cascade summary.
    totals = {f"n_{k}": counts[k].sum(axis=1) for k in HEAD_NAMES}
    per_cascade = pd.DataFrame({
        "cascade_id": np.arange(n_seeds, dtype=np.int64),
        **totals,
        "score": S,
        "score_rel": S_rel,
        "total_exposures": total_exp,
        "is_active": is_active,
        **{f"p_{k}": probs[k] for k in HEAD_NAMES},
    })

    logger.info(
        "simulated %d cascades under regime=%s: median total reply=%d retweet=%d like=%d",
        n_seeds, scoring_config.regime,
        int(np.median(counts["reply"].sum(axis=1))),
        int(np.median(counts["retweet"].sum(axis=1))),
        int(np.median(counts["like"].sum(axis=1))),
    )
    return SimResult(per_bin=per_bin, per_cascade=per_cascade, regime=scoring_config.regime)


__all__ = [
    "SimConfig",
    "SimResult",
    "fit_diurnal_weights",
    "simulate_cascades",
    "uniform_diurnal_weights",
]

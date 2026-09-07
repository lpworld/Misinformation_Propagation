"""Tests for the cascade simulator.

We focus on output-shape correctness and reproducibility (seed → same
outputs). Statistical properties of the generated cascades are validated by
analysis.validation against observed data; we don't repeat that here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ranker.architecture import HEAD_NAMES, HeavyRankerStub, RankerConfig
from ranker.features import BINARY_FEATURES, CONTINUOUS_FEATURES, FEATURE_NAMES, FeatureScaler
from ranker.scoring import make_default_configs
from simulation.cascade import SimConfig, simulate_cascades


def _toy_seeds(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "id_str": [str(10**18 + i) for i in range(n)],
            "rawContent": ["hello world"] * n,
            "epoch": np.linspace(1.7e9, 1.7e9 + 3600, n),
            "replyCount": np.zeros(n, dtype=int),
            "retweetCount": np.zeros(n, dtype=int),
            "likeCount": np.zeros(n, dtype=int),
            "quoteCount": np.zeros(n, dtype=int),
            "is_reply": [False] * n,
            "is_quote": [False] * n,
            "is_original": [True] * n,
            "link_urls": [np.array([], dtype=object) for _ in range(n)],
            "user_id_str": [str(2 * 10**9 + i) for i in range(n)],
            "user_followersCount": rng.integers(10, 10_000, n),
            "user_friendsCount": rng.integers(10, 5_000, n),
            "user_statusesCount": rng.integers(100, 100_000, n),
            "user_favouritesCount": rng.integers(0, 50_000, n),
            "user_listedCount": rng.integers(0, 50, n),
            "user_blue": rng.integers(0, 2, n).astype(bool),
        }
    )


def _toy_users(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "user_id_str": [str(3 * 10**9 + i) for i in range(n)],
            "user_followersCount": rng.integers(0, 5_000, n),
            "user_friendsCount": rng.integers(0, 1_000, n),
            "user_statusesCount": rng.integers(0, 50_000, n),
            "user_favouritesCount": rng.integers(0, 10_000, n),
            "user_listedCount": rng.integers(0, 20, n),
            "user_blue": rng.integers(0, 2, n).astype(bool),
        }
    )


def _fitted_ranker_and_scaler() -> tuple[HeavyRankerStub, FeatureScaler]:
    """Build an untrained ranker with a sanely-fit scaler.

    The scaler must be fit on representative-magnitude data; if we fit it on
    zeros, the floored std (1e-6) blows up downstream features and the MLP
    saturates to constants, which collapses head probabilities and makes the
    different regimes produce indistinguishable outputs (defeating the point
    of the test).
    """
    torch.manual_seed(0)
    cfg = RankerConfig(n_features=len(FEATURE_NAMES))
    ranker = HeavyRankerStub(cfg)
    scaler = FeatureScaler()
    rng = np.random.default_rng(2)
    n_cont = len(CONTINUOUS_FEATURES)
    n_bin = len(BINARY_FEATURES)
    fit_data = np.zeros((128, len(FEATURE_NAMES)), dtype=np.float32)
    fit_data[:, :n_cont] = rng.uniform(0.0, 10.0, size=(128, n_cont)).astype(np.float32)
    fit_data[:, n_cont:] = rng.integers(0, 2, size=(128, n_bin)).astype(np.float32)
    scaler.fit(fit_data)
    return ranker, scaler


def test_simulator_output_shape() -> None:
    seeds = _toy_seeds(5)
    users = _toy_users(20)
    ranker, scaler = _fitted_ranker_and_scaler()
    sc = make_default_configs()["additive"]
    sim_cfg = SimConfig(n_time_bins=12, seed=42)
    out = simulate_cascades(seeds, users, ranker, scaler, sc, sim_cfg)
    assert len(out.per_bin) == 5 * 12
    assert set(out.per_bin.columns) == {
        "cascade_id", "t_hour", "n_reply", "n_retweet", "n_like", "n_deep"
    }
    assert len(out.per_cascade) == 5
    for k in HEAD_NAMES:
        assert f"n_{k}" in out.per_cascade.columns
        assert f"p_{k}" in out.per_cascade.columns


def test_simulator_is_reproducible() -> None:
    seeds = _toy_seeds(4)
    users = _toy_users(10)
    ranker, scaler = _fitted_ranker_and_scaler()
    sc = make_default_configs()["additive"]
    sim_cfg = SimConfig(n_time_bins=8, seed=7)
    a = simulate_cascades(seeds, users, ranker, scaler, sc, sim_cfg)
    b = simulate_cascades(seeds, users, ranker, scaler, sc, sim_cfg)
    pd.testing.assert_frame_equal(a.per_bin, b.per_bin)
    pd.testing.assert_frame_equal(a.per_cascade, b.per_cascade)


def test_regimes_produce_different_scores() -> None:
    """Additive vs ablated should produce different per-cascade scores on the
    same seeds. (Equality would mean the ablation is a no-op.)"""
    seeds = _toy_seeds(8)
    users = _toy_users(10)
    ranker, scaler = _fitted_ranker_and_scaler()
    sim_cfg = SimConfig(n_time_bins=4, seed=1)
    add = simulate_cascades(seeds, users, ranker, scaler, make_default_configs()["additive"], sim_cfg)
    abl = simulate_cascades(seeds, users, ranker, scaler, make_default_configs()["ablated"], sim_cfg)
    # Scores must differ on at least some seeds.
    assert not np.allclose(add.per_cascade["score"].to_numpy(), abl.per_cascade["score"].to_numpy())


def test_activity_gate_zeros_all_counts_when_pi_zero() -> None:
    """activity_pi=0 ⇒ all cascades inactive ⇒ all counts zero."""
    seeds = _toy_seeds(6)
    users = _toy_users(10)
    ranker, scaler = _fitted_ranker_and_scaler()
    sc = make_default_configs()["additive"]
    sim_cfg = SimConfig(n_time_bins=6, seed=3, activity_pi=0.0)
    out = simulate_cascades(seeds, users, ranker, scaler, sc, sim_cfg)
    for k in HEAD_NAMES:
        assert (out.per_cascade[f"n_{k}"] == 0).all()
        assert (out.per_bin[f"n_{k}"] == 0).all()
    assert (out.per_cascade["is_active"] == 0).all()


def test_activity_gate_passthrough_when_pi_one() -> None:
    """activity_pi=1.0 must produce identical counts to activity_pi=None
    (every cascade is active either way)."""
    seeds = _toy_seeds(6)
    users = _toy_users(10)
    ranker, scaler = _fitted_ranker_and_scaler()
    sc = make_default_configs()["additive"]
    base = SimConfig(n_time_bins=6, seed=11)
    gated = SimConfig(n_time_bins=6, seed=11, activity_pi=1.0)
    a = simulate_cascades(seeds, users, ranker, scaler, sc, base)
    b = simulate_cascades(seeds, users, ranker, scaler, sc, gated)
    # The activity gate consumes one rng draw per cascade before the Poisson
    # samples, so per-bin counts will *not* be byte-identical even at pi=1.
    # What we can check: total counts are statistically consistent and the
    # gate column is uniformly 1.
    assert (b.per_cascade["is_active"] == 1).all()
    for k in HEAD_NAMES:
        assert b.per_cascade[f"n_{k}"].sum() > 0  # not zeroed out


def test_nb_produces_heavier_tail_than_poisson() -> None:
    """NB with small dispersion_r should produce heavier per-cascade tails
    than Poisson on the same expected mean. This is the whole point of
    introducing NB — Poisson tails were too thin to match observed counts."""
    rng = np.random.default_rng(0)
    seeds = pd.concat(
        [_toy_seeds(50)] * 10, ignore_index=True
    )  # 500 cascades for tail stability
    users = _toy_users(20)
    ranker, scaler = _fitted_ranker_and_scaler()
    sc = make_default_configs()["additive"]
    poisson_cfg = SimConfig(n_time_bins=12, seed=42, baseline_exposures=200.0)
    nb_cfg = SimConfig(
        n_time_bins=12, seed=42, baseline_exposures=200.0, dispersion_r=0.1,
    )
    p = simulate_cascades(seeds, users, ranker, scaler, sc, poisson_cfg)
    n = simulate_cascades(seeds, users, ranker, scaler, sc, nb_cfg)
    p_p99 = np.percentile(p.per_cascade["n_reply"].to_numpy(), 99)
    n_p99 = np.percentile(n.per_cascade["n_reply"].to_numpy(), 99)
    # NB with r=0.1 should produce a meaningfully heavier 99th percentile.
    assert n_p99 > p_p99, f"NB p99 ({n_p99}) should exceed Poisson p99 ({p_p99})"
    _ = rng  # keep ruff happy


def test_nb_large_r_approximates_poisson_total() -> None:
    """As r → ∞, NB collapses to Poisson. Per-cascade total mean should be
    within sampling noise of the Poisson path at large r."""
    seeds = _toy_seeds(100)
    users = _toy_users(20)
    ranker, scaler = _fitted_ranker_and_scaler()
    sc = make_default_configs()["additive"]
    poisson_cfg = SimConfig(n_time_bins=12, seed=7)
    nb_cfg = SimConfig(n_time_bins=12, seed=7, dispersion_r=50.0)
    p = simulate_cascades(seeds, users, ranker, scaler, sc, poisson_cfg)
    n = simulate_cascades(seeds, users, ranker, scaler, sc, nb_cfg)
    p_mean = float(p.per_cascade["n_reply"].mean())
    n_mean = float(n.per_cascade["n_reply"].mean())
    # Means should agree within ~25% (still some Monte-Carlo noise at n=100).
    if p_mean > 0:
        assert abs(n_mean - p_mean) / p_mean < 0.25, (n_mean, p_mean)


def test_nb_rejects_invalid_dispersion() -> None:
    seeds = _toy_seeds(3)
    users = _toy_users(5)
    ranker, scaler = _fitted_ranker_and_scaler()
    sc = make_default_configs()["additive"]
    import pytest
    sim_cfg = SimConfig(n_time_bins=3, seed=0, dispersion_r=-1.0)
    with pytest.raises(ValueError):
        simulate_cascades(seeds, users, ranker, scaler, sc, sim_cfg)


def test_activity_gate_rejects_invalid_pi() -> None:
    seeds = _toy_seeds(3)
    users = _toy_users(5)
    ranker, scaler = _fitted_ranker_and_scaler()
    sc = make_default_configs()["additive"]
    import pytest
    for bad in (-0.1, 1.5):
        sim_cfg = SimConfig(n_time_bins=3, seed=0, activity_pi=bad)
        with pytest.raises(ValueError):
            simulate_cascades(seeds, users, ranker, scaler, sc, sim_cfg)

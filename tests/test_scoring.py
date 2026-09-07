"""Tests for the score-aggregation layer.

The aggregation layer is the architectural locus of this project's claim;
its behavior is the thing the ablation will compare against. These tests
pin down the mathematical invariants we rely on so refactors can't drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from ranker.scoring import (
    PUBLISHED_WEIGHTS,
    ScoringConfig,
    aggregate_score,
    make_default_configs,
)


def _probs(n: int, p_reply: float, p_retweet: float, p_like: float, p_deep: float) -> dict[str, np.ndarray]:
    return {
        "reply": np.full(n, p_reply, dtype=np.float64),
        "retweet": np.full(n, p_retweet, dtype=np.float64),
        "like": np.full(n, p_like, dtype=np.float64),
        "deep": np.full(n, p_deep, dtype=np.float64),
    }


def test_additive_matches_dot_product() -> None:
    """Additive regime is exactly Σ w_k · p_k."""
    cfg = make_default_configs()["additive"]
    p = _probs(3, 0.1, 0.2, 0.3, 0.05)
    s = aggregate_score(p, cfg)
    expected = (
        PUBLISHED_WEIGHTS["reply"] * 0.1
        + PUBLISHED_WEIGHTS["retweet"] * 0.2
        + PUBLISHED_WEIGHTS["like"] * 0.3
        + PUBLISHED_WEIGHTS["deep"] * 0.05
    )
    assert np.allclose(s, expected)


def test_ablated_gate_zero_slow_kills_score() -> None:
    """If slow heads (reply, deep) are exactly 0, ablated score is 0 regardless
    of fast heads (retweet, like)."""
    cfg = make_default_configs()["ablated"]
    p = _probs(2, p_reply=0.0, p_retweet=0.99, p_like=0.99, p_deep=0.0)
    s = aggregate_score(p, cfg)
    assert np.allclose(s, 0.0)


def test_ablated_gate_nonzero_slow_amplified_by_fast() -> None:
    """With slow > 0, fast amplifies. Higher retweet → higher score."""
    cfg = make_default_configs()["ablated"]
    p_low = _probs(1, p_reply=0.5, p_retweet=0.0, p_like=0.0, p_deep=0.5)
    p_high = _probs(1, p_reply=0.5, p_retweet=0.9, p_like=0.0, p_deep=0.5)
    s_low = aggregate_score(p_low, cfg)
    s_high = aggregate_score(p_high, cfg)
    assert s_high > s_low > 0.0


def test_ablated_alpha_zero_collapses_to_slow_only() -> None:
    """alpha=0 makes the ablated form independent of fast heads."""
    cfg = ScoringConfig(
        regime="ablated",
        weights=dict(PUBLISHED_WEIGHTS),
        alpha=0.0,
        slow_heads=("reply", "deep"),
        fast_heads=("retweet", "like"),
    )
    # Fix slow heads, vary fast heads — score should be identical.
    p1 = _probs(1, p_reply=0.5, p_retweet=0.1, p_like=0.1, p_deep=0.5)
    p2 = _probs(1, p_reply=0.5, p_retweet=0.99, p_like=0.99, p_deep=0.5)
    assert np.allclose(aggregate_score(p1, cfg), aggregate_score(p2, cfg))


def test_retuned_lower_reply_weight_lowers_score_when_reply_high() -> None:
    """For a reply-dominated tweet, the retuned additive should score lower."""
    cfgs = make_default_configs()
    add = cfgs["additive"]
    retuned = cfgs["additive_retuned"]
    p = _probs(1, p_reply=0.8, p_retweet=0.05, p_like=0.05, p_deep=0.05)
    s_add = aggregate_score(p, add)
    s_ret = aggregate_score(p, retuned)
    assert s_add > s_ret


def test_aggregate_score_shape() -> None:
    """Output is 1-D with length matching input."""
    cfg = make_default_configs()["additive"]
    p = _probs(7, 0.1, 0.1, 0.1, 0.1)
    assert aggregate_score(p, cfg).shape == (7,)


def test_unknown_regime_raises() -> None:
    p = _probs(1, 0.1, 0.1, 0.1, 0.1)
    with pytest.raises(ValueError):
        aggregate_score(p, ScoringConfig(regime="bogus"))  # type: ignore[arg-type]


def test_ratio_correction_penalizes_fast_heavy_content() -> None:
    """ratio_correction is strictly less than additive when fast is high,
    and ≈ additive when fast is zero."""
    from ranker.scoring import make_robustness_configs

    cfg = make_robustness_configs()["ratio_correction"]
    add_cfg = make_default_configs()["additive"]
    p_low_fast = _probs(1, p_reply=0.5, p_retweet=0.0, p_like=0.0, p_deep=0.5)
    p_high_fast = _probs(1, p_reply=0.5, p_retweet=0.9, p_like=0.9, p_deep=0.5)
    s_lf_rc = aggregate_score(p_low_fast, cfg)
    s_hf_rc = aggregate_score(p_high_fast, cfg)
    s_lf_add = aggregate_score(p_low_fast, add_cfg)
    s_hf_add = aggregate_score(p_high_fast, add_cfg)
    # No fast: ratio_correction ≈ additive (denominator = 1).
    assert np.isclose(s_lf_rc, s_lf_add)
    # High fast: ratio_correction < additive (penalty kicks in).
    assert s_hf_rc < s_hf_add


def test_reflective_floor_suppresses_low_slow_content() -> None:
    """reflective_floor: when S_slow is far below floor, score → 0."""
    from ranker.scoring import make_robustness_configs

    cfg = make_robustness_configs()["reflective_floor"]
    p_low_slow = _probs(1, p_reply=0.0, p_retweet=0.9, p_like=0.9, p_deep=0.0)
    p_high_slow = _probs(1, p_reply=0.9, p_retweet=0.9, p_like=0.9, p_deep=0.9)
    s_low = aggregate_score(p_low_slow, cfg)
    s_high = aggregate_score(p_high_slow, cfg)
    # Low slow (no reply, no quote) should be heavily suppressed.
    # High slow should pass through close to additive.
    assert s_low < s_high
    # Specifically: low_slow is suppressed below the additive baseline by a lot.
    additive = aggregate_score(p_low_slow, make_default_configs()["additive"])
    assert s_low < 0.5 * additive  # gated below half of what additive would produce

"""Tests for Stage 2 metrics: gap computation + cross-regime contrasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.hypothesis import (
    HIGH,
    LOW,
    cascade_size,
    compute_regime_gap,
    cross_regime_contrast,
    run_stage2,
)


def _toy_per_cascade(
    n_per_cell: int = 50,
    regimes: tuple[str, ...] = ("additive", "ablated"),
    n_replicates: int = 5,
    low_size_offset: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic per_cascade with known low/high difference under each regime.

    Each cascade has n_reply, n_retweet, n_like, n_deep set so that
    cascade_size = n_reply + n_retweet + n_deep = a known value.
    """
    # Cascade ids are shared across regimes (the real harness pairs them by
    # seed). Assign once: the first n_per_cell cascade_ids are LOW, the next
    # n_per_cell are HIGH, then everything is replicated across (regime,
    # replicate) with the same id × label assignment.
    rng = np.random.default_rng(seed)
    cascade_assignments: list[tuple[int, str]] = []
    for i in range(n_per_cell):
        cascade_assignments.append((i, LOW))
    for i in range(n_per_cell):
        cascade_assignments.append((n_per_cell + i, HIGH))
    rows: list[dict] = []
    for regime in regimes:
        for rep in range(n_replicates):
            for cid, label in cascade_assignments:
                base = 10.0 + (low_size_offset if label == LOW else 0.0)
                n_reply = max(0, int(rng.normal(base, 1.0)))
                rows.append(
                    {
                        "regime": regime,
                        "replicate": rep,
                        "cascade_id": cid,
                        "credibility_label": label,
                        "n_reply": n_reply,
                        "n_retweet": 0,
                        "n_like": 0,
                        "n_deep": 0,
                        "total_exposures": 100.0,
                    }
                )
    return pd.DataFrame(rows)


def test_cascade_size_sums_three_heads_excluding_like() -> None:
    """size = n_reply + n_retweet + n_deep (not likeCount per the design specification formula)."""
    df = pd.DataFrame({
        "n_reply": [1, 2, 3], "n_retweet": [4, 5, 6],
        "n_like": [99, 99, 99], "n_deep": [7, 8, 9],
    })
    assert cascade_size(df).tolist() == [12.0, 15.0, 18.0]


def test_compute_regime_gap_recovers_known_offset() -> None:
    """If low cascades are systematically larger, the gap should be positive."""
    pc = _toy_per_cascade(low_size_offset=5.0, n_replicates=10, n_per_cell=200)
    size = cascade_size(pc)
    gap = compute_regime_gap(
        pc, regime="additive",
        metric_name="cascade_size", metric_values=size,
    )
    assert gap.gap_mean > 4.0  # observed offset, with sampling noise
    assert gap.gap_mean < 6.0
    assert gap.gap_ci_lo > 0  # CI excludes zero
    assert gap.n_replicates == 10


def test_compute_regime_gap_zero_when_no_offset() -> None:
    """No systematic offset → gap CI should overlap zero."""
    pc = _toy_per_cascade(low_size_offset=0.0, n_replicates=10, n_per_cell=200)
    size = cascade_size(pc)
    gap = compute_regime_gap(
        pc, regime="additive",
        metric_name="cascade_size", metric_values=size,
    )
    assert gap.gap_ci_lo < 0 < gap.gap_ci_hi


def test_compute_regime_gap_raises_when_replicate_missing_label() -> None:
    """If any replicate is missing one label, the gap is undefined."""
    pc = _toy_per_cascade(n_replicates=3)
    pc = pc[~((pc["replicate"] == 0) & (pc["credibility_label"] == LOW))]
    size = cascade_size(pc)
    with pytest.raises(ValueError, match="missing low or high"):
        compute_regime_gap(
            pc, regime="additive",
            metric_name="cascade_size", metric_values=size,
        )


def test_cross_regime_contrast_pairs_replicates() -> None:
    """The contrast should be the per-replicate (regime_b − regime_a) gap difference."""
    pc = _toy_per_cascade(low_size_offset=5.0, n_replicates=8, n_per_cell=150)
    size = cascade_size(pc)
    g_a = compute_regime_gap(
        pc, regime="additive",
        metric_name="cascade_size", metric_values=size,
    )
    g_b = compute_regime_gap(
        pc, regime="ablated",
        metric_name="cascade_size", metric_values=size,
    )
    contrast = cross_regime_contrast(g_a, g_b)
    np.testing.assert_allclose(
        contrast.per_replicate_diff,
        g_b.per_replicate_gap - g_a.per_replicate_gap,
    )
    assert contrast.regime_a == "additive"
    assert contrast.regime_b == "ablated"


def test_run_stage2_end_to_end_shape() -> None:
    pc = _toy_per_cascade(n_replicates=5, n_per_cell=100)
    # synthetic per_bin: every cascade has all events at t_hour=0 → trivial peak.
    pb = pd.DataFrame({
        "regime": pc["regime"],
        "replicate": pc["replicate"],
        "cascade_id": pc["cascade_id"],
        "t_hour": 0,
        "n_reply": pc["n_reply"],
    })
    out = run_stage2(
        pc, pb, regimes=("additive", "ablated"), baseline_regime="additive",
        n_bootstrap=50,  # small for test speed
    )
    assert set(out["gaps"].keys()) == {"cascade_size", "audience_reach", "time_to_peak"}
    assert set(out["gaps"]["cascade_size"].keys()) == {"additive", "ablated"}
    # Only one cross-regime contrast since baseline is additive.
    assert set(out["contrasts"]["cascade_size"].keys()) == {"ablated"}
    # Bootstrap layers also present.
    assert set(out["bootstrap_gaps"]["cascade_size"].keys()) == {"additive", "ablated"}
    assert set(out["bootstrap_contrasts"]["cascade_size"].keys()) == {"ablated"}

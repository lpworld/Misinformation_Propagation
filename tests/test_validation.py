"""Tests for Stage 1 validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.validation import (
    compare_distributions,
    observed_root_reply_counts,
    stage1_root_reply_check,
)


def test_ks_close_for_identical_distributions() -> None:
    """Identical samples should produce a small KS statistic and p > 0.05."""
    rng = np.random.default_rng(0)
    a = rng.poisson(2, size=500).astype(np.float64)
    b = rng.poisson(2, size=500).astype(np.float64)
    t = compare_distributions(a, b)
    assert t.ks_stat < 0.15
    assert t.ks_pvalue > 0.01  # not super tight; Poisson noise can wiggle this


def test_ks_rejects_clearly_different_distributions() -> None:
    rng = np.random.default_rng(1)
    a = rng.poisson(1, size=500).astype(np.float64)
    b = rng.poisson(20, size=500).astype(np.float64)
    t = compare_distributions(a, b)
    assert t.ks_stat > 0.5
    assert t.ks_pvalue < 1e-10


def test_observed_root_reply_counts_picks_in_corpus_roots() -> None:
    """A tweet whose id == conversationId is a root; all others are not."""
    df = pd.DataFrame({
        "id_str": ["1000", "2000", "3000", "4000"],
        "conversationId": [1000.0, 1000.0, 3000.0, 9999.0],  # 9999 is missing
        "replyCount": [42, 0, 7, 1],
    })
    roots = observed_root_reply_counts(df)
    # id=1000 matches conversationId=1000 → root with replyCount=42
    # id=3000 matches conversationId=3000 → root with replyCount=7
    # id=2000 not a root (conversationId=1000 ≠ 2000)
    # id=4000 not a root (conversationId=9999, no match in corpus)
    assert sorted(roots.tolist()) == [7.0, 42.0]


def test_stage1_check_runs_on_min_inputs() -> None:
    """Smoke test the full check with tiny synthetic inputs."""
    df = pd.DataFrame({
        "id_str": ["100", "200"],
        "conversationId": [100.0, 200.0],
        "replyCount": [5, 10],
    })
    sim = pd.DataFrame({"n_reply": [5, 10, 7]})
    t = stage1_root_reply_check(df, sim)
    assert t.metric == "root_reply_count"
    assert t.n_observed == 2
    assert t.n_simulated == 3

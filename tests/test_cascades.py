"""Hand-checked reply-tree reconstruction tests.

We construct tiny conversations by hand and verify depth, breadth,
direct replies, orphan counting, and root selection.
"""

from __future__ import annotations

import math

import pandas as pd

from analysis import cascades


def _make_conv(rows: list[tuple[float, float | None, float, float]]) -> pd.DataFrame:
    """Helper: rows are tuples of (id_float, parent_float_or_nan, conv_id, epoch)."""
    return pd.DataFrame(
        {
            "id_float": [r[0] for r in rows],
            "in_reply_to_status_id_str": [r[1] if r[1] is not None else float("nan") for r in rows],
            "conversationId": [r[2] for r in rows],
            "epoch": [r[3] for r in rows],
        }
    )


# ---- shape tests ---------------------------------------------------

def test_single_root_no_replies():
    df = _make_conv([(1.0, None, 1.0, 100.0)])
    s = cascades.cascade_stats(df)
    assert s.root_id == 1.0
    assert s.depth == 0
    assert s.breadth == 1
    assert s.direct_replies == 0
    assert s.n_orphans == 0


def test_star_one_level():
    # root 1 with three direct replies
    df = _make_conv(
        [
            (1.0, None, 1.0, 100.0),
            (2.0, 1.0, 1.0, 110.0),
            (3.0, 1.0, 1.0, 120.0),
            (4.0, 1.0, 1.0, 130.0),
        ]
    )
    s = cascades.cascade_stats(df)
    assert s.root_id == 1.0
    assert s.depth == 1
    assert s.breadth == 3
    assert s.direct_replies == 3
    assert s.n_orphans == 0


def test_chain_three_deep():
    # 1 -> 2 -> 3 -> 4
    df = _make_conv(
        [
            (1.0, None, 1.0, 100.0),
            (2.0, 1.0, 1.0, 110.0),
            (3.0, 2.0, 1.0, 120.0),
            (4.0, 3.0, 1.0, 130.0),
        ]
    )
    s = cascades.cascade_stats(df)
    assert s.depth == 3
    assert s.breadth == 1
    assert s.direct_replies == 1


def test_branching_tree():
    # 1 -> {2, 3}; 2 -> {4, 5}; 3 -> 6; 6 -> 7
    df = _make_conv(
        [
            (1.0, None, 1.0, 100.0),
            (2.0, 1.0, 1.0, 110.0),
            (3.0, 1.0, 1.0, 120.0),
            (4.0, 2.0, 1.0, 130.0),
            (5.0, 2.0, 1.0, 140.0),
            (6.0, 3.0, 1.0, 150.0),
            (7.0, 6.0, 1.0, 160.0),
        ]
    )
    s = cascades.cascade_stats(df)
    assert s.depth == 3                # 1->3->6->7
    assert s.breadth == 3              # depth-2 has nodes {4,5,6}
    assert s.direct_replies == 2


def test_orphan_count():
    # 5 replies but parent 99 is missing; only 1 reply to root
    df = _make_conv(
        [
            (1.0, None, 1.0, 100.0),
            (2.0, 1.0, 1.0, 110.0),
            (3.0, 99.0, 1.0, 120.0),  # parent absent → orphan
            (4.0, 99.0, 1.0, 130.0),  # ditto
        ]
    )
    s = cascades.cascade_stats(df)
    assert s.root_id == 1.0
    assert s.n_orphans == 2
    assert s.depth == 1
    assert s.direct_replies == 1


def test_root_fallback_when_id_missing():
    # conversationId is 99 but no tweet has id 99 — fall back to earliest tweet
    df = _make_conv(
        [
            (5.0, None, 99.0, 200.0),
            (6.0, 5.0, 99.0, 210.0),
            (7.0, 5.0, 99.0, 220.0),
        ]
    )
    s = cascades.cascade_stats(df)
    assert s.root_id == 5.0  # earliest by epoch
    assert s.depth == 1
    assert s.direct_replies == 2


# ---- time-to-peak --------------------------------------------------

def test_time_to_peak_simple():
    # 1 reply at +30 min, 5 replies at +1.5 h, 1 reply at +3 h.
    # Peak hour-bucket = bucket 1 (1 to 2h after root).
    rows = [(1.0, None, 1.0, 0.0)]
    rows.append((2.0, 1.0, 1.0, 1800.0))     # 0.5h
    for i in range(5):
        rows.append((10.0 + i, 1.0, 1.0, 5400.0 + i))  # ~1.5h
    rows.append((20.0, 1.0, 1.0, 10800.0))   # 3h
    df = _make_conv(rows)
    s = cascades.cascade_stats(df)
    assert s.time_to_peak_seconds == 3600.0  # bucket 1 = [3600, 7200)


# ---- batch over a DataFrame ---------------------------------------

def test_all_cascade_stats_filters_min_size():
    rows = [
        (1.0, None, 1.0, 100.0),
        (2.0, 1.0, 1.0, 110.0),
        (3.0, None, 3.0, 200.0),  # standalone — only 1 tweet in conv 3
    ]
    df = _make_conv(rows)
    out = cascades.all_cascade_stats(df, min_size=2)
    assert list(out["conversation_id"]) == [1.0]


def test_reply_tree_quality_aggregates():
    rows = [
        # conv 1: depth 2 (1->2->3)
        (1.0, None, 1.0, 100.0),
        (2.0, 1.0, 1.0, 110.0),
        (3.0, 2.0, 1.0, 120.0),
        # conv 2: depth 1 star
        (10.0, None, 10.0, 200.0),
        (11.0, 10.0, 10.0, 210.0),
        (12.0, 10.0, 10.0, 220.0),
    ]
    df = _make_conv(rows)
    stats = cascades.all_cascade_stats(df)
    q = cascades.reply_tree_quality(stats)
    assert q["n_cascades"] == 2
    assert math.isclose(q["frac_depth_ge_2"], 0.5)
    assert math.isclose(q["frac_breadth_ge_2"], 0.5)

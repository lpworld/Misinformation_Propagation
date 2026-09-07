"""Reply-tree reconstruction and cascade-level statistics.

Trees are built per ``conversationId``. The parent→child edge is from a
tweet's ``id`` to the child's ``in_reply_to_status_id_str``. Both are floats
in the USC chunks (precision loss noted in :mod:`analysis.loader`), so the
join uses float IDs throughout.

Each conversation may be partial — the root tweet is often outside the
chunked subset. We pick a root by:

1. The unique tweet whose ``id`` (as float) equals the conversation's
   ``conversationId``. (Twitter's convention: a conversation's id is the
   id of its starter tweet.)
2. Failing (1), the earliest tweet in the conversation by ``epoch``.

Some replies' parents are missing from the chunk; those become orphans
and are not connected to the tree. We report orphan counts so the user can
judge how lossy the reconstruction is.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CascadeStats:
    """Per-conversation cascade metrics."""

    conversation_id: float
    root_id: float | None       # float id of the root tweet, or None if absent
    n_tweets: int               # tweets in this conversation present in the data
    n_orphans: int              # tweets whose parent is not in the conversation
    depth: int                  # max edges from root to any descendant
    breadth: int                # max number of nodes at any single depth
    direct_replies: int         # number of direct replies to root
    time_to_peak_seconds: float | None  # seconds from root creation to peak hourly reply rate
    duration_seconds: float | None      # epoch span from earliest to latest tweet


# ---- tree construction --------------------------------------------------

def _conversation_root(group: pd.DataFrame) -> float | None:
    """Pick the root tweet's id as a float for the given conversation group.

    Strategy: prefer the tweet whose float id equals the conversationId; else
    earliest by epoch. Returns the root's float id, or ``None`` if the group
    is empty.
    """
    if group.empty:
        return None
    cid = group["conversationId"].iloc[0]
    if pd.notna(cid):
        match = group[np.isclose(group["id_float"], cid)]
        if len(match) >= 1:
            return float(match["id_float"].iloc[0])
    earliest = group["epoch"].idxmin()
    return float(group.at[earliest, "id_float"])


def build_reply_tree(
    conversation_df: pd.DataFrame,
) -> tuple[nx.DiGraph, float | None, int]:
    """Build the directed reply tree for one conversation.

    Returns ``(tree, root_id, n_orphans)``. The tree's edges go from parent → child.
    Orphans (tweets whose parent is absent from the group) are added as nodes
    but not connected; the count is returned separately.

    Required input columns: ``id_float``, ``in_reply_to_status_id_str``, ``epoch``.
    """
    g = nx.DiGraph()
    if conversation_df.empty:
        return g, None, 0

    g.add_nodes_from(conversation_df["id_float"].astype(float).tolist())

    present_ids = set(conversation_df["id_float"].astype(float).tolist())
    n_orphans = 0
    for child, parent in zip(
        conversation_df["id_float"].astype(float),
        conversation_df["in_reply_to_status_id_str"],
    ):
        if pd.isna(parent):
            continue
        parent_f = float(parent)
        if parent_f in present_ids:
            g.add_edge(parent_f, child)
        else:
            n_orphans += 1

    root = _conversation_root(conversation_df)
    return g, root, n_orphans


# ---- per-tree statistics -----------------------------------------------

def _tree_depth(tree: nx.DiGraph, root: float) -> int:
    if root not in tree:
        return 0
    depths = nx.single_source_shortest_path_length(tree, root)
    return max(depths.values()) if depths else 0


def _tree_breadth(tree: nx.DiGraph, root: float) -> int:
    """Max number of nodes at any single depth (i.e., widest level)."""
    if root not in tree:
        return 0
    depths = nx.single_source_shortest_path_length(tree, root)
    if not depths:
        return 0
    by_depth: dict[int, int] = defaultdict(int)
    for d in depths.values():
        by_depth[d] += 1
    return max(by_depth.values())


def _direct_replies(tree: nx.DiGraph, root: float) -> int:
    return tree.out_degree(root) if root in tree else 0


def _time_to_peak(group: pd.DataFrame, root_id: float | None) -> float | None:
    """Seconds from root creation to the hour-bucket with the most replies.

    Uses 1-hour bins of ``epoch``. If there are fewer than 2 tweets or the
    root timestamp is missing, returns None.
    """
    if root_id is None or len(group) < 2:
        return None
    root_rows = group[np.isclose(group["id_float"], root_id)]
    if root_rows.empty:
        return None
    root_epoch = float(root_rows["epoch"].iloc[0])
    if not np.isfinite(root_epoch):
        return None
    epochs = group["epoch"].dropna().to_numpy()
    if epochs.size < 2:
        return None
    bins = np.floor((epochs - root_epoch) / 3600.0).astype(int)
    if bins.size == 0:
        return None
    unique_bins, counts = np.unique(bins, return_counts=True)
    peak_bin = unique_bins[counts.argmax()]
    return float(peak_bin) * 3600.0


def cascade_stats(group: pd.DataFrame) -> CascadeStats:
    """Compute all per-conversation metrics for one group."""
    cid = float(group["conversationId"].iloc[0]) if "conversationId" in group else float("nan")
    tree, root, n_orphans = build_reply_tree(group)
    if root is None:
        return CascadeStats(
            conversation_id=cid,
            root_id=None,
            n_tweets=len(group),
            n_orphans=n_orphans,
            depth=0,
            breadth=0,
            direct_replies=0,
            time_to_peak_seconds=None,
            duration_seconds=None,
        )
    epochs = group["epoch"].dropna().to_numpy()
    duration = float(epochs.max() - epochs.min()) if epochs.size >= 2 else None
    return CascadeStats(
        conversation_id=cid,
        root_id=root,
        n_tweets=len(group),
        n_orphans=n_orphans,
        depth=_tree_depth(tree, root),
        breadth=_tree_breadth(tree, root),
        direct_replies=_direct_replies(tree, root),
        time_to_peak_seconds=_time_to_peak(group, root),
        duration_seconds=duration,
    )


# ---- batch over a DataFrame --------------------------------------------

def prepare_for_cascades(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``id_float`` column for join compatibility with reply-tree IDs.

    The CSV stores ``id_str`` as a 19-digit integer string but stores
    ``in_reply_to_status_id_str`` and ``conversationId`` as float64 (lossy).
    To match parents to children, we cast both sides to float here.
    """
    out = df.copy()
    if "id_float" not in out.columns:
        out["id_float"] = pd.to_numeric(out["id_str"], errors="coerce").astype(float)
    return out


def all_cascade_stats(
    df: pd.DataFrame, *, min_size: int = 2
) -> pd.DataFrame:
    """Compute :class:`CascadeStats` for every conversation with ≥ ``min_size`` tweets.

    Returns a DataFrame indexed by conversation_id with one row per cascade.
    """
    df = prepare_for_cascades(df)
    if "conversationId" not in df.columns:
        raise ValueError("dataframe missing 'conversationId' column")
    sizes = df.groupby("conversationId", sort=False).size()
    keep = sizes[sizes >= min_size].index
    sub = df[df["conversationId"].isin(keep)]
    rows: list[dict] = []
    for cid, group in sub.groupby("conversationId", sort=False):
        s = cascade_stats(group)
        rows.append(s.__dict__)
    return pd.DataFrame(rows)


def reply_tree_quality(stats_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate "is reply-tree structure workable?" diagnostics.

    Output keys:
        n_cascades, frac_depth_ge_2, frac_breadth_ge_2, mean_size,
        median_size, p95_size, mean_orphan_frac
    """
    if stats_df.empty:
        return {
            "n_cascades": 0,
            "frac_depth_ge_2": 0.0,
            "frac_breadth_ge_2": 0.0,
            "mean_size": 0.0,
            "median_size": 0.0,
            "p95_size": 0.0,
            "mean_orphan_frac": 0.0,
        }
    sizes = stats_df["n_tweets"].astype(float)
    orphan_frac = stats_df["n_orphans"].astype(float) / sizes.where(sizes > 0, 1)
    return {
        "n_cascades": int(len(stats_df)),
        "frac_depth_ge_2": float((stats_df["depth"] >= 2).mean()),
        "frac_breadth_ge_2": float((stats_df["breadth"] >= 2).mean()),
        "mean_size": float(sizes.mean()),
        "median_size": float(sizes.median()),
        "p95_size": float(np.percentile(sizes, 95)),
        "mean_orphan_frac": float(orphan_frac.mean()),
    }


__all__ = [
    "CascadeStats",
    "build_reply_tree",
    "cascade_stats",
    "prepare_for_cascades",
    "all_cascade_stats",
    "reply_tree_quality",
]

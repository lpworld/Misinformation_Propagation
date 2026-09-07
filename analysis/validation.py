"""Stage 1 validation: count-based cascade-distribution matching.

Stage 1 asks a label-free question: do simulated cascades produce
engagement-count distributions that look like observed cascades in USC?
Per the design specification §Two-stage analysis, the **primary metrics** are:

1. **Root-reply-count distribution** — ``root.replyCount`` per conversation
   vs. simulated total reply events per cascade.
2. **Aggregate-engagement-count distribution** — Σ (replyCount+retweetCount
   +quoteCount) per conversation vs. Σ (n_reply+n_retweet+n_deep) per
   simulated cascade.
3. **Reactive-to-reflective ratio** — per tweet (observed) /
   per cascade summary (simulated): ``reply / (retweet + quote + 1)``.
4. **Time-to-peak distribution** — hour offset (from root) at which
   reply rate peaks within each conversation.

Comparison is the two-sample KS test (``scipy.stats.ks_2samp``) plus
Mann–Whitney as backup. Per the design specification, no credibility labels enter this
module — Stage 1 is label-free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributionTest:
    """Result of a two-sample distribution comparison."""

    metric: str
    n_observed: int
    n_simulated: int
    obs_median: float
    sim_median: float
    obs_mean: float
    sim_mean: float
    obs_p90: float
    sim_p90: float
    ks_stat: float
    ks_pvalue: float
    mw_stat: float
    mw_pvalue: float


# ---- helpers -----------------------------------------------------------

def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _id_float_col(df: pd.DataFrame) -> pd.Series:
    """Return id_float either from a precomputed column or by casting id_str."""
    if "id_float" in df.columns:
        return df["id_float"]
    return pd.to_numeric(df["id_str"], errors="coerce").astype(float)


# ---- observed distributions --------------------------------------------

def observed_root_reply_counts(df: pd.DataFrame) -> np.ndarray:
    """``root.replyCount`` per conversation with an in-corpus root.

    A "root" is a tweet whose float id matches the conversation's
    ``conversationId``. Conversations without an in-corpus root are skipped.
    """
    if "conversationId" not in df.columns or "id_str" not in df.columns:
        raise ValueError("df missing required columns for root identification")
    work = pd.DataFrame({
        "id_float": _id_float_col(df),
        "conversationId": df["conversationId"],
        "replyCount": _to_numeric(df["replyCount"]),
    })
    is_root = (work["id_float"] == work["conversationId"]) & work["conversationId"].notna()
    roots = work.loc[is_root, "replyCount"].to_numpy(dtype=np.float64)
    logger.info(
        "observed root-reply: %d in-corpus roots from %d-row corpus (%.1f%%)",
        len(roots), len(df), 100.0 * len(roots) / max(len(df), 1),
    )
    return roots


def observed_aggregate_engagement(df: pd.DataFrame) -> np.ndarray:
    """Σ (replyCount + retweetCount + quoteCount) per conversation."""
    if "conversationId" not in df.columns:
        raise ValueError("df missing 'conversationId' column")
    work = pd.DataFrame({
        "conversationId": df["conversationId"],
        "agg": _to_numeric(df["replyCount"])
                + _to_numeric(df["retweetCount"])
                + _to_numeric(df["quoteCount"]),
    })
    return work.groupby("conversationId")["agg"].sum().to_numpy(dtype=np.float64)


def observed_reactive_to_reflective(df: pd.DataFrame) -> np.ndarray:
    """Per-tweet ratio ``replyCount / (retweetCount + quoteCount + 1)``."""
    rep = _to_numeric(df["replyCount"]).to_numpy(dtype=np.float64)
    rt = _to_numeric(df["retweetCount"]).to_numpy(dtype=np.float64)
    qt = _to_numeric(df["quoteCount"]).to_numpy(dtype=np.float64)
    return rep / (rt + qt + 1.0)


def observed_time_to_peak_hours(df: pd.DataFrame) -> np.ndarray:
    """Hour offset (from root's epoch) at which the reply rate peaks per conversation.

    For each conversation with an in-corpus root and ≥ 2 observed tweets,
    bin tweets into hour-of-cascade bins (relative to root's epoch) and
    return the bin index of the modal hour. Conversations failing those
    conditions are skipped.

    Note: with USC's ~76% orphan fraction, this metric is computed on the
    *observed* sub-tree — biased toward smaller cascades. We accept this
    bias (per the design specification, count-based primitives are still primary; the
    bias is the same one we already documented for reply-tree-shape metrics).
    """
    needed = {"id_str", "conversationId", "epoch"}
    if not needed.issubset(df.columns):
        raise ValueError(f"df missing one of {needed}")

    work = pd.DataFrame({
        "id_float": _id_float_col(df),
        "conversationId": df["conversationId"],
        "epoch": _to_numeric(df["epoch"]),
    })
    work = work[work["conversationId"].notna()]

    # Pick roots: id_float == conversationId, per-conversation.
    roots = work[work["id_float"] == work["conversationId"]][
        ["conversationId", "epoch"]
    ].rename(columns={"epoch": "root_epoch"})
    if roots.empty:
        return np.array([], dtype=np.float64)

    joined = work.merge(roots, on="conversationId", how="inner")
    joined["t_hour"] = np.floor((joined["epoch"] - joined["root_epoch"]) / 3600.0)
    # Drop the root's own bin (t_hour=0 is the post itself, not a reply); replies
    # have t_hour ≥ 0, but we want strictly the *peak reply hour*, so we
    # exclude the root tweet from the histogram.
    joined = joined[joined["id_float"] != joined["conversationId"]]
    if joined.empty:
        return np.array([], dtype=np.float64)

    grouped = joined.groupby("conversationId")["t_hour"]
    # For each conversation, find the modal hour bucket. We need ≥ 2 events
    # in the observed subset to call something a "peak."
    peaks: list[float] = []
    for _, hours in grouped:
        if len(hours) < 2:
            continue
        h = hours.to_numpy()
        h = h[(h >= 0) & np.isfinite(h)]
        if h.size < 2:
            continue
        hi = h.astype(int)
        bins = np.bincount(hi)
        peaks.append(float(bins.argmax()))
    return np.asarray(peaks, dtype=np.float64)


# ---- simulated distributions -------------------------------------------

def simulated_root_reply_counts(per_cascade: pd.DataFrame) -> np.ndarray:
    return per_cascade["n_reply"].to_numpy(dtype=np.float64)


def simulated_aggregate_engagement(per_cascade: pd.DataFrame) -> np.ndarray:
    return (
        per_cascade["n_reply"]
        + per_cascade["n_retweet"]
        + per_cascade["n_deep"]
    ).to_numpy(dtype=np.float64)


def simulated_reactive_to_reflective(per_cascade: pd.DataFrame) -> np.ndarray:
    """Per-cascade summary ratio (analog of the per-tweet observed metric)."""
    rep = per_cascade["n_reply"].to_numpy(dtype=np.float64)
    rt = per_cascade["n_retweet"].to_numpy(dtype=np.float64)
    qt = per_cascade["n_deep"].to_numpy(dtype=np.float64)
    return rep / (rt + qt + 1.0)


def simulated_time_to_peak_hours(per_bin: pd.DataFrame) -> np.ndarray:
    """For each simulated cascade, the t_hour with max n_reply."""
    if per_bin.empty:
        return np.array([], dtype=np.float64)
    # idxmax over a long-form frame: use groupby with reset index to find
    # the row of the max within each cascade.
    out = (
        per_bin.loc[per_bin.groupby("cascade_id")["n_reply"].idxmax(), ["cascade_id", "t_hour"]]
    )
    return out["t_hour"].to_numpy(dtype=np.float64)


# ---- comparison ---------------------------------------------------------

def compare_distributions(
    observed: np.ndarray,
    simulated: np.ndarray,
    *,
    metric: str = "unspecified",
) -> DistributionTest:
    """Run KS + Mann–Whitney two-sample tests with summary stats."""
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    if obs.size == 0 or sim.size == 0:
        raise ValueError(f"observed/simulated for {metric!r} must be non-empty")

    ks = stats.ks_2samp(obs, sim, alternative="two-sided", method="asymp")
    mw = stats.mannwhitneyu(obs, sim, alternative="two-sided")
    return DistributionTest(
        metric=metric,
        n_observed=int(obs.size),
        n_simulated=int(sim.size),
        obs_median=float(np.median(obs)),
        sim_median=float(np.median(sim)),
        obs_mean=float(np.mean(obs)),
        sim_mean=float(np.mean(sim)),
        obs_p90=float(np.percentile(obs, 90)),
        sim_p90=float(np.percentile(sim, 90)),
        ks_stat=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        mw_stat=float(mw.statistic),
        mw_pvalue=float(mw.pvalue),
    )


# ---- one-call wrappers --------------------------------------------------

def stage1_all_metrics(
    observed_df: pd.DataFrame,
    sim_per_cascade: pd.DataFrame,
    sim_per_bin: pd.DataFrame,
) -> dict[str, DistributionTest]:
    """Run all four Stage 1 metrics and return them as a dict."""
    return {
        "root_reply_count": compare_distributions(
            observed_root_reply_counts(observed_df),
            simulated_root_reply_counts(sim_per_cascade),
            metric="root_reply_count",
        ),
        "aggregate_engagement": compare_distributions(
            observed_aggregate_engagement(observed_df),
            simulated_aggregate_engagement(sim_per_cascade),
            metric="aggregate_engagement",
        ),
        "reactive_to_reflective": compare_distributions(
            observed_reactive_to_reflective(observed_df),
            simulated_reactive_to_reflective(sim_per_cascade),
            metric="reactive_to_reflective",
        ),
        "time_to_peak_hours": compare_distributions(
            observed_time_to_peak_hours(observed_df),
            simulated_time_to_peak_hours(sim_per_bin),
            metric="time_to_peak_hours",
        ),
    }


def stage1_root_reply_check(
    observed_df: pd.DataFrame,
    sim_per_cascade: pd.DataFrame,
) -> DistributionTest:
    """Phase-2 entry point. Returns just the root-reply-count comparison."""
    return compare_distributions(
        observed_root_reply_counts(observed_df),
        simulated_root_reply_counts(sim_per_cascade),
        metric="root_reply_count",
    )


__all__ = [
    "DistributionTest",
    "compare_distributions",
    "observed_aggregate_engagement",
    "observed_reactive_to_reflective",
    "observed_root_reply_counts",
    "observed_time_to_peak_hours",
    "simulated_aggregate_engagement",
    "simulated_reactive_to_reflective",
    "simulated_root_reply_counts",
    "simulated_time_to_peak_hours",
    "stage1_all_metrics",
    "stage1_root_reply_check",
]

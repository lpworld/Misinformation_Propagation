"""Sample seed content (cascade roots) from the observed USC tweet distribution.

For Phase 2 MVP we sample from any tweet — including replies and quotes —
because the simulator treats each seed as the *root* of a fresh cascade
regardless of its real-world position. Filtering to ``is_original`` is
available as a switch but not the default; original-only seeds underrepresent
the kind of content the production ranker actually fans out at scale.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# Tweet-side columns we carry through to the simulator + ranker feature extractor.
SEED_COLS: tuple[str, ...] = (
    "id_str",
    "rawContent",
    "epoch",
    "replyCount",
    "retweetCount",
    "likeCount",
    "quoteCount",
    "is_reply",
    "is_quote",
    "is_original",
    "link_urls",
    "user_id_str",
    "user_followersCount",
    "user_friendsCount",
    "user_statusesCount",
    "user_favouritesCount",
    "user_listedCount",
    "user_blue",
)


def sample_seeds(
    df: pd.DataFrame,
    n: int,
    *,
    seed: int = 1337,
    originals_only: bool = False,
) -> pd.DataFrame:
    """Random sample of ``n`` tweets to use as cascade seeds.

    ``originals_only`` filters to ``is_original`` tweets (excludes replies
    and quote-tweets). Default ``False`` because the production ranker fans
    out replies and quotes too, and our simulator treats each seed as a
    fresh root regardless.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    pool = df
    if originals_only and "is_original" in df.columns:
        pool = df[df["is_original"].fillna(False).astype(bool)]
    if len(pool) == 0:
        raise ValueError("seed pool is empty")

    cols = [c for c in SEED_COLS if c in pool.columns]
    seeds = pool[cols].sample(
        n=min(n, len(pool)), random_state=seed, replace=(len(pool) < n)
    ).reset_index(drop=True)
    logger.info(
        "sampled %d seeds from a pool of %d tweets (originals_only=%s)",
        len(seeds), len(pool), originals_only,
    )
    return seeds


__all__ = ["SEED_COLS", "sample_seeds"]

"""Sample a user pool from the USC user-feature distribution.

For Phase 2 MVP we keep this lightweight: dedupe by ``user_id_str``, then
stratified-sample across follower-count quintiles × ``user_blue``. Static
features only — there is no in-simulation user-feature evolution per the
design spec.

The output is a DataFrame indexed 0..n-1 with the same author-side columns
the ranker's feature extractor expects (``user_followersCount``,
``user_friendsCount``, etc.) plus ``user_blue``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Author-side columns required downstream by ranker.features.extract_features.
USER_COLS: tuple[str, ...] = (
    "user_id_str",
    "user_followersCount",
    "user_friendsCount",
    "user_statusesCount",
    "user_favouritesCount",
    "user_listedCount",
    "user_blue",
)


def _follower_strata(s: pd.Series, n_strata: int = 5) -> np.ndarray:
    """Return integer stratum labels by follower-count quantile.

    Quantile binning is robust to the heavy follower-count tail.
    """
    s_num = pd.to_numeric(s, errors="coerce").fillna(0.0)
    # qcut may collapse if there are many duplicate boundaries (lots of zeros);
    # fall back to rank-based binning in that case.
    try:
        codes = pd.qcut(s_num, q=n_strata, labels=False, duplicates="drop")
        codes = codes.fillna(0).astype(int).to_numpy()
        if len(np.unique(codes)) >= 2:
            return codes
    except ValueError:
        pass
    ranks = s_num.rank(method="first", pct=True).to_numpy()
    return np.clip((ranks * n_strata).astype(int), 0, n_strata - 1)


def sample_users(
    df: pd.DataFrame,
    n: int,
    *,
    seed: int = 1337,
    n_follower_strata: int = 5,
) -> pd.DataFrame:
    """Stratified sample of ``n`` distinct users from a USC DataFrame.

    Strata: follower-count quintile × ``user_blue`` (10 cells when n_follower_strata=5).
    Within-cell sampling is uniform with replacement only if the cell is
    smaller than its allocation; otherwise without.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if "user_id_str" not in df.columns:
        raise ValueError("expected 'user_id_str' column in df")

    # Dedupe by user; keep the first row we saw for each.
    users = df.drop_duplicates(subset=["user_id_str"], keep="first")
    users = users[list(USER_COLS)].copy().reset_index(drop=True)
    if len(users) == 0:
        raise ValueError("no users found in df after dedup")

    users["_blue"] = users["user_blue"].fillna(False).astype(bool).astype(int)
    users["_fstratum"] = _follower_strata(users["user_followersCount"], n_follower_strata)
    users["_cell"] = users["_fstratum"].astype(str) + "_" + users["_blue"].astype(str)

    rng = np.random.default_rng(seed)
    cells = users["_cell"].unique()
    n_cells = len(cells)
    base = n // n_cells
    rem = n - base * n_cells

    sampled_idx: list[int] = []
    # Distribute the remainder across the first ``rem`` cells deterministically
    # (sorted) for reproducibility.
    cells_sorted = sorted(cells)
    for i, cell in enumerate(cells_sorted):
        cell_idx = users.index[users["_cell"] == cell].to_numpy()
        alloc = base + (1 if i < rem else 0)
        if alloc <= 0:
            continue
        replace = len(cell_idx) < alloc
        chosen = rng.choice(cell_idx, size=alloc, replace=replace)
        sampled_idx.extend(chosen.tolist())

    out = users.iloc[sampled_idx].drop(columns=["_blue", "_fstratum", "_cell"])
    out = out.reset_index(drop=True)
    logger.info(
        "sampled %d users from %d distinct authors across %d strata cells",
        len(out), len(users), n_cells,
    )
    return out


__all__ = ["USER_COLS", "sample_users"]

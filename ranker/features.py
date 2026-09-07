"""Feature extraction + scaling for the ranker.

Phase 3 expansion: from 12 features to 17, adding clout / account-age /
posting-rate / weekly-cycle features that benefit MaskNet's instance-guided
masks. Still leak-free — nothing is derived from post-publication engagement
counts or view_count.

    Author      log1p of followers/friends/statuses/favourites/listed counts;
                log_clout = log((followers+1)/(friends+1));
                log_favs_per_status, log_statuses_per_day;
                log_account_age_days; user_blue flag.
    Tweet       is_reply, is_quote, has_url, log_text_length.
    Time        sin/cos of hour-of-day, sin/cos of hour-of-week.

Total: 13 continuous + 4 binary = 17 features. Continuous slots are
standardized by a fitted :class:`FeatureScaler`; binary slots pass through.

Targets (binary, head-wise) are unchanged: 1 if the corresponding engagement
count is > 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Names define both the order in the feature matrix and what the scaler
# standardizes. Continuous features come first, binary features last.
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "log_followers",
    "log_friends",
    "log_statuses",
    "log_favourites",
    "log_listed",
    "log_clout",
    "log_favs_per_status",
    "log_statuses_per_day",
    "log_account_age_days",
    "log_text_length",
    "hour_sin",
    "hour_cos",
    "weekhour_sin",
    "weekhour_cos",
)
BINARY_FEATURES: tuple[str, ...] = (
    "user_blue",
    "is_reply",
    "is_quote",
    "has_url",
)
FEATURE_NAMES: tuple[str, ...] = CONTINUOUS_FEATURES + BINARY_FEATURES

TARGET_COUNT_COLS: dict[str, str] = {
    "reply": "replyCount",
    "retweet": "retweetCount",
    "like": "likeCount",
    "deep": "quoteCount",
}


def _safe_log1p(s: pd.Series) -> np.ndarray:
    """log1p of a numeric series, with NaNs → 0."""
    arr = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    return np.log1p(arr).astype(np.float32)


def _epoch_seconds(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy().astype(np.float64)


def _user_created_epoch(df: pd.DataFrame) -> np.ndarray:
    """Convert ``user_created_at`` (tz-aware datetime) to unix-epoch seconds."""
    if "user_created_at" not in df.columns:
        return np.zeros(len(df), dtype=np.float64)
    s = pd.to_datetime(df["user_created_at"], utc=True, errors="coerce")
    secs = (s.astype("int64", copy=False) // 1_000_000_000).astype(np.float64)
    # Pandas marks NaT as a very-negative int64; clamp to 0 to avoid blowups.
    secs = np.where(np.isfinite(secs) & (secs > 0), secs, 0.0)
    return secs


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Build the (n, n_features) float32 matrix for a cleaned USC DataFrame.

    Column order matches :data:`FEATURE_NAMES`.
    """
    n = len(df)
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    followers = pd.to_numeric(df["user_followersCount"], errors="coerce").fillna(0.0).to_numpy()
    friends = pd.to_numeric(df["user_friendsCount"], errors="coerce").fillna(0.0).to_numpy()
    statuses = pd.to_numeric(df["user_statusesCount"], errors="coerce").fillna(0.0).to_numpy()
    favourites = pd.to_numeric(df["user_favouritesCount"], errors="coerce").fillna(0.0).to_numpy()
    listed = pd.to_numeric(df["user_listedCount"], errors="coerce").fillna(0.0).to_numpy()

    out[:, 0] = np.log1p(followers).astype(np.float32)
    out[:, 1] = np.log1p(friends).astype(np.float32)
    out[:, 2] = np.log1p(statuses).astype(np.float32)
    out[:, 3] = np.log1p(favourites).astype(np.float32)
    out[:, 4] = np.log1p(listed).astype(np.float32)

    # Clout: log((followers + 1) / (friends + 1)). Captures influence asymmetry
    # — celebrities have huge followers, low friends; bots often have the
    # opposite. The +1 floors avoid log(0). Signed, centered around 0.
    out[:, 5] = np.log((followers + 1.0) / (friends + 1.0)).astype(np.float32)

    # Engagement-tendency-of-the-author proxy: how many likes the author has
    # bestowed per post they've made.
    out[:, 6] = np.log1p(favourites / np.maximum(statuses, 1.0)).astype(np.float32)

    # Posting cadence: average posts/day. Computed from account age (in days)
    # — denominator clamped to 1 day to avoid blowups for fresh accounts.
    tweet_epoch = _epoch_seconds(df["epoch"])
    user_epoch = _user_created_epoch(df)
    age_seconds = np.maximum(tweet_epoch - user_epoch, 0.0)
    age_days = age_seconds / 86400.0
    out[:, 7] = np.log1p(statuses / np.maximum(age_days, 1.0)).astype(np.float32)
    out[:, 8] = np.log1p(age_days).astype(np.float32)

    # Tweet-side
    text_len = df["rawContent"].fillna("").astype(str).str.len()
    out[:, 9] = np.log1p(text_len.to_numpy()).astype(np.float32)

    # Time-of-day (24h cycle)
    hour = (tweet_epoch / 3600.0) % 24.0
    out[:, 10] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    out[:, 11] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    # Time-of-week (168h cycle): captures weekday/weekend rhythm
    weekhour = (tweet_epoch / 3600.0) % (24.0 * 7.0)
    out[:, 12] = np.sin(2 * np.pi * weekhour / 168.0).astype(np.float32)
    out[:, 13] = np.cos(2 * np.pi * weekhour / 168.0).astype(np.float32)

    # Binary block (offsets must match BINARY_FEATURES order).
    bbase = len(CONTINUOUS_FEATURES)
    out[:, bbase + 0] = df["user_blue"].fillna(False).astype(bool).astype(np.float32)
    out[:, bbase + 1] = df["is_reply"].fillna(False).astype(bool).astype(np.float32)
    out[:, bbase + 2] = df["is_quote"].fillna(False).astype(bool).astype(np.float32)
    out[:, bbase + 3] = (df["link_urls"].map(len) > 0).astype(np.float32)

    return out


def extract_targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Binary engagement targets per head: 1 if the count is > 0, else 0."""
    return {
        name: (pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy() > 0).astype(np.float32)
        for name, col in TARGET_COUNT_COLS.items()
    }


@dataclass
class FeatureScaler:
    """Standardize continuous features (binary features pass through).

    Fit on training data only; persist the mean/std alongside ranker weights
    so that inference uses the same transformation.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    n_continuous: int = len(CONTINUOUS_FEATURES)

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        cont = X[:, : self.n_continuous]
        self.mean = cont.mean(axis=0).astype(np.float32)
        # Floor std at 1e-6 to avoid division by zero on degenerate columns.
        self.std = np.maximum(cont.std(axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler must be fit before transform")
        out = X.copy()
        out[:, : self.n_continuous] = (out[:, : self.n_continuous] - self.mean) / self.std
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


__all__ = [
    "BINARY_FEATURES",
    "CONTINUOUS_FEATURES",
    "FEATURE_NAMES",
    "TARGET_COUNT_COLS",
    "FeatureScaler",
    "extract_features",
    "extract_targets",
]

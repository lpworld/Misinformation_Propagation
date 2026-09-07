"""Load USC X 2024 chunks and parse nested JSON-like fields.

USC chunks are gzipped CSVs with several columns serialized as Python repr
(not JSON): ``user``, ``viewCount``, ``mentionedUsers``, ``links``,
``hashtags``, ``media``. The ``user`` dict embeds ``datetime.datetime(...)``
calls, which ``ast.literal_eval`` rejects. We preprocess datetime expressions
into ISO-8601 string literals via regex, then use ``ast.literal_eval`` —
strictly safer than ``eval`` because no code execution is possible.

ID precision caveat: the columns ``id`` and ``id_str`` are stored as plain
integers (full 19-digit precision), but ``conversationId``,
``in_reply_to_status_id_str``, ``in_reply_to_user_id_str``, and
``retweetedTweetID`` are stored in scientific notation in the CSV itself,
losing precision past ~16 significant figures. Reply-tree linking therefore
operates on float IDs throughout. Collision probability across our universe
of tweets is small but non-zero — flag in any downstream analysis sensitive
to it.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---- nested-JSON parsing -------------------------------------------------

_DATETIME_RE = re.compile(
    r"datetime\.datetime\("
    r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*"          # year, month, day
    r"(?:,\s*(\d+)\s*)?"                            # hour
    r"(?:,\s*(\d+)\s*)?"                            # minute
    r"(?:,\s*(\d+)\s*)?"                            # second
    r"(?:,\s*(\d+)\s*)?"                            # microsecond
    r"(?:,\s*tzinfo\s*=\s*datetime\.timezone\.utc\s*)?"
    r"\)"
)


def _datetime_to_iso(m: re.Match[str]) -> str:
    """Replace a ``datetime.datetime(...)`` call with an ISO-8601 string literal."""
    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3))
    hour = int(m.group(4) or 0)
    minute = int(m.group(5) or 0)
    second = int(m.group(6) or 0)
    micro = int(m.group(7) or 0)
    iso = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
    if micro:
        iso += f".{micro:06d}"
    iso += "+00:00"
    return repr(iso)  # quoted string literal


def _safe_parse(s: Any) -> Any:
    """Parse a Python-repr string into the corresponding object.

    Returns ``None`` for null/empty inputs or unparseable strings.
    """
    if not isinstance(s, str) or not s:
        return None
    cleaned = _DATETIME_RE.sub(_datetime_to_iso, s)
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError, TypeError):
        logger.debug("safe_parse failed: %.120s", s)
        return None


# ---- field extractors ----------------------------------------------------

USER_FIELDS: tuple[str, ...] = (
    "id_str",
    "username",
    "followersCount",
    "friendsCount",
    "statusesCount",
    "favouritesCount",
    "listedCount",
    "mediaCount",
    "verified",
    "blue",
    "created",
)


def _extract_user_fields(user_dict: dict | None) -> dict[str, Any]:
    """Pull a fixed schema out of the parsed user dict."""
    if user_dict is None:
        return {f"user_{k}": None for k in USER_FIELDS}
    return {f"user_{k}": user_dict.get(k) for k in USER_FIELDS}


def _parse_view_count(vc_dict: dict | None) -> int | None:
    if vc_dict is None:
        return None
    c = vc_dict.get("count")
    if c is None or c == "None":
        return None
    try:
        return int(c)
    except (TypeError, ValueError):
        return None


def _extract_link_urls(links_list: list | None) -> list[str]:
    """Pick the ``expanded_url`` from each link dict; drop missing/empty."""
    if not links_list:
        return []
    out: list[str] = []
    for link in links_list:
        if isinstance(link, dict):
            url = link.get("expanded_url")
            if isinstance(url, str) and url:
                out.append(url)
    return out


# ---- chunk and part loaders ---------------------------------------------

# Columns we drop from the cleaned output: redundant, structurally empty,
# or superseded by parsed equivalents.
_DROP_COLS: tuple[str, ...] = (
    "Unnamed: 0",
    "user",
    "viewCount",
    "links",
    "user_dict",
    "_user_created_raw",
    "_type",
    "type",          # uniformly "tweet-" — useless
    "location",      # always NaN at the tweet level (user-level location lives on user dict)
    "cash_app_handle",
    "url",           # tweet's own URL — not used downstream
    "media",         # we don't analyze media
    "hashtags",      # parseable but unused for now
    "mentionedUsers",
    "conversationIdStr",  # duplicate of conversationId, same precision loss
)


def load_chunk(path: str | Path) -> pd.DataFrame:
    """Load one USC chunk, parse nested fields, return a clean DataFrame.

    Output schema (notable columns):
        id, id_str                          — full-precision tweet id (string + int)
        epoch, created_at                   — float seconds + tz-aware datetime
        rawContent, text, lang
        replyCount, retweetCount, likeCount, quoteCount, view_count
        conversationId                       — float (~16 sig figs, lossy)
        in_reply_to_status_id_str            — float (~16 sig figs, lossy)
        in_reply_to_user_id_str              — float
        link_urls                            — list[str] of expanded URLs
        is_reply, is_quote, is_retweet, is_original
        user_id_str, user_username, user_followersCount, user_friendsCount,
        user_statusesCount, user_favouritesCount, user_listedCount,
        user_mediaCount, user_verified, user_blue, user_created_at
    """
    path = Path(path)
    logger.info("loading chunk %s", path.name)
    df = pd.read_csv(
        path,
        compression="gzip",
        dtype={"id": str, "id_str": str},
        low_memory=False,
    )

    # Coerce the float-id columns to numeric explicitly. Pandas' inference
    # is unreliable across chunks: in some chunks (e.g. part_3+ with mixed
    # numeric/blank rows) it falls back to dtype=object, which pyarrow then
    # refuses to write to parquet. ``pd.to_numeric(errors="coerce")`` makes
    # blanks NaN and parses scientific-notation strings as floats — the same
    # ~16-sig-fig precision loss we already document.
    for col in (
        "conversationId",
        "in_reply_to_status_id_str",
        "in_reply_to_user_id_str",
        "replyCount",
        "retweetCount",
        "likeCount",
        "quoteCount",
        "epoch",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse the three Python-repr columns we care about
    user_parsed = df["user"].map(_safe_parse)
    df["view_count"] = df["viewCount"].map(_safe_parse).map(_parse_view_count)
    df["link_urls"] = df["links"].map(_safe_parse).map(_extract_link_urls)

    # Explode the user dict into top-level columns
    user_records = user_parsed.map(_extract_user_fields)
    user_df = pd.DataFrame(list(user_records), index=df.index)
    df = pd.concat([df, user_df], axis=1)

    # Convert user_created (already an ISO string after our preprocessing) to datetime
    df["user_created_at"] = pd.to_datetime(df["user_created"], errors="coerce", utc=True)
    df = df.drop(columns=["user_created"], errors="ignore")

    # Tweet-type flags. Note: 'type' column is uniformly 'tweet-' — useless;
    # derive from structural signals instead.
    # quotedTweet / retweetedTweet arrive as either embedded-dict strings
    # (truthy), the literal string 'False' (falsy), or NaN. We coerce to
    # bool here so the parquet write doesn't choke on mixed dtypes; and we
    # overwrite the raw columns with their bool form so that the parquet
    # output stays consistent with parts 1/2 (which preserve these columns).
    def _coerce_tweet_ref(s: pd.Series) -> pd.Series:
        return s.notna() & (s.astype(str).str.lower() != "false") & (s.astype(str) != "")
    df["quotedTweet"] = _coerce_tweet_ref(df["quotedTweet"])
    df["retweetedTweet"] = _coerce_tweet_ref(df["retweetedTweet"])
    df["is_reply"] = df["in_reply_to_status_id_str"].notna()
    df["is_quote"] = df["quotedTweet"].astype(bool)
    df["is_retweet"] = df["retweetedTweet"].astype(bool)
    df["is_original"] = ~(df["is_reply"] | df["is_quote"] | df["is_retweet"])

    # Tweet timestamp from epoch
    df["created_at"] = pd.to_datetime(df["epoch"], unit="s", errors="coerce", utc=True)

    # Drop columns we no longer need
    df = df.drop(columns=list(_DROP_COLS), errors="ignore")

    return df


def load_part(
    part_dir: str | Path,
    *,
    max_chunks: int | None = None,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load all chunks in one part directory, optionally caching to parquet.

    If ``cache_path`` exists, it is read and returned directly. Otherwise, all
    matching ``*.csv.gz`` chunks are loaded, concatenated, and (if ``cache_path``
    is given) written to disk for reuse.
    """
    part_dir = Path(part_dir)
    cache_path = Path(cache_path) if cache_path else None

    if cache_path is not None and cache_path.exists():
        logger.info("loading cached part from %s", cache_path)
        return pd.read_parquet(cache_path)

    chunk_paths = sorted(part_dir.glob("*.csv.gz"))
    if max_chunks is not None:
        chunk_paths = chunk_paths[:max_chunks]
    if not chunk_paths:
        raise FileNotFoundError(f"no *.csv.gz chunks under {part_dir}")

    logger.info("loading %d chunks from %s", len(chunk_paths), part_dir)
    df = pd.concat((load_chunk(p) for p in chunk_paths), ignore_index=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("caching parsed part to %s", cache_path)
        df.to_parquet(cache_path, index=False)

    return df


def iter_chunks(
    part_dir: str | Path, *, max_chunks: int | None = None
) -> Iterable[tuple[Path, pd.DataFrame]]:
    """Stream chunks one at a time. Use when full-part memory is a concern."""
    part_dir = Path(part_dir)
    chunk_paths = sorted(part_dir.glob("*.csv.gz"))
    if max_chunks is not None:
        chunk_paths = chunk_paths[:max_chunks]
    for p in chunk_paths:
        yield p, load_chunk(p)

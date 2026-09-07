"""Phase 1 driver: load part_1, compute descriptive + cascade stats, label, report.

Run::

    uv run python -m analysis.run_phase1                # all 20 chunks
    uv run python -m analysis.run_phase1 --max-chunks 2 # quick sanity run
    uv run python -m analysis.run_phase1 --no-cache     # rebuild parquet cache

Writes:
    data/processed/part_1.parquet                       — cached parsed data
    paper/phase1_report.md                              — human-readable summary

The decision gate at the end of Phase 1 (the design specification) asks: is the data shape
consistent with the design? This report is the answer.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import cascades, labeling, loader

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PART_DIR = ROOT / "data" / "raw" / "usc-x-24" / "part_1"
DEFAULT_CACHE = ROOT / "data" / "processed" / "part_1.parquet"
DEFAULT_IFFY = ROOT / "data" / "labels" / "iffy_plus.csv"
DEFAULT_MAINSTREAM = ROOT / "data" / "labels" / "mainstream_domains.txt"
DEFAULT_REPORT = ROOT / "paper" / "phase1_report.md"


# ---- descriptive stats --------------------------------------------------

def describe_tweet_types(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    return {
        "n_tweets": n,
        "frac_reply": float(df["is_reply"].mean()),
        "frac_quote": float(df["is_quote"].mean()),
        "frac_retweet": float(df["is_retweet"].mean()),
        "frac_original": float(df["is_original"].mean()),
    }


def describe_engagement(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["replyCount", "retweetCount", "likeCount", "quoteCount", "view_count"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    out = sub.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).T
    out["nonzero_frac"] = (sub > 0).mean()
    return out[["count", "mean", "50%", "90%", "95%", "99%", "max", "nonzero_frac"]]


def describe_users(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "user_followersCount",
        "user_friendsCount",
        "user_statusesCount",
        "user_favouritesCount",
    ]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    out = sub.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).T
    return out[["count", "mean", "50%", "90%", "95%", "99%", "max"]]


def describe_user_flags(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    if "user_verified" in df.columns:
        v = df["user_verified"]
        out["frac_verified"] = float(v.fillna(False).astype(bool).mean())
    if "user_blue" in df.columns:
        b = df["user_blue"]
        out["frac_blue"] = float(b.fillna(False).astype(bool).mean())
    return out


def describe_conversation_sizes(df: pd.DataFrame) -> dict[str, float]:
    sizes = df.groupby("conversationId", sort=False).size()
    return {
        "n_conversations": int(len(sizes)),
        "n_size_ge_2": int((sizes >= 2).sum()),
        "n_size_ge_5": int((sizes >= 5).sum()),
        "n_size_ge_20": int((sizes >= 20).sum()),
        "median_size": float(sizes.median()),
        "p95_size": float(np.percentile(sizes, 95)),
        "max_size": int(sizes.max()),
    }


def describe_url_coverage(df: pd.DataFrame) -> dict[str, float]:
    # link_urls round-trips through parquet as numpy arrays (not Python lists),
    # so bool(arr) is ambiguous. Use len > 0 throughout.
    lengths = df["link_urls"].map(len)
    has_url = lengths > 0
    return {
        "frac_with_any_url": float(has_url.mean()),
        "n_with_any_url": int(has_url.sum()),
        "mean_urls_per_tweet": float(lengths.mean()),
    }


# ---- formatting helpers ------------------------------------------------

def _fmt_dict(d: dict[str, float]) -> str:
    rows = []
    for k, v in d.items():
        if isinstance(v, float):
            if abs(v) < 1 and v != 0:
                rows.append(f"  {k:30s} {v:.4f}")
            else:
                rows.append(f"  {k:30s} {v:,.2f}")
        else:
            rows.append(f"  {k:30s} {v:,}")
    return "\n".join(rows)


def _fmt_df(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.3f}")


# ---- main --------------------------------------------------------------

def run(
    part_dir: Path,
    cache_path: Path | None,
    iffy_path: Path,
    mainstream_path: Path,
    report_path: Path,
    max_chunks: int | None,
) -> None:
    logger.info("loading part %s (max_chunks=%s)", part_dir, max_chunks)
    df = loader.load_part(part_dir, max_chunks=max_chunks, cache_path=cache_path)
    logger.info("loaded %s rows x %s cols", f"{len(df):,}", len(df.columns))

    # Section 1: tweet types
    types = describe_tweet_types(df)

    # Section 2: engagement
    eng = describe_engagement(df)

    # Section 3: user features
    users = describe_users(df)
    user_flags = describe_user_flags(df)

    # Section 4: conversation sizes
    convs = describe_conversation_sizes(df)

    # Section 5: URL coverage
    urls = describe_url_coverage(df)

    # Section 6: cascade stats
    logger.info("computing cascade stats on conversations with size >= 2")
    stats_df = cascades.all_cascade_stats(df, min_size=2)
    quality = cascades.reply_tree_quality(stats_df)

    # Section 7: labeling
    logger.info("labeling against iffy+ and mainstream lists")
    labeled = labeling.add_labels(
        df,
        iffy_path=iffy_path,
        mainstream_path=mainstream_path,
    )
    coverage = labeling.coverage_summary(labeled)
    # Coverage among URL-bearing tweets (more meaningful than overall)
    url_bearing = labeled[labeled["link_urls"].map(len) > 0]
    coverage_among_url = labeling.coverage_summary(url_bearing) if len(url_bearing) else {}

    # Compose report (no leading indentation — substituted blocks are flush-left)
    report = (
f"""# Phase 1 Report — USC X 2024, part_1

Run on: {pd.Timestamp.utcnow().isoformat()}
Source: `{part_dir.relative_to(ROOT)}`
Chunks loaded: {len(list(part_dir.glob("*.csv.gz"))) if max_chunks is None else max_chunks}
Total tweets: {len(df):,}

---

## 1. Tweet types

```
{_fmt_dict(types)}
```

Per the design specification the dataset note says "~70% replies, ~10% quotes, remainder
originals." Compare above.

## 2. Engagement count distributions

```
{_fmt_df(eng)}
```

## 3. User feature distributions

Numeric features:
```
{_fmt_df(users)}
```

Flags:
```
{_fmt_dict(user_flags)}
```

## 4. Conversation sizes

```
{_fmt_dict(convs)}
```

## 5. URL coverage

```
{_fmt_dict(urls)}
```

## 6. Reply-tree cascade structure

Computed over conversations with >= 2 tweets.

```
{_fmt_dict(quality)}
```

Decision-gate question (the design specification Phase 1 checkpoint): are there enough
reply trees with non-trivial structure (depth >= 2) for validation? The
fraction `frac_depth_ge_2` directly answers this.

## 7. Credibility labeling coverage

Overall (all tweets):
```
{_fmt_dict(coverage)}
```

Among URL-bearing tweets only:
```
{_fmt_dict(coverage_among_url) if coverage_among_url else "  (no URL-bearing tweets)"}
```

The "labeled fraction" sets the upper bound for Stage 2's hypothesis-test
sample. If it is too low (<< 5% of tweets), revisit the labeling source
before scaling up.
""")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info("wrote report to %s", report_path)
    # Use sys.stdout.buffer to bypass Windows cp1252 stdout encoding,
    # in case any data values contained non-ASCII chars.
    import sys
    sys.stdout.buffer.write(report.encode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part-dir", type=Path, default=DEFAULT_PART_DIR)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--no-cache", action="store_true",
                   help="ignore and overwrite any existing parquet cache")
    p.add_argument("--iffy-path", type=Path, default=DEFAULT_IFFY)
    p.add_argument("--mainstream-path", type=Path, default=DEFAULT_MAINSTREAM)
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="load only the first N chunks (for quick iteration)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache_path = None if args.no_cache else args.cache_path
    if args.no_cache and args.cache_path.exists():
        args.cache_path.unlink()
        logger.info("removed stale cache %s", args.cache_path)

    run(
        part_dir=args.part_dir,
        cache_path=cache_path,
        iffy_path=args.iffy_path,
        mainstream_path=args.mainstream_path,
        report_path=args.report_path,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()

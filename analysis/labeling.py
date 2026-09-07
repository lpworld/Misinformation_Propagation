"""Domain-based credibility labeling against Iffy+ (low-cred) and a curated
mainstream list (high-cred).

A tweet is labeled by the set of registered domains in its ``link_urls``:

- All matched domains in Iffy+    → ``low_credibility``
- All matched domains in mainstream → ``high_credibility``
- Mix of both                    → ``mixed``
- No URLs / no list matches     → ``None`` (unlabeled)

This approximates "low-credibility *source*" rather than "false claim" —
state explicitly in the paper (per the design specification §Labeling).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import tldextract

logger = logging.getLogger(__name__)

LOW_CRED = "low_credibility"
HIGH_CRED = "high_credibility"
MIXED = "mixed"

# Use an offline tldextract — avoid hitting the network during runs.
# The bundled suffix list is current enough for our purposes.
_extractor = tldextract.TLDExtract(suffix_list_urls=())


def load_iffy_domains(path: str | Path) -> set[str]:
    """Read the Iffy+ CSV and return the set of registered domains, lowercased."""
    df = pd.read_csv(path)
    if "Domain" not in df.columns:
        raise ValueError(f"expected 'Domain' column in {path}, got {list(df.columns)}")
    domains = (
        df["Domain"].dropna().astype(str).str.strip().str.lower()
    )
    return {d for d in domains if d}


def load_mainstream_domains(path: str | Path) -> set[str]:
    """Read the curated mainstream-domain text file (one domain per line, ``#`` comments)."""
    out: set[str] = set()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip().lower()
        if line:
            out.add(line)
    return out


def url_to_registered_domain(url: str) -> str:
    """Extract registered domain (e.g., ``foo.example.co.uk`` → ``example.co.uk``)."""
    if not isinstance(url, str) or not url:
        return ""
    e = _extractor(url)
    if not e.domain or not e.suffix:
        return ""
    return f"{e.domain}.{e.suffix}".lower()


def label_urls(
    urls: Iterable[str] | None,
    iffy: set[str],
    mainstream: set[str],
) -> str | None:
    """Label one tweet given its expanded URLs.

    Returns ``None`` when no URLs are present, or none match either list.

    Accepts list, tuple, or numpy array of URLs (parquet round-trips lists
    as numpy arrays, so we use ``len`` rather than truth-testing).
    """
    if urls is None:
        return None
    try:
        if len(urls) == 0:
            return None
    except TypeError:
        return None
    domains = {url_to_registered_domain(u) for u in urls}
    domains.discard("")
    has_low = bool(domains & iffy)
    has_high = bool(domains & mainstream)
    if has_low and has_high:
        return MIXED
    if has_low:
        return LOW_CRED
    if has_high:
        return HIGH_CRED
    return None


def add_labels(
    df: pd.DataFrame,
    *,
    iffy_path: str | Path,
    mainstream_path: str | Path,
    url_col: str = "link_urls",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``credibility_label`` column added."""
    iffy = load_iffy_domains(iffy_path)
    mainstream = load_mainstream_domains(mainstream_path)
    overlap = iffy & mainstream
    if overlap:
        logger.warning(
            "overlap between iffy and mainstream domain sets: %s", sorted(overlap)
        )
    out = df.copy()
    out["credibility_label"] = out[url_col].map(
        lambda urls: label_urls(urls, iffy, mainstream)
    )
    return out


def coverage_summary(df: pd.DataFrame, *, label_col: str = "credibility_label") -> dict[str, int | float]:
    """Counts and percentages by label, including the unlabeled bucket."""
    n = len(df)
    counts = df[label_col].value_counts(dropna=False).to_dict()
    return {
        "total": n,
        "low_credibility": int(counts.get(LOW_CRED, 0)),
        "high_credibility": int(counts.get(HIGH_CRED, 0)),
        "mixed": int(counts.get(MIXED, 0)),
        "unlabeled": int(counts.get(None, 0) + counts.get(float("nan"), 0))
        if None in counts or any(isinstance(k, float) for k in counts) else n - sum(
            counts.get(k, 0) for k in (LOW_CRED, HIGH_CRED, MIXED)
        ),
        "labeled_fraction": (
            counts.get(LOW_CRED, 0) + counts.get(HIGH_CRED, 0) + counts.get(MIXED, 0)
        ) / n if n else 0.0,
    }

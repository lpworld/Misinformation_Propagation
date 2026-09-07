"""Content-based moral-emotional labeling via the NRC Emotion Lexicon.

Phase 4 v1 used URL-domain credibility labels (Iffy+ vs mainstream) and
found the USC 2024 election corpus does *not* exhibit the Vosoughi-2018
asymmetry — high-credibility tweets get more engagement than low-cred,
so the architectural-mechanism test came back null on cascade_size.

Phase 4 v2 partitions content by **moral-emotional intensity** instead:
the count of NRC anger/disgust/fear words per tweet (normalized by
tweet length). This taps the dual-process / reactive-engagement lever
the architectural claim is built on, independent of the credibility
framing.

Lexicon: NRC Word-Emotion Association Lexicon (Mohammad & Turney 2013),
research-use license. Format: ``word\\temotion\\tflag`` per line, one
row per (word × emotion). Stored at
``data/labels/nrc_emotion_lexicon.txt``.

Operationalization:

    score(tweet) = #{tokens ∈ EMOTION_WORDS} / max(#tokens, 1)

where EMOTION_WORDS is the union of NRC anger ∪ disgust words by default
(both classic "high-arousal negative" emotions tied to reactive
engagement; fear is excluded because it correlates with reflective
sharing — see Berger & Milkman 2012).

Partition: top quintile of non-zero scores → ``high_emotion``; tweets
with score == 0 → ``low_emotion``; middle → unlabeled. This forces a
clean dichotomy at the extremes, matching how the credibility labeling
worked (binary: low vs high, with a large unlabeled middle).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HIGH_EMOTION = "high_emotion"
LOW_EMOTION = "low_emotion"

DEFAULT_EMOTIONS: tuple[str, ...] = ("anger", "disgust")

# Tokenize on word characters; lowercase. Keeps the matcher fast.
_TOKEN_RE = re.compile(r"[A-Za-z']+")


def load_emotion_words(
    lex_path: str | Path,
    emotions: Iterable[str] = DEFAULT_EMOTIONS,
) -> set[str]:
    """Read NRC EmoLex (TSV) and return the union of words across ``emotions``."""
    lex = pd.read_csv(lex_path, sep="\t", header=None, names=["word", "emotion", "flag"])
    keep = lex[(lex["emotion"].isin(list(emotions))) & (lex["flag"] == 1)]
    out = set(keep["word"].astype(str).str.lower().tolist())
    logger.info(
        "loaded %d emotion words across %s from %s",
        len(out), tuple(emotions), lex_path,
    )
    return out


def emotion_score(text: str, emotion_words: set[str]) -> float:
    """Fraction of tokens in ``text`` that are in ``emotion_words`` (in [0, 1])."""
    if not isinstance(text, str) or not text:
        return 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in emotion_words)
    return hits / len(tokens)


def add_emotion_labels(
    df: pd.DataFrame,
    *,
    lex_path: str | Path,
    text_col: str = "rawContent",
    emotions: Iterable[str] = DEFAULT_EMOTIONS,
    high_quantile: float = 0.80,
) -> pd.DataFrame:
    """Add ``emotion_score`` and ``emotion_label`` columns to ``df``.

    Labeling rule:
        * score == 0 → ``LOW_EMOTION``
        * score ≥ quantile(high_quantile, scores>0) → ``HIGH_EMOTION``
        * everything else → ``None`` (unlabeled middle band)

    The quantile is computed over the *non-zero* score distribution, so
    high_quantile=0.80 means "top 20% of tweets that contain any
    emotion word." Together with the score==0 rule, this carves a clean
    dichotomy with a buffer in the middle.
    """
    words = load_emotion_words(lex_path, emotions=emotions)
    out = df.copy()
    out["emotion_score"] = out[text_col].fillna("").map(
        lambda s: emotion_score(s, words)
    ).astype(np.float32)

    nonzero = out.loc[out["emotion_score"] > 0, "emotion_score"]
    if len(nonzero) == 0:
        raise ValueError("all emotion_scores are zero; corpus has no lexicon hits")
    threshold = float(np.quantile(nonzero, high_quantile))

    label = np.full(len(out), None, dtype=object)
    label[out["emotion_score"].to_numpy() == 0.0] = LOW_EMOTION
    label[out["emotion_score"].to_numpy() >= threshold] = HIGH_EMOTION
    out["emotion_label"] = label
    out.attrs["emotion_threshold"] = threshold

    counts = out["emotion_label"].value_counts(dropna=False).to_dict()
    logger.info(
        "emotion labels — high (score ≥ %.4f): %d, low (score == 0): %d, "
        "middle band: %d",
        threshold,
        int(counts.get(HIGH_EMOTION, 0)),
        int(counts.get(LOW_EMOTION, 0)),
        int(counts.get(None, 0)) + int(counts.get(float("nan"), 0)),
    )
    return out


def emotion_coverage_summary(
    df: pd.DataFrame,
    *,
    label_col: str = "emotion_label",
) -> dict[str, int | float]:
    n = len(df)
    counts = df[label_col].value_counts(dropna=False).to_dict()
    high = int(counts.get(HIGH_EMOTION, 0))
    low = int(counts.get(LOW_EMOTION, 0))
    return {
        "total": n,
        "high_emotion": high,
        "low_emotion": low,
        "unlabeled_middle": n - high - low,
        "labeled_fraction": (high + low) / n if n else 0.0,
    }


__all__ = [
    "DEFAULT_EMOTIONS",
    "HIGH_EMOTION",
    "LOW_EMOTION",
    "add_emotion_labels",
    "emotion_coverage_summary",
    "emotion_score",
    "load_emotion_words",
]

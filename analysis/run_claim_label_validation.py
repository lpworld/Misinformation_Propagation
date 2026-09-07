"""RB6 — claim-level labeling validation (revision item 6).

Validates the URL-domain credibility labeling used in § 4.3 by sampling
labeled tweets, classifying them with a large-language-model rater as
claim-level misinformation / not / unclear, and reporting agreement
with the URL-domain label.

This is a *validation* of the source-based labeling, not a
re-labeling — sampling 200 + 200 tweets is enough to estimate
agreement to ±5% precision. Disagreement examples are saved for
inspection. The rater model is selected via the ``LLM_MODEL_ID``
environment variable, and API credentials come from
``ANTHROPIC_API_KEY``.

Run::

    python -m analysis.run_claim_label_validation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from anthropic import Anthropic

from analysis.hypothesis import HIGH, LOW
from analysis.labeling import add_labels

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


MODEL_ID = os.environ.get("LLM_MODEL_ID", "")
SYSTEM_PROMPT = (
    "You are classifying tweets from the 2024 U.S. election Twitter/X "
    "discourse. For each tweet, judge whether the tweet's CLAIMS (not the "
    "links it contains, and not the credibility of the linked source) are "
    "false, misleading, or unverifiable speculation about politics, "
    "elections, candidates, or current events.\n\n"
    "Respond with EXACTLY ONE of these tokens:\n"
    "- YES — the tweet itself states or implies a false/misleading/"
    "unverified factual claim\n"
    "- NO — the tweet's claims are factually defensible, or it makes no "
    "factual claim (e.g., expresses opinion, asks a question)\n"
    "- UNCLEAR — the tweet is too ambiguous or context-dependent to judge\n\n"
    "Output only the single token YES, NO, or UNCLEAR. No explanation."
)


def _classify_one(client: Anthropic, text: str) -> str:
    """One classification call. Returns YES/NO/UNCLEAR/ERROR."""
    if not text or not isinstance(text, str):
        return "UNCLEAR"
    text = text[:1000]  # truncate very long tweets to keep input bounded
    try:
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=8,
            system=[
                {
                    "type": "text", "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": f"Tweet:\n{text}"}],
        )
        out = "".join(b.text for b in resp.content if hasattr(b, "text")).strip().upper()
        if out.startswith("YES"):
            return "YES"
        if out.startswith("NO"):
            return "NO"
        return "UNCLEAR"
    except Exception as e:
        logger.warning("classification failed: %s", e)
        return "ERROR"


def _classify_batch(
    client: Anthropic, texts: list[str], *, max_workers: int = 8,
) -> list[str]:
    """Parallel classify a batch of texts. Returns list aligned with input."""
    n = len(texts)
    out = ["ERROR"] * n
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_classify_one, client, texts[i]): i for i in range(n)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            out[i] = fut.result()
            done += 1
            if done % 50 == 0:
                logger.info("  classified %d/%d", done, n)
    return out


def _stratified_sample_for_validation(
    df: pd.DataFrame, *, n_per_class: int, seed: int,
) -> pd.DataFrame:
    """Take n_per_class tweets per credibility class, prioritizing those
    with non-empty rawContent of reasonable length."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for label in (LOW, HIGH):
        pool = df[
            (df["credibility_label"] == label)
            & df["rawContent"].notna()
            & (df["rawContent"].astype(str).str.len() >= 30)
        ]
        if len(pool) == 0:
            raise ValueError(f"empty pool for {label}")
        n = min(n_per_class, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        parts.append(pool.iloc[idx][["id_str", "rawContent", "credibility_label", "link_urls"]])
    out = pd.concat(parts, ignore_index=True)
    return out.reset_index(drop=True)


def _cohen_kappa(rater1: np.ndarray, rater2: np.ndarray) -> float:
    """Cohen's kappa between two raters with same labelset."""
    cats = sorted(set(rater1) | set(rater2))
    n = len(rater1)
    if n == 0:
        return float("nan")
    confusion = pd.crosstab(pd.Series(rater1), pd.Series(rater2)).reindex(
        index=cats, columns=cats, fill_value=0,
    ).to_numpy()
    po = float(np.diag(confusion).sum() / n)
    p1 = confusion.sum(axis=1) / n
    p2 = confusion.sum(axis=0) / n
    pe = float((p1 * p2).sum())
    if pe >= 1.0:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def run(*, n_per_class: int = 200, seed: int = 1337, max_workers: int = 8) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    if not MODEL_ID:
        raise RuntimeError("LLM_MODEL_ID not set")

    out_dir = ROOT / "data" / "processed" / "phase5_extra" / "claim_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "paper" / "phase5_extra_claim_validation_report.md"

    parquet = ROOT / "data" / "processed" / "part_1.parquet"
    logger.info("loading %s", parquet)
    df = pd.read_parquet(parquet)
    df = add_labels(
        df,
        iffy_path=ROOT / "data" / "labels" / "iffy_plus.csv",
        mainstream_path=ROOT / "data" / "labels" / "mainstream_domains.txt",
    )

    logger.info("sampling %d tweets per class", n_per_class)
    sample = _stratified_sample_for_validation(df, n_per_class=n_per_class, seed=seed)
    logger.info("sample shape: %s", sample.shape)

    client = Anthropic()
    logger.info("classifying with %s", MODEL_ID)
    t0 = time.time()
    classifications = _classify_batch(
        client, sample["rawContent"].astype(str).tolist(),
        max_workers=max_workers,
    )
    dt = time.time() - t0
    logger.info("classified %d tweets in %.1fs", len(classifications), dt)

    sample["llm_label"] = classifications

    # Map: rater YES = claim-level misinformation; URL low_credibility = source-level low-cred.
    # Compute agreement on the YES vs NO axis (treat UNCLEAR as ambiguous).
    sample["url_yes"] = (sample["credibility_label"] == LOW).astype(int)
    sample["llm_yes"] = (sample["llm_label"] == "YES").astype(int)
    sample["llm_no"] = (sample["llm_label"] == "NO").astype(int)
    sample["llm_unclear"] = (sample["llm_label"] == "UNCLEAR").astype(int)
    sample["llm_error"] = (sample["llm_label"] == "ERROR").astype(int)

    # Drop ERROR rows for kappa computation; keep them for transparency.
    valid = sample[sample["llm_label"] != "ERROR"].copy()

    # Per-class breakdown
    class_breakdown: dict[str, dict[str, int]] = {}
    for cred_label, sub in valid.groupby("credibility_label"):
        class_breakdown[str(cred_label)] = {
            "n": int(len(sub)),
            "llm_YES": int((sub["llm_label"] == "YES").sum()),
            "llm_NO": int((sub["llm_label"] == "NO").sum()),
            "llm_UNCLEAR": int((sub["llm_label"] == "UNCLEAR").sum()),
        }

    # Agreement on YES vs NO axis (drop UNCLEAR for kappa)
    yn = valid[valid["llm_label"].isin(["YES", "NO"])].copy()
    if len(yn) > 0:
        agreement = float((yn["url_yes"] == yn["llm_yes"]).mean())
        kappa = _cohen_kappa(yn["url_yes"].to_numpy(), yn["llm_yes"].to_numpy())
    else:
        agreement = float("nan")
        kappa = float("nan")

    # Disagreements: where URL-low says misinformation but the LLM rater says NO,
    # or URL-high says credible but the LLM rater says YES.
    disagree_low_url_no_claim = valid[
        (valid["credibility_label"] == LOW) & (valid["llm_label"] == "NO")
    ].head(10)
    disagree_high_url_yes_claim = valid[
        (valid["credibility_label"] == HIGH) & (valid["llm_label"] == "YES")
    ].head(10)

    summary = {
        "model": MODEL_ID,
        "n_per_class": n_per_class,
        "n_classified": int(len(valid)),
        "n_errors": int(sample["llm_error"].sum()),
        "agreement_yn": agreement,
        "cohen_kappa_yn": kappa,
        "class_breakdown": class_breakdown,
        "elapsed_seconds": dt,
    }

    sample.to_parquet(out_dir / "sample_with_classifications.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(
        summary, sample, disagree_low_url_no_claim, disagree_high_url_yes_claim,
        report_path,
    )
    logger.info("wrote report → %s", report_path)


def write_report(
    summary: dict, sample: pd.DataFrame,
    dis_low: pd.DataFrame, dis_high: pd.DataFrame, path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# RB6 — Claim-level labeling validation\n")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}\n")
    lines.append(
        f"**Method.** Stratified sample of {summary['n_per_class']} tweets per "
        "URL-credibility class (low / high), classified by "
        f"`{summary['model']}` as YES (claim-level misinformation), NO "
        "(claims defensible / no factual claim), or UNCLEAR. We compare the "
        "LLM-rater classification to the URL-domain label used in § 4.3 to "
        "validate the source-based labeling assumption.\n"
    )
    lines.append("## Class breakdown\n")
    lines.append("| URL-domain label | n | YES (LLM) | NO (LLM) | UNCLEAR (LLM) |")
    lines.append("|---|---|---|---|---|")
    for label, blob in summary["class_breakdown"].items():
        lines.append(
            f"| {label} | {blob['n']} | {blob['llm_YES']} | "
            f"{blob['llm_NO']} | {blob['llm_UNCLEAR']} |"
        )
    lines.append("")

    lines.append("## Agreement on YES-vs-NO axis (drops UNCLEAR)\n")
    lines.append(f"- **Raw agreement:** {summary['agreement_yn']:.3f}")
    lines.append(f"- **Cohen's κ:** {summary['cohen_kappa_yn']:.3f}")
    lines.append("")
    lines.append(
        "**Interpretation.** URL-domain labels and LLM claim-level labels "
        "measure different things: the URL label asks about source "
        "credibility (publisher reliability), while the LLM label asks "
        "about the tweet text's factual content. Perfect agreement is not "
        "expected — many URL-low tweets share factually-defensible "
        "headlines, and some URL-high tweets contain misleading "
        "interpretations. Agreement around 60-75% with positive κ would "
        "support URL-domain labeling as a reasonable but imperfect proxy; "
        "agreement near chance would suggest the two labels capture "
        "different phenomena.\n"
    )

    lines.append("## Disagreement examples\n")
    lines.append("### URL-low / LLM-NO (URL says misinformation source; tweet's claims defensible per the LLM rater)\n")
    if len(dis_low) > 0:
        for _, row in dis_low.iterrows():
            text = str(row.get("rawContent", ""))[:200].replace("\n", " ")
            lines.append(f"- `{row['id_str']}`: {text}")
    else:
        lines.append("*(none)*")
    lines.append("")
    lines.append("### URL-high / LLM-YES (URL says credible; tweet's claims flagged by the LLM rater)\n")
    if len(dis_high) > 0:
        for _, row in dis_high.iterrows():
            text = str(row.get("rawContent", ""))[:200].replace("\n", " ")
            lines.append(f"- `{row['id_str']}`: {text}")
    else:
        lines.append("*(none)*")
    lines.append("")
    lines.append(
        f"\n## Run notes\n"
        f"- Errors: {summary['n_errors']} of {summary['n_classified'] + summary['n_errors']} "
        "(API failures or parse issues; excluded from agreement calc)\n"
        f"- Elapsed: {summary['elapsed_seconds']:.1f}s for "
        f"{summary['n_classified']} classifications\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(n_per_class=args.n_per_class, seed=args.seed, max_workers=args.max_workers)


if __name__ == "__main__":
    main()

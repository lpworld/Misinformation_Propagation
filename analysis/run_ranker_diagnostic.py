"""Ranker-prediction diagnostic on Phase 4 cascade seeds (revision item C.1).

Resolves the §6.4 contradiction by exposing what the *trained ranker
predicts* for low-credibility vs high-credibility seeds, separately from
what the *observed engagement counts* show. The architectural mechanism
operates on the former; §6.4 currently conflates them.

Reproduces the Phase 4 seed sample (seed=1337, n_per_label=2500 per
``configs/experiment_phase4.yaml``), runs the trained Phase-3 MaskNet
forward, and reports:

* Per-head predicted probabilities by credibility label
  (mean / median / p10 / p90 / p99).
* Predicted reactive-to-reflective ratio (corrected slow/fast partition:
  slow = reply + deep, fast = retweet + like). Reported in BOTH
  directions to disambiguate the §6.4 column-header bug
  (validation.py's ``observed_reactive_to_reflective`` actually returns
  ``reply / (retweet + quote + 1)`` — that's reflective/reactive).
* Aggregated scores under the three Phase-4 regimes (additive, ablated,
  additive_retuned), so we can see whether the score ordering matches
  the predicted-ratio story.
* Observed per-tweet engagement on the same 5,000 seeds, with the ratio
  computed in both directions and partitions for clarity.

Run::

    uv run python -m analysis.run_ranker_diagnostic
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from analysis.hypothesis import HIGH, LOW
from analysis.labeling import add_labels, coverage_summary
from analysis.run_phase4 import _stratified_label_seeds
from ranker.scoring import (
    ScoringConfig,
    aggregate_score,
    make_default_configs,
)
from ranker.training import load_ranker
from simulation.cascade import _seed_probs

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

# Dual-process partition (corrected, design log 2026-04-26).
SLOW_HEADS = ("reply", "deep")
FAST_HEADS = ("retweet", "like")

# Quantile probes used in the per-class summaries.
QUANTILES = (0.10, 0.50, 0.90, 0.99)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """``num / (den + eps)`` — additive smoothing, never divides by zero."""
    return num.astype(np.float64) / (den.astype(np.float64) + eps)


def _summarize_array(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    out: dict[str, float] = {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
    }
    for q in QUANTILES:
        out[f"p{int(q*100)}"] = float(np.quantile(x, q))
    return out


def _per_label_stats(values: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    """Stratify ``values`` by ``labels`` (LOW/HIGH) and summarize each subset."""
    out: dict[str, dict[str, float]] = {}
    for label in (LOW, HIGH):
        mask = labels == label
        if not mask.any():
            out[label] = {"n": 0}
            continue
        out[label] = _summarize_array(values[mask])
    return out


# ---- diagnostic core -----------------------------------------------------


def compute_predicted(probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-seed scalar quantities derived from the per-head predictions."""
    p_reply = probs["reply"]
    p_retweet = probs["retweet"]
    p_like = probs["like"]
    p_deep = probs["deep"]
    s_slow = p_reply + p_deep
    s_fast = p_retweet + p_like
    return {
        "p_reply": p_reply,
        "p_retweet": p_retweet,
        "p_like": p_like,
        "p_deep": p_deep,
        "s_slow_unweighted": s_slow,
        "s_fast_unweighted": s_fast,
        # Both directions, additively smoothed so zeros don't blow up.
        # The *mechanism* operates on reactive/reflective: high values are
        # what the ablated regime suppresses.
        "predicted_reactive_over_reflective": _safe_ratio(s_fast, s_slow, eps=1e-3),
        "predicted_reflective_over_reactive": _safe_ratio(s_slow, s_fast, eps=1e-3),
    }


def compute_observed(seeds: pd.DataFrame) -> dict[str, np.ndarray]:
    """Observed per-tweet engagement on the seed tweets themselves.

    Returns ratios in both directions and under both partitions:

    * **legacy_reflective_over_reactive**: ``replyCount / (retweetCount
      + quoteCount + 1)`` — the formula validation.py calls
      ``observed_reactive_to_reflective``. Reproduces the §6.4 numbers
      0.144 / 0.237 (mislabeled in the draft).
    * **corrected_reactive_over_reflective**: ``(retweetCount + likeCount)
      / (replyCount + quoteCount + 1)`` — the dual-process direction
      under the corrected slow/fast partition (quote → slow).
    * **corrected_reflective_over_reactive**: inverse of above.
    """
    rep = seeds["replyCount"].astype(np.float64).to_numpy()
    rt = seeds["retweetCount"].astype(np.float64).to_numpy()
    lk = seeds["likeCount"].astype(np.float64).to_numpy()
    qt = seeds["quoteCount"].astype(np.float64).to_numpy()
    return {
        "obs_replyCount": rep,
        "obs_retweetCount": rt,
        "obs_likeCount": lk,
        "obs_quoteCount": qt,
        "legacy_reflective_over_reactive": _safe_ratio(rep, rt + qt, eps=1.0),
        "corrected_reactive_over_reflective": _safe_ratio(rt + lk, rep + qt, eps=1.0),
        "corrected_reflective_over_reactive": _safe_ratio(rep + qt, rt + lk, eps=1.0),
    }


def compute_scores(
    probs: dict[str, np.ndarray],
    scoring_configs: dict[str, ScoringConfig],
) -> dict[str, np.ndarray]:
    """Aggregated scores under each scoring regime."""
    return {regime: aggregate_score(probs, cfg) for regime, cfg in scoring_configs.items()}


# ---- summary tabulation --------------------------------------------------


def build_summary(
    *,
    probs: dict[str, np.ndarray],
    derived: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, Any]:
    """Build the JSON-serializable summary that downstream artefacts read."""
    sections: dict[str, Any] = {}

    # Per-head predicted probabilities, stratified by label.
    sections["predicted_probabilities"] = {
        head: _per_label_stats(probs[head], labels) for head in ("reply", "retweet", "like", "deep")
    }

    # Predicted ratios.
    sections["predicted_ratios"] = {
        "reactive_over_reflective": _per_label_stats(
            derived["predicted_reactive_over_reflective"], labels
        ),
        "reflective_over_reactive": _per_label_stats(
            derived["predicted_reflective_over_reactive"], labels
        ),
    }

    # Predicted slow / fast aggregates (unweighted, for interpretability).
    sections["predicted_aggregates_unweighted"] = {
        "s_slow": _per_label_stats(derived["s_slow_unweighted"], labels),
        "s_fast": _per_label_stats(derived["s_fast_unweighted"], labels),
    }

    # Aggregated scores under each regime.
    sections["regime_scores"] = {
        regime: _per_label_stats(arr, labels) for regime, arr in scores.items()
    }

    # Observed counts and ratios on the same seeds (for the §6.4 cross-check).
    sections["observed_counts"] = {
        col.replace("obs_", ""): _per_label_stats(observed[col], labels)
        for col in ("obs_replyCount", "obs_retweetCount", "obs_likeCount", "obs_quoteCount")
    }
    sections["observed_ratios"] = {
        "legacy_reflective_over_reactive": _per_label_stats(
            observed["legacy_reflective_over_reactive"], labels
        ),
        "corrected_reactive_over_reflective": _per_label_stats(
            observed["corrected_reactive_over_reflective"], labels
        ),
        "corrected_reflective_over_reactive": _per_label_stats(
            observed["corrected_reflective_over_reactive"], labels
        ),
    }

    # Verdict on the §C.1 three-way branch.
    pred_rr = sections["predicted_ratios"]["reactive_over_reflective"]
    low_mean = pred_rr.get(LOW, {}).get("mean", float("nan"))
    high_mean = pred_rr.get(HIGH, {}).get("mean", float("nan"))
    if np.isfinite(low_mean) and np.isfinite(high_mean):
        if low_mean > high_mean:
            branch = "A_higher_for_low_cred_mechanism_consistent"
        elif abs(low_mean - high_mean) / max(high_mean, 1e-9) < 0.05:
            branch = "B_similar_revisit_mechanism_explanation"
        else:
            branch = "C_lower_for_low_cred_diagnose_before_publishing"
    else:
        branch = "indeterminate"
    sections["c1_branch"] = {
        "branch": branch,
        "low_mean_predicted_reactive_over_reflective": float(low_mean),
        "high_mean_predicted_reactive_over_reflective": float(high_mean),
        "delta_low_minus_high": float(low_mean - high_mean),
    }

    return sections


def per_seed_frame(
    *,
    seeds: pd.DataFrame,
    probs: dict[str, np.ndarray],
    derived: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    """One row per seed with everything needed for follow-up cuts."""
    out = pd.DataFrame({"credibility_label": seeds["credibility_label"].to_numpy()})
    out["id_str"] = seeds["id_str"].astype(str).to_numpy()
    out["user_followersCount"] = seeds["user_followersCount"].astype(np.float64).to_numpy()
    for head in ("reply", "retweet", "like", "deep"):
        out[f"p_{head}"] = probs[head]
    for col, arr in derived.items():
        if col not in {"p_reply", "p_retweet", "p_like", "p_deep"}:
            out[col] = arr
    for col, arr in observed.items():
        out[col] = arr
    for regime, arr in scores.items():
        out[f"score_{regime}"] = arr
    return out


# ---- markdown report -----------------------------------------------------


def _fmt(v: float, digits: int = 4) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def _stats_row(stats: dict[str, float]) -> str:
    return (
        f"{_fmt(stats.get('mean', float('nan')))} | "
        f"{_fmt(stats.get('p50', float('nan')))} | "
        f"{_fmt(stats.get('p10', float('nan')))} | "
        f"{_fmt(stats.get('p90', float('nan')))} | "
        f"{_fmt(stats.get('p99', float('nan')))}"
    )


def write_markdown(summary: dict[str, Any], n_low: int, n_high: int, path: Path) -> None:
    branch_blob = summary["c1_branch"]
    branch = branch_blob["branch"]
    low_mean = branch_blob["low_mean_predicted_reactive_over_reflective"]
    high_mean = branch_blob["high_mean_predicted_reactive_over_reflective"]
    delta = branch_blob["delta_low_minus_high"]

    lines: list[str] = []
    lines.append("# Phase 4 ranker-prediction diagnostic\n")
    lines.append(
        f"Run: {pd.Timestamp.utcnow().isoformat()}  \n"
        "Resolves the §6.4 contradiction (revision item C.1) by separating "
        "*ranker-predicted* engagement profiles from *observed* engagement "
        "counts.\n"
    )

    lines.append(
        f"**Sample**: {n_low + n_high:,} Phase-4 seeds — {n_low:,} low-credibility "
        f"+ {n_high:,} high-credibility, drawn with `seed=1337` from `part_1.parquet`.  \n"
        "**Ranker**: `data/processed/phase3_full/ranker/` (MaskNet, 4 heads).  \n"
        f"**Slow/fast partition (corrected)**: slow = `{', '.join(SLOW_HEADS)}`, "
        f"fast = `{', '.join(FAST_HEADS)}`.\n"
    )

    lines.append("## 1. §C.1 verdict\n")
    branch_label = {
        "A_higher_for_low_cred_mechanism_consistent": (
            "**Branch A — predicted reactive/reflective is *higher* for low-cred "
            "than high-cred.** The ranker has learned the asymmetry from features "
            "beyond observed engagement counts; the architectural mechanism's "
            "suppression of low-cred is mechanistically consistent with the "
            "claim. §6.4 can report this directly."
        ),
        "B_similar_revisit_mechanism_explanation": (
            "**Branch B — predicted ratios are *similar* across classes.** The "
            "suppression mechanism is operating on something other than the ratio "
            "(probably base-rate engagement levels: ranker predicts lower overall "
            "engagement for low-cred, and the multiplicative gate punishes that). "
            "§6.4 needs to be rewritten around the base-rate finding, not the "
            "ratio story."
        ),
        "C_lower_for_low_cred_diagnose_before_publishing": (
            "**Branch C — predicted reactive/reflective is *lower* for low-cred.** "
            "Surprising; the simulator's suppression of low-cred is happening for "
            "reasons orthogonal to the reactive-to-reflective signature. **Do not "
            "publish without diagnosing.**"
        ),
        "indeterminate": "Indeterminate — manual review needed.",
    }[branch]
    lines.append(branch_label + "\n")
    lines.append(
        f"- mean predicted reactive/reflective, low-cred: **{low_mean:.4f}**  \n"
        f"- mean predicted reactive/reflective, high-cred: **{high_mean:.4f}**  \n"
        f"- delta (low − high): **{delta:+.4f}**\n"
    )

    lines.append("## 2. Per-head predicted probabilities (stratified by label)\n")
    lines.append("| head | label | mean | p50 | p10 | p90 | p99 |")
    lines.append("|------|-------|------|-----|-----|-----|-----|")
    for head, blob in summary["predicted_probabilities"].items():
        for label, stats in blob.items():
            lines.append(f"| `{head}` | {label} | {_stats_row(stats)} |")
    lines.append("")

    lines.append("## 3. Predicted reactive/reflective ratio (the mechanism's input)\n")
    lines.append(
        "Reactive/reflective = (P(retweet) + P(like)) / (P(reply) + P(deep)). "
        "Higher = more reactive-leaning by the ranker's prediction. The ablated "
        "regime suppresses high-ratio content via the multiplicative gate.\n"
    )
    lines.append("| direction | label | mean | p50 | p10 | p90 | p99 |")
    lines.append("|-----------|-------|------|-----|-----|-----|-----|")
    for direction, blob in summary["predicted_ratios"].items():
        for label, stats in blob.items():
            lines.append(f"| {direction} | {label} | {_stats_row(stats)} |")
    lines.append("")

    lines.append("## 4. Aggregated scores per regime (mean per label)\n")
    lines.append("| regime | low mean | high mean | low − high |")
    lines.append("|--------|---------:|----------:|-----------:|")
    for regime, blob in summary["regime_scores"].items():
        lo = blob.get(LOW, {}).get("mean", float("nan"))
        hi = blob.get(HIGH, {}).get("mean", float("nan"))
        lines.append(f"| `{regime}` | {_fmt(lo)} | {_fmt(hi)} | {_fmt(lo - hi)} |")
    lines.append("")

    lines.append("## 5. Observed per-tweet counts on the same 5,000 seeds\n")
    lines.append("| count | label | mean | p50 | p10 | p90 | p99 |")
    lines.append("|-------|-------|------|-----|-----|-----|-----|")
    for col, blob in summary["observed_counts"].items():
        for label, stats in blob.items():
            lines.append(f"| `{col}` | {label} | {_stats_row(stats)} |")
    lines.append("")

    lines.append("## 6. Observed per-tweet ratios — disambiguates the §6.4 column header\n")
    lines.append(
        "The §6.4 table currently labels its ratio column "
        "`reactive/reflective` but reports values that match "
        "`legacy_reflective_over_reactive` "
        "= `replyCount / (retweetCount + quoteCount + 1)` — the formula "
        "`analysis/validation.py:observed_reactive_to_reflective` actually "
        "uses, despite its name. The §6.4 rewrite should pick a single "
        "convention and label it correctly.\n"
    )
    lines.append("| ratio | label | mean | p50 | p10 | p90 | p99 |")
    lines.append("|-------|-------|------|-----|-----|-----|-----|")
    for ratio_name, blob in summary["observed_ratios"].items():
        for label, stats in blob.items():
            lines.append(f"| {ratio_name} | {label} | {_stats_row(stats)} |")
    lines.append("")

    lines.append("## 7. Reading guide for the R2 §6.4 rewrite\n")
    lines.append(
        "1. The §6.4 column **header** says reactive/reflective; the **values** "
        "come from a reflective/reactive formula. Pick one direction and "
        "label it consistently.\n"
        "2. The mechanism operates on **predicted** ratios (§3 above), not "
        "observed counts (§6 above). The §6.4 narrative needs to make this "
        "distinction explicit.\n"
        "3. The §C.1 verdict above tells you what story §6.4 can claim. "
        "Don't reuse the language `\"low-cred has the higher reactive-to-"
        "reflective ratio\"` without verifying it against the predicted "
        "table — that phrasing is what triggered the contradiction.\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---- main ---------------------------------------------------------------


def run(config_path: Path, out_dir: Path, report_path: Path) -> None:
    cfg = _load_yaml(config_path)
    seed = int(cfg["random_seed"])

    parquet = ROOT / cfg["data"]["part_parquet"]
    logger.info("loading %s", parquet)
    df = pd.read_parquet(parquet)
    df = add_labels(
        df,
        iffy_path=ROOT / cfg["labels"]["iffy_path"],
        mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
    )
    cov = coverage_summary(df)
    logger.info(
        "label coverage: low=%d high=%d mixed=%d unlabeled=%d",
        cov["low_credibility"], cov["high_credibility"], cov["mixed"], cov["unlabeled"],
    )

    n_per_label = int(cfg["simulation"]["n_per_label"])
    seeds = _stratified_label_seeds(df, n_per_label, seed)
    labels = seeds["credibility_label"].to_numpy()
    n_low = int((labels == LOW).sum())
    n_high = int((labels == HIGH).sum())
    logger.info("seeds: low=%d high=%d", n_low, n_high)

    ranker_dir = ROOT / cfg["ranker"]["load_dir"]
    logger.info("loading ranker from %s", ranker_dir)
    model, scaler, _meta = load_ranker(ranker_dir)

    logger.info("running ranker forward pass")
    probs = _seed_probs(seeds, model, scaler)
    derived = compute_predicted(probs)
    observed = compute_observed(seeds)
    scoring_configs = make_default_configs()
    scores = compute_scores(probs, scoring_configs)

    summary = build_summary(
        probs=probs, derived=derived, observed=observed,
        scores=scores, labels=labels,
    )
    summary["meta"] = {
        "config_path": str(config_path.relative_to(ROOT)),
        "ranker_dir": str(ranker_dir.relative_to(ROOT)),
        "seed": seed,
        "n_per_label": n_per_label,
        "n_low": n_low,
        "n_high": n_high,
        "scoring_configs": {regime: asdict(cfg) for regime, cfg in scoring_configs.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = per_seed_frame(
        seeds=seeds, probs=probs, derived=derived, observed=observed, scores=scores,
    )
    df_out.to_parquet(out_dir / "diagnostic.parquet", index=False)
    (out_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
        encoding="utf-8",
    )
    logger.info("wrote per-seed parquet → %s", out_dir / "diagnostic.parquet")
    logger.info("wrote summary json    → %s", out_dir / "diagnostic_summary.json")

    write_markdown(summary, n_low=n_low, n_high=n_high, path=report_path)
    logger.info("wrote markdown report → %s", report_path)

    # Surface the §C.1 verdict in the log.
    logger.info("§C.1 verdict: %s", summary["c1_branch"]["branch"])
    logger.info(
        "  predicted reactive/reflective: low=%.4f, high=%.4f, delta=%+.4f",
        summary["c1_branch"]["low_mean_predicted_reactive_over_reflective"],
        summary["c1_branch"]["high_mean_predicted_reactive_over_reflective"],
        summary["c1_branch"]["delta_low_minus_high"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_phase4.yaml",
        help="Phase-4 config (defines seed, n_per_label, ranker dir).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase4" / "ranker_predictions",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=ROOT / "paper" / "phase4_ranker_diagnostic.md",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args.config.resolve(), args.out_dir.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()

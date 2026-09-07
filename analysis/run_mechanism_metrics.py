"""Mechanism process-metrics: how the score-aggregation gate acts on low- vs
high-credibility content (paper mechanism subsection).

Whereas ``run_ranker_diagnostic.py`` reports *what the ranker predicts* and the
Stage-2 hypothesis test reports *cascade-size gaps*, this script exposes the
*internal mechanics* of the architectural score-aggregation layer — the
intermediate quantities that show **how** the slow-gates-fast / reflective-floor
forms reweight content relative to the additive baseline.

Operating on the cached Phase-4 per-seed predictions
(``data/processed/phase4/ranker_predictions/diagnostic.parquet``; one row per
seed, 2500 low + 2500 high), it:

1. Rebuilds the per-head probability dict from the ``p_*`` columns and
   recomputes scores for ALL FIVE regimes (additive, ablated, additive_retuned,
   ratio_correction, reflective_floor) via :func:`ranker.scoring.aggregate_score`,
   sanity-checking the recomputed additive/ablated scores against the cached
   columns to ~1e-6.
2. Computes per-seed mechanism quantities:
   * **Weighted S_slow / S_fast** under the published weights.
   * **Reflective-floor gate** ``g = sigmoid((S_slow - floor) / floor_scale)``.
   * **Ablated multiplier** ``score_ablated / score_additive`` and
     **reflective-floor multiplier** ``score_reflective_floor / score_additive``.
   * **Percentile rank** of each seed under additive / ablated / reflective_floor
     (rank over all seeds, in [0, 1]) and the **rank displacement** vs additive
     (negative = demoted by the regime).
3. Stratifies everything by credibility class and builds a low-vs-high
   comparison table plus distribution quantiles (p10/p50/p90) of the gate and
   the rank displacements.
4. Writes a per-seed parquet (for later plotting), a summary JSON, and a
   markdown report with the comparison table and a short interpretation.

The architectural prediction: the gate demotes/suppresses low-credibility
content *more* than high-credibility — low-cred should show a lower mean gate,
a higher floor-suppressed fraction, and a more negative mean rank displacement.

Run::

    .venv/Scripts/python.exe -m analysis.run_mechanism_metrics
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

from analysis.hypothesis import HIGH, LOW
from ranker.scoring import (
    PUBLISHED_WEIGHTS,
    ScoringConfig,
    aggregate_score,
    make_default_configs,
    make_robustness_configs,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

# Fixed seed in case any future randomness is introduced (none currently).
SEED = 1337

HEADS = ("reply", "retweet", "like", "deep")
ALL_REGIMES = (
    "additive",
    "ablated",
    "additive_retuned",
    "ratio_correction",
    "reflective_floor",
)
# Quantile probes for distribution summaries.
QUANTILES = (0.10, 0.50, 0.90)
# Tolerance for the recomputed-vs-cached score sanity check.
SCORE_TOL = 1e-6


# ---- helpers -------------------------------------------------------------


def _scoring_configs() -> dict[str, ScoringConfig]:
    """All five regimes: the three default + the two robustness forms."""
    configs = dict(make_default_configs())
    configs.update(make_robustness_configs())
    return configs


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    """Rank of each element in [0, 1] over the whole array.

    ``rank = (#elements <= x) / n`` via average-rank ordering, robust to ties.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    # Average ranks for ties.
    sorted_x = x[order]
    i = 0
    avg_ranks = np.empty(n, dtype=np.float64)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        # positions i..j (0-based) share a value; average 1-based rank
        avg = (i + j) / 2.0 + 1.0
        avg_ranks[i : j + 1] = avg
        i = j + 1
    ranks[order] = avg_ranks
    return ranks / n


def _summarize_array(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    out: dict[str, float] = {
        "n": int(x.size),
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
    }
    for q in QUANTILES:
        out[f"p{int(q * 100)}"] = float(np.quantile(x, q)) if x.size else float("nan")
    return out


def _per_label_stats(values: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in (LOW, HIGH):
        mask = labels == label
        out[label] = _summarize_array(values[mask]) if mask.any() else {"n": 0}
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ---- mechanism core ------------------------------------------------------


def recompute_scores(
    probs: dict[str, np.ndarray],
    configs: dict[str, ScoringConfig],
) -> dict[str, np.ndarray]:
    """Aggregated score under each of the five regimes."""
    return {regime: aggregate_score(probs, cfg) for regime, cfg in configs.items()}


def sanity_check_scores(
    recomputed: dict[str, np.ndarray],
    cached: pd.DataFrame,
    tol: float = SCORE_TOL,
) -> dict[str, float]:
    """Compare recomputed additive/ablated/additive_retuned to cached columns.

    Returns the max absolute deviation per checked regime; raises if any
    exceeds ``tol``.
    """
    checks = {
        "additive": "score_additive",
        "ablated": "score_ablated",
        "additive_retuned": "score_additive_retuned",
    }
    devs: dict[str, float] = {}
    for regime, col in checks.items():
        if col not in cached.columns:
            continue
        dev = float(np.max(np.abs(recomputed[regime] - cached[col].to_numpy(dtype=np.float64))))
        devs[regime] = dev
        if dev > tol:
            raise ValueError(
                f"sanity check failed: recomputed {regime!r} deviates from cached "
                f"{col!r} by {dev:.3e} > tol {tol:.0e}"
            )
    return devs


def compute_mechanism_quantities(
    probs: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    floor: float,
    floor_scale: float,
) -> dict[str, np.ndarray]:
    """Per-seed mechanism quantities (weighted aggregates, gate, multipliers, ranks)."""
    w = PUBLISHED_WEIGHTS
    s_slow = w["reply"] * probs["reply"] + w["deep"] * probs["deep"]
    s_fast = w["retweet"] * probs["retweet"] + w["like"] * probs["like"]

    # Reflective-floor gate (matches scoring.py reflective_floor regime).
    gate = _sigmoid((s_slow - floor) / max(floor_scale, 1e-6))

    add = scores["additive"]
    # Multipliers relative to the additive baseline. Additive scores are
    # strictly positive here (weighted sums of positive probabilities), so the
    # ratio is well defined; guard with a tiny epsilon defensively.
    eps = 1e-12
    mult_ablated = scores["ablated"] / (add + eps)
    mult_floor = scores["reflective_floor"] / (add + eps)

    # Percentile ranks over all seeds.
    rank_add = _percentile_rank(add)
    rank_abl = _percentile_rank(scores["ablated"])
    rank_floor = _percentile_rank(scores["reflective_floor"])

    return {
        "s_slow_weighted": s_slow,
        "s_fast_weighted": s_fast,
        "gate": gate,
        "mult_ablated": mult_ablated,
        "mult_reflective_floor": mult_floor,
        "rank_additive": rank_add,
        "rank_ablated": rank_abl,
        "rank_reflective_floor": rank_floor,
        "disp_ablated": rank_abl - rank_add,
        "disp_reflective_floor": rank_floor - rank_add,
    }


# ---- summary tabulation --------------------------------------------------


def build_summary(
    *,
    mech: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
    score_devs: dict[str, float],
) -> dict[str, Any]:
    """JSON-serializable summary, including the low-vs-high comparison table."""
    sections: dict[str, Any] = {}

    # Full per-quantity per-label distributions.
    sections["distributions"] = {
        "s_slow_weighted": _per_label_stats(mech["s_slow_weighted"], labels),
        "s_fast_weighted": _per_label_stats(mech["s_fast_weighted"], labels),
        "gate": _per_label_stats(mech["gate"], labels),
        "mult_ablated": _per_label_stats(mech["mult_ablated"], labels),
        "mult_reflective_floor": _per_label_stats(mech["mult_reflective_floor"], labels),
        "disp_ablated": _per_label_stats(mech["disp_ablated"], labels),
        "disp_reflective_floor": _per_label_stats(mech["disp_reflective_floor"], labels),
    }

    # Regime score distributions for completeness.
    sections["regime_scores"] = {
        regime: _per_label_stats(arr, labels) for regime, arr in scores.items()
    }

    # The headline by-class comparison table.
    def _by_class(fn) -> dict[str, float]:
        out: dict[str, float] = {}
        for label in (LOW, HIGH):
            mask = labels == label
            out[label] = float(fn(mask)) if mask.any() else float("nan")
        return out

    gate = mech["gate"]
    comparison: dict[str, dict[str, float]] = {
        "mean_s_slow": _by_class(lambda m: np.mean(mech["s_slow_weighted"][m])),
        "mean_s_fast": _by_class(lambda m: np.mean(mech["s_fast_weighted"][m])),
        "mean_gate": _by_class(lambda m: np.mean(gate[m])),
        "frac_gate_below_0.5": _by_class(lambda m: np.mean(gate[m] < 0.5)),
        "mean_mult_ablated": _by_class(lambda m: np.mean(mech["mult_ablated"][m])),
        "mean_mult_reflective_floor": _by_class(lambda m: np.mean(mech["mult_reflective_floor"][m])),
        "mean_disp_ablated": _by_class(lambda m: np.mean(mech["disp_ablated"][m])),
        "mean_disp_reflective_floor": _by_class(lambda m: np.mean(mech["disp_reflective_floor"][m])),
        "frac_demoted_ablated": _by_class(lambda m: np.mean(mech["disp_ablated"][m] < 0)),
        "frac_demoted_reflective_floor": _by_class(lambda m: np.mean(mech["disp_reflective_floor"][m] < 0)),
    }
    # Add low-minus-high deltas for convenience.
    for key, blob in comparison.items():
        blob["low_minus_high"] = blob[LOW] - blob[HIGH]
    sections["comparison"] = comparison

    # Verdict on the architectural direction.
    lower_gate_low = comparison["mean_gate"][LOW] < comparison["mean_gate"][HIGH]
    higher_supp_low = comparison["frac_gate_below_0.5"][LOW] > comparison["frac_gate_below_0.5"][HIGH]
    more_neg_disp_abl = comparison["mean_disp_ablated"][LOW] < comparison["mean_disp_ablated"][HIGH]
    more_neg_disp_floor = comparison["mean_disp_reflective_floor"][LOW] < comparison["mean_disp_reflective_floor"][HIGH]
    sections["verdict"] = {
        "low_cred_lower_mean_gate": bool(lower_gate_low),
        "low_cred_higher_suppressed_fraction": bool(higher_supp_low),
        "low_cred_more_negative_disp_ablated": bool(more_neg_disp_abl),
        "low_cred_more_negative_disp_reflective_floor": bool(more_neg_disp_floor),
        "mechanism_demotes_low_cred_more": bool(
            lower_gate_low and higher_supp_low and more_neg_disp_abl and more_neg_disp_floor
        ),
    }

    sections["sanity_check"] = {
        "max_abs_dev": score_devs,
        "tol": SCORE_TOL,
        "passed": all(d <= SCORE_TOL for d in score_devs.values()),
    }

    return sections


def per_seed_frame(
    *,
    labels: np.ndarray,
    mech: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    followers: np.ndarray,
) -> pd.DataFrame:
    """One row per seed with mechanism quantities, all five scores, ranks, displacements."""
    out = pd.DataFrame({"credibility_label": labels})
    out["user_followersCount"] = followers
    out["s_slow_weighted"] = mech["s_slow_weighted"]
    out["s_fast_weighted"] = mech["s_fast_weighted"]
    out["gate"] = mech["gate"]
    for regime in ALL_REGIMES:
        out[f"score_{regime}"] = scores[regime]
    out["mult_ablated"] = mech["mult_ablated"]
    out["mult_reflective_floor"] = mech["mult_reflective_floor"]
    out["rank_additive"] = mech["rank_additive"]
    out["rank_ablated"] = mech["rank_ablated"]
    out["rank_reflective_floor"] = mech["rank_reflective_floor"]
    out["disp_ablated"] = mech["disp_ablated"]
    out["disp_reflective_floor"] = mech["disp_reflective_floor"]
    return out


# ---- markdown report -----------------------------------------------------


def _fmt(v: float, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def write_markdown(summary: dict[str, Any], n_low: int, n_high: int, path: Path) -> None:
    comp = summary["comparison"]
    verdict = summary["verdict"]
    dist = summary["distributions"]

    rows = [
        ("mean S_slow (weighted)", "mean_s_slow"),
        ("mean S_fast (weighted)", "mean_s_fast"),
        ("mean gate g", "mean_gate"),
        ("frac suppressed by floor (g < 0.5)", "frac_gate_below_0.5"),
        ("mean ablated multiplier", "mean_mult_ablated"),
        ("mean reflective_floor multiplier", "mean_mult_reflective_floor"),
        ("mean rank-displacement (ablated)", "mean_disp_ablated"),
        ("mean rank-displacement (reflective_floor)", "mean_disp_reflective_floor"),
        ("frac demoted (ablated, disp < 0)", "frac_demoted_ablated"),
        ("frac demoted (reflective_floor, disp < 0)", "frac_demoted_reflective_floor"),
    ]

    lines: list[str] = []
    lines.append("# Mechanism process-metrics: how the score-aggregation gate acts by credibility\n")
    lines.append(
        f"Run: {pd.Timestamp.utcnow().isoformat()}  \n"
        "Internal quantities of the architectural score-aggregation layer, "
        "showing **how** the slow-gates-fast (`ablated`) and `reflective_floor` "
        "forms reweight content relative to the additive baseline. Operates on "
        "the cached Phase-4 per-seed ranker predictions.\n"
    )
    lines.append(
        f"**Sample**: {n_low + n_high:,} seeds — {n_low:,} low-credibility "
        f"+ {n_high:,} high-credibility.  \n"
        f"**Weights** (published): reply={PUBLISHED_WEIGHTS['reply']}, "
        f"deep={PUBLISHED_WEIGHTS['deep']}, retweet={PUBLISHED_WEIGHTS['retweet']}, "
        f"like={PUBLISHED_WEIGHTS['like']}.  \n"
        "**Gate**: `g = sigmoid((S_slow - floor) / floor_scale)`, floor=1.0, "
        "floor_scale=0.5. **Rank**: percentile over all seeds; displacement = "
        "rank_regime − rank_additive (negative = demoted).\n"
    )

    sc = summary["sanity_check"]
    lines.append("## Sanity check\n")
    devs = ", ".join(f"`{k}`={v:.2e}" for k, v in sc["max_abs_dev"].items())
    lines.append(
        f"Recomputed vs cached score max-abs-dev: {devs} "
        f"(tol {sc['tol']:.0e}). **Passed: {sc['passed']}**.\n"
    )

    lines.append("## By-class comparison (low vs high)\n")
    lines.append("| quantity | low-credibility | high-credibility | low − high |")
    lines.append("|----------|----------------:|-----------------:|-----------:|")
    for label, key in rows:
        blob = comp[key]
        lines.append(
            f"| {label} | {_fmt(blob[LOW])} | {_fmt(blob[HIGH])} | "
            f"{_fmt(blob['low_minus_high'])} |"
        )
    lines.append("")

    lines.append("## Distribution quantiles (p10 / p50 / p90) by class\n")
    lines.append("| quantity | label | p10 | p50 | p90 |")
    lines.append("|----------|-------|----:|----:|----:|")
    for qty in ("gate", "disp_ablated", "disp_reflective_floor"):
        for label in (LOW, HIGH):
            s = dist[qty][label]
            lines.append(
                f"| {qty} | {label} | {_fmt(s.get('p10', float('nan')))} | "
                f"{_fmt(s.get('p50', float('nan')))} | {_fmt(s.get('p90', float('nan')))} |"
            )
    lines.append("")

    lines.append("## Interpretation\n")
    direction_ok = verdict["mechanism_demotes_low_cred_more"]
    if direction_ok:
        interp = (
            "The mechanism acts asymmetrically in the predicted direction: "
            "low-credibility seeds carry a lower mean reflective-floor gate "
            f"({_fmt(comp['mean_gate'][LOW])} vs {_fmt(comp['mean_gate'][HIGH])}) and a "
            f"larger floor-suppressed fraction ({_fmt(comp['frac_gate_below_0.5'][LOW])} vs "
            f"{_fmt(comp['frac_gate_below_0.5'][HIGH])}), because their weighted slow "
            "(effortful) engagement is lower. "
            "Consequently both the multiplicative `ablated` gate and the "
            "`reflective_floor` gate demote low-credibility content more in "
            f"percentile rank (mean displacement {_fmt(comp['mean_disp_ablated'][LOW])} / "
            f"{_fmt(comp['mean_disp_reflective_floor'][LOW])} for low vs "
            f"{_fmt(comp['mean_disp_ablated'][HIGH])} / "
            f"{_fmt(comp['mean_disp_reflective_floor'][HIGH])} for high). "
            "This is the internal-mechanics counterpart to the Stage-2 cascade-gap "
            "result: the architectural gate, not the parameter values, is what "
            "differentially suppresses reactive-leaning low-credibility content."
        )
    else:
        interp = (
            "The mechanism does NOT cleanly demote low-credibility content more "
            "across all four indicators — see the comparison table above and "
            "diagnose before using these numbers in the mechanism subsection. "
            f"(low/high mean gate {_fmt(comp['mean_gate'][LOW])}/"
            f"{_fmt(comp['mean_gate'][HIGH])}; "
            f"low/high mean ablated displacement {_fmt(comp['mean_disp_ablated'][LOW])}/"
            f"{_fmt(comp['mean_disp_ablated'][HIGH])}.)"
        )
    lines.append(interp + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---- main ---------------------------------------------------------------


def run(diagnostic_path: Path, out_dir: Path, report_path: Path) -> None:
    np.random.seed(SEED)

    logger.info("loading %s", diagnostic_path)
    df = pd.read_parquet(diagnostic_path)
    labels = df["credibility_label"].to_numpy()
    followers = df["user_followersCount"].astype(np.float64).to_numpy()
    n_low = int((labels == LOW).sum())
    n_high = int((labels == HIGH).sum())
    logger.info("seeds: low=%d high=%d total=%d", n_low, n_high, len(df))

    probs = {head: df[f"p_{head}"].to_numpy(dtype=np.float64) for head in HEADS}

    configs = _scoring_configs()
    logger.info("recomputing scores for regimes: %s", ", ".join(configs))
    scores = recompute_scores(probs, configs)

    score_devs = sanity_check_scores(scores, df)
    logger.info(
        "sanity check passed: max-abs-dev %s",
        {k: f"{v:.2e}" for k, v in score_devs.items()},
    )

    floor_cfg = configs["reflective_floor"]
    mech = compute_mechanism_quantities(
        probs, scores, floor=floor_cfg.floor, floor_scale=floor_cfg.floor_scale
    )

    summary = build_summary(mech=mech, scores=scores, labels=labels, score_devs=score_devs)
    summary["meta"] = {
        "diagnostic_path": str(diagnostic_path.relative_to(ROOT)),
        "seed": SEED,
        "n_low": n_low,
        "n_high": n_high,
        "published_weights": dict(PUBLISHED_WEIGHTS),
        "scoring_configs": {regime: asdict(cfg) for regime, cfg in configs.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = per_seed_frame(labels=labels, mech=mech, scores=scores, followers=followers)
    per_seed_path = out_dir / "mechanism_per_seed.parquet"
    per_seed.to_parquet(per_seed_path, index=False)
    logger.info("wrote per-seed parquet → %s", per_seed_path)

    summary_path = out_dir / "mechanism_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x),
        ),
        encoding="utf-8",
    )
    logger.info("wrote summary json     → %s", summary_path)

    write_markdown(summary, n_low=n_low, n_high=n_high, path=report_path)
    logger.info("wrote markdown report  → %s", report_path)

    v = summary["verdict"]
    logger.info(
        "verdict: mechanism_demotes_low_cred_more=%s "
        "(lower_gate=%s, higher_suppressed=%s, more_neg_disp_ablated=%s, "
        "more_neg_disp_floor=%s)",
        v["mechanism_demotes_low_cred_more"],
        v["low_cred_lower_mean_gate"],
        v["low_cred_higher_suppressed_fraction"],
        v["low_cred_more_negative_disp_ablated"],
        v["low_cred_more_negative_disp_reflective_floor"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--diagnostic",
        type=Path,
        default=ROOT / "data" / "processed" / "phase4" / "ranker_predictions" / "diagnostic.parquet",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase4" / "mechanism",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=ROOT / "paper" / "mechanism_metrics_report.md",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args.diagnostic.resolve(), args.out_dir.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()

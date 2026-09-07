"""Collateral cost of the reflective floor on UNLABELED content (referee E5).

The paper's targeting evidence only ever scores the ~2.7% URL-labeled seeds.
This script measures who *else* gets demoted: 5,000 unlabeled seeds (no
credibility label) sampled from the same processed part_1 corpus, pooled with
the 5,000 labeled Phase-4 seeds, ranked under additive and reflective_floor.

Seed-eligibility filters, replicated from the Phase-4 seed construction
(``analysis.run_phase4._stratified_label_seeds`` on
``configs/experiment_phase4.yaml``):

* the pool is ``data/processed/part_1.parquet`` after ``analysis.labeling
  .add_labels`` — Phase 4 applies **no** eligibility filter beyond the label
  match (no originals-only filter, no language filter, no engagement floor;
  ``simulation.content.sample_seeds``'s ``originals_only`` switch is not used);
* the labeled pools are ``credibility_label == low/high_credibility``; the
  unlabeled pool is therefore ``credibility_label is null`` (tweets whose
  links match neither list, or that carry no links). Rows labeled ``mixed``
  are excluded — they are labeled, just not analyzed;
* sampling mirrors ``_stratified_label_seeds``: ``np.random.default_rng(1337)
  .choice`` over the pool, without replacement (pool ≫ n), same SEED_COLS.

Ranker probabilities for the unlabeled seeds come from the cached trained
ranker (+ scaler) via ``simulation.cascade._seed_probs``; the labeled seeds'
probabilities come from the cached Phase-4 diagnostic parquet.

Reported:

* demotion distribution (mean, median, q10/q90, % demoted > 1 pct point) for
  unlabeled vs low-cred vs high-cred, under POOLED (n = 10,000) ranking;
* composition of the most-demoted pooled decile (baseline shares: 50% / 25% /
  25%);
* substitutability-share distribution of demoted vs undemoted unlabeled
  content (is the collateral concentrated on the same fast-substitution
  signature?);
* mean predicted reactive/reflective ratio R of demoted vs undemoted
  unlabeled seeds.

Run from the repo root with the project venv::

    .venv/Scripts/python.exe -m analysis.run_unlabeled_collateral
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from analysis.hypothesis import HIGH, LOW
from analysis.labeling import add_labels
from analysis.run_solution_validation import (
    fast_contrib,
    percentile_rank,
    reflective_floor_score,
    slow_contrib,
)
from ranker.training import load_ranker
from simulation.cascade import _seed_probs
from simulation.content import SEED_COLS

logger = logging.getLogger("run_unlabeled_collateral")
ROOT = Path(__file__).resolve().parents[1]

SEED = 1337
N_UNLABELED = 5_000
RATIO_EPS = 1e-3  # paper's predicted_reactive_over_reflective smoothing
DEMOTED_THRESHOLD = 1.0  # percentile points; "% demoted > 1 pct point"

PHASE4_CONFIG = ROOT / "configs" / "experiment_phase4.yaml"
DIAGNOSTIC_PARQUET = (
    ROOT / "data" / "processed" / "phase4" / "ranker_predictions" / "diagnostic.parquet"
)
OUT_DIR = ROOT / "data" / "processed" / "phase5_extra" / "unlabeled_collateral"
REPORT_PATH = ROOT / "paper" / "unlabeled_collateral_report.md"

GROUP_UNLABELED = "unlabeled"


# ---- sampling --------------------------------------------------------------


def sample_unlabeled_seeds(cfg: dict[str, Any], n: int, seed: int) -> pd.DataFrame:
    """Sample ``n`` unlabeled seeds with the Phase-4 eligibility filters.

    Mirrors ``analysis.run_phase4._stratified_label_seeds`` exactly, except the
    pool predicate is ``credibility_label is null`` instead of a label match.
    """
    part_parquet = ROOT / cfg["data"]["part_parquet"]
    logger.info("loading corpus: %s", part_parquet)
    df = pd.read_parquet(part_parquet)
    df = add_labels(
        df,
        iffy_path=ROOT / cfg["labels"]["iffy_path"],
        mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
    )
    n_total = len(df)
    counts = df["credibility_label"].value_counts(dropna=False)
    logger.info("label counts:\n%s", counts.to_string())

    pool = df[df["credibility_label"].isna()]
    logger.info(
        "unlabeled pool: %s / %s rows (%.2f%%); 'mixed' rows excluded (labeled, not analyzed)",
        f"{len(pool):,}", f"{n_total:,}", 100.0 * len(pool) / n_total,
    )
    if len(pool) == 0:
        raise ValueError("unlabeled pool is empty")

    rng = np.random.default_rng(seed)
    replace = len(pool) < n
    idx = rng.choice(len(pool), size=n, replace=replace)
    seeds = pool.iloc[idx].copy()
    cols = [c for c in SEED_COLS if c in seeds.columns]
    return seeds[cols].reset_index(drop=True)


# ---- summaries ---------------------------------------------------------------


def demotion_stats(d: np.ndarray) -> dict[str, float]:
    """Distribution summary of per-seed demotion (percentile points)."""
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "q10": float(np.quantile(d, 0.10)),
        "q90": float(np.quantile(d, 0.90)),
        f"frac_demoted_gt_{DEMOTED_THRESHOLD:g}pp": float((d > DEMOTED_THRESHOLD).mean()),
        "frac_promoted_lt_-1pp": float((d < -DEMOTED_THRESHOLD).mean()),
    }


def dist_summary(x: np.ndarray) -> dict[str, float]:
    return {
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else float("nan"),
        "median": float(np.median(x)) if x.size else float("nan"),
        "q10": float(np.quantile(x, 0.10)) if x.size else float("nan"),
        "q90": float(np.quantile(x, 0.90)) if x.size else float("nan"),
    }


# ---- report ------------------------------------------------------------------


def _fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if np.isfinite(v) else "—"


def write_report(results: dict[str, Any], path: Path) -> None:
    meta = results["meta"]
    dm = results["demotion_by_group"]
    dec = results["most_demoted_decile"]
    sub = results["unlabeled_demoted_vs_not"]

    lines: list[str] = []
    lines.append("# Collateral cost of the reflective floor on unlabeled content\n")
    lines.append(
        f"Run: {pd.Timestamp.utcnow().isoformat()}  \n"
        "Referee question (E5): the paper's targeting evidence only scores the ~2.7% "
        "URL-labeled seeds — who *else* does the reflective floor demote? "
        f"{meta['n_unlabeled']:,} unlabeled seeds (Phase-4 eligibility filters, seed "
        f"{meta['seed']}) pooled with the {meta['n_labeled']:,} labeled Phase-4 seeds; "
        "ranking and demotion computed over the pooled n = "
        f"{meta['n_pooled']:,}.\n"
    )
    lines.append(
        "**Eligibility filters replicated.** Phase-4 seed sampling "
        "(`run_phase4._stratified_label_seeds`) draws from `part_1.parquet` after "
        "`add_labels` with *no* filter other than the label match (no originals-only, "
        "no language, no engagement floor). The unlabeled pool is therefore "
        "`credibility_label is null` "
        f"({meta['unlabeled_pool_size']:,} of {meta['corpus_size']:,} rows; `mixed` "
        "rows excluded as labeled-but-unanalyzed); sampling uses the same "
        "`np.random.default_rng(1337).choice` mechanics and SEED_COLS.\n"
    )
    lines.append(
        "**Scoring.** Cached trained ranker + scaler → per-head probabilities; "
        "additive S = 13.5·p_reply + 2.0·p_deep + 1.0·p_retweet + 0.5·p_like; "
        "reflective floor = S · σ((S_slow − 1.0)/0.5); "
        "demotion = rank-pct(additive) − rank-pct(floor), pooled percentiles.\n"
    )

    lines.append("## Demotion by group (pooled ranking)\n")
    lines.append("| group | n | mean | median | q10 | q90 | % demoted > 1 pp | % promoted < −1 pp |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for g in (GROUP_UNLABELED, LOW, HIGH):
        s = dm[g]
        lines.append(
            f"| {g} | {s['n']:,} | {_fmt(s['mean'], 2)} | {_fmt(s['median'], 2)} | "
            f"{_fmt(s['q10'], 2)} | {_fmt(s['q90'], 2)} | "
            f"{100 * s['frac_demoted_gt_1pp']:.1f}% | {100 * s['frac_promoted_lt_-1pp']:.1f}% |"
        )
    lines.append("")

    lines.append("## Composition of the most-demoted pooled decile\n")
    lines.append(
        f"Top decile by demotion (n = {dec['n_decile']:,}); baseline shares from pool "
        "composition: unlabeled 50.0%, low-cred 25.0%, high-cred 25.0%.\n"
    )
    lines.append("| group | count | share of decile | baseline share | enrichment |")
    lines.append("|---|---:|---:|---:|---:|")
    for g in (GROUP_UNLABELED, LOW, HIGH):
        c = dec["composition"][g]
        lines.append(
            f"| {g} | {c['count']:,} | {100 * c['share']:.1f}% | "
            f"{100 * c['baseline_share']:.1f}% | {c['enrichment']:.2f}× |"
        )
    lines.append("")

    lines.append("## Is the unlabeled collateral on the fast-substitution signature?\n")
    lines.append(
        f"Unlabeled seeds split at demotion > {DEMOTED_THRESHOLD:g} pp "
        f"(demoted n = {sub['n_demoted']:,}, undemoted n = {sub['n_undemoted']:,}).\n"
    )
    lines.append("| quantity | demoted unlabeled | undemoted unlabeled |")
    lines.append("|---|---:|---:|")
    for key, name in (
        ("substitutability_share", "substitutability share (fast / additive)"),
        ("predicted_R", "predicted reactive/reflective ratio R"),
        ("s_slow", "slow level L = S_slow"),
    ):
        a, b = sub["demoted"][key], sub["undemoted"][key]
        lines.append(
            f"| mean {name} | {_fmt(a['mean'])} | {_fmt(b['mean'])} |"
        )
        lines.append(
            f"| &nbsp;&nbsp;median (q10–q90) | {_fmt(a['median'])} "
            f"({_fmt(a['q10'])}–{_fmt(a['q90'])}) | {_fmt(b['median'])} "
            f"({_fmt(b['q10'])}–{_fmt(b['q90'])}) |"
        )
    lines.append("")

    # ---- honest reading -------------------------------------------------
    u, lo, hi = dm[GROUP_UNLABELED], dm[LOW], dm[HIGH]
    dec_u = dec["composition"][GROUP_UNLABELED]
    sub_gap = sub["demoted"]["substitutability_share"]["mean"] - sub["undemoted"]["substitutability_share"]["mean"]
    r_gap = sub["demoted"]["predicted_R"]["mean"] - sub["undemoted"]["predicted_R"]["mean"]

    lines.append("## Reading\n")
    reading: list[str] = []
    net_promoted = u["mean"] < 0
    reading.append(
        f"**Scale of the collateral.** "
        + (
            f"Unlabeled content is on net *promoted*, not demoted, by the floor: mean "
            f"demotion {_fmt(u['mean'], 2)} pp (median {_fmt(u['median'], 2)}), with "
            f"{100 * u['frac_promoted_lt_-1pp']:.1f}% gaining more than 1 percentile point "
            f"against only {100 * u['frac_demoted_gt_1pp']:.1f}% losing more than 1 pp — "
            f"versus {100 * lo['frac_demoted_gt_1pp']:.1f}% of low-credibility and "
            f"{100 * hi['frac_demoted_gt_1pp']:.1f}% of high-credibility seeds demoted. "
            "The typical unlabeled seed inherits rank that the floor strips from "
            "fast-loaded labeled content."
            if net_promoted
            else f"The reflective floor is not free for unlabeled content: "
            f"{100 * u['frac_demoted_gt_1pp']:.1f}% of unlabeled seeds lose more than "
            f"1 percentile point (mean demotion {_fmt(u['mean'], 2)} pp, q90 "
            f"{_fmt(u['q90'], 2)} pp), compared with "
            f"{100 * lo['frac_demoted_gt_1pp']:.1f}% of low-credibility and "
            f"{100 * hi['frac_demoted_gt_1pp']:.1f}% of high-credibility seeds."
        )
    )
    reading.append(
        f"**Who fills the most-demoted decile.** Unlabeled content makes up "
        f"{100 * dec_u['share']:.1f}% of the most-demoted pooled decile against a 50% "
        f"baseline ({dec_u['enrichment']:.2f}× enrichment); low-cred is "
        f"{100 * dec['composition'][LOW]['share']:.1f}% (baseline 25%, "
        f"{dec['composition'][LOW]['enrichment']:.2f}×) and high-cred "
        f"{100 * dec['composition'][HIGH]['share']:.1f}% (baseline 25%, "
        f"{dec['composition'][HIGH]['enrichment']:.2f}×). "
        + (
            "In absolute terms most of the demoted mass is unlabeled — unavoidable when "
            "97% of the corpus carries no URL label — so any deployment claim must be "
            "phrased as *signature-targeted*, not *label-targeted*."
            if dec_u["share"] >= 0.4
            else "The decile is dominated by labeled content despite the 50% unlabeled baseline."
        )
    )
    reading.append(
        f"**Signature check.** Demoted unlabeled seeds do sit on the fast-substitution "
        f"signature: mean substitutability share {_fmt(sub['demoted']['substitutability_share']['mean'])} "
        f"vs {_fmt(sub['undemoted']['substitutability_share']['mean'])} for undemoted "
        f"(gap {sub_gap:+.3f}), and mean predicted R "
        f"{_fmt(sub['demoted']['predicted_R']['mean'])} vs "
        f"{_fmt(sub['undemoted']['predicted_R']['mean'])} (gap {r_gap:+.3f}). "
        + (
            "The collateral is at least *consistent* — the floor demotes the same "
            "engagement signature regardless of label, which is exactly what an "
            "architecture-level (content-blind) intervention should do."
            if sub_gap > 0 and r_gap > 0
            else "The demoted unlabeled set does NOT show a clearly elevated "
            "fast-substitution signature — this would undercut the targeting claim "
            "and must be reported."
        )
    )
    reading.append(
        "**Honest framing for the paper.** The floor cannot distinguish an unlabeled "
        "seed with a fast-loaded engagement profile from a low-credibility one — by "
        "design. Whether the demoted unlabeled mass is a *cost* depends on its true "
        "credibility mix, which we cannot observe; the URL-label analysis shows the "
        "signature is enriched for low-credibility where labels exist, but that "
        "enrichment cannot be assumed to transfer to the unlabeled 97%. The paper "
        "should state the collateral numbers above explicitly rather than implying "
        "the reallocation touches only low-credibility content."
    )
    lines.append("\n\n".join(reading) + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote report → %s", path)


# ---- main --------------------------------------------------------------------


def run(
    *,
    config_path: Path,
    diagnostic_parquet: Path,
    out_dir: Path,
    report_path: Path,
    n_unlabeled: int = N_UNLABELED,
) -> dict[str, Any]:
    np.random.seed(SEED)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # --- corpus + unlabeled sample ---------------------------------------
    part_parquet = ROOT / cfg["data"]["part_parquet"]
    df = pd.read_parquet(part_parquet)
    df = add_labels(
        df,
        iffy_path=ROOT / cfg["labels"]["iffy_path"],
        mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
    )
    corpus_size = len(df)
    pool = df[df["credibility_label"].isna()]
    pool_size = len(pool)
    logger.info(
        "corpus %s rows; unlabeled pool %s (%.2f%%)",
        f"{corpus_size:,}", f"{pool_size:,}", 100.0 * pool_size / corpus_size,
    )
    rng = np.random.default_rng(SEED)
    idx = rng.choice(pool_size, size=n_unlabeled, replace=pool_size < n_unlabeled)
    unl = pool.iloc[idx].copy()
    cols = [c for c in SEED_COLS if c in unl.columns]
    unl = unl[cols].reset_index(drop=True)
    del df, pool

    # --- ranker probabilities ---------------------------------------------
    ranker_dir = ROOT / cfg["ranker"]["load_dir"]
    logger.info("loading ranker: %s", ranker_dir)
    model, scaler, _meta = load_ranker(ranker_dir)
    probs_u = _seed_probs(unl, model, scaler)

    logger.info("loading cached labeled diagnostic: %s", diagnostic_parquet)
    diag = pd.read_parquet(diagnostic_parquet)

    groups = np.concatenate(
        [
            np.full(len(unl), GROUP_UNLABELED, dtype=object),
            diag["credibility_label"].to_numpy(dtype=object),
        ]
    )
    p_reply = np.concatenate([probs_u["reply"], diag["p_reply"].to_numpy(np.float64)])
    p_retweet = np.concatenate([probs_u["retweet"], diag["p_retweet"].to_numpy(np.float64)])
    p_like = np.concatenate([probs_u["like"], diag["p_like"].to_numpy(np.float64)])
    p_deep = np.concatenate([probs_u["deep"], diag["p_deep"].to_numpy(np.float64)])

    # --- pooled scoring -----------------------------------------------------
    s_slow = slow_contrib(p_reply, p_deep)
    s_fast = fast_contrib(p_retweet, p_like)
    s_add = s_slow + s_fast
    s_rf = reflective_floor_score(s_add, s_slow)
    sub_share = s_fast / s_add
    r_ratio = (p_retweet + p_like) / (p_reply + p_deep + RATIO_EPS)

    rank_add = percentile_rank(s_add)
    rank_rf = percentile_rank(s_rf)
    demotion = rank_add - rank_rf

    n_pooled = len(demotion)
    u_mask = groups == GROUP_UNLABELED
    low_mask = groups == LOW
    high_mask = groups == HIGH

    # --- demotion distributions ---------------------------------------------
    dm = {
        GROUP_UNLABELED: demotion_stats(demotion[u_mask]),
        LOW: demotion_stats(demotion[low_mask]),
        HIGH: demotion_stats(demotion[high_mask]),
    }
    for g, s in dm.items():
        logger.info(
            "%s: mean=%.2f median=%.2f q10=%.2f q90=%.2f frac>1pp=%.3f",
            g, s["mean"], s["median"], s["q10"], s["q90"], s["frac_demoted_gt_1pp"],
        )

    # --- most-demoted pooled decile composition ------------------------------
    n_decile = n_pooled // 10
    decile_idx = np.argsort(demotion)[-n_decile:]
    dec_groups = groups[decile_idx]
    baseline = {
        GROUP_UNLABELED: float(u_mask.mean()),
        LOW: float(low_mask.mean()),
        HIGH: float(high_mask.mean()),
    }
    composition: dict[str, Any] = {}
    for g in (GROUP_UNLABELED, LOW, HIGH):
        cnt = int((dec_groups == g).sum())
        share = cnt / n_decile
        composition[g] = {
            "count": cnt,
            "share": share,
            "baseline_share": baseline[g],
            "enrichment": share / baseline[g] if baseline[g] > 0 else float("nan"),
        }
        logger.info(
            "decile composition %s: %d (%.1f%%, baseline %.1f%%, %.2fx)",
            g, cnt, 100 * share, 100 * baseline[g], composition[g]["enrichment"],
        )

    # --- demoted vs undemoted unlabeled: signature ----------------------------
    d_u = demotion[u_mask]
    demoted_u = d_u > DEMOTED_THRESHOLD
    sub_block: dict[str, Any] = {
        "threshold_pp": DEMOTED_THRESHOLD,
        "n_demoted": int(demoted_u.sum()),
        "n_undemoted": int((~demoted_u).sum()),
        "demoted": {
            "substitutability_share": dist_summary(sub_share[u_mask][demoted_u]),
            "predicted_R": dist_summary(r_ratio[u_mask][demoted_u]),
            "s_slow": dist_summary(s_slow[u_mask][demoted_u]),
        },
        "undemoted": {
            "substitutability_share": dist_summary(sub_share[u_mask][~demoted_u]),
            "predicted_R": dist_summary(r_ratio[u_mask][~demoted_u]),
            "s_slow": dist_summary(s_slow[u_mask][~demoted_u]),
        },
    }
    logger.info(
        "unlabeled demoted (n=%d) sub_share=%.3f R=%.3f | undemoted (n=%d) sub_share=%.3f R=%.3f",
        sub_block["n_demoted"],
        sub_block["demoted"]["substitutability_share"]["mean"],
        sub_block["demoted"]["predicted_R"]["mean"],
        sub_block["n_undemoted"],
        sub_block["undemoted"]["substitutability_share"]["mean"],
        sub_block["undemoted"]["predicted_R"]["mean"],
    )

    results: dict[str, Any] = {
        "meta": {
            "seed": SEED,
            "config": str(config_path.relative_to(ROOT)),
            "diagnostic_parquet": str(diagnostic_parquet.relative_to(ROOT)),
            "corpus_size": corpus_size,
            "unlabeled_pool_size": pool_size,
            "n_unlabeled": int(len(unl)),
            "n_labeled": int(len(diag)),
            "n_pooled": int(n_pooled),
            "eligibility_filters": (
                "Replicated from run_phase4._stratified_label_seeds: pool = "
                "part_1.parquet after add_labels; ONLY filter is the label predicate "
                "(here: credibility_label is null; 'mixed' rows excluded as labeled). "
                "No originals-only / language / engagement filters exist in Phase 4. "
                "Sampling: np.random.default_rng(1337).choice, replace only if pool < n; "
                "SEED_COLS columns."
            ),
            "scoring": (
                "slow = 13.5*p_reply + 2.0*p_deep; fast = 1.0*p_retweet + 0.5*p_like; "
                "additive = slow + fast; reflective_floor = additive * "
                "sigmoid((slow - 1.0)/0.5); demotion = pooled rank-pct(additive) - "
                "pooled rank-pct(reflective_floor)"
            ),
        },
        "demotion_by_group": dm,
        "most_demoted_decile": {"n_decile": int(n_decile), "composition": composition},
        "unlabeled_demoted_vs_not": sub_block,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote results → %s", out_dir / "results.json")

    # Per-seed parquet for downstream plotting / auditing.
    per_seed = pd.DataFrame(
        {
            "group": groups,
            "p_reply": p_reply,
            "p_retweet": p_retweet,
            "p_like": p_like,
            "p_deep": p_deep,
            "s_slow": s_slow,
            "s_fast": s_fast,
            "score_additive": s_add,
            "score_reflective_floor": s_rf,
            "substitutability_share": sub_share,
            "predicted_R": r_ratio,
            "rank_additive": rank_add,
            "rank_reflective_floor": rank_rf,
            "demotion": demotion,
        }
    )
    per_seed.to_parquet(out_dir / "pooled_per_seed.parquet", index=False)
    logger.info("wrote per-seed → %s", out_dir / "pooled_per_seed.parquet")

    write_report(results, report_path)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=PHASE4_CONFIG)
    p.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC_PARQUET)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--report", type=Path, default=REPORT_PATH)
    p.add_argument("--n-unlabeled", type=int, default=N_UNLABELED)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(
        config_path=args.config.resolve(),
        diagnostic_parquet=args.diagnostic.resolve(),
        out_dir=args.out_dir.resolve(),
        report_path=args.report.resolve(),
        n_unlabeled=args.n_unlabeled,
    )


if __name__ == "__main__":
    main()

"""Truncation sensitivity analysis for the Phase 4 ``cascade_size`` contrast.

Background (design log 2026-04-26 — Phase 4 v4): under the corrected slow/fast
partition, the bootstrap CI on the ablated regime's cascade_size contrast
came in at [-9.39, +1.71] — heavily skewed negative but not strictly excluding
zero. Separately, Stage 1 calibration shows the simulator over-produces
engagement at the p90 by ~3× from observed (design log 2026-04-26 — NB event
model entry). These two facts are potentially related: the borderline-null
on cascade_size may be driven by simulator behavior in exactly the regime
where the simulator is least calibrated.

This script asks: does the cascade_size contrast survive when we constrain
the analysis to the body of the distribution where the simulator is well
calibrated? Cap the per-cascade replicate-mean cascade_size at a quantile
drawn from the *observed* aggregate-engagement distribution
(``replyCount + retweetCount + quoteCount`` per conversation — the
observed analog of the simulator's ``n_reply + n_retweet + n_deep``), then
recompute the paired bootstrap contrast on the capped values.

Three possible outcomes, all valid:

1. **Truncated contrast strengthens (CI moves further from zero):**
   the original null was a tail-noise artifact; the architectural finding
   is real on cascade_size too.
2. **Truncated contrast weakens (CI moves closer to zero or reverses sign):**
   the original tentative-negative point estimate was driven by simulator
   tail behavior; the cascade_size finding doesn't survive sensitivity.
3. **Truncated contrast is similar to untruncated:** the tail isn't
   load-bearing in either direction; the borderline-null is what it is.

Whatever the outcome, document and report it. The quantile sweep is
pre-specified in the config; do not tune the cap to get a desired answer.

This is an additional analysis script. It does NOT modify ``analysis/
run_phase4.py``, ``analysis/run_phase5.py``, or any cached Phase 4
outputs — it reads ``data/processed/phase4/ablation_per_cascade.parquet``
and writes a new report ``paper/phase4_truncation_sensitivity.md``. The
original Phase 4 results remain exactly reproducible as run.

Run::

    uv run python -m analysis.run_truncation_sensitivity \\
        --config configs/experiment_truncation_sensitivity.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from analysis.hypothesis import (
    HIGH,
    LOW,
    cascade_bootstrap_contrast,
)
from analysis.validation import observed_aggregate_engagement

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


# ---- config + utilities ------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rel_to_root(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


@contextmanager
def _timed(label: str, timings: dict[str, float]):
    t0 = time.time()
    logger.info("▶ %s", label)
    try:
        yield
    finally:
        dt = time.time() - t0
        timings[label] = dt
        logger.info("✓ %s — %.2fs", label, dt)


# ---- core computation --------------------------------------------------

@dataclass(frozen=True)
class TruncatedContrast:
    """Paired bootstrap contrast on cascade_size after capping at a quantile."""

    quantile: float
    cap_value: float
    n_low_kept: int
    n_high_kept: int
    n_low_clipped: int           # cascades whose baseline value exceeded cap
    n_high_clipped: int
    regime_a: str
    regime_b: str
    diff_point: float
    diff_ci_lo: float
    diff_ci_hi: float


def replicate_mean_wide(
    per_cascade: pd.DataFrame,
    *,
    metric_col_expr,
    regimes: tuple[str, ...],
) -> pd.DataFrame:
    """Build a wide per-cascade frame: one row per cascade, one column per regime.

    ``metric_col_expr`` is a callable that takes the per-cascade DataFrame and
    returns a 1-D numpy array of per-row metric values (e.g.,
    ``n_reply + n_retweet + n_deep``). We average those values across
    replicates for each (regime, cascade_id) — this is the
    "AFTER replicate-averaging" choice from the truncation design.
    """
    work = per_cascade[
        ["regime", "replicate", "cascade_id", "credibility_label"]
    ].copy()
    work["_v"] = metric_col_expr(per_cascade)

    long = (
        work.groupby(
            ["regime", "cascade_id", "credibility_label"], sort=False
        )["_v"]
        .mean()
        .reset_index()
        .rename(columns={"_v": "value"})
    )

    wide = long.pivot_table(
        index=["cascade_id", "credibility_label"],
        columns="regime",
        values="value",
    ).dropna(subset=list(regimes))
    return wide.reset_index()


def _cascade_size_expr(pc: pd.DataFrame) -> np.ndarray:
    return (pc["n_reply"] + pc["n_retweet"] + pc["n_deep"]).to_numpy(dtype=np.float64)


def _audience_reach_expr(pc: pd.DataFrame) -> np.ndarray:
    return pc["total_exposures"].to_numpy(dtype=np.float64)


def winsorize_wide(
    wide: pd.DataFrame,
    *,
    regimes: tuple[str, ...],
    cap: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Clip per-regime values at ``cap`` and return a count of clipped rows.

    Winsorization preserves cascade pairing across regimes (every cascade
    still has an entry under every regime), which is required for the
    paired bootstrap to make sense. The returned counts split by label so
    the report can show whether clipping bites asymmetrically.
    """
    out = wide.copy()
    counts: dict[str, dict[str, int]] = {}
    for regime in regimes:
        col = out[regime].to_numpy(dtype=np.float64)
        clipped_mask = col > cap
        out[regime] = np.minimum(col, cap)
        counts[regime] = {
            "low": int(
                ((wide["credibility_label"] == LOW) & clipped_mask).sum()
            ),
            "high": int(
                ((wide["credibility_label"] == HIGH) & clipped_mask).sum()
            ),
        }
    return out, counts


def truncated_bootstrap_contrast(
    wide_clipped: pd.DataFrame,
    *,
    quantile: float,
    cap_value: float,
    regime_a: str,
    regime_b: str,
    n_bootstrap: int,
    rng: np.random.Generator,
    clip_counts: dict[str, dict[str, int]],
) -> TruncatedContrast:
    """Paired bootstrap on the clipped values; mirrors ``cascade_bootstrap_contrast``."""
    bc = cascade_bootstrap_contrast(
        wide_clipped,
        regime_a=regime_a,
        regime_b=regime_b,
        metric_name="cascade_size_truncated",
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    low_mask = wide_clipped["credibility_label"] == LOW
    high_mask = wide_clipped["credibility_label"] == HIGH
    return TruncatedContrast(
        quantile=quantile,
        cap_value=cap_value,
        n_low_kept=int(low_mask.sum()),
        n_high_kept=int(high_mask.sum()),
        n_low_clipped=clip_counts[regime_b]["low"],
        n_high_clipped=clip_counts[regime_b]["high"],
        regime_a=regime_a,
        regime_b=regime_b,
        diff_point=bc.diff_point,
        diff_ci_lo=bc.diff_ci_lo,
        diff_ci_hi=bc.diff_ci_hi,
    )


# ---- driver ------------------------------------------------------------

def run(config_path: Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    seed = int(cfg["random_seed"])
    timings: dict[str, float] = {}

    out_dir = ROOT / cfg["artifacts"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / cfg["artifacts"]["report_path"]

    pc_path = ROOT / cfg["phase4"]["per_cascade_parquet"]
    obs_path = ROOT / cfg["data"]["observed_part_parquet"]
    if not pc_path.exists():
        raise FileNotFoundError(
            f"Phase 4 cached output not found at {pc_path}. Run "
            "`uv run python -m analysis.run_phase4` first."
        )

    # --- 1. observed cap distribution ----------------------------------
    with _timed(f"load observed parquet ({obs_path.name})", timings):
        obs_df = pd.read_parquet(obs_path, columns=[
            "conversationId", "replyCount", "retweetCount", "quoteCount",
        ])
        obs_cascade_size = observed_aggregate_engagement(obs_df)
        logger.info(
            "  observed cascade_size: n=%d median=%.1f mean=%.2f p90=%.1f p99=%.1f",
            obs_cascade_size.size,
            float(np.median(obs_cascade_size)),
            float(np.mean(obs_cascade_size)),
            float(np.percentile(obs_cascade_size, 90)),
            float(np.percentile(obs_cascade_size, 99)),
        )

    # --- 2. cached Phase 4 per-cascade frame ---------------------------
    with _timed(f"load Phase 4 per-cascade ({pc_path.name})", timings):
        pc = pd.read_parquet(pc_path)
        if "credibility_label" not in pc.columns:
            raise ValueError(
                f"{pc_path} missing credibility_label column — was this "
                "produced by the current run_phase4.py?"
            )
        labeled = pc[pc["credibility_label"].isin([LOW, HIGH])].copy()
        regimes = tuple(sorted(labeled["regime"].unique().tolist()))
        baseline_regime = str(cfg["baseline_regime"])
        if baseline_regime not in regimes:
            raise ValueError(
                f"baseline regime {baseline_regime!r} not in cached regimes {regimes}"
            )
        comparator_regimes = tuple(r for r in regimes if r != baseline_regime)
        logger.info(
            "  regimes=%s baseline=%s comparators=%s n_replicates(per regime/cascade)≈%d",
            regimes, baseline_regime, comparator_regimes,
            int(
                labeled.groupby(["regime", "cascade_id"]).size().median()
            ),
        )

    # --- 3. per-cascade replicate means (cascade_size + audience_reach)
    with _timed("replicate-mean per cascade (cascade_size, audience_reach)", timings):
        wide_size = replicate_mean_wide(
            labeled, metric_col_expr=_cascade_size_expr, regimes=regimes,
        )
        wide_reach = replicate_mean_wide(
            labeled, metric_col_expr=_audience_reach_expr, regimes=regimes,
        )

    # --- 4. baseline (untruncated) bootstrap contrasts -----------------
    n_boot = int(cfg["bootstrap"]["iterations"])
    with _timed(f"baseline bootstrap (untruncated, n={n_boot})", timings):
        rng = np.random.default_rng(seed)
        baseline_size: dict[str, Any] = {}
        baseline_reach: dict[str, Any] = {}
        for regime in comparator_regimes:
            bc_size = cascade_bootstrap_contrast(
                wide_size, regime_a=baseline_regime, regime_b=regime,
                metric_name="cascade_size",
                n_bootstrap=n_boot, rng=rng,
            )
            bc_reach = cascade_bootstrap_contrast(
                wide_reach, regime_a=baseline_regime, regime_b=regime,
                metric_name="audience_reach",
                n_bootstrap=n_boot, rng=rng,
            )
            baseline_size[regime] = bc_size
            baseline_reach[regime] = bc_reach

    # --- 5. truncated bootstrap contrasts at each quantile -------------
    quantiles = [float(q) for q in cfg["truncation"]["quantiles"]]
    if cfg["truncation"].get("mode", "winsorize") != "winsorize":
        raise NotImplementedError(
            "Only mode='winsorize' is implemented. Drop-mode is documented "
            "in the report but intentionally not run as primary, since "
            "it breaks paired bootstrap when a cascade exceeds the cap "
            "under one regime but not another."
        )

    with _timed(f"truncated bootstrap × {len(quantiles)} quantiles", timings):
        truncated: dict[float, dict[str, TruncatedContrast]] = {}
        cap_values: dict[float, float] = {}
        for q in quantiles:
            cap = float(np.percentile(obs_cascade_size, q))
            cap_values[q] = cap
            wide_clipped, clip_counts = winsorize_wide(
                wide_size, regimes=regimes, cap=cap,
            )
            truncated[q] = {}
            # Reseed per quantile so each quantile's bootstrap is reproducible
            # given (seed, q) without entanglement to sweep ordering.
            rng_q = np.random.default_rng(seed + int(q))
            for regime in comparator_regimes:
                truncated[q][regime] = truncated_bootstrap_contrast(
                    wide_clipped,
                    quantile=q, cap_value=cap,
                    regime_a=baseline_regime, regime_b=regime,
                    n_bootstrap=n_boot, rng=rng_q,
                    clip_counts=clip_counts,
                )
            logger.info(
                "  q=%.0f cap=%.2f → %s",
                q, cap,
                ", ".join(
                    f"{r}: diff={truncated[q][r].diff_point:+.3f} "
                    f"CI=[{truncated[q][r].diff_ci_lo:+.3f}, "
                    f"{truncated[q][r].diff_ci_hi:+.3f}]"
                    for r in comparator_regimes
                ),
            )

    # --- 6. write report + JSON metrics --------------------------------
    write_report(
        config_path=config_path,
        cfg=cfg,
        regimes=regimes,
        baseline_regime=baseline_regime,
        comparator_regimes=comparator_regimes,
        obs_cascade_size=obs_cascade_size,
        baseline_size=baseline_size,
        baseline_reach=baseline_reach,
        truncated=truncated,
        cap_values=cap_values,
        timings=timings,
        report_path=report_path,
    )

    metrics_blob: dict[str, Any] = {
        "config_path": _rel_to_root(config_path),
        "config": cfg,
        "observed_cascade_size_summary": {
            "n": int(obs_cascade_size.size),
            "median": float(np.median(obs_cascade_size)),
            "mean": float(np.mean(obs_cascade_size)),
            "p50": float(np.percentile(obs_cascade_size, 50)),
            "p75": float(np.percentile(obs_cascade_size, 75)),
            "p90": float(np.percentile(obs_cascade_size, 90)),
            "p95": float(np.percentile(obs_cascade_size, 95)),
            "p99": float(np.percentile(obs_cascade_size, 99)),
            "max": float(np.max(obs_cascade_size)),
        },
        "baseline_untruncated": {
            "cascade_size": {
                regime: {
                    "diff_point": bc.diff_point,
                    "diff_ci_lo": bc.diff_ci_lo,
                    "diff_ci_hi": bc.diff_ci_hi,
                    "n_low": bc.n_low,
                    "n_high": bc.n_high,
                }
                for regime, bc in baseline_size.items()
            },
            "audience_reach": {
                regime: {
                    "diff_point": bc.diff_point,
                    "diff_ci_lo": bc.diff_ci_lo,
                    "diff_ci_hi": bc.diff_ci_hi,
                    "n_low": bc.n_low,
                    "n_high": bc.n_high,
                }
                for regime, bc in baseline_reach.items()
            },
        },
        "truncated": {
            str(q): {
                "cap_value": cap_values[q],
                "regimes": {
                    regime: {
                        "diff_point": tc.diff_point,
                        "diff_ci_lo": tc.diff_ci_lo,
                        "diff_ci_hi": tc.diff_ci_hi,
                        "n_low_kept": tc.n_low_kept,
                        "n_high_kept": tc.n_high_kept,
                        "n_low_clipped": tc.n_low_clipped,
                        "n_high_clipped": tc.n_high_clipped,
                    }
                    for regime, tc in regime_results.items()
                },
            }
            for q, regime_results in truncated.items()
        },
        "timings_seconds": timings,
    }
    (out_dir / "truncation_sensitivity_metrics.json").write_text(
        json.dumps(metrics_blob, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote metrics to %s", out_dir / "truncation_sensitivity_metrics.json")
    logger.info("wrote report to %s", report_path)

    logger.info("=== timing summary ===")
    for label, dt in timings.items():
        logger.info("  %-55s %8.2fs", label, dt)


# ---- report ------------------------------------------------------------

def _sign_marker(ci_lo: float, ci_hi: float) -> str:
    if ci_hi < 0:
        return "**< 0**"
    if ci_lo > 0:
        return "**> 0**"
    return "≈ 0"


def _interpret_shift(
    baseline_lo: float,
    baseline_hi: float,
    trunc_lo: float,
    trunc_hi: float,
) -> str:
    """Compare a truncated CI to the baseline CI (both for the same regime).

    We compare the sign-status of each CI; ties go to "similar".
    """
    base_sign = _sign_marker(baseline_lo, baseline_hi)
    trunc_sign = _sign_marker(trunc_lo, trunc_hi)
    if base_sign == trunc_sign:
        # Same sign-status; check if CI moved further from / closer to zero.
        base_dist = min(abs(baseline_lo), abs(baseline_hi))
        trunc_dist = min(abs(trunc_lo), abs(trunc_hi))
        if abs(trunc_dist - base_dist) < 0.5:
            return "similar"
        return "further from 0" if trunc_dist > base_dist else "closer to 0"
    # Sign-status flipped.
    if base_sign == "≈ 0" and trunc_sign == "**< 0**":
        return "**strengthens** (now strictly < 0)"
    if base_sign == "≈ 0" and trunc_sign == "**> 0**":
        return "**reverses** (now strictly > 0)"
    if base_sign == "**< 0**" and trunc_sign == "≈ 0":
        return "**weakens** (no longer strictly < 0)"
    if base_sign == "**< 0**" and trunc_sign == "**> 0**":
        return "**reverses sign**"
    if base_sign == "**> 0**" and trunc_sign == "≈ 0":
        return "**weakens** (no longer strictly > 0)"
    if base_sign == "**> 0**" and trunc_sign == "**< 0**":
        return "**reverses sign**"
    return "shift"


def write_report(
    *,
    config_path: Path,
    cfg: dict[str, Any],
    regimes: tuple[str, ...],
    baseline_regime: str,
    comparator_regimes: tuple[str, ...],
    obs_cascade_size: np.ndarray,
    baseline_size: dict[str, Any],
    baseline_reach: dict[str, Any],
    truncated: dict[float, dict[str, TruncatedContrast]],
    cap_values: dict[float, float],
    timings: dict[str, float],
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 4 Truncation Sensitivity — `cascade_size` contrast\n")
    lines.append(f"Config: `{_rel_to_root(config_path)}`")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(
        f"Cached Phase 4 source: `{cfg['phase4']['per_cascade_parquet']}`\n"
    )

    lines.append("## 1. Motivation\n")
    lines.append(
        "Phase 4 v4 returned a borderline-null on the **`cascade_size`** "
        f"primary metric (ablated bootstrap CI on the regime − {baseline_regime} "
        "contrast was [-9.39, +1.71] — heavily skewed negative but not "
        "strictly excluding zero). Stage 1 calibration shows the simulator "
        "over-produces engagement at the p90 by ~3× from observed. The "
        "cascade_size borderline is therefore plausibly a tail-noise "
        "artifact in the regime where the simulator is least calibrated.\n\n"
        "This sensitivity check caps the per-cascade replicate-mean "
        "cascade_size at quantiles drawn from the **observed** "
        "aggregate-engagement distribution (`replyCount + retweetCount + "
        "quoteCount` per conversation — the observed-side analog of the "
        "simulator's `n_reply + n_retweet + n_deep`), then recomputes the "
        "paired bootstrap contrast. The cap source is observed (not "
        "simulated) — using simulated quantiles would defeat the purpose.\n\n"
        "**Quantile sweep is pre-specified in the config**; the report "
        "below shows every quantile run, regardless of which one is the "
        "most flattering. Cap-and-recompute is performed AFTER "
        "replicate-averaging (the question is whether the contrast "
        "survives in the body of the distribution, not whether individual "
        "noisy event-counts cap differently). The bootstrap resamples the "
        "**capped** values, not the originals.\n"
    )

    lines.append("## 2. Observed cascade-size distribution\n")
    lines.append(
        "Source: `replyCount + retweetCount + quoteCount` summed per "
        "`conversationId` over the cached part_1 parquet. Used only to "
        "set the cap; not bootstrapped against directly.\n"
    )
    lines.append("```")
    lines.append(f"  n         {obs_cascade_size.size:>10,}")
    lines.append(f"  median    {float(np.median(obs_cascade_size)):>10.2f}")
    lines.append(f"  mean      {float(np.mean(obs_cascade_size)):>10.2f}")
    lines.append(f"  p50       {float(np.percentile(obs_cascade_size, 50)):>10.2f}")
    lines.append(f"  p75       {float(np.percentile(obs_cascade_size, 75)):>10.2f}")
    lines.append(f"  p90       {float(np.percentile(obs_cascade_size, 90)):>10.2f}")
    lines.append(f"  p95       {float(np.percentile(obs_cascade_size, 95)):>10.2f}")
    lines.append(f"  p99       {float(np.percentile(obs_cascade_size, 99)):>10.2f}")
    lines.append(f"  max       {float(np.max(obs_cascade_size)):>10.2f}")
    lines.append("```\n")

    lines.append("## 3. Baseline (untruncated) bootstrap contrasts — for reference\n")
    lines.append(
        f"Recomputed from cached Phase 4 parquet using the same paired-"
        f"bootstrap procedure as Phase 4 v4. Numbers should match Phase 4's "
        f"reported bootstrap CIs (within Monte-Carlo error of the bootstrap "
        f"resample). Sign convention: negative contrast = regime closes the "
        f"low/high gap.\n"
    )
    lines.append("| metric | regime | diff_point | 95% bootstrap CI | sign |")
    lines.append("|---|---|---|---|---|")
    for regime in comparator_regimes:
        bc = baseline_size[regime]
        lines.append(
            f"| cascade_size | `{regime}` | {bc.diff_point:+.3f} | "
            f"[{bc.diff_ci_lo:+.3f}, {bc.diff_ci_hi:+.3f}] | "
            f"{_sign_marker(bc.diff_ci_lo, bc.diff_ci_hi)} |"
        )
    for regime in comparator_regimes:
        bc = baseline_reach[regime]
        lines.append(
            f"| audience_reach | `{regime}` | {bc.diff_point:+.3f} | "
            f"[{bc.diff_ci_lo:+.3f}, {bc.diff_ci_hi:+.3f}] | "
            f"{_sign_marker(bc.diff_ci_lo, bc.diff_ci_hi)} |"
        )
    lines.append("")
    lines.append(
        "*Note:* `audience_reach` is included as untruncated context only "
        "(it has no observed-side analog the cap could be drawn from, and "
        "its Phase 4 CI was already strictly < 0). Truncation is applied "
        "to `cascade_size` only — that is the metric whose sensitivity "
        "this script tests.\n"
    )

    lines.append("## 4. Truncated `cascade_size` contrasts\n")
    lines.append(
        "For each pre-specified quantile *q* of the observed distribution, "
        "cap each cascade's replicate-mean `cascade_size` at p*q* and "
        "rerun the paired bootstrap. Winsorization (clip from above) "
        "preserves cascade pairing across regimes — every cascade keeps "
        "an entry under every regime, just with values bounded by the cap. "
        "The dropped-cascades alternative is not run as primary because "
        "it breaks pairing when a cascade exceeds the cap under one regime "
        "but not another.\n"
    )
    for regime in comparator_regimes:
        bc_base = baseline_size[regime]
        lines.append(f"### `{regime}` − `{baseline_regime}`\n")
        lines.append(
            "| quantile | cap | n_low (clipped) | n_high (clipped) | "
            "diff_point | 95% CI | sign | vs. baseline |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        # Baseline row first, for visual comparison.
        lines.append(
            f"| *(none)* | — | {bc_base.n_low} (—) | {bc_base.n_high} (—) | "
            f"{bc_base.diff_point:+.3f} | "
            f"[{bc_base.diff_ci_lo:+.3f}, {bc_base.diff_ci_hi:+.3f}] | "
            f"{_sign_marker(bc_base.diff_ci_lo, bc_base.diff_ci_hi)} | — |"
        )
        for q in sorted(truncated.keys()):
            tc = truncated[q][regime]
            lines.append(
                f"| p{q:.0f} | {tc.cap_value:.2f} | "
                f"{tc.n_low_kept} ({tc.n_low_clipped}) | "
                f"{tc.n_high_kept} ({tc.n_high_clipped}) | "
                f"{tc.diff_point:+.3f} | "
                f"[{tc.diff_ci_lo:+.3f}, {tc.diff_ci_hi:+.3f}] | "
                f"{_sign_marker(tc.diff_ci_lo, tc.diff_ci_hi)} | "
                f"{_interpret_shift(bc_base.diff_ci_lo, bc_base.diff_ci_hi, tc.diff_ci_lo, tc.diff_ci_hi)} |"
            )
        lines.append("")

    lines.append("## 5. Outcome interpretation\n")
    lines.append(
        "Pre-specified outcome categories (apply per comparator regime):\n\n"
        "1. **Strengthens** — truncated CI strictly < 0 where baseline "
        "overlapped 0, *or* truncated CI moves further from zero. The "
        "original null was a tail-noise artifact; the architectural "
        "finding holds on cascade_size.\n"
        "2. **Weakens** — truncated CI overlaps 0 (or reverses) where "
        "baseline was strictly < 0, *or* truncated CI moves closer to "
        "zero. The original tentative-negative was driven by simulator "
        "tail behavior; the cascade_size finding doesn't survive "
        "sensitivity. Audience_reach (already strictly < 0 in baseline) "
        "carries the architectural claim.\n"
        "3. **Similar** — sign-status unchanged and CI distance from 0 "
        "moves by < 0.5. The tail isn't load-bearing in either direction; "
        "the borderline-null is what it is.\n\n"
        "Read the per-quantile rows in §4 row-by-row against these "
        "categories. The point is **not** to pick the quantile that gives "
        "the cleanest answer — it's to see whether the answer is stable "
        "across the body-vs-tail spectrum.\n"
    )

    lines.append("## 6. Method notes\n")
    lines.append(
        "- **Cap source:** observed `replyCount + retweetCount + quoteCount` "
        "per conversation (`analysis.validation.observed_aggregate_engagement`).\n"
        "- **Cap target:** simulator per-cascade replicate-mean "
        "`cascade_size = n_reply + n_retweet + n_deep`. Computed AFTER "
        "averaging across replicates so the cap acts on each cascade's "
        "central tendency.\n"
        "- **Winsorization:** values exceeding the cap are clipped *to* the "
        "cap (not dropped). Preserves cascade pairing; bounds tail "
        "influence on means without losing data.\n"
        "- **Bootstrap:** identical procedure to "
        "`analysis.hypothesis.cascade_bootstrap_contrast` — paired resampling "
        "of cascade_ids within each label, applied to the **capped** "
        "values. Each quantile uses an independent RNG seed (config seed + "
        "quantile) so quantile sweeps are reproducible without "
        "entanglement.\n"
        "- **What this does NOT touch:** the original Phase 4 outputs, "
        "Phase 5 outputs, or the pre-registration. This is an "
        "additional sensitivity analysis. Phase 4's cached parquet is "
        "read-only.\n"
    )

    lines.append("## 7. Step timings\n")
    lines.append("```")
    for label, dt in timings.items():
        lines.append(f"  {label:55s} {dt:8.2f}s")
    lines.append("```\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_truncation_sensitivity.yaml",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args.config)


if __name__ == "__main__":
    main()

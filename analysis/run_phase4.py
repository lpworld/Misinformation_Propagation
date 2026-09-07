"""Phase 4 driver — Stage 2 hypothesis test.

Runs the multi-regime ablation harness on label-stratified seeds, computes
gap metrics per regime + cross-regime contrasts, and emits a report
keyed to the pre-registration in ``paper/phase4_preregistration.md``.

Run::

    uv run python -m analysis.run_phase4 --config configs/experiment_phase4.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from analysis.hypothesis import HIGH, LOW, run_stage2
from analysis.labeling import add_labels, coverage_summary
from ranker.scoring import ScoringConfig, make_default_configs
from ranker.training import load_ranker
from simulation.ablation import run_ablation
from simulation.calibration import calibrate_baseline_exposures
from simulation.cascade import (
    SimConfig,
    fit_diurnal_weights,
)
from simulation.cascade import _seed_probs as _seed_probs_internal
from simulation.content import SEED_COLS
from simulation.users import sample_users

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


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
    logger.info("▶ %s — start", label)
    try:
        yield
    finally:
        dt = time.time() - t0
        timings[label] = dt
        logger.info("✓ %s — done in %.2fs", label, dt)


def _build_scoring_configs(cfg: dict[str, Any]) -> dict[str, ScoringConfig]:
    defaults = make_default_configs()
    alpha = float(cfg["scoring"].get("alpha", 1.0))
    abl = defaults["ablated"]
    defaults["ablated"] = ScoringConfig(
        regime="ablated",
        weights=abl.weights,
        alpha=alpha,
        slow_heads=abl.slow_heads,
        fast_heads=abl.fast_heads,
    )
    return defaults


def _stratified_label_seeds(
    labeled_df: pd.DataFrame,
    n_per_label: int,
    seed: int,
) -> pd.DataFrame:
    """Sample ``n_per_label`` cascades from each of low- and high-credibility."""
    rng = np.random.default_rng(seed)
    out_chunks: list[pd.DataFrame] = []
    for label in (LOW, HIGH):
        pool = labeled_df[labeled_df["credibility_label"] == label]
        if len(pool) == 0:
            raise ValueError(f"no rows with credibility_label={label!r}")
        replace = len(pool) < n_per_label
        idx = rng.choice(len(pool), size=n_per_label, replace=replace)
        chunk = pool.iloc[idx].copy()
        out_chunks.append(chunk)
    seeds = pd.concat(out_chunks, ignore_index=True)
    cols = [c for c in SEED_COLS if c in seeds.columns]
    if "credibility_label" not in cols:
        cols = cols + ["credibility_label"]
    return seeds[cols].reset_index(drop=True)


def run(config_path: Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    seed = int(cfg["random_seed"])
    timings: dict[str, float] = {}

    out_dir = ROOT / cfg["artifacts"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / cfg["artifacts"]["report_path"]
    prereg_path = ROOT / cfg["artifacts"]["preregistration_path"]
    if not prereg_path.exists():
        raise FileNotFoundError(
            f"pre-registration not found at {prereg_path}; refusing to run "
            "Phase 4 without a locked analysis spec"
        )

    # --- 1. data + labels ----------------------------------------------
    parquet = ROOT / cfg["data"]["part_parquet"]
    with _timed(f"load parquet ({parquet.name})", timings):
        df = pd.read_parquet(parquet)
        logger.info("  loaded %s rows × %s cols", f"{len(df):,}", len(df.columns))

    with _timed("apply credibility labels", timings):
        df = add_labels(
            df,
            iffy_path=ROOT / cfg["labels"]["iffy_path"],
            mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
        )
        cov = coverage_summary(df)
        logger.info(
            "  label coverage: low=%d high=%d mixed=%d unlabeled=%d (labeled %.2f%%)",
            cov["low_credibility"], cov["high_credibility"],
            cov["mixed"], cov["unlabeled"], 100 * cov["labeled_fraction"],
        )

    # --- 2. ranker -----------------------------------------------------
    ranker_dir = ROOT / cfg["ranker"]["load_dir"]
    with _timed(f"load ranker ({ranker_dir.name})", timings):
        model, scaler, meta = load_ranker(ranker_dir)
        logger.info(
            "  ranker kind=%s n_params=%d",
            meta["architecture"]["kind"],
            meta["metrics"].get("n_params", -1),
        )

    # --- 3. user pool + label-stratified seeds -------------------------
    n_users = int(cfg["simulation"]["n_users"])
    n_per_label = int(cfg["simulation"]["n_per_label"])
    with _timed(f"sample {n_users} users", timings):
        user_pool = sample_users(df, n=n_users, seed=seed)
    with _timed(f"sample {n_per_label * 2} label-stratified seeds", timings):
        seeds = _stratified_label_seeds(df, n_per_label, seed)
        logger.info(
            "  seeds by label: %s",
            seeds["credibility_label"].value_counts().to_dict(),
        )

    # --- 4. diurnal + calibration --------------------------------------
    with _timed("fit diurnal weights", timings):
        diurnal = fit_diurnal_weights(df)

    scoring_configs = _build_scoring_configs(cfg)
    with _timed("calibrate (additive regime)", timings):
        # Calibration uses the same seed set the simulator will run on.
        seed_probs = _seed_probs_internal(seeds, model, scaler)
        calib = calibrate_baseline_exposures(
            observed_df=df,
            seed_probs=seed_probs,
            scoring_config=scoring_configs["additive"],
            target=str(cfg["calibration"].get("target", "zero_inflated_nb")),
            score_normalization=str(cfg["simulation"]["score_normalization"]),
        )

    sim_cfg = SimConfig(
        n_time_bins=int(cfg["simulation"]["n_time_bins"]),
        decay_tau_hours=float(cfg["simulation"]["decay_tau_hours"]),
        baseline_exposures=float(calib.baseline_exposures),
        exposure_min=float(cfg["simulation"]["exposure_min"]),
        exposure_max=float(cfg["simulation"]["exposure_max"]),
        score_normalization=str(cfg["simulation"]["score_normalization"]),
        use_circadian=bool(cfg["simulation"].get("use_circadian", True)),
        activity_pi=calib.activity_pi,
        dispersion_r=calib.dispersion_r,
        seed=seed,
    )

    # --- 5. multi-regime × multi-replicate ablation --------------------
    n_replicates = int(cfg["simulation"]["n_replicates"])
    with _timed(
        f"ablation: {len(scoring_configs)} regimes × {n_replicates} replicates",
        timings,
    ):
        ablation = run_ablation(
            seeds=seeds,
            user_pool=user_pool,
            ranker=model,
            scaler=scaler,
            scoring_configs=scoring_configs,
            sim_config=sim_cfg,
            n_replicates=n_replicates,
            diurnal_weights=diurnal,
        )

    # Attach credibility labels onto per_cascade by cascade_id (cascade_id
    # is the seed-row index; identical across regimes/replicates).
    label_lookup = seeds["credibility_label"].rename("credibility_label")
    pc = ablation.per_cascade.copy()
    pc["credibility_label"] = label_lookup.iloc[pc["cascade_id"].to_numpy()].to_numpy()

    pc.to_parquet(out_dir / "ablation_per_cascade.parquet", index=False)
    ablation.per_bin.to_parquet(out_dir / "ablation_per_bin.parquet", index=False)

    # --- 6. Stage 2 metrics + cross-regime contrasts -------------------
    regimes = tuple(scoring_configs.keys())
    n_bootstrap = int(cfg["simulation"].get("bootstrap_iterations", 1000))
    with _timed(
        f"compute Stage 2 gaps + contrasts (bootstrap n={n_bootstrap})", timings,
    ):
        stage2 = run_stage2(
            per_cascade=pc,
            per_bin=ablation.per_bin,
            regimes=regimes,
            baseline_regime="additive",
            n_bootstrap=n_bootstrap,
            bootstrap_seed=seed,
        )

    # --- 7. report -----------------------------------------------------
    write_report(
        config_path=config_path,
        cfg=cfg,
        ranker_meta=meta,
        scoring_configs=scoring_configs,
        calibration=calib,
        coverage=cov,
        stage2=stage2,
        timings=timings,
        report_path=report_path,
    )

    metrics_blob = {
        "config_path": _rel_to_root(config_path),
        "preregistration_path": _rel_to_root(prereg_path),
        "config": cfg,
        "calibration": asdict(calib),
        "label_coverage": cov,
        "stage2_gaps": {
            metric: {regime: asdict(gap) for regime, gap in regime_gaps.items()}
            for metric, regime_gaps in stage2["gaps"].items()
        },
        "stage2_contrasts": {
            metric: {regime: asdict(c) for regime, c in regime_contrasts.items()}
            for metric, regime_contrasts in stage2["contrasts"].items()
        },
        "stage2_bootstrap_gaps": {
            metric: {regime: asdict(g) for regime, g in regime_gaps.items()}
            for metric, regime_gaps in stage2["bootstrap_gaps"].items()
        },
        "stage2_bootstrap_contrasts": {
            metric: {regime: asdict(c) for regime, c in regime_contrasts.items()}
            for metric, regime_contrasts in stage2["bootstrap_contrasts"].items()
        },
        "timings_seconds": timings,
    }
    (out_dir / "phase4_metrics.json").write_text(
        json.dumps(metrics_blob, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
        encoding="utf-8",
    )
    logger.info("wrote metrics to %s", out_dir / "phase4_metrics.json")
    logger.info("wrote report to %s", report_path)

    logger.info("=== timing summary ===")
    for label, dt in timings.items():
        logger.info("  %-50s %8.2fs", label, dt)


def write_report(
    *,
    config_path: Path,
    cfg: dict[str, Any],
    ranker_meta: dict[str, Any],
    scoring_configs: dict[str, ScoringConfig],
    calibration,
    coverage: dict[str, Any],
    stage2: dict,
    timings: dict[str, float],
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 4 Report — Stage 2 Hypothesis Test\n")
    lines.append(f"Config: `{_rel_to_root(config_path)}`")
    lines.append(f"Pre-registration: `{cfg['artifacts']['preregistration_path']}`")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(
        f"Seed: {cfg['random_seed']} | "
        f"n_per_label={cfg['simulation']['n_per_label']:,} (×2 labels) | "
        f"n_replicates={cfg['simulation']['n_replicates']} | "
        f"n_users={cfg['simulation']['n_users']:,}\n"
    )

    lines.append("## 1. Setup\n")
    arch = ranker_meta.get("architecture", {})
    lines.append("```")
    lines.append(f"  ranker             {arch.get('kind')} ({ranker_meta['metrics'].get('n_params')} params)")
    lines.append(f"  calibration mode   {calibration.target_mode}")
    lines.append(f"  baseline_exposures {calibration.baseline_exposures:.2f}")
    if calibration.activity_pi is not None:
        lines.append(f"  activity_pi        {calibration.activity_pi:.4f}")
    if calibration.dispersion_r is not None:
        lines.append(f"  dispersion_r       {calibration.dispersion_r:.4f}")
    lines.append("```\n")

    lines.append("## 2. Label coverage (full corpus)\n")
    lines.append("```")
    for k, v in coverage.items():
        if isinstance(v, float):
            lines.append(f"  {k:24s} {v:.4f}")
        else:
            lines.append(f"  {k:24s} {v:,}")
    lines.append("```\n")

    # Per-regime gaps
    lines.append("## 3. Gap (low − high) per regime, per metric\n")
    lines.append(
        "Replicate-mean of (mean_low − mean_high), with 2.5%/97.5% percentile CI "
        "across replicates. Positive = low-credibility content propagates farther.\n"
    )
    for metric, regime_gaps in stage2["gaps"].items():
        lines.append(f"### {metric}\n")
        lines.append("| regime | gap_mean | 95% CI | n_low/rep | n_high/rep | n_replicates |")
        lines.append("|--------|----------|--------|-----------|------------|--------------|")
        for regime, gap in regime_gaps.items():
            lines.append(
                f"| `{regime}` | {gap.gap_mean:,.3f} | "
                f"[{gap.gap_ci_lo:,.3f}, {gap.gap_ci_hi:,.3f}] | "
                f"{gap.n_low_per_replicate:,} | {gap.n_high_per_replicate:,} | "
                f"{gap.n_replicates} |"
            )
        lines.append("")

    # Cross-regime contrasts (replicate-level + bootstrap-level)
    lines.append("## 4. Cross-regime contrasts (regime − additive)\n")
    lines.append(
        "Two uncertainty layers per pre-reg + 2026-04-26 amendment:\n\n"
        "- **Replicate CI** (pre-registered): variance across simulator RNG "
        "seeds, fixed cascade pool. Captures Monte-Carlo error from event "
        "sampling. For deterministic metrics (audience_reach), this CI is "
        "degenerate (point mass).\n"
        "- **Bootstrap CI** (amendment): replicate-mean per cascade, then "
        "paired bootstrap over cascade_ids. Captures seed-sampling "
        "uncertainty. The right CI for deterministic metrics; complementary "
        "for stochastic ones.\n\n"
        "Sign convention: **negative contrast on cascade_size = architectural-"
        "claim signature** (regime closes the propagation gap; pre-reg §6).\n"
    )
    for metric in stage2["contrasts"]:
        lines.append(f"### {metric}\n")
        lines.append("| regime | diff (replicate) | 95% replicate CI | diff (bootstrap) | 95% bootstrap CI | bootstrap CI sign |")
        lines.append("|--------|------------------|-------------------|-------------------|-------------------|--------------------|")
        for regime in stage2["contrasts"][metric]:
            rc = stage2["contrasts"][metric][regime]
            bc = stage2["bootstrap_contrasts"][metric][regime]
            if bc.diff_ci_hi < 0:
                bs_sign = "**< 0** (closes gap)"
            elif bc.diff_ci_lo > 0:
                bs_sign = "**> 0** (widens gap)"
            else:
                bs_sign = "overlaps 0"
            lines.append(
                f"| `{regime}` | "
                f"{rc.diff_mean:,.3f} | [{rc.diff_ci_lo:,.3f}, {rc.diff_ci_hi:,.3f}] | "
                f"{bc.diff_point:,.3f} | [{bc.diff_ci_lo:,.3f}, {bc.diff_ci_hi:,.3f}] | "
                f"{bs_sign} |"
            )
        lines.append("")

    # Decision
    lines.append("## 5. Decision rule outcome (primary: cascade_size; bootstrap CI)\n")
    lines.append(
        "Per pre-reg §6 applied to the **bootstrap CI** (the amendment-"
        "compliant uncertainty estimate). The replicate CI is reported "
        "alongside as Monte-Carlo error.\n"
    )
    primary_bs = stage2["bootstrap_contrasts"]["cascade_size"]
    abl = primary_bs.get("ablated")
    ret = primary_bs.get("additive_retuned")
    if abl is None or ret is None:
        lines.append("(cannot apply decision rule — missing contrast)\n")
    else:
        abl_strict_neg = abl.diff_ci_hi < 0
        abl_strict_pos = abl.diff_ci_lo > 0
        ret_strict_neg = ret.diff_ci_hi < 0
        if abl_strict_neg and not ret_strict_neg:
            outcome = ("**A — architectural claim supported.** ablated closes the gap "
                       "(bootstrap CI strictly < 0); additive_retuned does not.")
        elif not abl_strict_neg and not abl_strict_pos:
            outcome = ("**B — null on the architectural claim.** ablated bootstrap CI "
                       "overlaps 0; the architectural mechanism does not differentially "
                       "affect propagation in this setup.")
        elif abl_strict_neg and ret_strict_neg:
            outcome = ("**C — both regimes close the gap.** Diagnose: parameter "
                       "retuning is too aggressive, or the slow/fast partition isn't "
                       "capturing the right distinction.")
        elif abl_strict_pos:
            outcome = ("**D — ablation amplifies the gap.** Opposite of predicted; "
                       "revisit slow/fast partition or gate functional form.")
        else:
            outcome = "Unclassified outcome — manual review."
        lines.append(outcome + "\n")

    # Co-primary: audience_reach — clean test of architectural mechanism at
    # algorithm-output layer (no event-sampling noise).
    lines.append("## 5b. Co-primary: audience_reach (algorithm-output layer)\n")
    ar = stage2["bootstrap_contrasts"]["audience_reach"]
    lines.append(
        "Audience-reach is the algorithm-output proxy and is deterministic "
        "given the score, so its bootstrap CI captures the only meaningful "
        "uncertainty (seed sampling). It's the cleanest test of the "
        "architectural mechanism *as it acts on the ranker's behavior*, "
        "before downstream user-behavior stochasticity.\n"
    )
    lines.append("| regime | diff_point | 95% bootstrap CI | sign |")
    lines.append("|--------|------------|-------------------|------|")
    for regime, c in ar.items():
        if c.diff_ci_hi < 0:
            s = "**< 0** (closes gap)"
        elif c.diff_ci_lo > 0:
            s = "**> 0** (widens gap)"
        else:
            s = "overlaps 0"
        lines.append(
            f"| `{regime}` | {c.diff_point:,.3f} | "
            f"[{c.diff_ci_lo:,.3f}, {c.diff_ci_hi:,.3f}] | {s} |"
        )
    lines.append("")

    lines.append("## 6. Step timings\n")
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
        default=ROOT / "configs" / "experiment_phase4.yaml",
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

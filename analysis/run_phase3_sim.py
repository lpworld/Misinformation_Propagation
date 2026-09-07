"""Phase 3 simulation + Stage 1 validation driver.

Loads the trained MaskNet, samples user pool + seeds at scale, calibrates
exposure baseline against observed mean root.replyCount under the additive
regime, runs all three scoring regimes, and reports all four Stage 1
metrics per the design specification.

Stage 1 is label-free. Labels enter at Phase 4.

Run::

    uv run python -m analysis.run_phase3_sim --config configs/experiment_phase3_sim.yaml
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

from analysis.validation import (
    DistributionTest,
    compare_distributions,
    observed_aggregate_engagement,
    observed_reactive_to_reflective,
    observed_root_reply_counts,
    observed_time_to_peak_hours,
    simulated_aggregate_engagement,
    simulated_reactive_to_reflective,
    simulated_root_reply_counts,
    simulated_time_to_peak_hours,
)
from ranker.scoring import ScoringConfig, make_default_configs
from ranker.training import load_ranker
from simulation.calibration import calibrate_baseline_exposures
from simulation.cascade import (
    SimConfig,
    SimResult,
    fit_diurnal_weights,
    simulate_cascades,
)
from simulation.cascade import _seed_probs as _seed_probs_internal
from simulation.content import sample_seeds
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
    """All three pre-specified regimes; ablated alpha overridden from cfg."""
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


def _make_sim_config(
    cfg: dict[str, Any],
    baseline_exposures: float,
    activity_pi: float | None = None,
    dispersion_r: float | None = None,
) -> SimConfig:
    s = cfg["simulation"]
    return SimConfig(
        n_time_bins=int(s["n_time_bins"]),
        decay_tau_hours=float(s["decay_tau_hours"]),
        baseline_exposures=float(baseline_exposures),
        exposure_min=float(s["exposure_min"]),
        exposure_max=float(s["exposure_max"]),
        score_normalization=str(s["score_normalization"]),
        use_circadian=bool(s.get("use_circadian", True)),
        activity_pi=activity_pi,
        dispersion_r=dispersion_r,
        seed=int(cfg["random_seed"]),
    )


def run(config_path: Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    seed = int(cfg["random_seed"])
    timings: dict[str, float] = {}

    out_dir = ROOT / cfg["artifacts"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / cfg["artifacts"]["report_path"]

    # --- 1. data + ranker ----------------------------------------------
    parquet = ROOT / cfg["data"]["part_parquet"]
    with _timed(f"load parquet ({parquet.name})", timings):
        df = pd.read_parquet(parquet)
        logger.info("  loaded %s rows × %s cols", f"{len(df):,}", len(df.columns))

    ranker_dir = ROOT / cfg["ranker"]["load_dir"]
    with _timed(f"load ranker ({ranker_dir.name})", timings):
        model, scaler, meta = load_ranker(ranker_dir)
        # load_ranker pops n_features off the cfg dict during reconstruction;
        # read it back from the model itself.
        n_features = int(getattr(model.config, "n_features", -1))
        logger.info(
            "  ranker kind=%s n_features=%d n_params=%d",
            meta["architecture"]["kind"],
            n_features,
            meta["metrics"].get("n_params", -1),
        )

    # --- 2. user pool + seeds ------------------------------------------
    n_users = int(cfg["simulation"]["n_users"])
    n_cascades = int(cfg["simulation"]["n_cascades"])
    originals_only = bool(cfg["simulation"].get("originals_only_seeds", False))

    with _timed(f"sample {n_users} users", timings):
        user_pool = sample_users(df, n=n_users, seed=seed)
    with _timed(f"sample {n_cascades} seeds", timings):
        seeds = sample_seeds(df, n=n_cascades, seed=seed, originals_only=originals_only)

    # --- 3. diurnal weights from observed corpus -----------------------
    with _timed("fit diurnal weights", timings):
        diurnal = fit_diurnal_weights(df)
        logger.info(
            "  diurnal weights: min=%.2f max=%.2f mean=%.2f",
            float(diurnal.min()), float(diurnal.max()), float(diurnal.mean()),
        )

    # --- 4. calibrate baseline_exposures (and activity gate, if zero-inflated)
    scoring_configs = _build_scoring_configs(cfg)
    if cfg["calibration"].get("enabled", True):
        with _timed("calibrate baseline_exposures (additive)", timings):
            seed_probs = _seed_probs_internal(seeds, model, scaler)
            calib = calibrate_baseline_exposures(
                observed_df=df,
                seed_probs=seed_probs,
                scoring_config=scoring_configs["additive"],
                target=str(cfg["calibration"].get("target", "mean")),
                score_normalization=str(cfg["simulation"]["score_normalization"]),
            )
            baseline_exposures = float(calib.baseline_exposures)
            activity_pi = calib.activity_pi  # None for target="mean"
            dispersion_r = calib.dispersion_r  # None unless target="zero_inflated_nb"
    else:
        calib = None
        baseline_exposures = 200.0
        activity_pi = None
        dispersion_r = None
        logger.info("calibration disabled; using baseline_exposures=%s", baseline_exposures)

    sim_cfg = _make_sim_config(
        cfg, baseline_exposures,
        activity_pi=activity_pi,
        dispersion_r=dispersion_r,
    )

    # --- 5. simulate under all three regimes ---------------------------
    sim_results: dict[str, SimResult] = {}
    for name, sc in scoring_configs.items():
        with _timed(f"simulate regime={name}", timings):
            r = simulate_cascades(
                seeds=seeds,
                user_pool=user_pool,
                ranker=model,
                scaler=scaler,
                scoring_config=sc,
                sim_config=sim_cfg,
                diurnal_weights=diurnal,
            )
            sim_results[name] = r
            r.per_bin.to_parquet(out_dir / f"sim_{name}_per_bin.parquet", index=False)
            r.per_cascade.to_parquet(out_dir / f"sim_{name}_per_cascade.parquet", index=False)

    # --- 6. Stage 1 validation -----------------------------------------
    # Pre-compute observed once (some metrics are expensive on 1M rows).
    with _timed("compute observed Stage 1 distributions", timings):
        obs = {
            "root_reply_count": observed_root_reply_counts(df),
            "aggregate_engagement": observed_aggregate_engagement(df),
            "reactive_to_reflective": observed_reactive_to_reflective(df),
            "time_to_peak_hours": observed_time_to_peak_hours(df),
        }
        for k, arr in obs.items():
            logger.info("  obs[%s]: n=%d median=%.2f mean=%.2f p90=%.2f",
                        k, arr.size,
                        float(np.median(arr)) if arr.size else float("nan"),
                        float(np.mean(arr)) if arr.size else float("nan"),
                        float(np.percentile(arr, 90)) if arr.size else float("nan"))

    sim_dist_funcs = {
        "root_reply_count": (simulated_root_reply_counts, "per_cascade"),
        "aggregate_engagement": (simulated_aggregate_engagement, "per_cascade"),
        "reactive_to_reflective": (simulated_reactive_to_reflective, "per_cascade"),
        "time_to_peak_hours": (simulated_time_to_peak_hours, "per_bin"),
    }

    stage1: dict[str, dict[str, DistributionTest]] = {}
    with _timed("run Stage 1 comparisons (all regimes × all metrics)", timings):
        for regime_name, r in sim_results.items():
            stage1[regime_name] = {}
            for metric_name, (func, source) in sim_dist_funcs.items():
                sim_arr = func(r.per_cascade if source == "per_cascade" else r.per_bin)
                if sim_arr.size == 0 or obs[metric_name].size == 0:
                    logger.warning(
                        "skipping %s/%s — empty array (obs=%d sim=%d)",
                        regime_name, metric_name, obs[metric_name].size, sim_arr.size,
                    )
                    continue
                stage1[regime_name][metric_name] = compare_distributions(
                    obs[metric_name], sim_arr, metric=metric_name,
                )

    # --- 7. report -----------------------------------------------------
    write_report(
        config_path=config_path,
        cfg=cfg,
        ranker_meta=meta,
        scoring_configs=scoring_configs,
        sim_results=sim_results,
        stage1=stage1,
        calibration=calib,
        timings=timings,
        report_path=report_path,
    )

    metrics_blob = {
        "config_path": _rel_to_root(config_path),
        "config": cfg,
        "ranker_meta": {
            "architecture": meta.get("architecture"),
            "metrics": meta.get("metrics"),
        },
        "calibration": asdict(calib) if calib is not None else None,
        "stage1": {
            regime: {m: asdict(t) for m, t in metrics.items()}
            for regime, metrics in stage1.items()
        },
        "sim_per_cascade_summary": {
            name: r.per_cascade[
                ["n_reply", "n_retweet", "n_like", "n_deep", "score", "total_exposures"]
            ].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()
            for name, r in sim_results.items()
        },
        "timings_seconds": timings,
    }
    (out_dir / "phase3_sim_metrics.json").write_text(
        json.dumps(metrics_blob, indent=2, default=float), encoding="utf-8"
    )
    logger.info("wrote metrics to %s", out_dir / "phase3_sim_metrics.json")
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
    sim_results: dict[str, SimResult],
    stage1: dict[str, dict[str, DistributionTest]],
    calibration,
    timings: dict[str, float],
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 3 Report — Simulation + Stage 1 Validation\n")
    lines.append(f"Config: `{_rel_to_root(config_path)}`")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(
        f"Seed: {cfg['random_seed']} | "
        f"n_users={cfg['simulation']['n_users']:,} | "
        f"n_cascades={cfg['simulation']['n_cascades']:,}\n"
    )

    # Ranker info
    arch = ranker_meta.get("architecture", {})
    rmetrics = ranker_meta.get("metrics", {})
    lines.append("## 1. Ranker (loaded from disk)\n")
    lines.append("```")
    lines.append(f"  kind        {arch.get('kind')}")
    lines.append(f"  n_params    {rmetrics.get('n_params')}")
    for k, v in rmetrics.items():
        if k.startswith("auc_"):
            lines.append(f"  {k:24s} {v:.4f}")
    lines.append("```\n")

    # Calibration
    lines.append("## 2. Exposure calibration\n")
    if calibration is not None:
        lines.append(f"Mode: **{calibration.target_mode}**\n")
        lines.append("```")
        lines.append(f"  observed mean root.replyCount         {calibration.target_mean:.3f}")
        lines.append(f"  E[S_rel × p_reply] over seeds         {calibration.expected_factor:.4f}")
        lines.append(f"  baseline_exposures (calibrated)       {calibration.baseline_exposures:,.2f}")
        if calibration.activity_pi is not None:
            lines.append(f"  observed P(reply = 0)                 {calibration.target_p_zero:.4f}")
            lines.append(f"  observed E[reply | reply > 0]         {calibration.target_conditional_mean:.3f}")
            lines.append(f"  activity_pi (1 - P_obs(reply=0))      {calibration.activity_pi:.4f}")
        if calibration.dispersion_r is not None:
            lines.append(f"  observed Var[reply | reply > 0]       {calibration.target_conditional_var:.3f}")
            lines.append(f"  dispersion_r (NB)                     {calibration.dispersion_r:.4f}")
        lines.append(f"  n_observed                            {calibration.n_observed:,}")
        lines.append(f"  n_seeds                               {calibration.n_seeds:,}")
        lines.append("```\n")
        lines.append(
            "Calibration is run under the **additive** regime (production-style "
            "comparator) and the resulting parameters (baseline_exposures, "
            "activity_pi) are reused across all three regimes — regime differences "
            "should sit in the score-aggregation layer, not in the exposure scale "
            "or activity rate.\n"
        )
    else:
        lines.append("Calibration disabled.\n")

    # Per-regime cascade summaries
    lines.append("## 3. Simulated cascades — per regime\n")
    for name, r in sim_results.items():
        sc = scoring_configs[name]
        lines.append(f"### regime = `{name}` ({sc.regime}, alpha={sc.alpha})\n")
        desc = r.per_cascade[
            ["n_reply", "n_retweet", "n_like", "n_deep", "score", "total_exposures"]
        ].describe(percentiles=[0.5, 0.9, 0.99])
        lines.append("```")
        lines.append(desc.to_string(float_format=lambda x: f"{x:,.2f}"))
        lines.append("```\n")

    # Stage 1 — full table per regime × metric
    lines.append("## 4. Stage 1 validation\n")
    lines.append(
        "Two-sample KS + Mann–Whitney comparing observed vs. simulated for "
        "each regime and each of the four Stage 1 metrics from the design specification "
        "§Two-stage analysis. Stage 1 is label-free.\n"
    )
    lines.append("| regime | metric | n_obs | n_sim | obs_med | sim_med | obs_p90 | sim_p90 | KS | KS p | MW p |")
    lines.append("|--------|--------|-------|-------|---------|---------|---------|---------|------|--------|--------|")
    for regime_name, metrics in stage1.items():
        for metric_name, t in metrics.items():
            lines.append(
                f"| {regime_name} | {metric_name} | {t.n_observed:,} | {t.n_simulated:,} | "
                f"{t.obs_median:.2f} | {t.sim_median:.2f} | "
                f"{t.obs_p90:.2f} | {t.sim_p90:.2f} | "
                f"{t.ks_stat:.3f} | {t.ks_pvalue:.2e} | {t.mw_pvalue:.2e} |"
            )
    lines.append("")

    # Decision-gate framing
    lines.append("## 5. Decision-gate notes (per the design specification Phase 3)\n")
    lines.append(
        "Stage 1 asks: **does the simulation produce cascades that look like "
        "observed cascades?** With ~1M observed tweets the KS test will reject "
        "almost any null at vanishing p-values; what matters is whether the "
        "*shape* (medians, p90s, mean ratios) of the simulated distribution "
        "tracks observed within a reasonable order of magnitude.\n"
    )
    lines.append(
        "Read the table above against this rubric:\n\n"
        "- **Aligned medians + p90s within ~2× across all four metrics** → "
        "proceed to Phase 4.\n"
        "- **Some metrics aligned, others off by an order of magnitude** → "
        "diagnose the off ones (usually one or two simulator parameters); the "
        "the design specification \"things that will probably go wrong\" entry on first-pass "
        "Stage 1 failure applies here.\n"
        "- **Nothing aligned** → revisit calibration target and ranker "
        "calibration before scaling further.\n"
    )

    # Timings
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
        default=ROOT / "configs" / "experiment_phase3_sim.yaml",
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

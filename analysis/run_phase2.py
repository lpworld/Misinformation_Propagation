"""Phase 2 driver: train stub ranker, simulate cascades under all three regimes, validate.

Run::

    uv run python -m analysis.run_phase2
    uv run python -m analysis.run_phase2 --config configs/experiment_phase2_mvp.yaml

The driver runs the simulator under all three pre-specified regimes
(``additive``, ``ablated``, ``additive_retuned``) so we can eyeball whether
they produce qualitatively different cascades on the MVP scale. The Phase 2
*decision gate* per the design specification is "no errors, plausible outputs" — not the
hypothesis test. That comes in Phase 4 with labels.

Outputs:
    data/processed/phase2_mvp/ranker/        — trained ranker + scaler
    data/processed/phase2_mvp/sim_*.parquet  — per-bin & per-cascade outputs
    paper/phase2_report.md                   — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from analysis.validation import stage1_root_reply_check
from ranker.architecture import RankerConfig
from ranker.scoring import ScoringConfig, make_default_configs
from ranker.training import TrainConfig, save_ranker, train_stub_ranker
from simulation.cascade import SimConfig, simulate_cascades
from simulation.content import sample_seeds
from simulation.users import sample_users

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _make_train_config(cfg: dict[str, Any]) -> TrainConfig:
    t = cfg["training"]
    return TrainConfig(
        batch_size=int(t["batch_size"]),
        lr=float(t["lr"]),
        epochs=int(t["epochs"]),
        val_frac=float(t["val_frac"]),
        max_train_rows=t.get("max_train_rows"),
        seed=int(cfg["random_seed"]),
    )


def _make_sim_config(cfg: dict[str, Any]) -> SimConfig:
    s = cfg["simulation"]
    return SimConfig(
        n_time_bins=int(s["n_time_bins"]),
        decay_tau_hours=float(s["decay_tau_hours"]),
        baseline_exposures=float(s["baseline_exposures"]),
        exposure_min=float(s["exposure_min"]),
        exposure_max=float(s["exposure_max"]),
        score_normalization=str(s["score_normalization"]),
        seed=int(cfg["random_seed"]),
    )


def _build_scoring_configs(cfg: dict[str, Any]) -> dict[str, ScoringConfig]:
    """Build the three pre-specified regimes, with alpha overridden from yaml."""
    defaults = make_default_configs()
    alpha = float(cfg["scoring"].get("alpha", 1.0))
    # Re-create the ablated config with the requested alpha; the others
    # don't depend on alpha.
    abl = defaults["ablated"]
    defaults["ablated"] = ScoringConfig(
        regime="ablated",
        weights=abl.weights,
        alpha=alpha,
        slow_heads=abl.slow_heads,
        fast_heads=abl.fast_heads,
    )
    return defaults


def run(config_path: Path) -> None:
    cfg = _load_config(config_path)
    seed = int(cfg["random_seed"])

    out_dir = ROOT / cfg["artifacts"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ranker_dir = out_dir / cfg["artifacts"]["ranker_subdir"]
    report_path = ROOT / cfg["artifacts"]["report_path"]

    # --- 1. data --------------------------------------------------------
    parquet = ROOT / cfg["data"]["part_parquet"]
    logger.info("loading cleaned data from %s", parquet)
    df = pd.read_parquet(parquet)
    logger.info("loaded %s rows × %s cols", f"{len(df):,}", len(df.columns))

    # --- 2. train stub ranker ------------------------------------------
    train_cfg = _make_train_config(cfg)
    from ranker.features import FEATURE_NAMES
    n_features = len(FEATURE_NAMES)
    rcfg = RankerConfig(
        n_features=n_features,
        hidden_dims=tuple(cfg["ranker"]["hidden_dims"]),
        dropout=float(cfg["ranker"]["dropout"]),
    )
    result = train_stub_ranker(df, train_config=train_cfg, ranker_config=rcfg)
    save_ranker(result, ranker_dir)
    logger.info("ranker saved to %s", ranker_dir)

    # --- 3. sample users + seeds ---------------------------------------
    n_users = int(cfg["simulation"]["n_users"])
    n_cascades = int(cfg["simulation"]["n_cascades"])
    originals_only = bool(cfg["simulation"].get("originals_only_seeds", False))

    user_pool = sample_users(df, n=n_users, seed=seed)
    seeds = sample_seeds(df, n=n_cascades, seed=seed, originals_only=originals_only)

    # --- 4. simulate under all three regimes ---------------------------
    sim_cfg = _make_sim_config(cfg)
    scoring_configs = _build_scoring_configs(cfg)

    sim_results = {}
    for name, sc in scoring_configs.items():
        logger.info("simulating regime=%s", name)
        r = simulate_cascades(
            seeds=seeds,
            user_pool=user_pool,
            ranker=result.model,
            scaler=result.scaler,
            scoring_config=sc,
            sim_config=sim_cfg,
        )
        sim_results[name] = r
        r.per_bin.to_parquet(out_dir / f"sim_{name}_per_bin.parquet", index=False)
        r.per_cascade.to_parquet(out_dir / f"sim_{name}_per_cascade.parquet", index=False)

    # --- 5. Stage 1 validation on the additive regime -------------------
    primary = sim_results["additive"]
    stage1 = stage1_root_reply_check(df, primary.per_cascade)
    logger.info(
        "Stage 1 (additive) — KS=%.4f p=%.3g  obs_median=%.1f sim_median=%.1f",
        stage1.ks_stat, stage1.ks_pvalue, stage1.obs_median, stage1.sim_median,
    )

    # Run the same Stage 1 check on the other regimes too — useful for the
    # report even though it isn't used as the gate.
    stage1_per_regime = {
        name: stage1_root_reply_check(df, r.per_cascade)
        for name, r in sim_results.items()
    }

    # --- 6. report ------------------------------------------------------
    write_report(
        config_path=config_path,
        cfg=cfg,
        train_metrics=result.metrics,
        scoring_configs=scoring_configs,
        sim_results=sim_results,
        stage1_per_regime=stage1_per_regime,
        report_path=report_path,
    )
    logger.info("wrote report to %s", report_path)

    # --- 7. machine-readable metrics (for downstream scripting) --------
    metrics = {
        "config": cfg,
        "train_metrics": result.metrics,
        "stage1_per_regime": {
            name: asdict(t) for name, t in stage1_per_regime.items()
        },
        "sim_per_cascade_summary": {
            name: r.per_cascade[
                ["n_reply", "n_retweet", "n_like", "n_deep", "score"]
            ].describe().to_dict()
            for name, r in sim_results.items()
        },
    }
    (out_dir / "phase2_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )


def write_report(
    *,
    config_path: Path,
    cfg: dict[str, Any],
    train_metrics: dict[str, float],
    scoring_configs: dict[str, ScoringConfig],
    sim_results: dict[str, Any],
    stage1_per_regime: dict[str, Any],
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 2 Report — MVP end-to-end pipeline\n")
    lines.append(f"Config: `{config_path.resolve().relative_to(ROOT)}`")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(
        f"Seed: {cfg['random_seed']} | "
        f"n_users={cfg['simulation']['n_users']} | "
        f"n_cascades={cfg['simulation']['n_cascades']}\n"
    )

    # Training
    lines.append("## 1. Stub Heavy Ranker training\n")
    lines.append("```")
    for k, v in train_metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:24s} {v:.4f}")
        else:
            lines.append(f"  {k:24s} {v}")
    lines.append("```\n")
    lines.append(
        "AUCs are on the held-out validation slice. Stub-level performance is "
        "expected — calibrated MaskNet replaces this in Phase 3.\n"
    )

    # Per-regime simulation summaries
    lines.append("## 2. Simulated cascades — per regime\n")
    for name, r in sim_results.items():
        cfg_r = scoring_configs[name]
        lines.append(f"### regime = `{name}`  ({cfg_r.regime}, alpha={cfg_r.alpha})\n")
        desc = r.per_cascade[
            ["n_reply", "n_retweet", "n_like", "n_deep", "score", "total_exposures"]
        ].describe(percentiles=[0.5, 0.9, 0.99])
        lines.append("```")
        lines.append(desc.to_string(float_format=lambda x: f"{x:,.2f}"))
        lines.append("```\n")

    # Stage 1
    lines.append("## 3. Stage 1 validation — root-reply-count distribution\n")
    lines.append(
        "Two-sample KS + Mann–Whitney between observed root.replyCount "
        "(per-conversation, in-corpus root only) and simulated total reply "
        "events per cascade.\n"
    )
    lines.append("| regime | n_obs | n_sim | obs_median | sim_median | KS | KS p | MW p |")
    lines.append("|--------|-------|-------|------------|------------|------|--------|--------|")
    for name, t in stage1_per_regime.items():
        lines.append(
            f"| {name} | {t.n_observed:,} | {t.n_simulated:,} | "
            f"{t.obs_median:.1f} | {t.sim_median:.1f} | "
            f"{t.ks_stat:.3f} | {t.ks_pvalue:.2e} | {t.mw_pvalue:.2e} |"
        )
    lines.append("")
    lines.append(
        "Phase 2's decision gate is **\"no errors, plausible outputs\"**, not "
        "passing a goodness-of-fit test. With 100 simulated cascades and "
        "~600K observed roots, KS will reject almost any null at the third "
        "decimal of the p-value — this is expected. What we want here is "
        "(a) the loop ran end-to-end and (b) simulated counts are in a "
        "plausible order of magnitude relative to observed. Stage 1's "
        "*real* test happens at Phase 3 scale (50K–100K cascades) when "
        "we'll judge whether the *shape* of the simulated distribution "
        "matches.\n"
    )

    lines.append("## 4. Decision gate notes\n")
    lines.append(
        "- Pipeline runs end-to-end without error: ✓ (this report exists)."
    )
    lines.append(
        "- Three scoring regimes wired and producing non-degenerate scores: "
        "see § 2 (variance in per-cascade `score` and downstream counts)."
    )
    lines.append(
        "- Refactor opportunities to flag for Phase 3:"
    )
    lines.append(
        "  - Ranker stub does not personalize per viewer; Phase 3 should "
        "introduce viewer-side features so the ranker can produce per-(user, "
        "tweet) scores rather than per-tweet."
    )
    lines.append(
        "  - Cascade simulator uses a single global decay; circadian "
        "patterns in the real data are visible and worth fitting."
    )
    lines.append(
        "  - User pool isn't yet used by the simulator (size only). Phase 3 "
        "exposure modeling needs to consume it."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_phase2_mvp.yaml",
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

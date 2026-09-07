"""Phase 5 driver — robustness sensitivity sweeps.

Runs the Phase 4 pipeline along four sensitivity axes:

1. **Ablation form:** the three pre-specified regimes plus
   ``ratio_correction`` and ``reflective_floor`` (Phase 5 robustness
   variants). All compared against ``additive``.
2. **Slow/fast partition:** four defensible partition assignments around
   the corrected default (design log 2026-04-26).
3. **Ranker training seed:** re-trains MaskNet with several seeds; runs
   Phase 4 against each.
4. **Data part:** runs Phase 4 separately on part_1 and part_2 (the two
   adjacent time slices in the cached USC sparse-checkout).

For each variant, records the bootstrap-CI contrast (regime − additive)
on cascade_size and audience_reach. The aggregate report shows whether
the architectural-claim signature (ablated audience_reach contrast
strictly < 0; additive_retuned > 0 or near 0) holds across all
variants.

Run::

    uv run python -m analysis.run_phase5 --config configs/experiment_phase5.yaml
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

from analysis.hypothesis import run_stage2
from analysis.labeling import add_labels
from analysis.run_phase4 import _stratified_label_seeds
from ranker.scoring import (
    PUBLISHED_WEIGHTS,
    ScoringConfig,
    make_default_configs,
    make_robustness_configs,
)
from ranker.training import (
    TrainConfig,
    load_ranker,
    save_ranker,
    train_ranker,
)
from simulation.ablation import run_ablation
from simulation.calibration import calibrate_baseline_exposures
from simulation.cascade import (
    SimConfig,
    fit_diurnal_weights,
)
from simulation.cascade import _seed_probs as _seed_probs_internal
from simulation.users import sample_users

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


# ---- one-variant pipeline -----------------------------------------------

def _build_scoring_set(
    cfg: dict[str, Any],
    *,
    extra_regimes: dict[str, ScoringConfig] | None = None,
    partition_override: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
) -> dict[str, ScoringConfig]:
    """Build the regime dict used for one ablation run.

    Always includes the three default regimes (additive, ablated,
    additive_retuned) plus any extras. If ``partition_override`` is given,
    the ``ablated`` entry's slow_heads/fast_heads are replaced.
    """
    out: dict[str, ScoringConfig] = {}
    defaults = make_default_configs()
    alpha = float(cfg["scoring_alpha"]) if "scoring_alpha" in cfg else 1.0

    abl = defaults["ablated"]
    if partition_override is not None:
        slow_heads, fast_heads = partition_override
        defaults["ablated"] = ScoringConfig(
            regime="ablated", weights=abl.weights, alpha=alpha,
            slow_heads=slow_heads, fast_heads=fast_heads,
        )
    else:
        defaults["ablated"] = ScoringConfig(
            regime="ablated", weights=abl.weights, alpha=alpha,
            slow_heads=abl.slow_heads, fast_heads=abl.fast_heads,
        )

    for k in ("additive", "ablated", "additive_retuned"):
        out[k] = defaults[k]
    if extra_regimes:
        for k, v in extra_regimes.items():
            out[k] = v
    return out


def _run_variant(
    *,
    df: pd.DataFrame,
    user_pool: pd.DataFrame,
    ranker,
    scaler,
    diurnal: np.ndarray,
    seeds: pd.DataFrame,
    cfg: dict[str, Any],
    scoring_configs: dict[str, ScoringConfig],
    seed: int,
) -> dict[str, Any]:
    """Run one Phase-4-style pipeline; return the bootstrap contrasts."""
    s = cfg["simulation"]

    # Calibrate using the additive regime (per project convention).
    seed_probs = _seed_probs_internal(seeds, ranker, scaler)
    calib = calibrate_baseline_exposures(
        observed_df=df,
        seed_probs=seed_probs,
        scoring_config=scoring_configs["additive"],
        target=str(cfg["calibration"].get("target", "zero_inflated_nb")),
        score_normalization=str(s["score_normalization"]),
    )

    sim_cfg = SimConfig(
        n_time_bins=int(s["n_time_bins"]),
        decay_tau_hours=float(s["decay_tau_hours"]),
        baseline_exposures=float(calib.baseline_exposures),
        exposure_min=float(s["exposure_min"]),
        exposure_max=float(s["exposure_max"]),
        score_normalization=str(s["score_normalization"]),
        use_circadian=bool(s.get("use_circadian", True)),
        activity_pi=calib.activity_pi,
        dispersion_r=calib.dispersion_r,
        seed=seed,
    )

    ablation = run_ablation(
        seeds=seeds, user_pool=user_pool, ranker=ranker, scaler=scaler,
        scoring_configs=scoring_configs, sim_config=sim_cfg,
        n_replicates=int(s["n_replicates"]),
        diurnal_weights=diurnal,
    )

    pc = ablation.per_cascade.copy()
    pc["credibility_label"] = seeds["credibility_label"].iloc[
        pc["cascade_id"].to_numpy()
    ].to_numpy()

    stage2 = run_stage2(
        per_cascade=pc, per_bin=ablation.per_bin,
        regimes=tuple(scoring_configs.keys()),
        baseline_regime="additive",
        n_bootstrap=int(s.get("bootstrap_iterations", 1000)),
        bootstrap_seed=seed,
    )

    # Extract just the bootstrap contrasts we care about per regime.
    out: dict[str, Any] = {
        "calibration": {
            "baseline_exposures": float(calib.baseline_exposures),
            "activity_pi": float(calib.activity_pi or 0.0),
            "dispersion_r": float(calib.dispersion_r or 0.0),
        },
        "contrasts": {},
    }
    for metric in ("cascade_size", "audience_reach", "time_to_peak"):
        out["contrasts"][metric] = {}
        for regime, c in stage2["bootstrap_contrasts"][metric].items():
            out["contrasts"][metric][regime] = {
                "diff_point": c.diff_point,
                "diff_ci_lo": c.diff_ci_lo,
                "diff_ci_hi": c.diff_ci_hi,
            }
    return out


# ---- driver -------------------------------------------------------------

def run(config_path: Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    seed = int(cfg["random_seed"])
    timings: dict[str, float] = {}

    out_dir = ROOT / cfg["artifacts"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir = out_dir / cfg["artifacts"]["ranker_seeds_subdir"]
    seeds_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / cfg["artifacts"]["report_path"]

    primary_parquet = ROOT / cfg["data"]["primary_part"]
    with _timed(f"load primary parquet ({primary_parquet.name})", timings):
        df = pd.read_parquet(primary_parquet)
        df = add_labels(
            df,
            iffy_path=ROOT / cfg["labels"]["iffy_path"],
            mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
        )

    ranker_dir = ROOT / cfg["ranker"]["load_dir"]
    with _timed(f"load baseline ranker ({ranker_dir.name})", timings):
        model, scaler, _ = load_ranker(ranker_dir)

    n_users = int(cfg["simulation"]["n_users"])
    n_per_label = int(cfg["simulation"]["n_per_label"])
    user_pool = sample_users(df, n=n_users, seed=seed)
    seeds = _stratified_label_seeds(df, n_per_label, seed)
    diurnal = fit_diurnal_weights(df)

    sweep_results: dict[str, list[dict[str, Any]]] = {
        "ablation_form": [],
        "partition": [],
        "ranker_seed": [],
        "data_part": [],
    }

    # ---- Sweep 1: Alternative ablation forms --------------------------
    with _timed("Sweep 1: ablation forms", timings):
        # Build a regime set that includes the two robustness variants
        # alongside the three defaults.
        extras = make_robustness_configs()
        sc_set = _build_scoring_set(cfg, extra_regimes=extras)
        out = _run_variant(
            df=df, user_pool=user_pool, ranker=model, scaler=scaler,
            diurnal=diurnal, seeds=seeds, cfg=cfg,
            scoring_configs=sc_set, seed=seed,
        )
        for regime in sc_set:
            if regime == "additive":
                continue
            entry = {
                "regime": regime,
                "audience_reach": out["contrasts"]["audience_reach"][regime],
                "cascade_size": out["contrasts"]["cascade_size"][regime],
            }
            sweep_results["ablation_form"].append(entry)

    # ---- Sweep 2: Slow/fast partition variants ------------------------
    with _timed("Sweep 2: partitions", timings):
        for spec in cfg["sweeps"]["partitions"]:
            slow = tuple(spec["slow"])
            fast = tuple(spec["fast"])
            sc_set = _build_scoring_set(cfg, partition_override=(slow, fast))
            out = _run_variant(
                df=df, user_pool=user_pool, ranker=model, scaler=scaler,
                diurnal=diurnal, seeds=seeds, cfg=cfg,
                scoring_configs=sc_set, seed=seed,
            )
            entry = {
                "partition": spec["name"],
                "slow_heads": list(slow),
                "fast_heads": list(fast),
                "audience_reach_ablated": out["contrasts"]["audience_reach"]["ablated"],
                "cascade_size_ablated": out["contrasts"]["cascade_size"]["ablated"],
            }
            sweep_results["partition"].append(entry)

    # ---- Sweep 3: Ranker training seeds -------------------------------
    with _timed("Sweep 3: ranker seeds", timings):
        # Reuse the existing trained ranker for its own seed (1337);
        # re-train for the others.
        for r_seed in cfg["sweeps"]["ranker_seeds"]:
            seed_dir = seeds_dir / f"seed_{r_seed}"
            if r_seed == 1337 and ranker_dir.exists():
                # Reuse the baseline ranker for the canonical seed.
                this_model, this_scaler = model, scaler
            else:
                if not (seed_dir / "ranker.pt").exists():
                    logger.info("training ranker for seed=%d", r_seed)
                    tcfg = TrainConfig(
                        batch_size=4096, lr=1e-3, epochs=5, val_frac=0.1,
                        max_train_rows=None, device="cuda",
                        seed=int(r_seed), model_kind="masknet",
                    )
                    result = train_ranker(df, train_config=tcfg)
                    save_ranker(result, seed_dir)
                    this_model, this_scaler = result.model, result.scaler
                else:
                    this_model, this_scaler, _ = load_ranker(seed_dir)
            sc_set = _build_scoring_set(cfg)
            out = _run_variant(
                df=df, user_pool=user_pool, ranker=this_model, scaler=this_scaler,
                diurnal=diurnal, seeds=seeds, cfg=cfg,
                scoring_configs=sc_set, seed=seed,
            )
            entry = {
                "ranker_seed": int(r_seed),
                "audience_reach_ablated": out["contrasts"]["audience_reach"]["ablated"],
                "cascade_size_ablated": out["contrasts"]["cascade_size"]["ablated"],
            }
            sweep_results["ranker_seed"].append(entry)

    # ---- Sweep 4: Data parts ------------------------------------------
    with _timed("Sweep 4: data parts", timings):
        for part_path_str in cfg["data"]["parts"]:
            part_path = ROOT / part_path_str
            df_part = pd.read_parquet(part_path)
            df_part = add_labels(
                df_part,
                iffy_path=ROOT / cfg["labels"]["iffy_path"],
                mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
            )
            user_pool_part = sample_users(df_part, n=n_users, seed=seed)
            seeds_part = _stratified_label_seeds(df_part, n_per_label, seed)
            diurnal_part = fit_diurnal_weights(df_part)
            sc_set = _build_scoring_set(cfg)
            out = _run_variant(
                df=df_part, user_pool=user_pool_part, ranker=model, scaler=scaler,
                diurnal=diurnal_part, seeds=seeds_part, cfg=cfg,
                scoring_configs=sc_set, seed=seed,
            )
            entry = {
                "data_part": Path(part_path_str).name,
                "audience_reach_ablated": out["contrasts"]["audience_reach"]["ablated"],
                "cascade_size_ablated": out["contrasts"]["cascade_size"]["ablated"],
            }
            sweep_results["data_part"].append(entry)

    # ---- Report -------------------------------------------------------
    write_report(
        config_path=config_path,
        cfg=cfg,
        sweep_results=sweep_results,
        timings=timings,
        report_path=report_path,
    )
    metrics_blob = {
        "config_path": str(config_path.resolve().relative_to(ROOT)),
        "config": cfg,
        "sweep_results": sweep_results,
        "timings_seconds": timings,
    }
    (out_dir / "phase5_metrics.json").write_text(
        json.dumps(metrics_blob, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
        encoding="utf-8",
    )
    logger.info("wrote metrics to %s", out_dir / "phase5_metrics.json")
    logger.info("wrote report to %s", report_path)

    logger.info("=== timing summary ===")
    for label, dt in timings.items():
        logger.info("  %-50s %8.2fs", label, dt)


def _sign_marker(ci_lo: float, ci_hi: float) -> str:
    if ci_hi < 0:
        return "**< 0**"
    if ci_lo > 0:
        return "**> 0**"
    return "≈ 0"


def write_report(
    *,
    config_path: Path,
    cfg: dict[str, Any],
    sweep_results: dict[str, list[dict[str, Any]]],
    timings: dict[str, float],
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 5 Report — Robustness Sweeps\n")
    lines.append(f"Config: `{config_path.name}`")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(
        f"Per-variant: n_per_label={cfg['simulation']['n_per_label']:,} × 2, "
        f"n_replicates={cfg['simulation']['n_replicates']}, "
        f"bootstrap={cfg['simulation']['bootstrap_iterations']:,}\n"
    )

    lines.append(
        "Each sweep varies one knob and re-runs the Phase 4 pipeline. The "
        "architectural-claim signature (per Phase 4 v4) is **ablated **"
        "audience_reach contrast strictly < 0**. Cascade_size contrast is "
        "the noisier propagation-outcome metric.\n"
    )

    # Sweep 1: ablation forms
    lines.append("## Sweep 1 — Alternative ablation forms\n")
    lines.append(
        "Variants beyond the pre-registered three. ``ratio_correction`` and "
        "``reflective_floor`` share the architectural property "
        "(privileging slow/effortful engagement) in different functional "
        "forms; the claim predicts they should produce contrasts in the "
        "same direction as ``ablated``.\n"
    )
    lines.append("| regime | audience_reach diff | 95% CI | sign | cascade_size diff | 95% CI | sign |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sweep_results["ablation_form"]:
        ar = r["audience_reach"]
        cs = r["cascade_size"]
        lines.append(
            f"| `{r['regime']}` | {ar['diff_point']:.3f} | "
            f"[{ar['diff_ci_lo']:.3f}, {ar['diff_ci_hi']:.3f}] | "
            f"{_sign_marker(ar['diff_ci_lo'], ar['diff_ci_hi'])} | "
            f"{cs['diff_point']:.3f} | "
            f"[{cs['diff_ci_lo']:.3f}, {cs['diff_ci_hi']:.3f}] | "
            f"{_sign_marker(cs['diff_ci_lo'], cs['diff_ci_hi'])} |"
        )
    lines.append("")

    # Sweep 2: partitions
    lines.append("## Sweep 2 — Slow/fast partition variants\n")
    lines.append(
        "All hold the ``ablated`` regime functional form fixed; vary which "
        "engagement heads count as slow vs. fast. The architectural claim "
        "is robust if the audience_reach contrast stays negative across "
        "defensible partitions.\n"
    )
    lines.append("| partition | audience_reach diff | 95% CI | sign | cascade_size diff | 95% CI | sign |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sweep_results["partition"]:
        ar = r["audience_reach_ablated"]
        cs = r["cascade_size_ablated"]
        lines.append(
            f"| {r['partition']} | {ar['diff_point']:.3f} | "
            f"[{ar['diff_ci_lo']:.3f}, {ar['diff_ci_hi']:.3f}] | "
            f"{_sign_marker(ar['diff_ci_lo'], ar['diff_ci_hi'])} | "
            f"{cs['diff_point']:.3f} | "
            f"[{cs['diff_ci_lo']:.3f}, {cs['diff_ci_hi']:.3f}] | "
            f"{_sign_marker(cs['diff_ci_lo'], cs['diff_ci_hi'])} |"
        )
    lines.append("")

    # Sweep 3: ranker seeds
    lines.append("## Sweep 3 — Ranker training seed sensitivity\n")
    lines.append(
        "Re-train MaskNet with different RNG seeds; run Phase 4 against "
        "each. Tests whether the architectural finding is robust to the "
        "ranker's training stochasticity.\n"
    )
    lines.append("| ranker_seed | audience_reach diff | 95% CI | sign | cascade_size diff | 95% CI | sign |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sweep_results["ranker_seed"]:
        ar = r["audience_reach_ablated"]
        cs = r["cascade_size_ablated"]
        lines.append(
            f"| {r['ranker_seed']} | {ar['diff_point']:.3f} | "
            f"[{ar['diff_ci_lo']:.3f}, {ar['diff_ci_hi']:.3f}] | "
            f"{_sign_marker(ar['diff_ci_lo'], ar['diff_ci_hi'])} | "
            f"{cs['diff_point']:.3f} | "
            f"[{cs['diff_ci_lo']:.3f}, {cs['diff_ci_hi']:.3f}] | "
            f"{_sign_marker(cs['diff_ci_lo'], cs['diff_ci_hi'])} |"
        )
    lines.append("")

    # Sweep 4: data parts
    lines.append("## Sweep 4 — Cross-temporal generalization (data parts)\n")
    lines.append(
        "Same trained ranker (part_1) applied to each part's seeds + "
        "labels + calibration. Tests whether the architectural finding "
        "generalizes across adjacent time slices.\n"
    )
    lines.append("| data_part | audience_reach diff | 95% CI | sign | cascade_size diff | 95% CI | sign |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sweep_results["data_part"]:
        ar = r["audience_reach_ablated"]
        cs = r["cascade_size_ablated"]
        lines.append(
            f"| {r['data_part']} | {ar['diff_point']:.3f} | "
            f"[{ar['diff_ci_lo']:.3f}, {ar['diff_ci_hi']:.3f}] | "
            f"{_sign_marker(ar['diff_ci_lo'], ar['diff_ci_hi'])} | "
            f"{cs['diff_point']:.3f} | "
            f"[{cs['diff_ci_lo']:.3f}, {cs['diff_ci_hi']:.3f}] | "
            f"{_sign_marker(cs['diff_ci_lo'], cs['diff_ci_hi'])} |"
        )
    lines.append("")

    lines.append("## Step timings\n")
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
        default=ROOT / "configs" / "experiment_phase5.yaml",
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

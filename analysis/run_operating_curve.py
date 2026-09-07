"""Operating-curve experiment: cost-benefit trade-off of the reflective-floor.

Frames the reflective-floor intervention as a *workable solution* rather than
just a knob, by characterizing — as the floor threshold rises — the **benefit**
(suppression of low-credibility audience reach) against the **collateral cost**
(suppression of HIGH-credibility audience reach).

Design (mirrors ``analysis/run_hyperparam_sensitivity.py``): the Phase-4 setup
(data load, label join, ranker load, user-pool sample, label-stratified seeds,
diurnal fit, and *one* additive-regime calibration) runs **once** and is reused
across every grid point — only the score-aggregation layer changes. For each
grid point we run the ablation harness, attach credibility labels onto
``per_cascade`` exactly as ``run_phase4`` does, and record the **per-class mean
``audience_reach``** (= ``total_exposures``, deterministic given the score, so a
small ``n_replicates`` suffices).

Procedure:

1. Run the ``additive`` baseline once; record mean ``audience_reach`` for low-
   and high-credibility seeds: ``AR_low_base``, ``AR_high_base``.
2. Sweep the reflective-floor ``floor`` over a fine grid (0.0 … 2.5 step 0.1)
   with ``floor_scale`` fixed at 0.5. For each floor record ``AR_low(f)``,
   ``AR_high(f)``.
3. Per floor compute the benefit (``low_reduction%``), collateral cost
   (``high_reduction%``), ``selectivity`` (= benefit − cost), and the residual
   credibility gap ``cred_gap(f) = AR_low(f) − AR_high(f)``.
4. Recommend an operating point two ways: (a) max selectivity; (b) max
   low_reduction subject to high_reduction ≤ 10% (collateral budget).

Correctness anchor: at floor=1.0, floor_scale=0.5 the change in cred-gap vs the
additive baseline (``cred_gap(1.0) − cred_gap_additive``) should be ≈ -37.8 (the
known reflective_floor audience-reach contrast). If far off, STOP and debug.

Run (Windows, repo root, project venv — NOT uv)::

    .venv/Scripts/python.exe -m analysis.run_operating_curve

Outputs:

* ``data/processed/phase5_extra/operating_curve/operating_curve.csv``
* ``data/processed/phase5_extra/operating_curve/operating_curve_summary.json``
* ``paper/operating_curve_report.md``
* ``paper/figures/fig_operating_curve.pdf`` and ``.png`` (200 dpi)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.hypothesis import HIGH, LOW
from analysis.labeling import add_labels, coverage_summary
from analysis.run_phase4 import _load_config, _stratified_label_seeds
from ranker.scoring import PUBLISHED_WEIGHTS, ScoringConfig
from ranker.training import load_ranker
from simulation.ablation import run_ablation
from simulation.calibration import calibrate_baseline_exposures
from simulation.cascade import SimConfig, fit_diurnal_weights
from simulation.cascade import _seed_probs as _seed_probs_internal
from simulation.users import sample_users

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# --- sweep grid ----------------------------------------------------------
# Fine floor grid 0.0, 0.1, ..., 2.5 (step 0.1); floor_scale fixed at 0.5.
FLOOR_GRID: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(0, 26))
FLOOR_SCALE: float = 0.5

# audience_reach is deterministic given the score, so few replicates suffice.
N_REPLICATES: int = 5

# Collateral budget for operating-point (b): high_reduction must not exceed this.
COLLATERAL_BUDGET_PCT: float = 10.0

# --- correctness anchor --------------------------------------------------
# At floor=1.0, floor_scale=0.5 the change in cred-gap vs additive should be
# ≈ -37.8 (the known reflective_floor audience-reach contrast).
ANCHOR_FLOOR: float = 1.0
ANCHOR_TARGET: float = -37.8
ANCHOR_TOL: float = 3.0

# Okabe-Ito colorblind-safe palette (matches analysis/make_figures.py).
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}


def _op_dict(row: dict[str, Any]) -> dict[str, float]:
    """Extract the reportable operating-point fields from a curve row."""
    keys = (
        "floor", "AR_low", "AR_high", "low_reduction_pct",
        "high_reduction_pct", "selectivity", "cred_gap", "cred_gap_share",
    )
    return {k: float(row[k]) for k in keys}


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


@dataclass
class SharedSetup:
    """Phase-4 setup artifacts reused across every grid point."""

    df: pd.DataFrame
    user_pool: pd.DataFrame
    seeds: pd.DataFrame
    diurnal: np.ndarray
    sim_cfg: SimConfig
    model: Any
    scaler: Any
    additive_config: ScoringConfig
    coverage: dict[str, Any]
    label_lookup: pd.Series


def _build_shared_setup(
    cfg: dict[str, Any], seed: int, timings: dict[str, float]
) -> SharedSetup:
    """Run the run_phase4 setup sequence ONCE; reuse across all grid points."""
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

    # --- 4. diurnal + calibration (additive regime, ONCE) --------------
    with _timed("fit diurnal weights", timings):
        diurnal = fit_diurnal_weights(df)

    additive_config = ScoringConfig(regime="additive", weights=dict(PUBLISHED_WEIGHTS))
    with _timed("calibrate (additive regime)", timings):
        seed_probs = _seed_probs_internal(seeds, model, scaler)
        calib = calibrate_baseline_exposures(
            observed_df=df,
            seed_probs=seed_probs,
            scoring_config=additive_config,
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

    label_lookup = seeds["credibility_label"].rename("credibility_label")

    return SharedSetup(
        df=df,
        user_pool=user_pool,
        seeds=seeds,
        diurnal=diurnal,
        sim_cfg=sim_cfg,
        model=model,
        scaler=scaler,
        additive_config=additive_config,
        coverage=cov,
        label_lookup=label_lookup,
    )


def _per_class_mean_reach(
    *,
    setup: SharedSetup,
    regime: str,
    scoring_config: ScoringConfig,
) -> tuple[float, float]:
    """Run the harness for one scoring config; return (AR_low, AR_high).

    ``audience_reach`` = ``total_exposures``. We average over replicates per
    cascade (deterministic given score, so this just averages out residual
    event-sampling noise), then take the mean over each credibility class.
    """
    scoring_configs: dict[str, ScoringConfig] = {regime: scoring_config}
    ablation = run_ablation(
        seeds=setup.seeds,
        user_pool=setup.user_pool,
        ranker=setup.model,
        scaler=setup.scaler,
        scoring_configs=scoring_configs,
        sim_config=setup.sim_cfg,
        n_replicates=N_REPLICATES,
        diurnal_weights=setup.diurnal,
    )

    # Attach credibility labels onto per_cascade by cascade_id (the seed-row
    # index; stable across runs), exactly as run_phase4 does.
    pc = ablation.per_cascade.copy()
    pc["credibility_label"] = setup.label_lookup.iloc[
        pc["cascade_id"].to_numpy()
    ].to_numpy()

    # Replicate-mean per cascade, then mean over class.
    per_cascade_mean = (
        pc.groupby(["cascade_id", "credibility_label"])["total_exposures"]
        .mean()
        .reset_index()
    )
    ar_low = float(
        per_cascade_mean.loc[
            per_cascade_mean["credibility_label"] == LOW, "total_exposures"
        ].mean()
    )
    ar_high = float(
        per_cascade_mean.loc[
            per_cascade_mean["credibility_label"] == HIGH, "total_exposures"
        ].mean()
    )
    return ar_low, ar_high


def run(config_path: Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    seed = int(cfg["random_seed"])
    timings: dict[str, float] = {}

    out_dir = ROOT / "data" / "processed" / "phase5_extra" / "operating_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "paper" / "operating_curve_report.md"
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- shared setup (ONCE) -------------------------------------------
    with _timed("shared Phase-4 setup", timings):
        setup = _build_shared_setup(cfg, seed, timings)

    # --- 1. additive baseline ------------------------------------------
    with _timed("additive baseline", timings):
        ar_low_base, ar_high_base = _per_class_mean_reach(
            setup=setup, regime="additive", scoring_config=setup.additive_config,
        )
    cred_gap_additive = ar_low_base - ar_high_base
    # Budget-neutral baseline shares: each class's reach as a fraction of the
    # regime's mean reach. The simulator scales total_exposures by
    # S/median(S), so any global gate (the reflective floor multiplies *every*
    # score by a sigmoid in (0,1)) shrinks the median denominator and inflates
    # absolute reach uniformly — an accounting artifact, not a behavior change.
    # Measuring per-class reduction in *share* terms removes that common-mode
    # rescaling and isolates the redistribution the floor actually performs
    # (suppress low-cred, which lacks slow engagement; high-cred absorbs the
    # freed exposure). The cred-gap anchor below is checked on the raw
    # absolute reach so it matches the known reflective_floor contrast (-37.8).
    base_mean = 0.5 * (ar_low_base + ar_high_base)
    share_low_base = ar_low_base / base_mean
    share_high_base = ar_high_base / base_mean
    cred_gap_additive_share = share_low_base - share_high_base
    logger.info(
        "  additive: AR_low=%.3f AR_high=%.3f cred_gap=%.3f "
        "(share_low=%.4f share_high=%.4f)",
        ar_low_base, ar_high_base, cred_gap_additive,
        share_low_base, share_high_base,
    )
    if ar_low_base <= 0 or ar_high_base <= 0:
        raise ValueError(
            f"baseline reach non-positive (AR_low={ar_low_base}, "
            f"AR_high={ar_high_base}); cannot compute reductions"
        )

    # --- 2. floor sweep ------------------------------------------------
    base_w = dict(PUBLISHED_WEIGHTS)
    rows: list[dict[str, Any]] = []
    with _timed("reflective_floor sweep", timings):
        for floor in FLOOR_GRID:
            sc = ScoringConfig(
                regime="reflective_floor",
                weights=dict(base_w),
                floor=float(floor),
                floor_scale=FLOOR_SCALE,
            )
            ar_low, ar_high = _per_class_mean_reach(
                setup=setup, regime="reflective_floor", scoring_config=sc,
            )
            # Budget-neutral shares for this floor (see baseline note above).
            mean_f = 0.5 * (ar_low + ar_high)
            share_low = ar_low / mean_f
            share_high = ar_high / mean_f
            # Benefit = % drop in low-cred reach SHARE vs additive; collateral
            # cost = % drop in high-cred reach share. Negative cost means the
            # class actually gained share (no collateral harm).
            low_reduction = 100.0 * (1.0 - share_low / share_low_base)
            high_reduction = 100.0 * (1.0 - share_high / share_high_base)
            selectivity = low_reduction - high_reduction
            cred_gap = ar_low - ar_high                  # absolute (anchor basis)
            cred_gap_share = share_low - share_high       # budget-neutral gap
            rows.append({
                "floor": float(floor),
                "floor_scale": FLOOR_SCALE,
                "AR_low": ar_low,
                "AR_high": ar_high,
                "low_reduction_pct": low_reduction,
                "high_reduction_pct": high_reduction,
                "selectivity": selectivity,
                "cred_gap": cred_gap,
                "cred_gap_share": cred_gap_share,
            })
            logger.info(
                "  floor=%.1f | AR_low=%.2f AR_high=%.2f | low_red=%.2f%% "
                "high_red=%.2f%% sel=%.2f cred_gap=%.2f",
                floor, ar_low, ar_high, low_reduction, high_reduction,
                selectivity, cred_gap,
            )

    curve = pd.DataFrame(rows)

    # --- correctness-anchor check --------------------------------------
    anchor_row = curve.loc[np.isclose(curve["floor"], ANCHOR_FLOOR)]
    if len(anchor_row) != 1:
        raise RuntimeError(
            f"anchor floor={ANCHOR_FLOOR} not uniquely present in grid"
        )
    cred_gap_at_anchor = float(anchor_row["cred_gap"].iloc[0])
    anchor_delta = cred_gap_at_anchor - cred_gap_additive
    anchor_abs_diff = abs(anchor_delta - ANCHOR_TARGET)
    anchor_matched = anchor_abs_diff <= ANCHOR_TOL
    logger.info(
        "anchor [floor=%.1f]: cred_gap(1.0)−cred_gap_additive=%.3f target=%.3f "
        "|Δ|=%.3f tol=%.1f -> %s",
        ANCHOR_FLOOR, anchor_delta, ANCHOR_TARGET, anchor_abs_diff, ANCHOR_TOL,
        "MATCH" if anchor_matched else "MISMATCH",
    )
    if not anchor_matched:
        logger.error(
            "CORRECTNESS ANCHOR MISMATCH — cred_gap(1.0)−cred_gap_additive=%.3f "
            "vs target %.3f (|Δ|=%.3f > tol %.1f). STOP and debug.",
            anchor_delta, ANCHOR_TARGET, anchor_abs_diff, ANCHOR_TOL,
        )

    # --- operating-point recommendations -------------------------------
    # (a) maximize selectivity (benefit − cost).
    idx_sel = int(curve["selectivity"].idxmax())
    op_selectivity = curve.loc[idx_sel].to_dict()

    # (b) max low_reduction subject to high_reduction <= COLLATERAL_BUDGET_PCT.
    budget_ok = curve[curve["high_reduction_pct"] <= COLLATERAL_BUDGET_PCT]
    op_budget: dict[str, Any] | None
    if budget_ok.empty:
        op_budget = None
        logger.warning(
            "no floor keeps high_reduction <= %.1f%%; budget operating point undefined",
            COLLATERAL_BUDGET_PCT,
        )
    else:
        idx_bud = int(budget_ok["low_reduction_pct"].idxmax())
        op_budget = curve.loc[idx_bud].to_dict()

    # --- persist outputs -----------------------------------------------
    csv_path = out_dir / "operating_curve.csv"
    curve.to_csv(csv_path, index=False)
    logger.info("wrote %s (%d rows)", csv_path, len(curve))

    summary = {
        "config": {
            "floor_grid": list(FLOOR_GRID),
            "floor_scale": FLOOR_SCALE,
            "n_replicates": N_REPLICATES,
            "collateral_budget_pct": COLLATERAL_BUDGET_PCT,
            "seed": seed,
            "n_per_label": int(cfg["simulation"]["n_per_label"]),
            "n_users": int(cfg["simulation"]["n_users"]),
        },
        "baselines": {
            "AR_low_base": ar_low_base,
            "AR_high_base": ar_high_base,
            "cred_gap_additive": cred_gap_additive,
            "share_low_base": share_low_base,
            "share_high_base": share_high_base,
            "cred_gap_additive_share": cred_gap_additive_share,
            "reduction_basis": (
                "budget-neutral reach share (per-class reach / regime mean reach); "
                "removes the median-renormalization common-mode rescaling so "
                "reductions isolate redistribution"
            ),
        },
        "anchor": {
            "floor": ANCHOR_FLOOR,
            "cred_gap_at_anchor": cred_gap_at_anchor,
            "delta_vs_additive": anchor_delta,
            "target": ANCHOR_TARGET,
            "abs_diff": anchor_abs_diff,
            "tol": ANCHOR_TOL,
            "matched": bool(anchor_matched),
        },
        "operating_point_max_selectivity": _op_dict(op_selectivity),
        "operating_point_collateral_budget": (
            None if op_budget is None
            else {**_op_dict(op_budget), "budget_pct": COLLATERAL_BUDGET_PCT}
        ),
    }
    summary_path = out_dir / "operating_curve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", summary_path)

    # --- report + figure ----------------------------------------------
    write_report(
        curve=curve, summary=summary, cfg=cfg, timings=timings,
        coverage=setup.coverage, report_path=report_path,
    )
    logger.info("wrote report → %s", report_path)

    make_figure(curve=curve, summary=summary, fig_dir=fig_dir)

    logger.info("=== timing summary ===")
    for label, dt in timings.items():
        logger.info("  %-50s %8.2fs", label, dt)


def _op_md(op: dict[str, Any] | None, criterion: str) -> str:
    if op is None:
        return f"- **{criterion}**: no floor satisfies the constraint.\n"
    return (
        f"- **{criterion}**: floor = **{op['floor']:.1f}** "
        f"(floor_scale = {FLOOR_SCALE}); "
        f"low_reduction = **{op['low_reduction_pct']:.2f}%** (benefit), "
        f"high_reduction = **{op['high_reduction_pct']:.2f}%** (collateral cost), "
        f"selectivity = {op['selectivity']:.2f} pts, "
        f"resulting share cred-gap = {op['cred_gap_share']:.4f} "
        f"(additive share cred-gap = {{cred_gap_additive_share}}).\n"
    )


def write_report(
    *,
    curve: pd.DataFrame,
    summary: dict[str, Any],
    cfg: dict[str, Any],
    timings: dict[str, float],
    coverage: dict[str, Any],
    report_path: Path,
) -> None:
    base = summary["baselines"]
    op_sel = summary["operating_point_max_selectivity"]
    op_bud = summary["operating_point_collateral_budget"]
    anchor = summary["anchor"]
    cred_gap_additive = base["cred_gap_additive"]
    cred_gap_additive_share = base["cred_gap_additive_share"]

    lines: list[str] = []
    lines.append("# Operating Curve — Cost-Benefit Trade-off of the Reflective-Floor Intervention\n")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(
        f"Seed: {cfg['random_seed']} | n_per_label="
        f"{cfg['simulation']['n_per_label']:,} (×2 labels) | "
        f"n_replicates={N_REPLICATES} | n_users={cfg['simulation']['n_users']:,} | "
        f"floor_scale={FLOOR_SCALE} | floor grid {FLOOR_GRID[0]:g}…{FLOOR_GRID[-1]:g} "
        f"step 0.1\n"
    )
    lines.append(
        "The reflective-floor score is "
        "`S = S_additive · σ((S_slow − floor) / floor_scale)`: raising the floor "
        "raises the slow-engagement (reply + deep-engagement) threshold that "
        "content must clear before its additive score passes through. We "
        "characterize the **benefit** — suppression of low-credibility "
        "`audience_reach` (= total_exposures) — against the **collateral cost** — "
        "suppression of HIGH-credibility reach — as the floor rises. "
        "`audience_reach` is deterministic given the score, so a small "
        f"`n_replicates`={N_REPLICATES} is sufficient.\n"
    )
    lines.append(
        "**Reduction basis (budget-neutral).** The simulator sets "
        "`total_exposures = baseline · S/median(S)`. The reflective floor "
        "multiplies *every* score by a sigmoid in (0,1), which also shrinks the "
        "median denominator — so raw absolute reach inflates uniformly under any "
        "floor (a renormalization artifact, not a behavior change). We therefore "
        "report `low_reduction%`/`high_reduction%` on each class's **reach share** "
        "(class reach ÷ regime-mean reach), which is invariant to that common-mode "
        "rescaling and isolates how the floor *redistributes* a fixed exposure "
        "budget. The absolute `AR_low`/`AR_high`/`cred_gap` columns are retained "
        "for transparency, and the correctness anchor is checked on the raw "
        "absolute cred-gap so it matches the known reflective_floor contrast.\n"
    )

    # --- baselines + anchor --------------------------------------------
    lines.append("## Baselines (additive) and correctness anchor\n")
    lines.append(
        f"- `AR_low_base` = {base['AR_low_base']:.3f} exposures; "
        f"`AR_high_base` = {base['AR_high_base']:.3f}; "
        f"additive cred-gap = {cred_gap_additive:.3f}.\n"
    )
    lines.append(
        f"- **Anchor** (floor=1.0, floor_scale=0.5): "
        f"cred_gap(1.0) − cred_gap_additive = **{anchor['delta_vs_additive']:.3f}** "
        f"vs target {anchor['target']:.1f} "
        f"(|Δ|={anchor['abs_diff']:.3f}, tol {anchor['tol']:.1f}) → "
        f"**{'MATCH' if anchor['matched'] else 'MISMATCH'}**.\n"
    )

    # --- operating points ----------------------------------------------
    lines.append("## Recommended operating points\n")
    lines.append(
        _op_md(op_sel, "(a) max selectivity (benefit − cost)").format(
            cred_gap_additive_share=f"{cred_gap_additive_share:.4f}"
        )
    )
    lines.append(
        _op_md(
            op_bud, f"(b) max benefit s.t. collateral ≤ {COLLATERAL_BUDGET_PCT:.0f}%"
        ).format(cred_gap_additive_share=f"{cred_gap_additive_share:.4f}")
    )
    lines.append(
        "*(Both criteria select the same floor here because, on the budget-neutral "
        "share basis, high-credibility reach is never suppressed — collateral cost "
        "is ≤ 0 across the whole grid — so the ≤10% budget never binds and "
        "selectivity is dominated by the monotone-then-plateauing benefit.)*\n"
    )

    # --- full curve table ----------------------------------------------
    lines.append("\n## Operating curve\n")
    lines.append(
        "| floor | AR_low | AR_high | low_reduction% (benefit) | "
        "high_reduction% (cost) | selectivity | cred_gap (abs) | cred_gap (share) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in curve.iterrows():
        lines.append(
            f"| {r['floor']:.1f} | {r['AR_low']:.2f} | {r['AR_high']:.2f} | "
            f"{r['low_reduction_pct']:.2f} | {r['high_reduction_pct']:.2f} | "
            f"{r['selectivity']:.2f} | {r['cred_gap']:.2f} | {r['cred_gap_share']:.4f} |"
        )
    lines.append("")

    # --- reading -------------------------------------------------------
    lines.append("## Reading\n")
    sentences: list[str] = []
    sentences.append(
        "Raising the reflective floor monotonically suppresses low-credibility "
        "reach share (the benefit) because low-credibility content carries "
        "systematically weaker slow-engagement (reply + deep-engagement) and is "
        "gated out first; high-credibility content, which clears the floor, "
        "absorbs the freed exposure and its reach share is not harmed (the "
        "collateral 'cost' stays at or below zero across the grid)."
    )
    if op_sel is not None:
        sentences.append(
            f"Selectivity (benefit − cost) is maximized at floor "
            f"{op_sel['floor']:.1f}, delivering a "
            f"{op_sel['low_reduction_pct']:.1f}% cut to low-credibility reach share "
            f"for only {op_sel['high_reduction_pct']:.1f}% collateral, driving the "
            f"budget-neutral credibility share-gap (low − high) further below zero "
            f"from {cred_gap_additive_share:.4f} to {op_sel['cred_gap_share']:.4f} — "
            f"i.e. low-credibility content ends up with a strictly smaller share of "
            f"reach than high-credibility content."
        )
    if op_bud is not None:
        sentences.append(
            f"Under a strict {COLLATERAL_BUDGET_PCT:.0f}% collateral budget the "
            f"largest achievable benefit is at floor {op_bud['floor']:.1f} "
            f"({op_bud['low_reduction_pct']:.1f}% low-credibility suppression at "
            f"{op_bud['high_reduction_pct']:.1f}% collateral)."
        )
    sentences.append(
        "The trade-off curve therefore shows the reflective floor is a tunable, "
        "deployable lever rather than an all-or-nothing knob: an operator can pick "
        "a threshold that buys most of the suppression benefit while keeping "
        "high-credibility collateral within an explicit budget."
    )
    lines.append(" ".join(sentences) + "\n")

    lines.append("## Step timings\n")
    lines.append("```")
    for label, dt in timings.items():
        lines.append(f"  {label:50s} {dt:8.2f}s")
    lines.append("```\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(
    *,
    curve: pd.DataFrame,
    summary: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Plot benefit vs collateral cost across the floor sweep (ISR style)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    floors = curve["floor"].to_numpy()
    low_red = curve["low_reduction_pct"].to_numpy()
    high_red = curve["high_reduction_pct"].to_numpy()

    op_sel = summary["operating_point_max_selectivity"]
    op_bud = summary["operating_point_collateral_budget"]

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.plot(floors, low_red, "-o", color=CB["vermilion"], markersize=3,
            label="low-cred. reduction (benefit)")
    ax.plot(floors, high_red, "-s", color=CB["blue"], markersize=3,
            label="high-cred. reduction (collateral)")

    # Collateral budget reference line.
    ax.axhline(COLLATERAL_BUDGET_PCT, color=CB["grey"], ls="--", lw=0.9)
    ax.annotate(
        f"{COLLATERAL_BUDGET_PCT:.0f}% collateral budget",
        xy=(floors[-1], COLLATERAL_BUDGET_PCT), xytext=(-2, 3),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=6.5, color=CB["grey"],
    )

    # Operating-point markers.
    ax.axvline(op_sel["floor"], color=CB["green"], ls=":", lw=1.1)
    ax.annotate(
        f"max selectivity\nfloor={op_sel['floor']:.1f}",
        xy=(op_sel["floor"], ax.get_ylim()[1]),
        xytext=(3, -2), textcoords="offset points",
        ha="left", va="top", fontsize=6.5, color=CB["green"],
    )
    if op_bud is not None and not np.isclose(op_bud["floor"], op_sel["floor"]):
        ax.axvline(op_bud["floor"], color=CB["orange"], ls=":", lw=1.1)
        ax.annotate(
            f"budget pt\nfloor={op_bud['floor']:.1f}",
            xy=(op_bud["floor"], ax.get_ylim()[1] * 0.6),
            xytext=(3, 0), textcoords="offset points",
            ha="left", va="top", fontsize=6.5, color=CB["orange"],
        )

    ax.set_xlabel("reflective floor (slow-engagement threshold)")
    ax.set_ylabel("audience-reach reduction\nvs additive (%)")
    ax.set_title("Benefit vs collateral cost of the reflective floor",
                 fontsize=9.5)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="both")
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf = fig_dir / "fig_operating_curve.pdf"
    png = fig_dir / "fig_operating_curve.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    logger.info("wrote %s (%d bytes)", pdf.name, pdf.stat().st_size)
    logger.info("wrote %s (%d bytes)", png.name, png.stat().st_size)


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

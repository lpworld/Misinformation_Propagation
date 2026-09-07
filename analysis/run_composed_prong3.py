"""Prong-3 simulation of the composed score: does the bounded content gate
survive the full cascade simulator?

The dual-lever frontier (``analysis/run_dual_lever_frontier.py``) established
at the *rank* level that a bounded content-credibility gate composed onto the
reflective floor — S = S_rf · g^η — passes the claim-level check (Prong 2) at
η ≈ 0.15 while retaining the architectural targeting (Prong 1). What it did
NOT establish is the system-level question: when the composed score actually
drives the cascade simulator, does the architectural gap-closure (Prong 3,
the Phase-4/5 result) survive the composition?

This driver answers that by re-running the exact Phase-4 harness (same data,
labels, ranker, calibration, seeds, replicate structure) with five regimes:

- ``additive``           — production-style baseline.
- ``reflective_floor``   — architecture-only headline (floor=1.0, scale=0.5).
- ``composed_eta015``    — reflective_floor × g^0.15 (the frontier knee).
- ``composed_eta100``    — reflective_floor × g^1.0 (unbounded endpoint).
- ``composed_tail010``   — reflective_floor × min(1, g/g_(0.10)) (tail-only).

Content gate g: the run_prong2_solution text classifier fit on all 400
claim-labeled tweets, applied to each seed's ``rawContent``;
g = quality_gate(1 − P(misinfo)) with label-free thresholds computed on the
seed distribution itself. Identical recipe (and identical caveat — seed-level
p_cred is unvalidated against claim labels) to the frontier's 5,000-seed step.

The gate enters the simulator via the ``score_multiplier`` pass-through added
to ``simulation.cascade.simulate_cascades`` — the engagement-scoring layer is
untouched; the gate is a per-seed multiplicative factor on the aggregate
score, exactly the deployment shape described in the manuscript.

Anchor: the ``reflective_floor`` audience_reach contrast vs ``additive`` must
reproduce the known Phase-5 value ≈ −37.8 (±3). If it does not, STOP.

Run (Windows, repo root, project venv — NOT uv)::

    .venv/Scripts/python.exe -m analysis.run_composed_prong3

Outputs:

* ``data/processed/phase5_extra/composed_prong3/results.json``
* ``data/processed/phase5_extra/composed_prong3/per_cascade.parquet``
* ``paper/composed_prong3_report.md``
* ``paper/figures/fig_composed_prong3.pdf`` / ``.png``
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.hypothesis import HIGH, LOW, run_stage2
from analysis.labeling import add_labels
from analysis.run_phase4 import _load_config, _stratified_label_seeds
from analysis.run_dual_lever_frontier import load_claims, tail_gate, tempered_gate
from analysis.run_prong2_solution import (
    CB,
    DOUBLE,
    SEED,
    build_text_classifier,
    quality_gate,
)
from ranker.scoring import PUBLISHED_WEIGHTS, ScoringConfig
from ranker.training import load_ranker
from simulation.ablation import run_ablation
from simulation.calibration import calibrate_baseline_exposures
from simulation.cascade import SimConfig, fit_diurnal_weights
from simulation.cascade import _seed_probs as _seed_probs_internal
from simulation.users import sample_users

logger = logging.getLogger("composed_prong3")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_phase4.yaml"
OUT_DIR = ROOT / "data" / "processed" / "phase5_extra" / "composed_prong3"
RESULTS_JSON = OUT_DIR / "results.json"
REPORT_MD = ROOT / "paper" / "composed_prong3_report.md"
FIGDIR = ROOT / "paper" / "figures"

# Pre-specified composed variants — fixed before the simulator runs. The η
# values are the frontier's recommended knee (0.15), the unbounded endpoint
# (1.0), and the tail-only flag quantile (0.10). No other cells are run; this
# is a confirmation of the frontier's recommendation at the system level, not
# a second search.
ETA_KNEE = 0.15
ETA_FULL = 1.0
TAIL_Q = 0.10
FLOOR, SCALE = 1.0, 0.5

N_REPLICATES = 100   # matches Phase 4
N_BOOTSTRAP = 1000   # matches Phase 4

# Correctness anchor (Phase-5 known value for reflective_floor vs additive
# on audience_reach; see paper/operating_curve_report.md).
ANCHOR_TARGET = -37.8
ANCHOR_TOL = 3.0

REGIME_ORDER = (
    "additive",
    "reflective_floor",
    "composed_eta015",
    "composed_eta100",
    "composed_tail010",
)
REGIME_LABELS = {
    "additive": "additive (baseline)",
    "reflective_floor": "reflective floor (η=0)",
    "composed_eta015": f"composed η={ETA_KNEE:g}",
    "composed_eta100": f"composed η={ETA_FULL:g}",
    "composed_tail010": f"composed tail q={TAIL_Q:g}",
}


def build_seed_gate(seeds: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Content gate g for every simulation seed.

    Classifier fit on all 400 claim-labeled tweets (allowed: the seeds carry
    no claim labels, so nothing leaks), applied to seed ``rawContent``;
    g = quality_gate(1 − P(misinfo)) with label-free thresholds (median/std)
    from the seed distribution itself.
    """
    claims = load_claims()
    text400 = claims["rawContent"].fillna("").to_numpy()
    y400 = claims["claim_misinfo"].to_numpy().astype(int)
    clf = build_text_classifier()
    clf.fit(text400, y400)

    seed_text = seeds["rawContent"].fillna("").astype(str)
    n_empty = int((seed_text.str.len() == 0).sum())
    p_cred = 1.0 - clf.predict_proba(seed_text.to_numpy())[:, 1]
    g = quality_gate(p_cred)

    meta = {
        "n_claim_tweets": int(len(claims)),
        "n_claim_positives": int(y400.sum()),
        "n_seeds": int(len(seeds)),
        "n_seeds_empty_text": n_empty,
        "gate_mean_low": float(g[seeds["credibility_label"].to_numpy() == LOW].mean()),
        "gate_mean_high": float(g[seeds["credibility_label"].to_numpy() == HIGH].mean()),
    }
    logger.info(
        "seed gate: n=%d (empty text: %d) — mean g | low=%0.4f high=%0.4f",
        meta["n_seeds"], n_empty, meta["gate_mean_low"], meta["gate_mean_high"],
    )
    return g, meta


def reach_table(pc: pd.DataFrame) -> pd.DataFrame:
    """Per-regime mean audience_reach (= total_exposures) by credibility class,
    plus reductions vs the additive baseline — the operating-curve quantities."""
    m = (
        pc.groupby(["regime", "credibility_label"])["total_exposures"]
        .mean()
        .unstack()
    )
    base_low = m.loc["additive", LOW]
    base_high = m.loc["additive", HIGH]
    out = pd.DataFrame({
        "AR_low": m[LOW],
        "AR_high": m[HIGH],
    })
    out["low_reduction_pct"] = 100.0 * (base_low - out["AR_low"]) / base_low
    out["high_reduction_pct"] = 100.0 * (base_high - out["AR_high"]) / base_high
    out["selectivity"] = out["low_reduction_pct"] - out["high_reduction_pct"]
    out["cred_gap"] = out["AR_low"] - out["AR_high"]
    return out.loc[list(REGIME_ORDER)]


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(CONFIG)
    seed = int(cfg["random_seed"])
    assert seed == SEED, "config seed must match the project seed (1337)"

    # ---- Phase-4 setup, verbatim ---------------------------------------
    df = pd.read_parquet(ROOT / cfg["data"]["part_parquet"])
    df = add_labels(
        df,
        iffy_path=ROOT / cfg["labels"]["iffy_path"],
        mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
    )
    model, scaler, _meta = load_ranker(ROOT / cfg["ranker"]["load_dir"])
    user_pool = sample_users(df, n=int(cfg["simulation"]["n_users"]), seed=seed)
    seeds = _stratified_label_seeds(
        df, int(cfg["simulation"]["n_per_label"]), seed
    )
    diurnal = fit_diurnal_weights(df)

    seed_probs = _seed_probs_internal(seeds, model, scaler)
    calib = calibrate_baseline_exposures(
        observed_df=df,
        seed_probs=seed_probs,
        scoring_config=ScoringConfig(regime="additive", weights=dict(PUBLISHED_WEIGHTS)),
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

    # ---- content gate ---------------------------------------------------
    g, gate_meta = build_seed_gate(seeds)

    # ---- regimes + per-seed multipliers ----------------------------------
    base_w = dict(PUBLISHED_WEIGHTS)
    rf = ScoringConfig(
        regime="reflective_floor", weights=base_w, floor=FLOOR, floor_scale=SCALE
    )
    scoring_configs = {
        "additive": ScoringConfig(regime="additive", weights=base_w),
        "reflective_floor": rf,
        "composed_eta015": rf,
        "composed_eta100": rf,
        "composed_tail010": rf,
    }
    multipliers = {
        "composed_eta015": tempered_gate(g, ETA_KNEE),
        "composed_eta100": tempered_gate(g, ETA_FULL),
        "composed_tail010": tail_gate(g, TAIL_Q),
    }

    ablation = run_ablation(
        seeds=seeds,
        user_pool=user_pool,
        ranker=model,
        scaler=scaler,
        scoring_configs=scoring_configs,
        sim_config=sim_cfg,
        n_replicates=N_REPLICATES,
        diurnal_weights=diurnal,
        score_multipliers=multipliers,
    )
    pc = ablation.per_cascade.copy()
    pc["credibility_label"] = (
        seeds["credibility_label"].iloc[pc["cascade_id"].to_numpy()].to_numpy()
    )
    pc.to_parquet(OUT_DIR / "per_cascade.parquet", index=False)

    # ---- Stage-2 contrasts: vs additive and vs reflective_floor ----------
    regimes = tuple(REGIME_ORDER)
    stage2_add = run_stage2(
        per_cascade=pc, per_bin=ablation.per_bin, regimes=regimes,
        baseline_regime="additive", n_bootstrap=N_BOOTSTRAP, bootstrap_seed=seed,
    )
    stage2_rf = run_stage2(
        per_cascade=pc, per_bin=ablation.per_bin, regimes=regimes,
        baseline_regime="reflective_floor", n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=seed,
    )

    # ---- anchor check -----------------------------------------------------
    anchor = stage2_add["bootstrap_contrasts"]["audience_reach"]["reflective_floor"]
    anchor_ok = abs(anchor.diff_point - ANCHOR_TARGET) <= ANCHOR_TOL
    logger.info(
        "ANCHOR: reflective_floor audience_reach contrast = %0.3f (target %0.1f ± %0.1f) → %s",
        anchor.diff_point, ANCHOR_TARGET, ANCHOR_TOL, "OK" if anchor_ok else "FAIL",
    )
    if not anchor_ok:
        logger.error("Anchor failed — results below are suspect. Debug before use.")

    # ---- operating-curve-style table --------------------------------------
    reach = reach_table(pc)

    results: dict[str, Any] = {
        "meta": {
            "seed": seed,
            "n_replicates": N_REPLICATES,
            "n_bootstrap": N_BOOTSTRAP,
            "eta_knee": ETA_KNEE,
            "eta_full": ETA_FULL,
            "tail_q": TAIL_Q,
            "floor": FLOOR,
            "floor_scale": SCALE,
            "calibration": asdict(calib),
            "gate": gate_meta,
            "anchor": {
                "value": float(anchor.diff_point),
                "target": ANCHOR_TARGET,
                "tol": ANCHOR_TOL,
                "ok": bool(anchor_ok),
            },
            "discipline": (
                "Variants fixed a priori from the frontier recommendation "
                "(knee η=0.15, endpoint η=1.0, tail q=0.10); single "
                "confirmation run, no system-level grid search. Seed-level "
                "p_cred is unvalidated against claim labels (same caveat as "
                "the frontier's 5,000-seed step)."
            ),
        },
        "reach_table": reach.reset_index().to_dict(orient="records"),
        "contrasts_vs_additive": {
            metric: {r: asdict(c) for r, c in regime_c.items()}
            for metric, regime_c in stage2_add["bootstrap_contrasts"].items()
        },
        "contrasts_vs_reflective_floor": {
            metric: {r: asdict(c) for r, c in regime_c.items()}
            for metric, regime_c in stage2_rf["bootstrap_contrasts"].items()
        },
        "gaps": {
            metric: {r: asdict(gp) for r, gp in regime_g.items()}
            for metric, regime_g in stage2_add["bootstrap_gaps"].items()
        },
    }
    RESULTS_JSON.write_text(
        json.dumps(results, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
        encoding="utf-8",
    )
    logger.info("wrote %s", RESULTS_JSON)

    make_figure(results)
    write_report(results)
    return results


def make_figure(results: dict[str, Any]) -> None:
    """Two panels: (a) audience-reach cred gap per regime with bootstrap CI;
    (b) low/high reach reduction vs additive (selectivity view)."""
    gaps = results["gaps"]["audience_reach"]
    reach = {r["regime"]: r for r in results["reach_table"]}
    regs = list(REGIME_ORDER)
    labels = [REGIME_LABELS[r] for r in regs]
    x = np.arange(len(regs))

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9))

    ax = axes[0]
    pts = [gaps[r]["gap_point"] for r in regs]
    lo = [gaps[r]["gap_point"] - gaps[r]["gap_ci_lo"] for r in regs]
    hi = [gaps[r]["gap_ci_hi"] - gaps[r]["gap_point"] for r in regs]
    colors = [CB["grey"], CB["blue"], CB["green"], CB["vermilion"], CB["purple"]]
    ax.bar(x, pts, yerr=[lo, hi], capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("audience-reach gap (low − high)", fontsize=8)
    ax.set_title("(a) credibility gap by regime", fontsize=8.5)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    width = 0.38
    lows = [reach[r]["low_reduction_pct"] for r in regs]
    highs = [reach[r]["high_reduction_pct"] for r in regs]
    ax.bar(x - width / 2, lows, width, color=CB["vermilion"], edgecolor="black",
           linewidth=0.5, label="low-credibility reach reduction")
    ax.bar(x + width / 2, highs, width, color=CB["sky"], edgecolor="black",
           linewidth=0.5, label="high-credibility reach reduction")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("reach reduction vs additive (%)", fontsize=8)
    ax.set_title("(b) benefit vs collateral cost", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_composed_prong3.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig_composed_prong3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote figure fig_composed_prong3.{pdf,png}")


def write_report(results: dict[str, Any]) -> None:
    meta = results["meta"]
    L: list[str] = []
    L.append("# Composed-score Prong 3: the bounded content gate in the full simulator\n")
    L.append(f"Run: {pd.Timestamp.utcnow().isoformat()}  ")
    L.append(f"Seed: {meta['seed']} | n_replicates={meta['n_replicates']} | "
             f"bootstrap n={meta['n_bootstrap']}\n")
    L.append(
        "**Question.** The dual-lever frontier established at the rank level that "
        f"S = S_rf · g^η passes the claim-level check at η={meta['eta_knee']:g} while "
        "retaining Prong-1 targeting. Does the *system-level* architectural result "
        "(Prong 3: gap closure in the calibrated cascade simulator) survive the "
        "composition?\n"
    )
    a = meta["anchor"]
    L.append(
        f"**Anchor.** reflective_floor audience_reach contrast vs additive = "
        f"{a['value']:.3f} (known Phase-5 value {a['target']:.1f} ± {a['tol']:.1f}) → "
        f"{'**OK**' if a['ok'] else '**FAIL — do not use these results**'}.\n"
    )
    g = meta["gate"]
    L.append(
        f"**Gate.** Classifier fit on all {g['n_claim_tweets']} claim-labeled tweets "
        f"({g['n_claim_positives']} positives), applied to {g['n_seeds']} seed texts "
        f"({g['n_seeds_empty_text']} empty). Mean g: low-cred {g['gate_mean_low']:.4f} "
        f"vs high-cred {g['gate_mean_high']:.4f}. Seed-level p_cred is unvalidated "
        "against claim labels — same caveat as the frontier's 5,000-seed step.\n"
    )

    L.append("## Per-class audience reach (operating-curve view)\n")
    L.append("| regime | AR_low | AR_high | low_reduction% | high_reduction% | selectivity | cred_gap |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in results["reach_table"]:
        L.append(
            f"| `{row['regime']}` | {row['AR_low']:.2f} | {row['AR_high']:.2f} | "
            f"{row['low_reduction_pct']:+.2f} | {row['high_reduction_pct']:+.2f} | "
            f"{row['selectivity']:+.2f} | {row['cred_gap']:+.2f} |"
        )
    L.append("")

    for metric in ("audience_reach", "cascade_size", "time_to_peak"):
        L.append(f"## Contrasts — {metric}\n")
        L.append("Bootstrap (cascade-level, paired) CIs. Negative = closes the "
                 "low-minus-high gap relative to the baseline named in the column.\n")
        L.append("| regime | vs additive | 95% CI | vs reflective_floor | 95% CI |")
        L.append("|---|---:|---|---:|---|")
        va = results["contrasts_vs_additive"][metric]
        vr = results["contrasts_vs_reflective_floor"][metric]
        for r in REGIME_ORDER:
            ca = va.get(r)
            cr = vr.get(r)
            sa = (f"{ca['diff_point']:+.3f}", f"[{ca['diff_ci_lo']:+.3f}, {ca['diff_ci_hi']:+.3f}]") if ca else ("—", "")
            sr = (f"{cr['diff_point']:+.3f}", f"[{cr['diff_ci_lo']:+.3f}, {cr['diff_ci_hi']:+.3f}]") if cr else ("—", "")
            L.append(f"| `{r}` | {sa[0]} | {sa[1]} | {sr[0]} | {sr[1]} |")
        L.append("")

    L.append("## Reading\n")
    ar = results["contrasts_vs_additive"]["audience_reach"]
    rf_pt = ar["reflective_floor"]["diff_point"]
    knee = ar["composed_eta015"]
    full = ar["composed_eta100"]
    tail = ar["composed_tail010"]
    knee_vs_rf = results["contrasts_vs_reflective_floor"]["audience_reach"]["composed_eta015"]
    L.append(
        f"- reflective_floor alone closes the audience-reach gap by {rf_pt:+.2f}. "
        f"Composed η={meta['eta_knee']:g}: {knee['diff_point']:+.2f} "
        f"[{knee['diff_ci_lo']:+.2f}, {knee['diff_ci_hi']:+.2f}]; "
        f"unbounded η=1: {full['diff_point']:+.2f}; tail-only: {tail['diff_point']:+.2f}."
    )
    L.append(
        f"- Composition increment (η={meta['eta_knee']:g} vs reflective_floor alone): "
        f"{knee_vs_rf['diff_point']:+.3f} "
        f"[{knee_vs_rf['diff_ci_lo']:+.3f}, {knee_vs_rf['diff_ci_hi']:+.3f}]. "
        "A CI excluding zero in the negative direction means the bounded gate "
        "*adds* gap closure on top of the architectural fix; a CI overlapping "
        "zero means it leaves the architectural Prong-3 result intact (the "
        "claim the manuscript needs is the latter — non-interference)."
    )
    L.append(f"\n**Discipline.** {meta['discipline']}\n")
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", REPORT_MD)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run()


if __name__ == "__main__":
    main()

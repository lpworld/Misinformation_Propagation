"""Truncation-sensitivity completion — referee item E3.

The manuscript applies a body-truncation sensitivity check (cap the
per-cascade replicate-mean ``cascade_size`` at quantiles of the *observed*
aggregate-engagement distribution, winsorize, paired bootstrap on the capped
values — ``analysis/run_truncation_sensitivity.py``) to the headline
``cascade_size`` contrast (ablated vs additive) only. A referee notes the
check was never applied to:

(a) the **reflective_floor** regime's ``cascade_size`` contrast vs additive —
    the strongest variant and the paper's prescription; and
(b) **Sweep 7** (enlarged training corpus: parts_1_2 and parts_1_to_5) —
    its ``cascade_size`` contrasts (ablated vs additive).

This script closes both gaps using the **identical** procedure and truncation
grid: it imports the winsorize/bootstrap machinery directly from
``analysis.run_truncation_sensitivity`` (no re-implementation), uses the same
pre-specified quantiles {p50, p75, p90, p95, p99} of the observed
``replyCount + retweetCount + quoteCount`` per-conversation distribution
(part_1), truncates AFTER replicate-averaging, and re-seeds the bootstrap per
quantile as ``default_rng(seed + int(q))``.

Data sources (no new experimental simulation):

* **reflective_floor** — cached per-cascade frame
  ``data/processed/phase5_extra/composed_prong3/per_cascade.parquet``
  (regimes ``additive`` + ``reflective_floor`` among others; same Phase-4
  seeds/calibration/replicate structure; anchor-checked at write time).
* **Sweep 7** — the original RB3 run (``analysis/run_phase5_extra.py``)
  persisted only the aggregate contrasts, not the per-cascade frames. The
  frames are **deterministically re-materialized** here: same cached rankers
  (``phase5_extra/ranker_parts_1_2``, ``ranker_parts_1_to_5``), same eval
  data (part_1), same seeds (stratified, seed=1337), same calibration, and
  the simulator's per-(regime, replicate) RNG is ``default_rng(1337 + rep)``
  — independent of which other regimes run. This is a regeneration of an
  unpersisted intermediate, not a new experiment; fidelity is VERIFIED by
  requiring the untruncated cascade_size and audience_reach contrast point
  estimates to match the cached Sweep-7 JSON to < 1e-6 before any truncation
  cell is reported. Ranker inference is attempted on CPU first and then CUDA
  (the original RB3 run inferred on CUDA for rankers it trained in-process
  and on CPU for checkpoint-loaded ones; float32 device rounding shifts the
  contrasts by ~1e-5, which the gate distinguishes). Frames are cached under
  the output dir only after passing the gate, so reruns are read-only.

Pre-specification: the quantile grid is fixed (identical to the headline
check); whatever each cell shows — survives, weakens, or goes null — is
reported. The headline's own pattern (null under body-truncation, conceded
in the paper) is the reporting standard here too.

Run (Windows, repo root, project venv)::

    .venv\\Scripts\\python.exe -m analysis.run_truncation_completion

Outputs:

* ``data/processed/phase5_extra/truncation_completion/results.json``
* ``data/processed/phase5_extra/truncation_completion/sweep7_<run>_per_cascade.parquet``
* ``paper/truncation_completion_report.md``
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

from analysis.hypothesis import HIGH, LOW, cascade_bootstrap_contrast
from analysis.labeling import add_labels
from analysis.run_phase4 import _stratified_label_seeds
from analysis.run_phase5 import _build_scoring_set
from analysis.run_truncation_sensitivity import (
    TruncatedContrast,
    _cascade_size_expr,
    _interpret_shift,
    _sign_marker,
    replicate_mean_wide,
    truncated_bootstrap_contrast,
    winsorize_wide,
)
from analysis.validation import observed_aggregate_engagement
from ranker.training import load_ranker
from simulation.ablation import run_ablation
from simulation.calibration import calibrate_baseline_exposures
from simulation.cascade import SimConfig, fit_diurnal_weights
from simulation.cascade import _seed_probs as _seed_probs_internal

logger = logging.getLogger("truncation_completion")

ROOT = Path(__file__).resolve().parents[1]
SEED = 1337
QUANTILES = (50.0, 75.0, 90.0, 95.0, 99.0)   # identical to the headline check
N_BOOTSTRAP = 1000                            # identical to the headline check

OBSERVED_PARQUET = ROOT / "data" / "processed" / "part_1.parquet"
COMPOSED_PRONG3_PC = (
    ROOT / "data" / "processed" / "phase5_extra" / "composed_prong3" / "per_cascade.parquet"
)
PHASE5_EXTRA_METRICS = (
    ROOT / "data" / "processed" / "phase5_extra" / "phase5_extra_metrics.json"
)
PHASE5_EXTRA_CONFIG = ROOT / "configs" / "experiment_phase5_extra.yaml"
OUT_DIR = ROOT / "data" / "processed" / "phase5_extra" / "truncation_completion"
REPORT_MD = ROOT / "paper" / "truncation_completion_report.md"

# Fidelity tolerance for the deterministic Sweep-7 re-materialization: the
# untruncated contrast point estimates must match the cached RB3 JSON values.
SWEEP7_MATCH_TOL = 1e-6


@contextmanager
def _timed(label: str, timings: dict[str, float]) -> Iterator[None]:
    t0 = time.time()
    logger.info("- %s", label)
    try:
        yield
    finally:
        dt = time.time() - t0
        timings[label] = dt
        logger.info("+ %s - %.2fs", label, dt)


# ---- shared truncation runner -------------------------------------------

def truncation_cells(
    labeled_pc: pd.DataFrame,
    *,
    regime_a: str,
    regime_b: str,
    obs_cascade_size: np.ndarray,
    seed: int = SEED,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    """Run the identical truncation procedure on one (regime_b vs regime_a) pair.

    ``labeled_pc`` must contain exactly the per-cascade long frame (one row
    per regime x replicate x cascade) restricted to LOW/HIGH labels, with
    both regimes present. Returns baseline (untruncated) contrast plus one
    cell per pre-specified quantile.
    """
    regimes = (regime_a, regime_b)
    wide = replicate_mean_wide(
        labeled_pc, metric_col_expr=_cascade_size_expr, regimes=regimes,
    )
    rng = np.random.default_rng(seed)
    baseline = cascade_bootstrap_contrast(
        wide, regime_a=regime_a, regime_b=regime_b,
        metric_name="cascade_size", n_bootstrap=n_bootstrap, rng=rng,
    )

    cells: dict[float, TruncatedContrast] = {}
    for q in QUANTILES:
        cap = float(np.percentile(obs_cascade_size, q))
        wide_clipped, clip_counts = winsorize_wide(wide, regimes=regimes, cap=cap)
        rng_q = np.random.default_rng(seed + int(q))
        cells[q] = truncated_bootstrap_contrast(
            wide_clipped,
            quantile=q, cap_value=cap,
            regime_a=regime_a, regime_b=regime_b,
            n_bootstrap=n_bootstrap, rng=rng_q,
            clip_counts=clip_counts,
        )
        logger.info(
            "  %s vs %s | q=p%.0f cap=%.2f -> diff=%+.3f CI=[%+.3f, %+.3f] %s",
            regime_b, regime_a, q, cap,
            cells[q].diff_point, cells[q].diff_ci_lo, cells[q].diff_ci_hi,
            "(< 0)" if cells[q].diff_ci_hi < 0 else "(overlaps 0)",
        )

    return {
        "regime_a": regime_a,
        "regime_b": regime_b,
        "baseline_untruncated": {
            "diff_point": baseline.diff_point,
            "diff_ci_lo": baseline.diff_ci_lo,
            "diff_ci_hi": baseline.diff_ci_hi,
            "n_low": baseline.n_low,
            "n_high": baseline.n_high,
        },
        "truncated": {str(q): asdict(tc) for q, tc in cells.items()},
    }


# ---- Sweep-7 per-cascade re-materialization ------------------------------

def _sweep7_per_cascade(
    *,
    run_name: str,
    ranker_dir: Path,
    df: pd.DataFrame,
    seeds: pd.DataFrame,
    diurnal: np.ndarray,
    cfg: dict[str, Any],
    device: str,
) -> pd.DataFrame:
    """Deterministically regenerate the Sweep-7 per-cascade frame for one run.

    Mirrors ``analysis.run_phase5._run_variant`` (same calibration recipe,
    SimConfig, replicate seeds) but keeps the per-cascade frame. Only the
    ``additive`` and ``ablated`` regimes are simulated — the simulator RNG is
    seeded per (replicate) independently of the regime set, so omitting
    ``additive_retuned`` leaves these two regimes bit-identical to the
    original RB3 run.

    ``device`` selects where ranker inference runs. The original RB3 run
    inferred on whatever device the model happened to be on: CPU for rankers
    loaded from a checkpoint, CUDA for rankers trained in-process. Float32
    inference differs between the two by ~2e-7 in the head probabilities,
    which shifts contrast point estimates by ~1e-5 — enough to fail the
    fidelity gate. The driver tries CPU first, then CUDA.
    """
    import torch

    model, scaler, _meta = load_ranker(ranker_dir)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA fallback requested but CUDA is unavailable")
        model = model.to("cuda")
    s = cfg["simulation"]

    seed_probs = _seed_probs_internal(seeds, model, scaler)
    sc_full = _build_scoring_set(cfg)
    calib = calibrate_baseline_exposures(
        observed_df=df,
        seed_probs=seed_probs,
        scoring_config=sc_full["additive"],
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
        seed=SEED,
    )
    scoring_configs = {k: sc_full[k] for k in ("additive", "ablated")}
    ablation = run_ablation(
        seeds=seeds, user_pool=pd.DataFrame(), ranker=model, scaler=scaler,
        scoring_configs=scoring_configs, sim_config=sim_cfg,
        n_replicates=int(s["n_replicates"]), diurnal_weights=diurnal,
    )
    pc = ablation.per_cascade.copy()
    pc["credibility_label"] = (
        seeds["credibility_label"].iloc[pc["cascade_id"].to_numpy()].to_numpy()
    )
    logger.info("regenerated %s frame on %s (%d rows)", run_name, device, len(pc))
    return pc


def _point_contrast(
    labeled_pc: pd.DataFrame, *, regime_a: str, regime_b: str, metric_expr,
) -> float:
    """Deterministic (no-bootstrap) gap-of-gaps point estimate."""
    wide = replicate_mean_wide(
        labeled_pc, metric_col_expr=metric_expr, regimes=(regime_a, regime_b),
    )
    low = wide["credibility_label"] == LOW
    high = wide["credibility_label"] == HIGH
    gap_a = wide.loc[low, regime_a].mean() - wide.loc[high, regime_a].mean()
    gap_b = wide.loc[low, regime_b].mean() - wide.loc[high, regime_b].mean()
    return float(gap_b - gap_a)


# ---- driver ---------------------------------------------------------------

def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    cfg = yaml.safe_load(PHASE5_EXTRA_CONFIG.read_text(encoding="utf-8"))
    assert int(cfg["random_seed"]) == SEED

    # --- observed cap distribution (identical source to the headline check)
    with _timed("observed cap distribution (part_1)", timings):
        obs_df = pd.read_parquet(OBSERVED_PARQUET, columns=[
            "conversationId", "replyCount", "retweetCount", "quoteCount",
        ])
        obs_cascade_size = observed_aggregate_engagement(obs_df)
        obs_summary = {
            "n": int(obs_cascade_size.size),
            **{f"p{int(q)}": float(np.percentile(obs_cascade_size, q)) for q in QUANTILES},
        }
        logger.info("observed caps: %s", obs_summary)

    results: dict[str, Any] = {
        "meta": {
            "seed": SEED,
            "quantiles": list(QUANTILES),
            "n_bootstrap": N_BOOTSTRAP,
            "mode": "winsorize (after replicate-averaging)",
            "procedure_source": "analysis/run_truncation_sensitivity.py (functions imported, not re-implemented)",
            "observed_cap_source": "part_1 replyCount+retweetCount+quoteCount per conversationId",
            "observed_summary": obs_summary,
        },
    }

    # --- (a) reflective_floor vs additive --------------------------------
    with _timed("(a) reflective_floor vs additive (composed_prong3 cache)", timings):
        pc_rf = pd.read_parquet(COMPOSED_PRONG3_PC)
        pc_rf = pc_rf[
            pc_rf["regime"].isin(["additive", "reflective_floor"])
            & pc_rf["credibility_label"].isin([LOW, HIGH])
        ].copy()
        results["reflective_floor"] = truncation_cells(
            pc_rf,
            regime_a="additive", regime_b="reflective_floor",
            obs_cascade_size=obs_cascade_size,
        )

    # --- (b) Sweep 7: enlarged training corpus ---------------------------
    cached_rb3 = {
        c["run"]: c
        for c in json.loads(PHASE5_EXTRA_METRICS.read_text(encoding="utf-8"))
        ["results"]["enlarged_training"]
        if c.get("status") == "ok"
    }
    with _timed("load eval data (part_1 + labels, seeds, diurnal)", timings):
        df = pd.read_parquet(ROOT / cfg["data"]["primary_part"])
        df = add_labels(
            df,
            iffy_path=ROOT / cfg["labels"]["iffy_path"],
            mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
        )
        seeds = _stratified_label_seeds(df, int(cfg["simulation"]["n_per_label"]), SEED)
        diurnal = fit_diurnal_weights(df)

    results["sweep7"] = {}
    for run_spec in cfg["sweeps"]["enlarged_training"]["runs"]:
        name = str(run_spec["name"])
        if name not in cached_rb3:
            logger.warning("skipping Sweep-7 run %s (not in cached RB3 results)", name)
            continue
        with _timed(f"(b) Sweep 7 [{name}]", timings):
            cs_cached = float(cached_rb3[name]["cascade_size_ablated"]["diff_point"])
            ar_cached = float(cached_rb3[name]["audience_reach_ablated"]["diff_point"])
            cache_path = OUT_DIR / f"sweep7_{name}_per_cascade.parquet"

            labeled7: pd.DataFrame | None = None
            fidelity: dict[str, Any] | None = None
            if cache_path.exists():
                candidates: list[tuple[str, pd.DataFrame]] = [
                    ("cache", pd.read_parquet(cache_path)),
                ]
                logger.info("loaded cached regenerated frame: %s", cache_path.name)
            else:
                candidates = []

            # Try CPU inference first, then CUDA (original RB3 inferred on
            # CUDA for rankers trained in-process; float32 device rounding
            # shifts contrasts by ~1e-5).
            attempt_iter = iter(["cpu", "cuda"])
            while True:
                if candidates:
                    provenance, pc7 = candidates.pop(0)
                else:
                    try:
                        device = next(attempt_iter)
                    except StopIteration:
                        break
                    provenance = device
                    pc7 = _sweep7_per_cascade(
                        run_name=name,
                        ranker_dir=ROOT / cfg["artifacts"]["out_dir"] / run_spec["out_subdir"],
                        df=df, seeds=seeds, diurnal=diurnal, cfg=cfg,
                        device=device,
                    )
                cand = pc7[pc7["credibility_label"].isin([LOW, HIGH])].copy()
                cs_point = _point_contrast(
                    cand, regime_a="additive", regime_b="ablated",
                    metric_expr=_cascade_size_expr,
                )
                ar_point = _point_contrast(
                    cand, regime_a="additive", regime_b="ablated",
                    metric_expr=lambda f: f["total_exposures"].to_numpy(np.float64),
                )
                cs_ok = abs(cs_point - cs_cached) < SWEEP7_MATCH_TOL
                ar_ok = abs(ar_point - ar_cached) < SWEEP7_MATCH_TOL
                logger.info(
                    "fidelity [%s/%s]: CS %.9f vs cached %.9f (%s) | AR %.9f vs cached %.9f (%s)",
                    name, provenance, cs_point, cs_cached, "OK" if cs_ok else "FAIL",
                    ar_point, ar_cached, "OK" if ar_ok else "FAIL",
                )
                if cs_ok and ar_ok:
                    labeled7 = cand
                    fidelity = {
                        "cascade_size_point": cs_point,
                        "cascade_size_cached": cs_cached,
                        "audience_reach_point": ar_point,
                        "audience_reach_cached": ar_cached,
                        "tolerance": SWEEP7_MATCH_TOL,
                        "inference_device": provenance,
                        "ok": True,
                    }
                    if provenance != "cache":
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        pc7.to_parquet(cache_path, index=False)
                        logger.info("cached validated frame -> %s", cache_path.name)
                    break

            if labeled7 is None or fidelity is None:
                raise RuntimeError(
                    f"Sweep-7 regeneration for {name!r} does not match cached RB3 "
                    f"contrasts on either inference device. The frame is NOT the "
                    f"original run — stop."
                )

            entry = truncation_cells(
                labeled7,
                regime_a="additive", regime_b="ablated",
                obs_cascade_size=obs_cascade_size,
            )
            entry["fidelity"] = fidelity
            entry["n_train"] = int(cached_rb3[name]["n_train"])
            results["sweep7"][name] = entry

    results["timings_seconds"] = timings
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    logger.info("wrote %s", OUT_DIR / "results.json")

    write_report(results)
    return results


# ---- report ---------------------------------------------------------------

def _cell_rows(entry: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    b = entry["baseline_untruncated"]
    rows.append(
        f"| *(none)* | — | {b['n_low']} (—) | {b['n_high']} (—) | "
        f"{b['diff_point']:+.3f} | [{b['diff_ci_lo']:+.3f}, {b['diff_ci_hi']:+.3f}] | "
        f"{_sign_marker(b['diff_ci_lo'], b['diff_ci_hi'])} | — |"
    )
    for q in sorted(entry["truncated"], key=float):
        t = entry["truncated"][q]
        rows.append(
            f"| p{float(q):.0f} | {t['cap_value']:.2f} | "
            f"{t['n_low_kept']} ({t['n_low_clipped']}) | "
            f"{t['n_high_kept']} ({t['n_high_clipped']}) | "
            f"{t['diff_point']:+.3f} | "
            f"[{t['diff_ci_lo']:+.3f}, {t['diff_ci_hi']:+.3f}] | "
            f"{_sign_marker(t['diff_ci_lo'], t['diff_ci_hi'])} | "
            f"{_interpret_shift(b['diff_ci_lo'], b['diff_ci_hi'], t['diff_ci_lo'], t['diff_ci_hi'])} |"
        )
    return rows


def _survival(entry: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Quantiles where the truncated CI is strictly < 0, and where it overlaps 0."""
    survive, null = [], []
    for q in sorted(entry["truncated"], key=float):
        t = entry["truncated"][q]
        (survive if t["diff_ci_hi"] < 0 else null).append(float(q))
    return survive, null


def write_report(results: dict[str, Any]) -> None:
    meta = results["meta"]
    L: list[str] = []
    L.append("# Truncation-sensitivity completion (referee item E3)\n")
    L.append(f"Run: {pd.Timestamp.utcnow().isoformat()}  ")
    L.append(
        f"Seed: {meta['seed']} | quantiles: {meta['quantiles']} | "
        f"bootstrap n={meta['n_bootstrap']} | mode: {meta['mode']}\n"
    )
    L.append(
        "**Question.** The body-truncation sensitivity check applied to the "
        "headline `cascade_size` contrast (ablated vs additive; result: null "
        "under body-truncation, conceded in the paper) was never applied to "
        "(a) the `reflective_floor` regime — the strongest variant and the "
        "paper's prescription — or (b) Sweep 7 (enlarged training corpus). "
        "This run applies the **identical** procedure (functions imported from "
        "`analysis/run_truncation_sensitivity.py`; same observed-quantile cap "
        "grid, same winsorize-after-replicate-mean ordering, same per-quantile "
        "bootstrap seeding) to both.\n"
    )
    obs = meta["observed_summary"]
    L.append(
        "Cap source: observed per-conversation `replyCount + retweetCount + "
        f"quoteCount` on part_1 (n={obs['n']:,}; caps "
        + ", ".join(f"p{int(q)}={obs[f'p{int(q)}']:.0f}" for q in meta["quantiles"])
        + ").\n"
    )

    header = (
        "| quantile | cap | n_low (clipped) | n_high (clipped) | "
        "diff_point | 95% CI | sign | vs. baseline |"
    )
    sep = "|---|---|---|---|---|---|---|---|"

    # (a)
    rf = results["reflective_floor"]
    L.append("## (a) `reflective_floor` − `additive`, cascade_size\n")
    L.append(
        "Source: cached per-cascade frame from the composed-Prong-3 run "
        "(`data/processed/phase5_extra/composed_prong3/per_cascade.parquet`; "
        "same Phase-4 seeds, calibration, and 100-replicate structure).\n"
    )
    L.append(header)
    L.append(sep)
    L.extend(_cell_rows(rf))
    s, n = _survival(rf)
    L.append(
        f"\n**Survival:** strictly < 0 at quantiles {s or 'none'}; "
        f"overlaps 0 at {n or 'none'}.\n"
    )

    # (b)
    L.append("## (b) Sweep 7 (enlarged training corpus): `ablated` − `additive`, cascade_size\n")
    L.append(
        "Per-cascade frames deterministically re-materialized from the cached "
        "Sweep-7 rankers (same seeds, calibration recipe, and per-replicate "
        "RNG as the original RB3 run; the harness seeds each replicate "
        "independently of the regime set). **Fidelity gate:** the untruncated "
        "cascade_size and audience_reach contrast point estimates must match "
        "the cached Sweep-7 JSON to < 1e-6 — both runs passed (exact values "
        "in `results.json`); otherwise this script aborts.\n"
    )
    for name, entry in results["sweep7"].items():
        f = entry["fidelity"]
        L.append(f"### {name} (n_train = {entry['n_train']:,})\n")
        L.append(
            f"Fidelity: CS point {f['cascade_size_point']:+.6f} vs cached "
            f"{f['cascade_size_cached']:+.6f}; AR point "
            f"{f['audience_reach_point']:+.6f} vs cached "
            f"{f['audience_reach_cached']:+.6f} — **match** "
            f"(ranker inference device: {f['inference_device']}; the original "
            f"RB3 run inferred on CUDA for rankers trained in-process and CPU "
            f"for checkpoint-loaded ones).\n"
        )
        L.append(header)
        L.append(sep)
        L.extend(_cell_rows(entry))
        s, n = _survival(entry)
        L.append(
            f"\n**Survival:** strictly < 0 at quantiles {s or 'none'}; "
            f"overlaps 0 at {n or 'none'}.\n"
        )

    # Reading
    L.append("## Reading\n")
    L.append(
        "*Degenerate cell:* the p50 cap is 0 (the median observed "
        "conversation has zero aggregate engagement), so every value is "
        "winsorized to 0 and the p50 contrast is identically zero by "
        "construction in **all** configurations — including the headline's "
        "own truncation table. p50 is reported for grid completeness but "
        "carries no information; the informative body cells are p75–p95.\n"
    )
    L.append(
        "*Headline reference (from the original "
        "`paper/phase4_truncation_sensitivity.md`):* the ablated-vs-additive "
        "Phase-4 contrast was strictly < 0 only at p95 "
        "(−0.124 [−0.212, −0.033]) and overlapped zero at p75, p90, and "
        "p99 — the body-truncation null the paper concedes.\n"
    )
    rf_s, rf_n = _survival(rf)
    b = rf["baseline_untruncated"]
    L.append(
        f"- **reflective_floor:** untruncated contrast {b['diff_point']:+.2f} "
        f"[{b['diff_ci_lo']:+.2f}, {b['diff_ci_hi']:+.2f}]. Under truncation "
        f"it is strictly negative at {len(rf_s)}/{len(meta['quantiles'])} "
        f"quantiles ({rf_s or '—'}) and overlaps zero at {rf_n or '—'}."
    )
    for name, entry in results["sweep7"].items():
        s7, n7 = _survival(entry)
        b7 = entry["baseline_untruncated"]
        L.append(
            f"- **Sweep 7 / {name}:** untruncated contrast {b7['diff_point']:+.2f} "
            f"[{b7['diff_ci_lo']:+.2f}, {b7['diff_ci_hi']:+.2f}]. Strictly "
            f"negative at {len(s7)}/{len(meta['quantiles'])} truncation "
            f"quantiles ({s7 or '—'}); overlaps zero at {n7 or '—'}."
        )
    L.append(
        "\nThe reporting standard is the same as for the headline metric: a "
        "cell that overlaps zero under body-truncation means the cascade_size "
        "evidence in that configuration rests on the simulator's poorly "
        "calibrated tail and should not be cited as body-robust. Cells that "
        "remain strictly negative at p75–p90 are robust in the regime where "
        "the simulator is well calibrated. The `audience_reach` contrast — "
        "the architectural-claim signature — is unaffected by this analysis "
        "(it has no observed-side cap analog and was already strictly "
        "negative; see the original truncation report).\n"
    )

    L.append("## Step timings\n")
    L.append("```")
    for label, dt in results["timings_seconds"].items():
        L.append(f"  {label:55s} {dt:8.2f}s")
    L.append("```\n")

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

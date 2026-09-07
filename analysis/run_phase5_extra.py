"""Phase 5 extra — supplementary robustness sweeps (RB1, RB2, RB3a/b).

Adds three sensitivity sweeps beyond the original Phase 5 four:

* **RB1 — cross-product (partition × ranker_seed).** Runs every cell of
  4 partitions × 5 ranker seeds = 20 Phase-4 pipelines. Tests whether
  the architectural-claim signature survives interactions between the
  two largest pre-existing sensitivity axes (Sweep 2 × Sweep 3).
* **RB2 — user-sample-size sensitivity.** Varies ``n_users`` across
  {25K, 50K, 100K} on the default partition + baseline ranker. Tests
  whether the audience_reach contrast scales sensibly with simulator
  pool size.
* **RB3 — enlarged training corpus.** Trains MaskNet on the union of
  several USC parts (parts 1+2 minimally; parts 1-5 if available on
  disk), then reruns Phase 4 with the new ranker. Addresses the §E
  "stronger defense" question: does the architectural finding survive
  on a better-trained ranker?

All three sweeps share the Phase-4 backbone via
``analysis.run_phase5._run_variant``.

Run::

    uv run python -m analysis.run_phase5_extra \
        --config configs/experiment_phase5_extra.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from analysis.labeling import add_labels
from analysis.run_phase4 import _stratified_label_seeds
from analysis.run_phase5 import _build_scoring_set, _run_variant
from ranker.training import (
    TrainConfig,
    load_ranker,
    save_ranker,
    train_ranker,
)
from simulation.cascade import fit_diurnal_weights
from simulation.users import sample_users

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
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


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_or_train_seed_ranker(
    *,
    df: pd.DataFrame,
    seed_id: int,
    seeds_root: Path,
    baseline_load_dir: Path,
):
    """Resolve a ranker for a given seed_id.

    seed=1337 → the baseline ranker checkpoint (Phase 3 full).
    Other seeds → checkpoint under ``seeds_root/seed_{seed_id}/`` if exists,
    otherwise train and save it there.
    """
    if seed_id == 1337:
        return load_ranker(baseline_load_dir)
    seed_dir = seeds_root / f"seed_{seed_id}"
    if (seed_dir / "ranker.pt").exists():
        return load_ranker(seed_dir)
    logger.info("training MaskNet for seed=%d (will save to %s)", seed_id, seed_dir)
    tcfg = TrainConfig(
        batch_size=4096, lr=1e-3, epochs=5, val_frac=0.1,
        max_train_rows=None, device=_device(),
        seed=seed_id, model_kind="masknet",
    )
    result = train_ranker(df, train_config=tcfg)
    save_ranker(result, seed_dir)
    return result.model, result.scaler, {}


# ---- RB1: cross-product (partition × ranker_seed) -----------------------

def _run_cross_product(
    *,
    df: pd.DataFrame,
    user_pool: pd.DataFrame,
    seeds: pd.DataFrame,
    diurnal: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    seeds_root: Path,
    baseline_load_dir: Path,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cp = cfg["sweeps"]["cross_product"]
    partitions = cp["partitions"]
    ranker_seeds = cp["ranker_seeds"]
    for r_seed in ranker_seeds:
        model, scaler, _ = _load_or_train_seed_ranker(
            df=df, seed_id=int(r_seed),
            seeds_root=seeds_root, baseline_load_dir=baseline_load_dir,
        )
        for spec in partitions:
            slow = tuple(spec["slow"])
            fast = tuple(spec["fast"])
            sc_set = _build_scoring_set(cfg, partition_override=(slow, fast))
            out_v = _run_variant(
                df=df, user_pool=user_pool, ranker=model, scaler=scaler,
                diurnal=diurnal, seeds=seeds, cfg=cfg,
                scoring_configs=sc_set, seed=seed,
            )
            cell = {
                "ranker_seed": int(r_seed),
                "partition": spec["name"],
                "slow_heads": list(slow),
                "fast_heads": list(fast),
                "audience_reach_ablated": out_v["contrasts"]["audience_reach"]["ablated"],
                "cascade_size_ablated": out_v["contrasts"]["cascade_size"]["ablated"],
                "audience_reach_retuned": out_v["contrasts"]["audience_reach"]["additive_retuned"],
            }
            out.append(cell)
            logger.info(
                "  seed=%d partition=%s | AR_abl=%.2f [%.2f, %.2f]",
                int(r_seed), spec["name"],
                cell["audience_reach_ablated"]["diff_point"],
                cell["audience_reach_ablated"]["diff_ci_lo"],
                cell["audience_reach_ablated"]["diff_ci_hi"],
            )
    return out


# ---- RB2: user-sample-size sensitivity ---------------------------------

def _run_user_pool_sweep(
    *,
    df: pd.DataFrame,
    seeds: pd.DataFrame,
    diurnal: np.ndarray,
    model,
    scaler,
    cfg: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sizes = cfg["sweeps"]["user_sample_sizes"]
    for n in sizes:
        user_pool = sample_users(df, n=int(n), seed=seed)
        sc_set = _build_scoring_set(cfg)
        out_v = _run_variant(
            df=df, user_pool=user_pool, ranker=model, scaler=scaler,
            diurnal=diurnal, seeds=seeds, cfg=cfg,
            scoring_configs=sc_set, seed=seed,
        )
        cell = {
            "n_users": int(n),
            "audience_reach_ablated": out_v["contrasts"]["audience_reach"]["ablated"],
            "cascade_size_ablated": out_v["contrasts"]["cascade_size"]["ablated"],
            "audience_reach_retuned": out_v["contrasts"]["audience_reach"]["additive_retuned"],
        }
        out.append(cell)
        logger.info(
            "  n_users=%d | AR_abl=%.2f [%.2f, %.2f]",
            int(n),
            cell["audience_reach_ablated"]["diff_point"],
            cell["audience_reach_ablated"]["diff_ci_lo"],
            cell["audience_reach_ablated"]["diff_ci_hi"],
        )
    return out


# ---- RB3: enlarged training corpus -------------------------------------

def _run_enlarged_training(
    *,
    eval_df: pd.DataFrame,
    eval_seeds: pd.DataFrame,
    eval_user_pool: pd.DataFrame,
    eval_diurnal: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Train ranker on enlarged corpus, rerun Phase-4 against eval_df.

    ``eval_*`` are fixed across runs (we always evaluate on part_1) so the
    contrast is purely "did training data quality change the result?".
    """
    out: list[dict[str, Any]] = []
    runs = cfg["sweeps"]["enlarged_training"]["runs"]
    iffy = ROOT / cfg["labels"]["iffy_path"]
    mainstream = ROOT / cfg["labels"]["mainstream_path"]

    for run_spec in runs:
        name = run_spec["name"]
        train_paths = [ROOT / p for p in run_spec["train_parquets"]]
        missing = [p for p in train_paths if not p.exists()]
        if missing:
            logger.warning(
                "skipping run '%s': missing parquets %s",
                name, [str(m.relative_to(ROOT)) for m in missing],
            )
            out.append({
                "run": name,
                "status": "skipped",
                "missing": [str(m.relative_to(ROOT)) for m in missing],
            })
            continue

        ranker_subdir = out_dir / run_spec["out_subdir"]
        if (ranker_subdir / "ranker.pt").exists():
            logger.info("loading cached ranker for run '%s'", name)
            model, scaler, meta = load_ranker(ranker_subdir)
            n_train = sum(len(pd.read_parquet(p, columns=["id_str"])) for p in train_paths)
        else:
            # Concatenate parts; train one MaskNet on the union.
            logger.info("loading + concatenating training parts: %s", [p.name for p in train_paths])
            frames = [pd.read_parquet(p) for p in train_paths]
            train_df = pd.concat(frames, ignore_index=True)
            n_train = len(train_df)
            logger.info("training corpus has %s rows", f"{n_train:,}")
            # Labels not needed for ranker training but cheap to include.
            train_df = add_labels(train_df, iffy_path=iffy, mainstream_path=mainstream)

            tcfg = TrainConfig(
                batch_size=4096, lr=1e-3, epochs=5, val_frac=0.1,
                max_train_rows=None, device=_device(),
                seed=seed, model_kind="masknet",
            )
            t0 = time.time()
            result = train_ranker(train_df, train_config=tcfg)
            train_dt = time.time() - t0
            logger.info("training done in %.1fs", train_dt)
            save_ranker(result, ranker_subdir)
            model, scaler = result.model, result.scaler
            meta = {"metrics": result.metrics}

        sc_set = _build_scoring_set(cfg)
        out_v = _run_variant(
            df=eval_df, user_pool=eval_user_pool, ranker=model, scaler=scaler,
            diurnal=eval_diurnal, seeds=eval_seeds, cfg=cfg,
            scoring_configs=sc_set, seed=seed,
        )
        cell = {
            "run": name,
            "status": "ok",
            "n_train": int(n_train),
            "ranker_metrics": meta.get("metrics", {}),
            "audience_reach_ablated": out_v["contrasts"]["audience_reach"]["ablated"],
            "cascade_size_ablated": out_v["contrasts"]["cascade_size"]["ablated"],
            "audience_reach_retuned": out_v["contrasts"]["audience_reach"]["additive_retuned"],
        }
        out.append(cell)
        logger.info(
            "  run=%s n_train=%d | AR_abl=%.2f [%.2f, %.2f]",
            name, int(n_train),
            cell["audience_reach_ablated"]["diff_point"],
            cell["audience_reach_ablated"]["diff_ci_lo"],
            cell["audience_reach_ablated"]["diff_ci_hi"],
        )
    return out


# ---- driver ------------------------------------------------------------

def run(config_path: Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = _load_yaml(config_path)
    seed = int(cfg["random_seed"])
    timings: dict[str, float] = {}

    out_dir = ROOT / cfg["artifacts"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / cfg["artifacts"]["report_path"]
    seeds_root = ROOT / cfg["ranker"]["seeds_dir"]
    baseline_load_dir = ROOT / cfg["ranker"]["load_dir"]

    primary_parquet = ROOT / cfg["data"]["primary_part"]
    with _timed(f"load primary parquet ({primary_parquet.name})", timings):
        df = pd.read_parquet(primary_parquet)
        df = add_labels(
            df,
            iffy_path=ROOT / cfg["labels"]["iffy_path"],
            mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
        )

    n_users = int(cfg["simulation"]["n_users"])
    n_per_label = int(cfg["simulation"]["n_per_label"])
    user_pool_default = sample_users(df, n=n_users, seed=seed)
    seeds = _stratified_label_seeds(df, n_per_label, seed)
    diurnal = fit_diurnal_weights(df)

    # Pre-load baseline ranker (used by RB2 + as a reference).
    with _timed("load baseline ranker", timings):
        baseline_model, baseline_scaler, _ = load_ranker(baseline_load_dir)

    results: dict[str, Any] = {}

    with _timed("RB1: cross-product (partition × ranker_seed)", timings):
        results["cross_product"] = _run_cross_product(
            df=df, user_pool=user_pool_default, seeds=seeds, diurnal=diurnal,
            cfg=cfg, seed=seed,
            seeds_root=seeds_root, baseline_load_dir=baseline_load_dir,
        )

    with _timed("RB2: user-sample-size sweep", timings):
        results["user_pool_sweep"] = _run_user_pool_sweep(
            df=df, seeds=seeds, diurnal=diurnal,
            model=baseline_model, scaler=baseline_scaler,
            cfg=cfg, seed=seed,
        )

    with _timed("RB3: enlarged training corpus", timings):
        results["enlarged_training"] = _run_enlarged_training(
            eval_df=df, eval_seeds=seeds,
            eval_user_pool=user_pool_default, eval_diurnal=diurnal,
            cfg=cfg, seed=seed, out_dir=out_dir,
        )

    # Persist + report
    metrics_blob = {
        "config_path": str(config_path.relative_to(ROOT)),
        "config": cfg,
        "results": results,
        "timings_seconds": timings,
    }
    (out_dir / "phase5_extra_metrics.json").write_text(
        json.dumps(metrics_blob, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
        encoding="utf-8",
    )
    write_report(cfg=cfg, results=results, timings=timings, report_path=report_path)
    logger.info("wrote metrics → %s", out_dir / "phase5_extra_metrics.json")
    logger.info("wrote report  → %s", report_path)

    logger.info("=== timing summary ===")
    for label, dt in timings.items():
        logger.info("  %-50s %8.2fs", label, dt)


def _sign(ci_lo: float, ci_hi: float) -> str:
    if ci_hi < 0:
        return "**< 0**"
    if ci_lo > 0:
        return "**> 0**"
    return "≈ 0"


def write_report(
    *,
    cfg: dict[str, Any],
    results: dict[str, Any],
    timings: dict[str, float],
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 5 Extra — Supplementary Robustness Sweeps\n")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(
        f"Per-cell: n_per_label={cfg['simulation']['n_per_label']:,} × 2, "
        f"n_replicates={cfg['simulation']['n_replicates']}, "
        f"bootstrap={cfg['simulation']['bootstrap_iterations']:,}\n"
    )
    lines.append(
        "Three sweeps beyond the original Phase 5 four. Sign convention: "
        "**`audience_reach` ablated contrast strictly < 0 = architectural-claim "
        "signature**.\n"
    )

    # RB1
    lines.append("## RB1 — Cross-product (partition × ranker_seed)\n")
    lines.append(
        "Tests whether the architectural-claim signature survives "
        "interactions between Phase 5 Sweep 2 (partition) and Sweep 3 "
        "(ranker training seed). Each cell is one Phase-4 pipeline; 4 "
        "partitions × 5 seeds = 20 cells.\n"
    )
    lines.append("| ranker_seed | partition | AR diff | 95% CI | sign | CS diff | 95% CI | sign |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in results["cross_product"]:
        ar = c["audience_reach_ablated"]
        cs = c["cascade_size_ablated"]
        lines.append(
            f"| {c['ranker_seed']} | {c['partition']} | "
            f"{ar['diff_point']:.2f} | [{ar['diff_ci_lo']:.2f}, {ar['diff_ci_hi']:.2f}] | "
            f"{_sign(ar['diff_ci_lo'], ar['diff_ci_hi'])} | "
            f"{cs['diff_point']:.2f} | [{cs['diff_ci_lo']:.2f}, {cs['diff_ci_hi']:.2f}] | "
            f"{_sign(cs['diff_ci_lo'], cs['diff_ci_hi'])} |"
        )
    n_cells = len(results["cross_product"])
    n_ar_neg = sum(
        1 for c in results["cross_product"]
        if c["audience_reach_ablated"]["diff_ci_hi"] < 0
    )
    n_cs_neg = sum(
        1 for c in results["cross_product"]
        if c["cascade_size_ablated"]["diff_ci_hi"] < 0
    )
    lines.append(
        f"\n**Summary:** AR contrast strictly < 0 in **{n_ar_neg}/{n_cells}** "
        f"cells; CS contrast strictly < 0 in **{n_cs_neg}/{n_cells}** cells.\n"
    )

    # RB2
    lines.append("## RB2 — User-sample-size sensitivity\n")
    lines.append(
        "Default partition + baseline ranker (seed=1337). Vary "
        "``n_users``; each cell is one Phase-4 pipeline.\n"
    )
    lines.append("| n_users | AR diff | 95% CI | sign | CS diff | 95% CI | sign |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in results["user_pool_sweep"]:
        ar = c["audience_reach_ablated"]
        cs = c["cascade_size_ablated"]
        lines.append(
            f"| {c['n_users']:,} | "
            f"{ar['diff_point']:.2f} | [{ar['diff_ci_lo']:.2f}, {ar['diff_ci_hi']:.2f}] | "
            f"{_sign(ar['diff_ci_lo'], ar['diff_ci_hi'])} | "
            f"{cs['diff_point']:.2f} | [{cs['diff_ci_lo']:.2f}, {cs['diff_ci_hi']:.2f}] | "
            f"{_sign(cs['diff_ci_lo'], cs['diff_ci_hi'])} |"
        )
    lines.append("")

    # RB3
    lines.append("## RB3 — Enlarged training corpus\n")
    lines.append(
        "Train MaskNet on the union of multiple USC parts; evaluate Phase 4 "
        "on part_1 (held fixed across runs so the contrast is "
        "training-corpus-only). Per-head AUCs are reported as ranker-quality "
        "evidence.\n"
    )
    for c in results["enlarged_training"]:
        if c.get("status") == "skipped":
            lines.append(
                f"- **{c['run']}**: skipped — missing parquets "
                f"{c['missing']}.\n"
            )
            continue
        m = c["ranker_metrics"]
        lines.append(f"### {c['run']}\n")
        lines.append(
            f"Training corpus: **{c['n_train']:,} tweets**. "
            f"Per-head AUC (val 10%): "
            f"reply {m.get('auc_reply', float('nan')):.3f}, "
            f"retweet {m.get('auc_retweet', float('nan')):.3f}, "
            f"like {m.get('auc_like', float('nan')):.3f}, "
            f"deep {m.get('auc_deep', float('nan')):.3f}.\n"
        )
        ar = c["audience_reach_ablated"]
        cs = c["cascade_size_ablated"]
        lines.append("| metric | regime | diff | 95% CI | sign |")
        lines.append("|---|---|---|---|---|")
        lines.append(
            f"| audience_reach | ablated | {ar['diff_point']:.2f} | "
            f"[{ar['diff_ci_lo']:.2f}, {ar['diff_ci_hi']:.2f}] | "
            f"{_sign(ar['diff_ci_lo'], ar['diff_ci_hi'])} |"
        )
        lines.append(
            f"| cascade_size | ablated | {cs['diff_point']:.2f} | "
            f"[{cs['diff_ci_lo']:.2f}, {cs['diff_ci_hi']:.2f}] | "
            f"{_sign(cs['diff_ci_lo'], cs['diff_ci_hi'])} |"
        )
        rr = c["audience_reach_retuned"]
        lines.append(
            f"| audience_reach | additive_retuned | {rr['diff_point']:.2f} | "
            f"[{rr['diff_ci_lo']:.2f}, {rr['diff_ci_hi']:.2f}] | "
            f"{_sign(rr['diff_ci_lo'], rr['diff_ci_hi'])} |"
        )
        lines.append("")

    # Timings
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
        default=ROOT / "configs" / "experiment_phase5_extra.yaml",
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

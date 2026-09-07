"""Phase 3 driver: train MaskNet on the cleaned USC corpus, with explicit
per-step timing.

Designed to diagnose and unblock the prior 10-hour-no-output hang. Each step
is timed and logged; if anything stalls, the timing breakdown will point at
the offending step instead of leaving us guessing.

Run::

    uv run python -m analysis.run_phase3 --config configs/experiment_phase3_smoke.yaml
    uv run python -m analysis.run_phase3 --config configs/experiment_phase3_full.yaml

Phase 3 scope per the design specification is broader than just training (also: scale users,
add Stage 1 metrics, run validation). This driver currently covers the
training half; simulation/validation is wired in once training is confirmed
sound.
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

import pandas as pd
import torch
import yaml

from ranker.features import FEATURE_NAMES, extract_features
from ranker.training import TrainConfig, save_ranker, train_ranker

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rel_to_root(path: Path) -> str:
    """Path relative to project root if possible, else the absolute path."""
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


def _make_train_config(cfg: dict[str, Any]) -> TrainConfig:
    t = cfg["training"]
    r = cfg["ranker"]
    return TrainConfig(
        batch_size=int(t["batch_size"]),
        lr=float(t["lr"]),
        epochs=int(t["epochs"]),
        val_frac=float(t["val_frac"]),
        max_train_rows=t.get("max_train_rows"),
        device=str(t.get("device", "cpu")),
        seed=int(cfg["random_seed"]),
        model_kind=str(r.get("model_kind", "masknet")),
        model_kwargs=dict(r.get("model_kwargs", {})),
    )


def run(config_path: Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    timings: dict[str, float] = {}

    out_dir = ROOT / cfg["artifacts"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ranker_dir = out_dir / cfg["artifacts"]["ranker_subdir"]
    report_path = ROOT / cfg["artifacts"]["report_path"]

    # Environment snapshot — useful when comparing CPU vs CUDA runs.
    logger.info(
        "torch=%s cuda_available=%s device_requested=%s",
        torch.__version__, torch.cuda.is_available(),
        cfg["training"].get("device", "cpu"),
    )
    if torch.cuda.is_available():
        logger.info("cuda device: %s", torch.cuda.get_device_name(0))

    # --- 1. data -------------------------------------------------------
    parquet = ROOT / cfg["data"]["part_parquet"]
    with _timed(f"load parquet ({parquet.name})", timings):
        df = pd.read_parquet(parquet)
        logger.info("  loaded %s rows × %s cols", f"{len(df):,}", len(df.columns))

    # --- 2. feature-extraction probe on full corpus --------------------
    # The prior hang most likely happened inside extract_features (in
    # particular pd.to_datetime on user_created_at over 1M rows). Time
    # this step independently before committing to a full training run.
    if cfg.get("diagnostics", {}).get("feature_probe_full_corpus", False):
        with _timed(f"extract_features probe ({len(df):,} rows)", timings):
            X_probe = extract_features(df)
            logger.info(
                "  feature matrix shape=%s dtype=%s",
                X_probe.shape, X_probe.dtype,
            )
            del X_probe

    # --- 3. train ------------------------------------------------------
    train_cfg = _make_train_config(cfg)
    logger.info(
        "training: kind=%s device=%s max_rows=%s epochs=%d batch=%d",
        train_cfg.model_kind, train_cfg.device, train_cfg.max_train_rows,
        train_cfg.epochs, train_cfg.batch_size,
    )
    with _timed("train_ranker", timings):
        result = train_ranker(df, train_config=train_cfg)
    with _timed("save_ranker", timings):
        save_ranker(result, ranker_dir)
        logger.info("  saved to %s", ranker_dir)

    # --- 4. report -----------------------------------------------------
    write_report(
        config_path=config_path,
        cfg=cfg,
        train_metrics=result.metrics,
        history=result.history,
        timings=timings,
        report_path=report_path,
    )

    metrics = {
        "config_path": str(_rel_to_root(config_path)),
        "config": cfg,
        "train_metrics": result.metrics,
        "history": result.history,
        "timings_seconds": timings,
        "n_features": len(FEATURE_NAMES),
    }
    (out_dir / "phase3_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )
    logger.info("wrote metrics to %s", out_dir / "phase3_metrics.json")
    logger.info("wrote report to %s", report_path)

    # Final timing summary, easy to eyeball in the terminal.
    logger.info("=== timing summary ===")
    for label, dt in timings.items():
        logger.info("  %-50s %8.2fs", label, dt)


def write_report(
    *,
    config_path: Path,
    cfg: dict[str, Any],
    train_metrics: dict[str, float],
    history: list[dict[str, float]],
    timings: dict[str, float],
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 3 Report — MaskNet training\n")
    lines.append(f"Config: `{_rel_to_root(config_path)}`")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(
        f"Seed: {cfg['random_seed']} | model_kind={cfg['ranker'].get('model_kind', 'masknet')} | "
        f"device={cfg['training'].get('device', 'cpu')}\n"
    )

    lines.append("## 1. Step timings\n")
    lines.append("```")
    for label, dt in timings.items():
        lines.append(f"  {label:50s} {dt:8.2f}s")
    lines.append("```\n")

    lines.append("## 2. Training metrics\n")
    lines.append("```")
    for k, v in train_metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:24s} {v:.4f}")
        else:
            lines.append(f"  {k:24s} {v}")
    lines.append("```\n")

    lines.append("## 3. Per-epoch history\n")
    lines.append("```")
    for rec in history:
        parts = [f"epoch={rec['epoch']:>2}"]
        for k in ("train_loss", "auc_reply", "auc_retweet", "auc_like", "auc_deep", "elapsed_s"):
            if k in rec:
                parts.append(f"{k}={rec[k]:.4f}")
        lines.append("  " + "  ".join(parts))
    lines.append("```\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_phase3_smoke.yaml",
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

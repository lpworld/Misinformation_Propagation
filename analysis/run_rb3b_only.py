"""One-off: run only the parts_1_to_5 retraining (RB3b) and merge into the
phase5_extra report. Useful when RB1/RB2/RB3a have already been computed
and we don't want to repeat them.

Reads the existing phase5_extra_metrics.json, runs RB3b, splices the new
result into the enlarged_training list, and rewrites the report + JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from analysis.labeling import add_labels
from analysis.run_phase4 import _stratified_label_seeds
from analysis.run_phase5_extra import (
    _run_enlarged_training,
    write_report,
    ROOT,
)
from ranker.training import load_ranker  # noqa: F401 (used indirectly)
from simulation.cascade import fit_diurnal_weights
from simulation.users import sample_users
import pandas as pd

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = ROOT / "configs" / "experiment_phase5_extra.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg["random_seed"])
    out_dir = ROOT / cfg["artifacts"]["out_dir"]
    report_path = ROOT / cfg["artifacts"]["report_path"]

    # Rebuild the same eval state used by the original sweep.
    primary_parquet = ROOT / cfg["data"]["primary_part"]
    df = pd.read_parquet(primary_parquet)
    df = add_labels(
        df,
        iffy_path=ROOT / cfg["labels"]["iffy_path"],
        mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
    )
    n_users = int(cfg["simulation"]["n_users"])
    n_per_label = int(cfg["simulation"]["n_per_label"])
    user_pool = sample_users(df, n=n_users, seed=seed)
    seeds = _stratified_label_seeds(df, n_per_label, seed)
    diurnal = fit_diurnal_weights(df)

    # Keep only the parts_1_to_5 run (parts_1_2 was already done).
    only_full_cfg = dict(cfg)
    only_full_cfg["sweeps"] = dict(cfg["sweeps"])
    only_full_cfg["sweeps"]["enlarged_training"] = {
        "runs": [
            r for r in cfg["sweeps"]["enlarged_training"]["runs"]
            if r["name"] == "parts_1_to_5"
        ]
    }

    new_results = _run_enlarged_training(
        eval_df=df, eval_seeds=seeds, eval_user_pool=user_pool,
        eval_diurnal=diurnal, cfg=only_full_cfg, seed=seed, out_dir=out_dir,
    )

    # Splice into the existing metrics blob.
    metrics_path = out_dir / "phase5_extra_metrics.json"
    blob = json.loads(metrics_path.read_text(encoding="utf-8"))
    blob["results"]["enlarged_training"] = [
        # Keep parts_1_2; replace any parts_1_to_5 entry; append new ones.
        e for e in blob["results"]["enlarged_training"]
        if e["run"] != "parts_1_to_5"
    ] + new_results
    metrics_path.write_text(
        json.dumps(blob, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)),
        encoding="utf-8",
    )

    write_report(
        cfg=cfg, results=blob["results"],
        timings=blob.get("timings_seconds", {}), report_path=report_path,
    )
    logger.info("merged RB3b into phase5_extra report")


if __name__ == "__main__":
    main()

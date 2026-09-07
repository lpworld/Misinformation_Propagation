"""RB5 — subgroup analyses (revision item 5).

Pure post-hoc analysis on the cached Phase-4 per-cascade outputs. For
each subgroup cut, recompute the audience_reach and cascade_size
contrasts (ablated − additive, additive_retuned − additive) within
each quintile of the cut variable. Tests whether the architectural
finding holds within strata defined by:

* **Follower count** of the seed's author (small / mid / large
  influencers, etc.).
* **Account age** of the seed's author (new / established).
* **Realized cascade size** of the seed (dampens / amplifies the
  architectural effect at the propagation tail?).

No new simulator runs; reuses ``ablation_per_cascade.parquet`` and the
``cascade_bootstrap_contrast`` machinery.

Run::

    uv run python -m analysis.run_subgroup_analyses
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.hypothesis import (
    HIGH,
    LOW,
    _replicate_mean_per_cascade,
    _wide_per_regime,
    cascade_bootstrap_contrast,
)
from analysis.labeling import add_labels
from analysis.run_phase4 import _stratified_label_seeds

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


REGIMES = ("additive", "ablated", "additive_retuned")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _seeds_with_features(cfg: dict) -> pd.DataFrame:
    """Reproduce Phase-4 seed sample with extra columns for subgrouping."""
    parquet = ROOT / cfg["data"]["part_parquet"]
    df = pd.read_parquet(parquet)
    df = add_labels(
        df,
        iffy_path=ROOT / cfg["labels"]["iffy_path"],
        mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
    )
    seeds = _stratified_label_seeds(df, int(cfg["simulation"]["n_per_label"]), int(cfg["random_seed"]))
    # Reattach features for subgrouping. _stratified_label_seeds keeps
    # SEED_COLS only — it includes user_followersCount but not the date.
    # We re-merge user_created_at by id_str.
    user_dates = df.set_index("id_str")["user_created_at"]
    seeds["user_created_at"] = seeds["id_str"].map(user_dates)
    # Cascade-size proxy from observed counts (used only as a subgroup cut).
    for col in ("replyCount", "retweetCount", "quoteCount"):
        if col in seeds.columns:
            seeds[col] = pd.to_numeric(seeds[col], errors="coerce").fillna(0.0)
    seeds["observed_aggregate_engagement"] = (
        seeds.get("replyCount", 0) + seeds.get("retweetCount", 0) + seeds.get("quoteCount", 0)
    )
    seeds["account_age_days"] = (
        (pd.Timestamp("2024-06-15", tz="UTC") - pd.to_datetime(seeds["user_created_at"], utc=True))
        .dt.total_seconds() / 86400.0
    )
    return seeds


def _quintile_cuts(values: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_edges, bin_indices) for n_bins equal-frequency cuts."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([]), np.zeros_like(values, dtype=int)
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)  # collapse if duplicates (heavily skewed dists)
    if edges.size < 2:
        return edges, np.zeros_like(values, dtype=int)
    bin_idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)
    return edges, bin_idx


def _bootstrap_contrasts_within(
    wide: pd.DataFrame, mask: np.ndarray, regime_a: str, regime_b: str,
    metric_name: str, n_bootstrap: int, seed: int,
):
    sub = wide[mask].reset_index(drop=True)
    if (sub["credibility_label"] == LOW).sum() == 0 or (sub["credibility_label"] == HIGH).sum() == 0:
        return None
    return cascade_bootstrap_contrast(
        sub,
        regime_a=regime_a, regime_b=regime_b,
        metric_name=metric_name, n_bootstrap=n_bootstrap,
        rng=np.random.default_rng(seed),
    )


def _format_contrast(c) -> str:
    if c is None:
        return "—"
    sign = "**< 0**" if c.diff_ci_hi < 0 else ("**> 0**" if c.diff_ci_lo > 0 else "≈ 0")
    return f"{c.diff_point:.2f} [{c.diff_ci_lo:.2f}, {c.diff_ci_hi:.2f}] {sign}"


def run(config_path: Path) -> None:
    cfg = _load_yaml(config_path)
    seed = int(cfg["random_seed"])
    n_bootstrap = int(cfg["simulation"]["bootstrap_iterations"])

    out_dir = ROOT / cfg["artifacts"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase-4 per-cascade parquet
    per_cascade_path = ROOT / "data" / "processed" / "phase4" / "ablation_per_cascade.parquet"
    logger.info("loading %s", per_cascade_path)
    pc = pd.read_parquet(per_cascade_path)
    logger.info("per_cascade: %s rows, regimes=%s", f"{len(pc):,}", pc["regime"].unique())

    # Reproduce seeds and add subgroup features.
    logger.info("rebuilding seed feature frame")
    seeds = _seeds_with_features(cfg)

    # Build per-cascade replicate means for AR + CS, then wide.
    logger.info("computing replicate means")
    pc_ar = _replicate_mean_per_cascade(pc, pc["total_exposures"].to_numpy())
    pc_cs = _replicate_mean_per_cascade(pc, (pc["n_reply"] + pc["n_retweet"] + pc["n_deep"]).to_numpy())
    wide_ar = _wide_per_regime(pc_ar, REGIMES)
    wide_cs = _wide_per_regime(pc_cs, REGIMES)

    # Attach seed-level features by cascade_id.
    seed_feats = seeds[["user_followersCount", "account_age_days", "observed_aggregate_engagement"]].copy()
    for col in seed_feats.columns:
        seed_feats[col] = pd.to_numeric(seed_feats[col], errors="coerce")
    wide_ar = wide_ar.merge(
        seed_feats.reset_index().rename(columns={"index": "cascade_id"}),
        on="cascade_id", how="left",
    )
    wide_cs = wide_cs.merge(
        seed_feats.reset_index().rename(columns={"index": "cascade_id"}),
        on="cascade_id", how="left",
    )

    # Subgroup cuts
    cuts = {
        "follower_count": "user_followersCount",
        "account_age_days": "account_age_days",
        "observed_engagement": "observed_aggregate_engagement",
    }
    n_bins = int(cfg.get("subgroup_n_bins", 5))

    results: dict = {"cuts": {}, "n_bootstrap": n_bootstrap, "n_bins": n_bins}
    for cut_name, col in cuts.items():
        logger.info("cut: %s (col=%s)", cut_name, col)
        values = wide_ar[col].to_numpy(dtype=np.float64)
        edges, bin_idx = _quintile_cuts(values, n_bins=n_bins)
        if edges.size < 2:
            logger.warning("skipping cut %s — degenerate quantiles", cut_name)
            continue
        cut_results: list[dict] = []
        for b in range(edges.size - 1):
            mask = bin_idx == b
            n_low = int(((wide_ar["credibility_label"] == LOW) & mask).sum())
            n_high = int(((wide_ar["credibility_label"] == HIGH) & mask).sum())
            if n_low == 0 or n_high == 0:
                cut_results.append({
                    "bin": b, "edge_lo": float(edges[b]), "edge_hi": float(edges[b+1]),
                    "n_low": n_low, "n_high": n_high,
                    "ar_ablated": None, "ar_retuned": None,
                    "cs_ablated": None, "cs_retuned": None,
                })
                continue
            cs_ablated = _bootstrap_contrasts_within(
                wide_cs, mask, "additive", "ablated", "cascade_size",
                n_bootstrap=n_bootstrap, seed=seed,
            )
            ar_ablated = _bootstrap_contrasts_within(
                wide_ar, mask, "additive", "ablated", "audience_reach",
                n_bootstrap=n_bootstrap, seed=seed,
            )
            ar_retuned = _bootstrap_contrasts_within(
                wide_ar, mask, "additive", "additive_retuned", "audience_reach",
                n_bootstrap=n_bootstrap, seed=seed,
            )
            cut_results.append({
                "bin": b, "edge_lo": float(edges[b]), "edge_hi": float(edges[b+1]),
                "n_low": n_low, "n_high": n_high,
                "ar_ablated": {
                    "diff_point": ar_ablated.diff_point,
                    "diff_ci_lo": ar_ablated.diff_ci_lo,
                    "diff_ci_hi": ar_ablated.diff_ci_hi,
                } if ar_ablated else None,
                "ar_retuned": {
                    "diff_point": ar_retuned.diff_point,
                    "diff_ci_lo": ar_retuned.diff_ci_lo,
                    "diff_ci_hi": ar_retuned.diff_ci_hi,
                } if ar_retuned else None,
                "cs_ablated": {
                    "diff_point": cs_ablated.diff_point,
                    "diff_ci_lo": cs_ablated.diff_ci_lo,
                    "diff_ci_hi": cs_ablated.diff_ci_hi,
                } if cs_ablated else None,
            })
        results["cuts"][cut_name] = {
            "edges": edges.tolist(),
            "bins": cut_results,
        }

    # Persist
    (out_dir / "subgroup_metrics.json").write_text(
        json.dumps(results, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x),
        encoding="utf-8",
    )
    write_report(results, out_dir / "subgroup_report.md")
    logger.info("wrote subgroup report → %s", out_dir / "subgroup_report.md")


def write_report(results: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# RB5 — Subgroup analyses\n")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(
        f"Cascade-level paired bootstrap (n={results['n_bootstrap']}) on existing Phase-4 "
        f"outputs (`ablation_per_cascade.parquet`). Cuts at {results['n_bins']}-quantile boundaries.\n"
    )
    lines.append(
        "Sign convention: **`audience_reach` ablated contrast strictly < 0 = "
        "architectural-claim signature within that subgroup**.\n"
    )
    for cut_name, blob in results["cuts"].items():
        lines.append(f"## Cut: {cut_name}\n")
        edges = blob["edges"]
        lines.append("| bin | range | n_low | n_high | AR ablated | AR retuned | CS ablated |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in blob["bins"]:
            ar_a = r.get("ar_ablated")
            ar_r = r.get("ar_retuned")
            cs_a = r.get("cs_ablated")
            def _fmt(d):
                if d is None:
                    return "—"
                sign = "**< 0**" if d["diff_ci_hi"] < 0 else ("**> 0**" if d["diff_ci_lo"] > 0 else "≈ 0")
                return f"{d['diff_point']:.2f} [{d['diff_ci_lo']:.2f}, {d['diff_ci_hi']:.2f}] {sign}"
            lines.append(
                f"| Q{r['bin']+1} | [{r['edge_lo']:.0f}, {r['edge_hi']:.0f}] | "
                f"{r['n_low']:,} | {r['n_high']:,} | "
                f"{_fmt(ar_a)} | {_fmt(ar_r)} | {_fmt(cs_a)} |"
            )
        n_strict = sum(
            1 for r in blob["bins"]
            if r.get("ar_ablated") and r["ar_ablated"]["diff_ci_hi"] < 0
        )
        n_total = sum(1 for r in blob["bins"] if r.get("ar_ablated"))
        lines.append(
            f"\n**Summary:** AR ablated contrast strictly < 0 in **{n_strict}/{n_total}** "
            f"non-empty {cut_name} bins.\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_phase4.yaml",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase5_extra" / "subgroup",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = _load_yaml(args.config)
    cfg.setdefault("artifacts", {})["out_dir"] = str(args.out_dir.relative_to(ROOT))
    cfg.setdefault("subgroup_n_bins", 5)
    run(args.config)


if __name__ == "__main__":
    main()

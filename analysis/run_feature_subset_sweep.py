"""RB4 — feature-subset robustness (revision item 4).

Trains MaskNet variants on different ranker feature subsets, evaluates the
score-level architectural contrast on the labeled Phase-4 seed pool, and
reports whether the ablated/retuned sign asymmetry survives across feature
designs.

Score-level contrast vs. audience_reach: under fixed simulator calibration
the audience_reach contrast (§ 6.2.1) is monotone in the per-seed score,
so the SIGN of the score-level contrast tracks the audience_reach sign.
This is the right comparison for a robustness check across feature
subsets — we test whether the architectural-property signature
(opposite signs of ablated vs retuned) survives, not whether magnitudes
exactly match. See § 6.5 Sweep 8 for the result and limitations.

Subsets tested:

* **full** — baseline, all 17 features (sanity check).
* **tweet_time_only** — no account features. Tests "does the architecture
  still target reactive vs reflective in the absence of any
  prestige/popularity signal?"
* **account_only** — no tweet- or time-side features. Tests "does the
  architecture work purely off author-level features?"
* **minimal_3** — log_followers, log_text_length, has_url. Stripped to
  the bare minimum.

Run::

    uv run python -m analysis.run_feature_subset_sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from analysis.hypothesis import HIGH, LOW
from analysis.labeling import add_labels
from analysis.run_phase4 import _stratified_label_seeds
from ranker.architecture import HEAD_NAMES
from ranker.features import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_NAMES,
    FeatureScaler,
    extract_features,
    extract_targets,
)
from ranker.masknet import MaskNetConfig, MaskNetRanker
from ranker.scoring import ScoringConfig, aggregate_score, make_default_configs
from sklearn.metrics import roc_auc_score
from torch import nn

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


SUBSETS: dict[str, list[str]] = {
    "full": list(FEATURE_NAMES),
    "tweet_time_only": [
        "log_text_length",
        "hour_sin", "hour_cos", "weekhour_sin", "weekhour_cos",
        "is_reply", "is_quote", "has_url",
    ],
    "account_only": [
        "log_followers", "log_friends", "log_statuses", "log_favourites",
        "log_listed", "log_clout", "log_favs_per_status",
        "log_statuses_per_day", "log_account_age_days",
        "user_blue",
    ],
    "minimal_3": ["log_followers", "log_text_length", "has_url"],
}


def _subset_indices(names: list[str]) -> tuple[list[int], int]:
    """Return (indices into full FEATURE_NAMES, n_continuous_in_subset)."""
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    indices = [name_to_idx[n] for n in names]
    n_continuous = sum(1 for n in names if n in CONTINUOUS_FEATURES)
    # Ensure subset is in canonical order (continuous first, binary after).
    indices_sorted = sorted(indices, key=lambda i: (i >= len(CONTINUOUS_FEATURES), i))
    return indices_sorted, n_continuous


def _train_subset_masknet(
    df: pd.DataFrame, *, subset_names: list[str], seed: int = 1337,
    epochs: int = 5, batch_size: int = 4096, max_train_rows: int | None = None,
) -> tuple[MaskNetRanker, FeatureScaler, dict[str, float]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if max_train_rows is not None and len(df) > max_train_rows:
        df = df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)

    indices, n_continuous = _subset_indices(subset_names)
    X_full = extract_features(df)
    X = X_full[:, indices].astype(np.float32, copy=True)
    Y = extract_targets(df)

    rng = np.random.default_rng(seed)
    n = len(df)
    perm = np.arange(n)
    rng.shuffle(perm)
    val_n = int(n * 0.1)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    scaler = FeatureScaler(n_continuous=n_continuous)
    scaler.fit(X[train_idx])
    Xs = scaler.transform(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(Xs).float().to(device)
    Yt = {k: torch.from_numpy(v).float().to(device) for k, v in Y.items()}

    cfg = MaskNetConfig(n_features=X.shape[1])
    model = MaskNetRanker(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    train_idx_t = torch.from_numpy(train_idx).long().to(device)
    val_idx_t = torch.from_numpy(val_idx).long().to(device)

    for epoch in range(epochs):
        model.train()
        rng_e = np.random.default_rng(seed + epoch)
        rng_e.shuffle(train_idx)
        for start in range(0, train_idx.size, batch_size):
            sel = torch.from_numpy(train_idx[start : start + batch_size]).long().to(device)
            xb = Xt[sel]
            yb = {k: v[sel] for k, v in Yt.items()}
            logits = model(xb)
            loss = sum(bce(logits[k], yb[k]) for k in HEAD_NAMES) / len(HEAD_NAMES)
            optim.zero_grad()
            loss.backward()
            optim.step()

    # Validation AUCs
    aucs: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        val_logits = model(Xt[val_idx_t])
        for k in HEAD_NAMES:
            y_true = Yt[k][val_idx_t].cpu().numpy()
            y_score = torch.sigmoid(val_logits[k]).cpu().numpy()
            if y_true.min() == y_true.max():
                aucs[f"auc_{k}"] = float("nan")
            else:
                aucs[f"auc_{k}"] = float(roc_auc_score(y_true, y_score))
    return model, scaler, aucs


def _predict_subset(
    model: MaskNetRanker, scaler: FeatureScaler, df: pd.DataFrame,
    subset_names: list[str],
) -> dict[str, np.ndarray]:
    indices, _ = _subset_indices(subset_names)
    X_full = extract_features(df)
    X = X_full[:, indices].astype(np.float32, copy=True)
    Xs = scaler.transform(X)
    device = next(model.parameters()).device
    with torch.no_grad():
        model.eval()
        xb = torch.from_numpy(Xs).float().to(device)
        probs = model.predict_proba(xb)
    return {k: v.cpu().numpy().astype(np.float64) for k, v in probs.items()}


def _bootstrap_score_contrast(
    scores_low: dict[str, np.ndarray], scores_high: dict[str, np.ndarray],
    *, regime_b: str, regime_a: str = "additive",
    n_bootstrap: int = 1000, seed: int = 1337,
) -> dict[str, float]:
    """Paired bootstrap: ((mean_low_b - mean_high_b) - (mean_low_a - mean_high_a))."""
    rng = np.random.default_rng(seed)
    a_low = scores_low[regime_a]
    a_high = scores_high[regime_a]
    b_low = scores_low[regime_b]
    b_high = scores_high[regime_b]
    n_low, n_high = a_low.size, a_high.size

    point = (b_low.mean() - b_high.mean()) - (a_low.mean() - a_high.mean())
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx_low = rng.integers(0, n_low, size=n_low)
        idx_high = rng.integers(0, n_high, size=n_high)
        gap_a = a_low[idx_low].mean() - a_high[idx_high].mean()
        gap_b = b_low[idx_low].mean() - b_high[idx_high].mean()
        boots[i] = gap_b - gap_a
    return {
        "diff_point": float(point),
        "diff_ci_lo": float(np.percentile(boots, 2.5)),
        "diff_ci_hi": float(np.percentile(boots, 97.5)),
    }


def run(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg["random_seed"])
    n_bootstrap = int(cfg["simulation"]["bootstrap_iterations"])

    out_dir = ROOT / "data" / "processed" / "phase5_extra" / "feature_subsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "paper" / "phase5_extra_feature_subsets_report.md"

    primary_parquet = ROOT / cfg["data"]["part_parquet"]
    logger.info("loading %s", primary_parquet)
    df = pd.read_parquet(primary_parquet)
    df = add_labels(
        df,
        iffy_path=ROOT / cfg["labels"]["iffy_path"],
        mainstream_path=ROOT / cfg["labels"]["mainstream_path"],
    )
    seeds = _stratified_label_seeds(df, int(cfg["simulation"]["n_per_label"]), seed)
    labels = seeds["credibility_label"].to_numpy()
    low_mask = labels == LOW
    high_mask = labels == HIGH

    scoring_cfgs = make_default_configs()
    regimes = ("additive", "ablated", "additive_retuned")

    results: dict[str, Any] = {"subsets": {}}
    for subset_name, subset_features in SUBSETS.items():
        logger.info("=== subset: %s (%d features) ===", subset_name, len(subset_features))
        t0 = time.time()
        model, scaler, aucs = _train_subset_masknet(
            df, subset_names=subset_features, seed=seed,
            epochs=int(cfg.get("masknet_epochs", 5)),
            max_train_rows=cfg.get("masknet_max_train_rows"),
        )
        logger.info("  trained in %.1fs | AUCs %s", time.time() - t0, aucs)

        probs = _predict_subset(model, scaler, seeds, subset_features)
        scores_low: dict[str, np.ndarray] = {}
        scores_high: dict[str, np.ndarray] = {}
        for regime in regimes:
            S = aggregate_score(probs, scoring_cfgs[regime])
            scores_low[regime] = S[low_mask]
            scores_high[regime] = S[high_mask]
        contrasts: dict[str, dict[str, float]] = {}
        for regime in regimes:
            if regime == "additive":
                continue
            contrasts[regime] = _bootstrap_score_contrast(
                scores_low, scores_high, regime_b=regime, regime_a="additive",
                n_bootstrap=n_bootstrap, seed=seed,
            )
        # Mean scores too — useful for absolute-level inspection.
        mean_low = {r: float(scores_low[r].mean()) for r in regimes}
        mean_high = {r: float(scores_high[r].mean()) for r in regimes}
        results["subsets"][subset_name] = {
            "n_features": len(subset_features),
            "feature_names": list(subset_features),
            "auc": aucs,
            "score_contrast": contrasts,
            "mean_score_low": mean_low,
            "mean_score_high": mean_high,
        }
        logger.info(
            "  AR-equivalent score contrast (ablated vs additive): "
            "%.4f [%.4f, %.4f]",
            contrasts["ablated"]["diff_point"],
            contrasts["ablated"]["diff_ci_lo"],
            contrasts["ablated"]["diff_ci_hi"],
        )

    (out_dir / "feature_subsets_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    write_report(results, report_path)


def write_report(results: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# RB4 — Feature-subset robustness\n")
    lines.append(f"Run: {pd.Timestamp.utcnow().isoformat()}\n")
    lines.append(
        "Score-level paired bootstrap on regime-vs-additive contrasts under "
        "different MaskNet feature subsets. Score-level contrast tracks the "
        "audience_reach contrast in sign by construction (audience_reach is "
        "monotone in score under fixed calibration); magnitudes are not "
        "directly comparable to the audience_reach numbers in §§ 6.2 / 6.5 "
        "because there is no simulator pass here. The test is whether the "
        "**ablated/retuned sign asymmetry** (the architectural-property "
        "signature) survives feature-design changes.\n"
    )
    lines.append("| subset | n_features | AUC reply / retweet / like / deep | mean S low → high (additive) | ablated score-contrast vs additive | retuned score-contrast vs additive |")
    lines.append("|---|---|---|---|---|---|")
    for name, blob in results["subsets"].items():
        a = blob["auc"]
        ml = blob["mean_score_low"]; mh = blob["mean_score_high"]
        cab = blob["score_contrast"].get("ablated", {})
        cre = blob["score_contrast"].get("additive_retuned", {})
        def _fmt(d):
            if not d:
                return "—"
            sign = "**< 0**" if d["diff_ci_hi"] < 0 else ("**> 0**" if d["diff_ci_lo"] > 0 else "≈ 0")
            return f"{d['diff_point']:.3f} [{d['diff_ci_lo']:.3f}, {d['diff_ci_hi']:.3f}] {sign}"
        lines.append(
            f"| `{name}` | {blob['n_features']} | "
            f"{a.get('auc_reply', float('nan')):.3f} / {a.get('auc_retweet', float('nan')):.3f} / "
            f"{a.get('auc_like', float('nan')):.3f} / {a.get('auc_deep', float('nan')):.3f} | "
            f"{ml['additive']:.3f} → {mh['additive']:.3f} | {_fmt(cab)} | {_fmt(cre)} |"
        )
    n_strict_neg = sum(
        1 for blob in results["subsets"].values()
        if blob["score_contrast"].get("ablated", {}).get("diff_ci_hi", 0) < 0
    )
    n_strict_pos = sum(
        1 for blob in results["subsets"].values()
        if blob["score_contrast"].get("additive_retuned", {}).get("diff_ci_lo", 0) > 0
    )
    n_total = len(results["subsets"])
    lines.append(
        f"\n**Summary:** ablated score contrast strictly < 0 in **{n_strict_neg}/{n_total}** "
        f"subsets; additive_retuned score contrast strictly > 0 in **{n_strict_pos}/{n_total}** "
        f"subsets. Architectural-property signature (opposite signs) holds when both conditions "
        f"are met simultaneously.\n"
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
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args.config)


if __name__ == "__main__":
    main()

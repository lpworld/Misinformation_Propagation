"""Model-class robustness — MaskNet vs MLP vs feature-token Transformer.

The 2026 open-source release of the X algorithm replaced the MaskNet-style
Heavy Ranker with a transformer prediction stack while keeping the additive
score aggregation. Referee-facing question: does the architectural-property
signature (ablated < 0, retuned > 0 on the score-level contrast) depend on
the prediction model's architecture class?

Protocol mirrors RB4 (``analysis/run_feature_subset_sweep.py``) with the
model class as the swept axis instead of the feature subset: identical
training recipe (seed 1337, 5 epochs, batch 4096, Adam 1e-3, BCE per head,
90/10 split), identical stratified Phase-4 seed pool, identical paired
score-level bootstrap. All 18 features throughout.

Classes:

* **masknet** — the paper's canonical architecture (anchor; must reproduce
  the RB4 ``full`` row).
* **mlp** — the Phase-2 stub (shared-body MLP), the simplest class.
* **transformer** — a feature-token transformer: each feature becomes a
  learned token (per-feature affine embedding of its value), a CLS token is
  prepended, a 2-layer TransformerEncoder attends over the 19 tokens, and
  the CLS output feeds the four sigmoid heads. This is an
  architecture-class probe in the direction of the 2026 stack, not a
  reconstruction of it (the production model attends over user engagement
  *sequences*, which the USC schema cannot supply).

Run (repo root)::

    .venv\\Scripts\\python.exe -m analysis.run_model_class_sweep

Outputs:

* ``data/processed/phase5_extra/model_classes/model_classes_metrics.json``
* ``paper/model_class_report.md``
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn

from analysis.hypothesis import HIGH, LOW
from analysis.labeling import add_labels
from analysis.run_feature_subset_sweep import _bootstrap_score_contrast
from analysis.run_phase4 import _stratified_label_seeds
from ranker.architecture import HEAD_NAMES, HeavyRankerStub, RankerConfig
from ranker.features import (
    CONTINUOUS_FEATURES,
    FEATURE_NAMES,
    FeatureScaler,
    extract_features,
    extract_targets,
)
from ranker.masknet import MaskNetConfig, MaskNetRanker
from ranker.scoring import aggregate_score, make_default_configs

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


class FeatureTokenTransformer(nn.Module):
    """Feature-token transformer with CLS pooling and per-head sigmoid heads."""

    def __init__(
        self, n_features: int, *, d_model: int = 32, n_heads: int = 4,
        n_layers: int = 2, ff_mult: int = 2, dropout: float = 0.1,
        head_names: tuple[str, ...] = HEAD_NAMES,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        # Per-feature affine value embedding: token_i = value_i * w_i + b_i.
        self.value_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.value_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict({name: nn.Linear(d_model, 1) for name in head_names})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, n_features) -> tokens (B, n_features, d_model)
        tokens = x.unsqueeze(-1) * self.value_weight + self.value_bias
        cls = self.cls.expand(x.shape[0], -1, -1)
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        h = self.norm(h[:, 0])
        return {name: head(h).squeeze(-1) for name, head in self.heads.items()}

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        logits = self.forward(x)
        return {k: torch.sigmoid(v) for k, v in logits.items()}


def _build_model(model_class: str, n_features: int) -> nn.Module:
    if model_class == "masknet":
        return MaskNetRanker(MaskNetConfig(n_features=n_features))
    if model_class == "mlp":
        return HeavyRankerStub(RankerConfig(n_features=n_features))
    if model_class == "transformer":
        return FeatureTokenTransformer(n_features)
    raise ValueError(model_class)


def _train_model(
    df: pd.DataFrame, model_class: str, *, seed: int = 1337, epochs: int = 5,
    batch_size: int = 4096, max_train_rows: int | None = None,
) -> tuple[nn.Module, FeatureScaler, dict[str, float]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if max_train_rows is not None and len(df) > max_train_rows:
        df = df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)

    X = extract_features(df)
    Y = extract_targets(df)

    rng = np.random.default_rng(seed)
    n = len(df)
    perm = np.arange(n)
    rng.shuffle(perm)
    val_n = int(n * 0.1)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    scaler = FeatureScaler(n_continuous=len(CONTINUOUS_FEATURES))
    scaler.fit(X[train_idx])
    Xs = scaler.transform(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(Xs).float().to(device)
    Yt = {k: torch.from_numpy(v).float().to(device) for k, v in Y.items()}

    model = _build_model(model_class, X.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

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

    aucs: dict[str, float] = {}
    model.eval()
    val_idx_t = torch.from_numpy(val_idx).long().to(device)
    with torch.no_grad():
        val_logits = model(Xt[val_idx_t])
        for k in HEAD_NAMES:
            y_true = Yt[k][val_idx_t].cpu().numpy()
            y_score = torch.sigmoid(val_logits[k]).cpu().numpy()
            aucs[f"auc_{k}"] = (
                float("nan") if y_true.min() == y_true.max()
                else float(roc_auc_score(y_true, y_score))
            )
    return model, scaler, aucs


def _predict(model: nn.Module, scaler: FeatureScaler, df: pd.DataFrame) -> dict[str, np.ndarray]:
    Xs = scaler.transform(extract_features(df))
    device = next(model.parameters()).device
    with torch.no_grad():
        model.eval()
        probs = model.predict_proba(torch.from_numpy(Xs).float().to(device))
    return {k: v.cpu().numpy().astype(np.float64) for k, v in probs.items()}


def run(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg["random_seed"])
    n_bootstrap = int(cfg["simulation"]["bootstrap_iterations"])

    out_dir = ROOT / "data" / "processed" / "phase5_extra" / "model_classes"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "paper" / "model_class_report.md"

    df = pd.read_parquet(ROOT / cfg["data"]["part_parquet"])
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

    results: dict[str, Any] = {"classes": {}}
    for model_class in ("masknet", "mlp", "transformer"):
        logger.info("=== model class: %s ===", model_class)
        t0 = time.time()
        model, scaler, aucs = _train_model(
            df, model_class, seed=seed,
            epochs=int(cfg.get("masknet_epochs", 5)),
            max_train_rows=cfg.get("masknet_max_train_rows"),
        )
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("  trained in %.1fs | %d params | AUCs %s", time.time() - t0, n_params, aucs)

        probs = _predict(model, scaler, seeds)
        scores_low: dict[str, np.ndarray] = {}
        scores_high: dict[str, np.ndarray] = {}
        for regime in regimes:
            S = aggregate_score(probs, scoring_cfgs[regime])
            scores_low[regime] = S[low_mask]
            scores_high[regime] = S[high_mask]
        contrasts = {
            regime: _bootstrap_score_contrast(
                scores_low, scores_high, regime_b=regime, regime_a="additive",
                n_bootstrap=n_bootstrap, seed=seed,
            )
            for regime in regimes if regime != "additive"
        }
        results["classes"][model_class] = {
            "n_params": n_params,
            "auc": aucs,
            "score_contrast": contrasts,
        }
        logger.info(
            "  ablated %.4f [%.4f, %.4f] | retuned %.4f [%.4f, %.4f]",
            contrasts["ablated"]["diff_point"],
            contrasts["ablated"]["diff_ci_lo"], contrasts["ablated"]["diff_ci_hi"],
            contrasts["additive_retuned"]["diff_point"],
            contrasts["additive_retuned"]["diff_ci_lo"],
            contrasts["additive_retuned"]["diff_ci_hi"],
        )

    (out_dir / "model_classes_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_report(results, report_path)


def write_report(results: dict, path: Path) -> None:
    L = [
        "# Model-class robustness — MaskNet / MLP / feature-token transformer\n",
        f"Run: {pd.Timestamp.utcnow().isoformat()}\n",
        "Score-level paired bootstrap on regime-vs-additive contrasts with the "
        "prediction-model architecture class as the swept axis (all 18 "
        "features, RB4 protocol otherwise unchanged). The transformer row is "
        "an architecture-class probe in the direction of the 2026 X stack, "
        "not a reconstruction of it. Sign of the score contrast tracks the "
        "audience_reach sign under fixed calibration.\n",
        "| class | params | AUC reply / retweet / like / deep | ablated score-contrast vs additive | retuned score-contrast vs additive |",
        "|---|---|---|---|---|",
    ]

    def _fmt(d: dict[str, float]) -> str:
        sign = "**< 0**" if d["diff_ci_hi"] < 0 else ("**> 0**" if d["diff_ci_lo"] > 0 else "= 0")
        return f"{d['diff_point']:.3f} [{d['diff_ci_lo']:.3f}, {d['diff_ci_hi']:.3f}] {sign}"

    n_sig = 0
    for name, blob in results["classes"].items():
        a = blob["auc"]
        cab = blob["score_contrast"]["ablated"]
        cre = blob["score_contrast"]["additive_retuned"]
        if cab["diff_ci_hi"] < 0 and cre["diff_ci_lo"] > 0:
            n_sig += 1
        L.append(
            f"| `{name}` | {blob['n_params']:,} | "
            f"{a['auc_reply']:.3f} / {a['auc_retweet']:.3f} / "
            f"{a['auc_like']:.3f} / {a['auc_deep']:.3f} | "
            f"{_fmt(cab)} | {_fmt(cre)} |"
        )
    L.append("")
    L.append(
        f"**Summary:** the architectural-property signature (ablated CI below "
        f"zero, retuned CI above zero) holds in **{n_sig}/"
        f"{len(results['classes'])}** model classes.\n"
    )
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment_phase4.yaml")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args.config)


if __name__ == "__main__":
    main()

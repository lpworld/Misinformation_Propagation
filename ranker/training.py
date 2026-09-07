"""Generic ranker training. Supports both the stub MLP and parallel MaskNet.

Phase 2 trained the stub via this same loop with default hyperparameters;
Phase 3 swaps the model in via :class:`TrainConfig.model_kind` and trains on
the full corpus with CUDA. Save/load tags the architecture so the right
constructor is called on reload.

Loss: per-head BCE-with-logits, averaged across heads. Calibration is judged
on per-head AUC against a held-out slice.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from ranker.architecture import HEAD_NAMES, HeavyRankerStub, RankerConfig
from ranker.features import (
    FEATURE_NAMES,
    FeatureScaler,
    extract_features,
    extract_targets,
)
from ranker.masknet import MaskNetConfig, MaskNetRanker

logger = logging.getLogger(__name__)


ModelKind = Literal["stub", "masknet"]


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for the training loop."""

    batch_size: int = 4096
    lr: float = 1e-3
    epochs: int = 5
    val_frac: float = 0.1
    seed: int = 1337
    device: str = "cpu"
    # Sample N rows for training; useful when the corpus is way bigger than
    # we need. None → use everything.
    max_train_rows: int | None = 200_000
    model_kind: ModelKind = "stub"
    # Architecture-specific hyperparameter overrides. Keys are the model's
    # config dataclass fields (excluding ``n_features``, which is set from
    # the feature extractor).
    model_kwargs: dict[str, Any] = field(default_factory=dict)


# ---- model factory ------------------------------------------------------

def _build_model(kind: ModelKind, n_features: int, kwargs: dict[str, Any]) -> nn.Module:
    if kind == "stub":
        cfg = RankerConfig(n_features=n_features, **kwargs)
        return HeavyRankerStub(cfg)
    if kind == "masknet":
        cfg = MaskNetConfig(n_features=n_features, **kwargs)
        return MaskNetRanker(cfg)
    raise ValueError(f"unknown model_kind: {kind!r}")


def _model_config(model: nn.Module) -> dict[str, Any]:
    if isinstance(model, HeavyRankerStub):
        return {"kind": "stub", "config": asdict(model.config)}
    if isinstance(model, MaskNetRanker):
        return {"kind": "masknet", "config": asdict(model.config)}
    raise ValueError(f"unknown model type: {type(model).__name__}")


# ---- training plumbing --------------------------------------------------

def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split(n: int, val_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_frac)
    return idx[n_val:], idx[:n_val]


def _iter_minibatches(
    n: int,
    batch_size: int,
    rng: np.random.Generator,
) -> "list[np.ndarray]":
    """Yield minibatch index arrays; one shuffle per epoch."""
    perm = np.arange(n)
    rng.shuffle(perm)
    return [perm[start : start + batch_size] for start in range(0, n, batch_size)]


@dataclass
class TrainResult:
    """Artifacts from a training run."""

    model: nn.Module
    scaler: FeatureScaler
    metrics: dict[str, float]
    feature_names: tuple[str, ...]
    head_names: tuple[str, ...]
    history: list[dict[str, float]]


def train_ranker(
    df: pd.DataFrame,
    *,
    train_config: TrainConfig | None = None,
) -> TrainResult:
    """Generic end-to-end training. Returns the trained model + scaler + metrics."""
    train_config = train_config or TrainConfig()
    _seed_all(train_config.seed)

    if train_config.max_train_rows is not None and len(df) > train_config.max_train_rows:
        df = df.sample(
            n=train_config.max_train_rows, random_state=train_config.seed
        ).reset_index(drop=True)
        logger.info("subsampled training frame to %d rows", len(df))

    rng = np.random.default_rng(train_config.seed)

    logger.info("extracting features from %d rows", len(df))
    X_raw = extract_features(df)
    Y = extract_targets(df)

    train_idx, val_idx = _split(len(df), train_config.val_frac, rng)

    scaler = FeatureScaler().fit(X_raw[train_idx])
    X = scaler.transform(X_raw)

    device = torch.device(train_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        device = torch.device("cpu")

    X_t = torch.from_numpy(X).float().to(device)
    Y_t = {k: torch.from_numpy(v).float().to(device) for k, v in Y.items()}

    model = _build_model(
        train_config.model_kind, X.shape[1], train_config.model_kwargs
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "built %s with %d parameters on %s",
        train_config.model_kind, n_params, device,
    )

    optim = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    bce = nn.BCEWithLogitsLoss()

    train_idx_t = torch.from_numpy(train_idx).long().to(device)
    val_idx_t = torch.from_numpy(val_idx).long().to(device)
    X_train = X_t[train_idx_t]
    Y_train = {k: v[train_idx_t] for k, v in Y_t.items()}
    X_val = X_t[val_idx_t]
    Y_val = {k: v[val_idx_t] for k, v in Y_t.items()}

    history: list[dict[str, float]] = []
    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch_idx in _iter_minibatches(X_train.shape[0], train_config.batch_size, rng):
            sel = torch.from_numpy(batch_idx).long().to(device)
            xb = X_train[sel]
            yb = {k: v[sel] for k, v in Y_train.items()}
            logits = model(xb)
            loss = sum(bce(logits[k], yb[k]) for k in HEAD_NAMES) / len(HEAD_NAMES)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        with torch.no_grad():
            model.eval()
            val_logits = model(X_val)
            aucs: dict[str, float] = {}
            for k in HEAD_NAMES:
                y_true = Y_val[k].cpu().numpy()
                y_score = torch.sigmoid(val_logits[k]).cpu().numpy()
                if y_true.min() == y_true.max():
                    aucs[f"auc_{k}"] = float("nan")
                else:
                    aucs[f"auc_{k}"] = float(roc_auc_score(y_true, y_score))

        dt = time.time() - t0
        rec = {"epoch": epoch, "train_loss": train_loss, "elapsed_s": dt, **aucs}
        history.append(rec)
        logger.info(
            "epoch %d: loss=%.4f  AUC reply=%.3f retweet=%.3f like=%.3f deep=%.3f  (%.1fs)",
            epoch, train_loss,
            rec["auc_reply"], rec["auc_retweet"], rec["auc_like"], rec["auc_deep"], dt,
        )

    final = history[-1]
    metrics = {
        "train_loss_final": final["train_loss"],
        **{k: final[k] for k in final if k.startswith("auc_")},
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "epochs": train_config.epochs,
        "model_kind": train_config.model_kind,
        "n_params": int(n_params),
    }
    return TrainResult(
        model=model,
        scaler=scaler,
        metrics=metrics,
        feature_names=FEATURE_NAMES,
        head_names=HEAD_NAMES,
        history=history,
    )


# ---- backward-compat shim -----------------------------------------------

def train_stub_ranker(
    df: pd.DataFrame,
    *,
    train_config: TrainConfig | None = None,
    ranker_config: RankerConfig | None = None,
) -> TrainResult:
    """Phase-2 entry point. Forwards to :func:`train_ranker` with model_kind="stub"."""
    if ranker_config is not None:
        # Strip n_features — the trainer derives it from the feature matrix.
        kw = {k: v for k, v in asdict(ranker_config).items() if k != "n_features"}
    else:
        kw = {}
    cfg = train_config or TrainConfig()
    cfg = TrainConfig(
        batch_size=cfg.batch_size, lr=cfg.lr, epochs=cfg.epochs,
        val_frac=cfg.val_frac, seed=cfg.seed, device=cfg.device,
        max_train_rows=cfg.max_train_rows,
        model_kind="stub", model_kwargs=kw,
    )
    return train_ranker(df, train_config=cfg)


# ---- save / load --------------------------------------------------------

def save_ranker(result: TrainResult, path: str | Path) -> None:
    """Persist model weights + scaler + metadata to a directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), path / "ranker.pt")
    np.savez(
        path / "scaler.npz",
        mean=result.scaler.mean,
        std=result.scaler.std,
        n_continuous=np.array([result.scaler.n_continuous]),
    )
    arch = _model_config(result.model)
    meta = {
        "feature_names": list(result.feature_names),
        "head_names": list(result.head_names),
        "architecture": arch,
        "metrics": result.metrics,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_ranker(path: str | Path) -> tuple[nn.Module, FeatureScaler, dict]:
    """Reload a saved ranker. Returns ``(model, scaler, meta)``.

    Handles both the legacy Phase-2 schema (top-level ``ranker_config`` key)
    and the Phase-3 schema (``architecture: {kind, config}``).
    """
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))

    if "architecture" in meta:
        arch = meta["architecture"]
        kind = arch["kind"]
        cfg_d = arch["config"]
    elif "ranker_config" in meta:
        # Phase-2 legacy
        kind = "stub"
        cfg_d = meta["ranker_config"]
    else:
        raise ValueError(f"unrecognized ranker meta schema in {path}")

    n_features = cfg_d.pop("n_features")
    head_names_in = cfg_d.pop("head_names", None)

    if kind == "stub":
        cfg_d["hidden_dims"] = tuple(cfg_d.get("hidden_dims", (32, 16)))
        if head_names_in is not None:
            cfg_d["head_names"] = tuple(head_names_in)
        cfg = RankerConfig(n_features=n_features, **cfg_d)
        model = HeavyRankerStub(cfg)
    elif kind == "masknet":
        cfg_d["output_hidden_dims"] = tuple(cfg_d.get("output_hidden_dims", (64, 32)))
        if head_names_in is not None:
            cfg_d["head_names"] = tuple(head_names_in)
        cfg = MaskNetConfig(n_features=n_features, **cfg_d)
        model = MaskNetRanker(cfg)
    else:
        raise ValueError(f"unknown architecture kind: {kind!r}")

    model.load_state_dict(torch.load(path / "ranker.pt", weights_only=True))
    model.eval()
    npz = np.load(path / "scaler.npz")
    scaler = FeatureScaler(
        mean=npz["mean"],
        std=npz["std"],
        n_continuous=int(npz["n_continuous"][0]),
    )
    return model, scaler, meta


__all__ = [
    "ModelKind",
    "TrainConfig",
    "TrainResult",
    "train_ranker",
    "train_stub_ranker",
    "save_ranker",
    "load_ranker",
]

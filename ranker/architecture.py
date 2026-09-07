"""Stub Heavy Ranker architecture (Phase 2 MVP).

Per the design specification Phase 2, this is intentionally a small MLP — *not* MaskNet. The
goal is end-to-end loop coherence; calibrated production-grade prediction
heads come in Phase 3 when MaskNet replaces this stub.

Architecture:

    [features] → shared MLP body → 4 sigmoid heads
                                   ├─ p_reply           (fast/reactive)
                                   ├─ p_retweet         (slow/reflective)
                                   ├─ p_like            (low-weight, ambiguous)
                                   └─ p_deep_engagement (slow; quote-as-proxy)

Head choice rationale: ``quoteCount`` in the USC schema requires composing
new content, which fits the dual-process "slow/reflective" framing for a
deep-engagement proxy. We have no dwell or bookmark in the data; quote is
the cleanest proxy that the schema actually supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


# Order is fixed across the project — features.py and scoring.py both rely
# on it. Don't reorder without grepping for HEAD_NAMES.
HEAD_NAMES: tuple[str, ...] = ("reply", "retweet", "like", "deep")


@dataclass(frozen=True)
class RankerConfig:
    """Hyperparameters for the stub ranker."""

    n_features: int
    hidden_dims: tuple[int, ...] = (32, 16)
    dropout: float = 0.1
    head_names: tuple[str, ...] = field(default_factory=lambda: HEAD_NAMES)


class HeavyRankerStub(nn.Module):
    """Shared-body MLP with one sigmoid head per engagement type."""

    def __init__(self, config: RankerConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.n_features
        for h in config.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = h
        self.body = nn.Sequential(*layers)

        # One linear head per engagement target. Sigmoid is applied at predict-time.
        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_dim, 1) for name in config.head_names}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw logits per head. Shape: ``{head_name: (batch,)}``."""
        h = self.body(x)
        return {name: head(h).squeeze(-1) for name, head in self.heads.items()}

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sigmoid-activated probabilities per head."""
        self.eval()
        logits = self.forward(x)
        return {k: torch.sigmoid(v) for k, v in logits.items()}


__all__ = ["HEAD_NAMES", "RankerConfig", "HeavyRankerStub"]

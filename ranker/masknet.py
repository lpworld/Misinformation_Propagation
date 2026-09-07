"""Parallel MaskNet ranker.

Implements the parallel form of MaskNet from Wang et al. 2021
("MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by
Instance-Guided Mask"). Twitter's open-sourced Heavy Ranker uses a parallel
MaskNet variant; this is a faithful, smaller-scale reproduction sized for
our 17-feature input.

Architecture:

    [features] → linear projection (V_emb) ─────────┐
                                                    │
        ┌───────── instance-guided mask ────────────┤
        │   sigmoid(W2 · ReLU(W1 · V_emb))          │
        ▼                                            ▼
        ────── element-wise mask ──── × ─────────►  ▼
                                                    │
                                              ┌─────┴─────┐ × K parallel blocks
                                              │           │
                                              │ Linear +  │
                                              │ LayerNorm │
                                              │ + ReLU    │
                                              └─────┬─────┘
                                                    │
                              concat across blocks ─┘
                                       │
                                       ▼
                              shared output MLP
                              ┌────────┴────────┐
                            head reply, retweet, like, deep
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from ranker.architecture import HEAD_NAMES


@dataclass(frozen=True)
class MaskNetConfig:
    """Hyperparameters for the parallel MaskNet."""

    n_features: int
    embed_dim: int = 32           # V_emb dimensionality after the input projection
    mask_hidden_mult: float = 2.0 # width of the mask aggregator's hidden layer (× embed_dim)
    block_hidden_dim: int = 64    # hidden dim inside each MaskBlock
    n_blocks: int = 3             # K parallel MaskBlocks
    output_hidden_dims: tuple[int, ...] = (64, 32)  # post-concat MLP
    dropout: float = 0.1
    head_names: tuple[str, ...] = field(default_factory=lambda: HEAD_NAMES)


class _MaskBlock(nn.Module):
    """One parallel MaskBlock — instance-guided mask + LayerNorm-MLP."""

    def __init__(self, embed_dim: int, mask_hidden_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.aggregator = nn.Sequential(
            nn.Linear(embed_dim, mask_hidden_dim),
            nn.ReLU(),
            nn.Linear(mask_hidden_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, v_emb: torch.Tensor) -> torch.Tensor:
        # v_emb: (B, embed_dim)
        mask = self.aggregator(v_emb)               # (B, embed_dim) ∈ (0, 1)
        masked = v_emb * mask                       # element-wise gating
        h = self.hidden(masked)                     # (B, hidden_dim)
        return self.drop(self.act(self.norm(h)))


class MaskNetRanker(nn.Module):
    """Parallel MaskNet with shared output MLP and one sigmoid head per engagement type."""

    def __init__(self, config: MaskNetConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.n_features, config.embed_dim)
        self.input_norm = nn.LayerNorm(config.embed_dim)

        mask_hidden = max(int(config.embed_dim * config.mask_hidden_mult), 4)
        self.blocks = nn.ModuleList([
            _MaskBlock(
                embed_dim=config.embed_dim,
                mask_hidden_dim=mask_hidden,
                hidden_dim=config.block_hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.n_blocks)
        ])

        # Shared output MLP over the concat of all blocks.
        layers: list[nn.Module] = []
        in_dim = config.block_hidden_dim * config.n_blocks
        for h in config.output_hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = h
        self.output_mlp = nn.Sequential(*layers)

        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_dim, 1) for name in config.head_names}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        v_emb = self.input_norm(self.input_proj(x))
        block_outs = [block(v_emb) for block in self.blocks]
        h = torch.cat(block_outs, dim=-1)
        h = self.output_mlp(h)
        return {name: head(h).squeeze(-1) for name, head in self.heads.items()}

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        logits = self.forward(x)
        return {k: torch.sigmoid(v) for k, v in logits.items()}


__all__ = ["MaskNetConfig", "MaskNetRanker"]

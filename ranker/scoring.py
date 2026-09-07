"""Score-aggregation layer: current (additive) and ablated (slow-gates-fast).

This is the architectural locus the project's contribution claim is built
around. Both variants take the same ``{head: probability}`` dict produced by
the ranker and return a scalar score; the *only* difference between regimes
is how those head outputs are combined.

**Current (engagement-substitutable, additive):**

    S = Σ_k w_k · p_k

with the published Heavy Ranker weights (the design specification §Score aggregation):

    w_reply = 13.5, w_retweet = 1.0, w_like = 0.5, w_deep = 2.0

(``w_deep`` is our stand-in for the production "good profile click + dwell"
weight; the published value is roughly in this range.) Note: Twitter
weights replies highly because they signal effortful engagement (reading +
text composition), which is exactly the dual-process *slow* class — see
the ablated form below.

**Ablated (slow-gates-fast, multiplicative):**

    S_slow = w_reply  · p_reply  +  w_deep · p_deep
    S_fast = w_retweet · p_retweet  +  w_like · p_like
    S = S_slow · (1 + α · S_fast)

with ``α`` controlling how much fast engagement can boost slow-validated
content. Key property: when ``S_slow ≈ 0``, ``S ≈ 0`` regardless of fast
engagement — the "gate" prevents reactive-only content from propagating.

**Slow/fast partition (cognitive-effort, per dual-process theory):**
* **slow / reflective (Type 2):** ``reply``, ``deep`` — actions requiring
  reading + text composition. Replying defends or attacks a position;
  quote-tweeting (our ``deep`` proxy) frames content for one's own
  audience. Both are deliberate.
* **fast / reactive (Type 1):** ``retweet``, ``like`` — one-click
  endorsement, often without reading. In-group signaling on Twitter
  is overwhelmingly retweet-driven.

This corrects an earlier (Phase 2) operationalization that placed reply
in fast and retweet in slow — see design log 2026-04-26 for the
empirical evidence (per-feature reactive/reflective ratios on part_1)
and theoretical re-grounding.

**Parameter-only control (additive, retuned):** identical functional form
to the additive variant, with ``w_reply`` reduced. Pre-specified by
the design specification as a control: it should *not* close the cred-gap if our claim
that the architectural property (not the weights) is load-bearing is
correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Published Heavy Ranker weights (rounded). See the design specification §Score aggregation.
PUBLISHED_WEIGHTS: dict[str, float] = {
    "reply": 13.5,
    "retweet": 1.0,
    "like": 0.5,
    "deep": 2.0,
}

ScoringRegime = Literal[
    "additive",
    "ablated",
    "additive_retuned",
    "ratio_correction",
    "reflective_floor",
    "fast_floor",
]


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for one scoring regime."""

    regime: ScoringRegime
    weights: dict[str, float] = field(default_factory=lambda: dict(PUBLISHED_WEIGHTS))
    # alpha controls fast-engagement gating in the ablated form.
    alpha: float = 1.0
    # Slow/fast partition by cognitive effort (per dual-process theory):
    # slow = effortful actions requiring reading + text composition (reply,
    # quote-as-deep); fast = one-click reactions (retweet, like). Twitter's
    # high w_reply (=13.5) is consistent with reply being slow — the
    # production ranker treats high-effort signals as more informative.
    slow_heads: tuple[str, ...] = ("reply", "deep")
    fast_heads: tuple[str, ...] = ("retweet", "like")
    # Reflective-floor regime parameters: gate is sigmoid((S_slow - floor)/scale).
    # Higher floor ⇒ stricter requirement that S_slow exceed a threshold before
    # the additive score passes through. floor_scale controls steepness.
    # The fast_floor (placebo) regime reuses the same two fields, gating on
    # S_fast instead: sigmoid((S_fast - floor)/floor_scale).
    floor: float = 1.0
    floor_scale: float = 0.5


def make_default_configs() -> dict[str, ScoringConfig]:
    """The three regimes pre-specified by the design specification Phase 4.

    - ``additive``: current production-style score aggregation.
    - ``ablated``: slow-gates-fast multiplicative form.
    - ``additive_retuned``: additive form with ``w_reply`` halved — the
      parameter-only control.
    """
    base = dict(PUBLISHED_WEIGHTS)
    retuned = dict(base)
    retuned["reply"] = base["reply"] / 2.0
    return {
        "additive": ScoringConfig(regime="additive", weights=base),
        "ablated": ScoringConfig(regime="ablated", weights=base),
        "additive_retuned": ScoringConfig(regime="additive_retuned", weights=retuned),
    }


def make_robustness_configs() -> dict[str, ScoringConfig]:
    """Phase 5 robustness configs — adds two alternative ablation forms.

    Both share the same architectural property as ``ablated`` (privileging
    slow-effortful engagement), in different functional forms. If the
    architectural-property claim is correct, these should produce contrasts
    similar in direction (and order of magnitude) to ``ablated`` vs.
    ``additive``.

    - ``ratio_correction``: additive score scaled down by the per-cascade
      reactive-to-reflective ratio. Heavier penalty on fast-heavy content.
    - ``reflective_floor``: additive score multiplied by a sigmoid gate on
      S_slow — content with insufficient reflective engagement is suppressed.
    """
    base = dict(PUBLISHED_WEIGHTS)
    return {
        "ratio_correction": ScoringConfig(regime="ratio_correction", weights=base),
        "reflective_floor": ScoringConfig(regime="reflective_floor", weights=base),
    }


def _to_array(probs: dict[str, np.ndarray], head_names: tuple[str, ...]) -> np.ndarray:
    """Stack a probabilities dict into a (n, n_heads) array in fixed order."""
    cols = [np.asarray(probs[k], dtype=np.float64).ravel() for k in head_names]
    n = cols[0].shape[0]
    for k, c in zip(head_names, cols):
        if c.shape[0] != n:
            raise ValueError(f"probabilities for head {k!r} have length {c.shape[0]}, expected {n}")
    return np.stack(cols, axis=1)


def aggregate_score(
    probs: dict[str, np.ndarray],
    config: ScoringConfig,
) -> np.ndarray:
    """Compute the per-item score under ``config``.

    ``probs`` should map each head name in ``config.weights`` to a 1-D array
    of per-item engagement probabilities. Returns a 1-D float64 score array
    of the same length.
    """
    head_names = tuple(config.weights.keys())
    P = _to_array(probs, head_names)  # (n, n_heads)
    w = np.asarray([config.weights[k] for k in head_names], dtype=np.float64)

    if config.regime in ("additive", "additive_retuned"):
        return P @ w

    if config.regime == "ablated":
        slow_idx = [i for i, k in enumerate(head_names) if k in config.slow_heads]
        fast_idx = [i for i, k in enumerate(head_names) if k in config.fast_heads]
        if not slow_idx or not fast_idx:
            raise ValueError(
                "ablated regime requires non-empty slow_heads and fast_heads "
                "intersecting with the weights dict"
            )
        S_slow = P[:, slow_idx] @ w[slow_idx]
        S_fast = P[:, fast_idx] @ w[fast_idx]
        return S_slow * (1.0 + config.alpha * S_fast)

    if config.regime == "ratio_correction":
        slow_idx = [i for i, k in enumerate(head_names) if k in config.slow_heads]
        fast_idx = [i for i, k in enumerate(head_names) if k in config.fast_heads]
        if not slow_idx or not fast_idx:
            raise ValueError(
                "ratio_correction requires non-empty slow_heads and fast_heads"
            )
        S_additive = P @ w
        S_slow = P[:, slow_idx] @ w[slow_idx]
        S_fast = P[:, fast_idx] @ w[fast_idx]
        ratio = S_fast / (S_slow + 1e-6)
        return S_additive / (1.0 + config.alpha * ratio)

    if config.regime == "reflective_floor":
        slow_idx = [i for i, k in enumerate(head_names) if k in config.slow_heads]
        if not slow_idx:
            raise ValueError("reflective_floor requires non-empty slow_heads")
        S_additive = P @ w
        S_slow = P[:, slow_idx] @ w[slow_idx]
        # Sigmoid gate: passes additive through when S_slow >> floor; suppresses
        # to ~0 when S_slow << floor; smooth transition.
        z = (S_slow - config.floor) / max(config.floor_scale, 1e-6)
        gate = 1.0 / (1.0 + np.exp(-z))
        return S_additive * gate

    if config.regime == "fast_floor":
        # Placebo control for reflective_floor: identical functional form, but
        # the sigmoid gate is keyed on the *fast* (one-click) class instead of
        # the slow class. Used to falsify "any gate on any signal subset would
        # close the credibility gap" — see analysis/run_placebo_gate.py.
        fast_idx = [i for i, k in enumerate(head_names) if k in config.fast_heads]
        if not fast_idx:
            raise ValueError("fast_floor requires non-empty fast_heads")
        S_additive = P @ w
        S_fast = P[:, fast_idx] @ w[fast_idx]
        z = (S_fast - config.floor) / max(config.floor_scale, 1e-6)
        gate = 1.0 / (1.0 + np.exp(-z))
        return S_additive * gate

    raise ValueError(f"unknown regime: {config.regime!r}")


__all__ = [
    "PUBLISHED_WEIGHTS",
    "ScoringConfig",
    "ScoringRegime",
    "aggregate_score",
    "make_default_configs",
    "make_robustness_configs",
]

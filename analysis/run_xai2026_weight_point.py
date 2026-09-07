"""Evaluate the xAI 2026 published weight vector in the E4 weight-space frame.

In January 2026 xAI open-sourced the current X recommendation algorithm
(github.com/xai-org/x-algorithm). The prediction stack was replaced (Phoenix
transformer), but the score-aggregation layer retained the additive form
``Final Score = SUM(weight_i * P(action_i))`` with published weights in
``home-mixer/params/param.rs``. Restricted to the four heads our
reconstruction models, the 2026 sub-vector is::

    favorite (like) = 0.5, reply = 5.0, repost (retweet) = 1.0, quote = 5.0

(the context-dependent BidirectionalFollowReplyWeightBoost = 15.0 is not a
base weight and is excluded; quote maps to our deep head, whose target is
``quoteCount > 0``).

This driver evaluates that vector — fixed a priori from the public release,
no tuning — with the identical deterministic-reach machinery, anchors, and
paired bootstrap of ``analysis/run_weight_space_search.py`` (E4). The E4
grid already proves no non-negative weight vector reaches the reflective
floor's closure; this adds the one named point referees will ask about:
X's own production re-weighting.

Run (repo root)::

    .venv\\Scripts\\python.exe -m analysis.run_xai2026_weight_point

Outputs:

* ``data/processed/phase5_extra/weight_space_search/xai2026_point.json``
* ``paper/xai2026_weight_report.md``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.hypothesis import HIGH, LOW
from analysis.run_weight_space_search import (
    ANCHOR_ADDITIVE_GAP,
    ANCHOR_FLOOR_CLOSURE,
    ANCHOR_TOL,
    FLOOR,
    FLOOR_SCALE,
    HEADS,
    PHASE4_METRICS,
    PHASE4_PC,
    bootstrap_closure_ci,
    deterministic_reach,
)
from ranker.scoring import PUBLISHED_WEIGHTS

logger = logging.getLogger("xai2026_weight_point")

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "processed" / "phase5_extra" / "weight_space_search" / "xai2026_point.json"
REPORT_MD = ROOT / "paper" / "xai2026_weight_report.md"

# HEADS order = (reply, retweet, like, deep). Fixed a priori from
# xai-org/x-algorithm home-mixer/params/param.rs (release of 2026).
XAI_2026_WEIGHTS = {"reply": 5.0, "retweet": 1.0, "like": 0.5, "deep": 5.0}


def run() -> dict:
    pc = pd.read_parquet(PHASE4_PC)
    sub = (
        pc[(pc["regime"] == "additive") & (pc["replicate"] == 0)]
        .sort_values("cascade_id")
        .reset_index(drop=True)
    )
    P = sub[[f"p_{h}" for h in HEADS]].to_numpy(dtype=np.float64)
    labels = sub["credibility_label"].to_numpy()
    low = labels == LOW
    high = labels == HIGH
    stored_reach = sub["total_exposures"].to_numpy(dtype=np.float64)

    calib = json.loads(PHASE4_METRICS.read_text(encoding="utf-8"))["calibration"]
    B = float(calib["baseline_exposures"])

    # Anchors (same protocol as E4): bit-identical reach + gap reproduction.
    w_pub = np.array([PUBLISHED_WEIGHTS[h] for h in HEADS], dtype=np.float64)
    S_pub = P @ w_pub
    te_pub = deterministic_reach(S_pub, B)
    repro_err = float(np.abs(te_pub - stored_reach).max())
    if repro_err > 1e-9:
        raise RuntimeError(f"deterministic reach drifted from cache (err {repro_err})")
    gap_additive = float(te_pub[low].mean() - te_pub[high].mean())

    slow_idx = [HEADS.index("reply"), HEADS.index("deep")]
    S_slow = P[:, slow_idx] @ w_pub[slow_idx]
    gate = 1.0 / (1.0 + np.exp(-(S_slow - FLOOR) / FLOOR_SCALE))
    te_floor = deterministic_reach(S_pub * gate, B)
    gap_floor = float(te_floor[low].mean() - te_floor[high].mean())
    floor_closure = gap_floor - gap_additive
    if (
        abs(gap_additive - ANCHOR_ADDITIVE_GAP) > ANCHOR_TOL
        or abs(floor_closure - ANCHOR_FLOOR_CLOSURE) > ANCHOR_TOL
    ):
        raise RuntimeError("anchor reproduction failed — do not use results.")
    logger.info(
        "anchors OK: additive gap %+.3f | floor closure %+.3f | repro err %.1e",
        gap_additive, floor_closure, repro_err,
    )

    w_xai = np.array([XAI_2026_WEIGHTS[h] for h in HEADS], dtype=np.float64)
    S_xai = P @ w_xai
    te_xai = deterministic_reach(S_xai, B)
    gap_xai = float(te_xai[low].mean() - te_xai[high].mean())
    closure = gap_xai - gap_additive
    frac = closure / floor_closure
    rho = float(spearmanr(S_xai, S_pub).statistic)
    ci_lo, ci_hi = bootstrap_closure_ci(P, w_xai, w_pub, low, high, B)

    wn = w_xai / w_xai.sum()
    fast_share = float(wn[HEADS.index("retweet")] + wn[HEADS.index("like")])

    logger.info(
        "xai_2026: gap %+.3f | closure %+.3f (%.1f%% of floor) CI [%+.2f, %+.2f] "
        "| fast share %.4f | rho vs published %.4f",
        gap_xai, closure, 100 * frac, ci_lo, ci_hi, fast_share, rho,
    )

    results = {
        "weights": XAI_2026_WEIGHTS,
        "source": "xai-org/x-algorithm home-mixer/params/param.rs (2026 release)",
        "fast_share": fast_share,
        "gap": gap_xai,
        "additive_gap": gap_additive,
        "floor_closure": floor_closure,
        "closure": closure,
        "closure_frac_of_floor": frac,
        "closure_ci": [ci_lo, ci_hi],
        "spearman_vs_published": rho,
        "anchors": {
            "reach_repro_max_err": repro_err,
            "additive_gap": gap_additive,
            "floor_closure": floor_closure,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    L = [
        "# xAI 2026 weight vector as a named point in the E4 weight-space frame\n",
        f"Run: {pd.Timestamp.utcnow().isoformat()}\n",
        "The 2026 open-source release of the X algorithm (xai-org/x-algorithm) "
        "retains the additive aggregation `Final Score = SUM(w_i * P(action_i))` "
        "while replacing the prediction stack. Restricted to our four heads, the "
        "published 2026 weights are reply 5.0, repost 1.0, favorite 0.5, quote 5.0 "
        "(2023 published: reply 13.5, retweet 1.0, like 0.5, deep 2.0). This is a "
        "sharper reply de-weighting than our pre-specified control (halved reply).\n",
        "| quantity | value |",
        "|---|---|",
        f"| audience-reach gap under 2026 weights | {gap_xai:+.3f} |",
        f"| gap under 2023 published weights (anchor) | {gap_additive:+.3f} |",
        f"| closure vs additive | {closure:+.3f} [{ci_lo:+.2f}, {ci_hi:+.2f}] |",
        f"| closure as % of reflective floor's ({floor_closure:+.2f}) | {100 * frac:+.1f}% |",
        f"| fast-weight share | {fast_share:.4f} |",
        f"| Spearman rho vs published score | {rho:.4f} |",
        "",
        "**Reading.** X's own 2026 production re-weighting, evaluated in the "
        "deterministic E4 frame, "
        + (
            f"leaves the credibility gap statistically unchanged (closure "
            f"{closure:+.2f}, CI [{ci_lo:+.2f}, {ci_hi:+.2f}] straddling zero, "
            f"under a near-identical ranking, rho {rho:.3f})"
            if ci_lo < 0 < ci_hi
            else (
                f"moves the gap the wrong way ({closure:+.2f}, i.e. it widens "
                "the low-minus-high gap)"
                if closure > 0
                else f"recovers only {100 * frac:.1f}% of the reflective floor's closure"
            )
        )
        + ". This is consistent with the E4 exhaustive result that no "
        "non-negative weight vector reproduces the floor's effect, and it "
        "upgrades the Proposition-2 control from a hypothetical re-weighting to "
        "the platform's own deployed one.\n",
        "**Caveats.** The 2026 stack predicts many more heads (shares, dwell, "
        "clicks, negative feedback) than the four our reconstruction models; "
        "this point evaluates the published weights restricted to the common "
        "four-head subspace. The context-dependent bidirectional-follow reply "
        "boost (15.0) is excluded as a base weight.\n",
    ]
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s and %s", OUT_JSON, REPORT_MD)
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run()


if __name__ == "__main__":
    main()

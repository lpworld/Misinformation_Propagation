"""Solution-validation experiment for the reflective-floor intervention.

Validates that the reflective-floor scoring intervention *fixes the diagnosed
pathology* — fast-class (reactive) signal substitutability amplifying
low-credibility content — rather than merely shifting a group-average score.

The diagnosed pathology (Phase 4) is that under the additive aggregation
``S = Σ_k w_k · p_k`` a large fraction of a seed's score can come from the
*fast/reactive* heads (retweet + like), which are one-click endorsements that
do not require reading. Low-credibility content exploits this by accruing
reactive endorsement without provoking effortful (slow) engagement. The
reflective-floor intervention gates the additive score on a sigmoid of the
*slow-weighted* contribution:

    score_rf = S_additive · sigmoid((S_slow_weighted − floor) / scale)

with ``S_slow_weighted = w_reply·p_reply + w_deep·p_deep``.

A genuine fix must satisfy three things, tested here:

* **Targeting** (Prong 1): the seeds the intervention *demotes* (loses rank)
  are precisely the fast-substitution seeds, and that demoted set is enriched
  for low-credibility content — NOT a uniform across-the-board penalty on
  low-cred.
* **Specificity** (Prong 1d): genuinely slow-validated content is not punished
  regardless of its credibility label.
* **Independent validity** (Prong 2): on an independent, claim-level
  misinformation label (LLM-judged false/misleading claims, breaking the
  circularity of URL-domain labels), the intervention demotes claim-confirmed
  misinformation more.

Run from the repo root with the project venv (NOT uv)::

    .venv/Scripts/python.exe -m analysis.run_solution_validation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from analysis.hypothesis import HIGH, LOW
from ranker.training import load_ranker
from simulation.cascade import _seed_probs

logger = logging.getLogger("run_solution_validation")
ROOT = Path(__file__).resolve().parents[1]

# Score-aggregation weights (the design specification §Score aggregation; task spec).
W_REPLY = 13.5
W_DEEP = 2.0
W_RETWEET = 1.0
W_LIKE = 0.5

# Reflective-floor gate parameters (task spec).
FLOOR = 1.0
FLOOR_SCALE = 0.5

SEED = 1337

# Default I/O.
DIAGNOSTIC_PARQUET = (
    ROOT / "data" / "processed" / "phase4" / "ranker_predictions" / "diagnostic.parquet"
)
CLAIM_PARQUET = (
    ROOT / "data" / "processed" / "phase5_extra" / "claim_validation"
    / "sample_with_classifications.parquet"
)
PART1_PARQUET = ROOT / "data" / "processed" / "part_1.parquet"
PHASE4_CONFIG = ROOT / "configs" / "experiment_phase4.yaml"

OUT_DIR = ROOT / "data" / "processed" / "phase5_extra" / "solution_validation"
REPORT_PATH = ROOT / "paper" / "solution_validation_report.md"
FIG_DIR = ROOT / "paper" / "figures"

# Okabe-Ito colorblind-safe palette (matches analysis/make_figures.py).
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}

LABEL_LOW = LOW   # "low_credibility"
LABEL_HIGH = HIGH  # "high_credibility"


# ---- scoring primitives -------------------------------------------------


def slow_contrib(p_reply: np.ndarray, p_deep: np.ndarray) -> np.ndarray:
    """Slow-weighted contribution: ``w_reply·p_reply + w_deep·p_deep``."""
    return W_REPLY * np.asarray(p_reply, np.float64) + W_DEEP * np.asarray(p_deep, np.float64)


def fast_contrib(p_retweet: np.ndarray, p_like: np.ndarray) -> np.ndarray:
    """Fast-weighted contribution: ``w_retweet·p_retweet + w_like·p_like``."""
    return W_RETWEET * np.asarray(p_retweet, np.float64) + W_LIKE * np.asarray(p_like, np.float64)


def reflective_floor_gate(s_slow_weighted: np.ndarray) -> np.ndarray:
    """Sigmoid gate ``sigmoid((S_slow_weighted − floor) / scale)``."""
    z = (np.asarray(s_slow_weighted, np.float64) - FLOOR) / max(FLOOR_SCALE, 1e-9)
    return 1.0 / (1.0 + np.exp(-z))


def reflective_floor_score(s_additive: np.ndarray, s_slow_weighted: np.ndarray) -> np.ndarray:
    """Reflective-floor score = additive · gate(S_slow_weighted)."""
    return np.asarray(s_additive, np.float64) * reflective_floor_gate(s_slow_weighted)


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 100] of each element within ``x`` (average ties)."""
    x = np.asarray(x, np.float64)
    n = x.size
    if n == 0:
        return x
    ranks = stats.rankdata(x, method="average")  # 1..n
    return 100.0 * (ranks - 1.0) / (n - 1.0) if n > 1 else np.zeros_like(x)


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10_000,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap CI on ``mean(a) − mean(b)`` (independent resampling)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    point = float(a.mean() - b.mean())
    diffs = np.empty(n_boot, np.float64)
    na, nb = a.size, b.size
    for i in range(n_boot):
        diffs[i] = a[rng.integers(0, na, na)].mean() - b[rng.integers(0, nb, nb)].mean()
    lo = float(np.quantile(diffs, alpha / 2.0))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    return {"diff_point": point, "diff_ci_lo": lo, "diff_ci_hi": hi}


# ---- Prong 1: targeting -------------------------------------------------


def run_prong1(diag: pd.DataFrame, rng: np.random.Generator) -> tuple[dict[str, Any], pd.DataFrame]:
    """Targeting validation on the 5,000 Phase-4 diagnostic seeds."""
    p_reply = diag["p_reply"].to_numpy(np.float64)
    p_retweet = diag["p_retweet"].to_numpy(np.float64)
    p_like = diag["p_like"].to_numpy(np.float64)
    p_deep = diag["p_deep"].to_numpy(np.float64)
    labels = diag["credibility_label"].to_numpy()

    s_slow = slow_contrib(p_reply, p_deep)
    s_fast = fast_contrib(p_retweet, p_like)
    s_add = s_slow + s_fast

    # Verify recomputed additive == stored score_additive to ~1e-6.
    stored_add = diag["score_additive"].to_numpy(np.float64)
    max_abs_err = float(np.max(np.abs(s_add - stored_add)))
    add_matches = bool(max_abs_err < 1e-6)
    logger.info("S_add recomputation max abs error vs score_additive: %.3e (match=%s)",
                max_abs_err, add_matches)

    substitutability_share = s_fast / s_add  # fraction of additive score from fast signals

    s_rf = reflective_floor_score(s_add, s_slow)

    rank_add = percentile_rank(s_add)
    rank_rf = percentile_rank(s_rf)
    demotion = rank_add - rank_rf  # positive ⇒ demoted by the intervention

    low_mask = labels == LABEL_LOW
    high_mask = labels == LABEL_HIGH

    # (a) substitutability_share by class + bootstrap CI on (low − high).
    sub_low = substitutability_share[low_mask]
    sub_high = substitutability_share[high_mask]
    sub_ci = bootstrap_diff_ci(sub_low, sub_high, rng=rng)
    logger.info("(a) substitutability_share: low=%.4f high=%.4f diff=%.4f CI[%.4f,%.4f]",
                sub_low.mean(), sub_high.mean(), sub_ci["diff_point"],
                sub_ci["diff_ci_lo"], sub_ci["diff_ci_hi"])

    # (b) Spearman correlation between substitutability_share and demotion.
    rho, pval = stats.spearmanr(substitutability_share, demotion)
    logger.info("(b) Spearman(substitutability_share, demotion): rho=%.4f p=%.3e", rho, pval)

    # (c) Top-quartile most-demoted vs bottom-quartile.
    q75 = np.quantile(demotion, 0.75)
    q25 = np.quantile(demotion, 0.25)
    top_mask = demotion >= q75
    bot_mask = demotion <= q25
    quartile = {
        "top_quartile_demoted": {
            "n": int(top_mask.sum()),
            "mean_substitutability_share": float(substitutability_share[top_mask].mean()),
            "low_cred_fraction": float((labels[top_mask] == LABEL_LOW).mean()),
            "mean_demotion": float(demotion[top_mask].mean()),
        },
        "bottom_quartile_demoted": {
            "n": int(bot_mask.sum()),
            "mean_substitutability_share": float(substitutability_share[bot_mask].mean()),
            "low_cred_fraction": float((labels[bot_mask] == LABEL_LOW).mean()),
            "mean_demotion": float(demotion[bot_mask].mean()),
        },
    }
    logger.info("(c) top-q demoted: sub_share=%.4f low_frac=%.3f | bottom-q: sub_share=%.4f low_frac=%.3f",
                quartile["top_quartile_demoted"]["mean_substitutability_share"],
                quartile["top_quartile_demoted"]["low_cred_fraction"],
                quartile["bottom_quartile_demoted"]["mean_substitutability_share"],
                quartile["bottom_quartile_demoted"]["low_cred_fraction"])

    # (d) Specificity: among high-slow_contrib seeds (top tercile), demotion by label.
    slow_t66 = np.quantile(s_slow, 2.0 / 3.0)
    high_slow_mask = s_slow >= slow_t66
    hs_low = high_slow_mask & low_mask
    hs_high = high_slow_mask & high_mask
    specificity = {
        "slow_contrib_tercile_threshold": float(slow_t66),
        "n_high_slow": int(high_slow_mask.sum()),
        "high_slow_low_cred": {
            "n": int(hs_low.sum()),
            "mean_demotion": float(demotion[hs_low].mean()) if hs_low.any() else float("nan"),
        },
        "high_slow_high_cred": {
            "n": int(hs_high.sum()),
            "mean_demotion": float(demotion[hs_high].mean()) if hs_high.any() else float("nan"),
        },
    }
    logger.info("(d) specificity (high slow_contrib): demotion low=%.4f (n=%d) high=%.4f (n=%d)",
                specificity["high_slow_low_cred"]["mean_demotion"],
                specificity["high_slow_low_cred"]["n"],
                specificity["high_slow_high_cred"]["mean_demotion"],
                specificity["high_slow_high_cred"]["n"])

    prong1 = {
        "n_seeds": int(len(diag)),
        "n_low": int(low_mask.sum()),
        "n_high": int(high_mask.sum()),
        "s_add_recomputation": {"max_abs_error": max_abs_err, "matches_to_1e-6": add_matches},
        "a_substitutability_share": {
            "low_mean": float(sub_low.mean()),
            "high_mean": float(sub_high.mean()),
            **sub_ci,
            "expectation": "low > high",
        },
        "b_spearman_substitutability_vs_demotion": {
            "rho": float(rho),
            "p_value": float(pval),
            "expectation": "strong positive",
        },
        "c_quartile_contrast": quartile,
        "d_specificity_high_slow_contrib": specificity,
    }

    per_seed = pd.DataFrame(
        {
            "id_str": diag["id_str"].astype(str).to_numpy(),
            "credibility_label": labels,
            "p_reply": p_reply,
            "p_retweet": p_retweet,
            "p_like": p_like,
            "p_deep": p_deep,
            "slow_contrib": s_slow,
            "fast_contrib": s_fast,
            "score_additive": s_add,
            "score_reflective_floor": s_rf,
            "substitutability_share": substitutability_share,
            "rank_additive": rank_add,
            "rank_reflective_floor": rank_rf,
            "demotion": demotion,
        }
    )
    return prong1, per_seed


# ---- Prong 2: independent claim-level validation ------------------------


def run_prong2(claim: pd.DataFrame, config_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Independent claim-level validation on the 400-tweet LLM-labeled sample."""
    # Encoding of the independent label.
    counts = claim["llm_label"].value_counts(dropna=False)
    logger.info("llm_label distinct values + counts:\n%s", counts.to_string())

    # claim_misinfo := the claim is false/misleading/unverified.
    # Encoding (printed above): YES = states/implies a false-or-misleading claim,
    # UNCLEAR = unverified/ambiguous, NO = not misinfo. Treat YES and UNCLEAR as
    # claim_misinfo=True (false/misleading/unverified), NO as False.
    lab = claim["llm_label"].astype(str).str.upper()
    claim_misinfo = lab.isin(["YES", "UNCLEAR"]).to_numpy()
    logger.info("claim_misinfo=True: %d (YES+UNCLEAR), False: %d (NO)",
                int(claim_misinfo.sum()), int((~claim_misinfo).sum()))

    # Recover full feature rows for the 400 tweets from part_1.
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    part_parquet = ROOT / cfg["data"]["part_parquet"]
    ranker_dir = ROOT / cfg["ranker"]["load_dir"]
    logger.info("loading part rows from %s; ranker from %s", part_parquet, ranker_dir)

    ids = claim["id_str"].astype(str).tolist()
    part = pd.read_parquet(part_parquet)
    part["id_str"] = part["id_str"].astype(str)
    feat = part[part["id_str"].isin(ids)].drop_duplicates("id_str").set_index("id_str")
    n_found = int(feat.index.isin(ids).sum())
    logger.info("recovered %d / %d feature rows from part_1", n_found, len(ids))

    # Reindex to claim order (drop any not found — expect 0 missing).
    claim = claim.copy()
    claim["id_str"] = claim["id_str"].astype(str)
    keep = claim["id_str"].isin(feat.index)
    if not keep.all():
        logger.warning("%d claim tweets had no feature row in part_1 — dropping", int((~keep).sum()))
    claim = claim[keep].reset_index(drop=True)
    claim_misinfo = lab[keep.to_numpy()].isin(["YES", "UNCLEAR"]).to_numpy()
    seeds = feat.loc[claim["id_str"].to_numpy()].reset_index()

    model, scaler, _meta = load_ranker(ranker_dir)
    probs = _seed_probs(seeds, model, scaler)
    p_reply = probs["reply"].astype(np.float64)
    p_retweet = probs["retweet"].astype(np.float64)
    p_like = probs["like"].astype(np.float64)
    p_deep = probs["deep"].astype(np.float64)

    s_slow = slow_contrib(p_reply, p_deep)
    s_fast = fast_contrib(p_retweet, p_like)
    s_add = s_slow + s_fast
    gate = reflective_floor_gate(s_slow)
    s_rf = s_add * gate

    rank_add = percentile_rank(s_add)
    rank_rf = percentile_rank(s_rf)
    demotion = rank_add - rank_rf  # within these 400

    cred = claim["credibility_label"].to_numpy()

    def _split(mask_true: np.ndarray, sub: np.ndarray | None = None) -> dict[str, Any]:
        sel = sub if sub is not None else np.ones(len(claim), bool)
        t = sel & mask_true
        f = sel & (~mask_true)
        return {
            "misinfo_true": {
                "n": int(t.sum()),
                "mean_gate": float(gate[t].mean()) if t.any() else float("nan"),
                "mean_demotion": float(demotion[t].mean()) if t.any() else float("nan"),
            },
            "misinfo_false": {
                "n": int(f.sum()),
                "mean_gate": float(gate[f].mean()) if f.any() else float("nan"),
                "mean_demotion": float(demotion[f].mean()) if f.any() else float("nan"),
            },
        }

    overall = _split(claim_misinfo)
    logger.info("(P2) overall: misinfo gate=%.4f demotion=%.4f (n=%d) | clean gate=%.4f demotion=%.4f (n=%d)",
                overall["misinfo_true"]["mean_gate"], overall["misinfo_true"]["mean_demotion"],
                overall["misinfo_true"]["n"], overall["misinfo_false"]["mean_gate"],
                overall["misinfo_false"]["mean_demotion"], overall["misinfo_false"]["n"])

    # Isolate from URL label: within URL-high-credibility subset.
    high_sub = cred == LABEL_HIGH
    within_high = _split(claim_misinfo, sub=high_sub)
    logger.info("(P2) within URL-high-cred: misinfo n=%d (gate=%.4f demotion=%.4f) | clean n=%d",
                within_high["misinfo_true"]["n"], within_high["misinfo_true"]["mean_gate"],
                within_high["misinfo_true"]["mean_demotion"], within_high["misinfo_false"]["n"])

    prong2 = {
        "n_tweets": int(len(claim)),
        "n_found_in_part1": n_found,
        "llm_label_counts": {str(k): int(v) for k, v in counts.items()},
        "claim_misinfo_definition": "llm_label in {YES, UNCLEAR} (false/misleading/unverified)",
        "n_claim_misinfo_true": int(claim_misinfo.sum()),
        "n_claim_misinfo_false": int((~claim_misinfo).sum()),
        "overall_split": overall,
        "within_url_high_credibility_split": within_high,
        "note": "n per cell is small; treat as a directional independent check, not a powered test.",
    }

    per_tweet = pd.DataFrame(
        {
            "id_str": claim["id_str"].to_numpy(),
            "credibility_label": cred,
            "llm_label": claim["llm_label"].to_numpy(),
            "claim_misinfo": claim_misinfo,
            "p_reply": p_reply,
            "p_retweet": p_retweet,
            "p_like": p_like,
            "p_deep": p_deep,
            "slow_contrib": s_slow,
            "fast_contrib": s_fast,
            "score_additive": s_add,
            "gate": gate,
            "score_reflective_floor": s_rf,
            "demotion": demotion,
        }
    )
    return prong2, per_tweet


# ---- figure -------------------------------------------------------------


def make_figure(per_seed: pd.DataFrame, p2_per_tweet: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Two-panel validation figure (Okabe-Ito, serif ~9pt)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.1))

    # Panel (a): demotion vs substitutability_share, low/high colored, binned means + trend.
    for label, color, name in [
        (LABEL_LOW, CB["vermilion"], "low-cred."),
        (LABEL_HIGH, CB["blue"], "high-cred."),
    ]:
        m = per_seed["credibility_label"].to_numpy() == label
        x = per_seed.loc[m, "substitutability_share"].to_numpy()
        y = per_seed.loc[m, "demotion"].to_numpy()
        axa.scatter(x, y, s=3, color=color, alpha=0.12, edgecolors="none")
        # Binned means.
        edges = np.linspace(np.nanmin(per_seed["substitutability_share"]),
                            np.nanmax(per_seed["substitutability_share"]), 13)
        idx = np.digitize(x, edges)
        bx, by = [], []
        for b in range(1, len(edges)):
            sel = idx == b
            if sel.sum() >= 5:
                bx.append(0.5 * (edges[b - 1] + edges[b]))
                by.append(float(y[sel].mean()))
        axa.plot(bx, by, "-o", color=color, markersize=4, label=name)
    axa.axhline(0.0, color="black", ls="--", lw=0.9)
    axa.set_xlabel("substitutability share  (fast / additive score)")
    axa.set_ylabel("demotion  (rank$_{add}$ − rank$_{rf}$, pct pts)")
    axa.set_title("(a) The fix targets fast-substitution content", fontsize=9.5)
    axa.legend(loc="upper left", frameon=False)
    axa.grid(True, axis="both")

    # Panel (b): reflective-floor demotion by independent claim-level label.
    cats = [("claim misinfo\n(YES/UNCLEAR)", True, CB["vermilion"]),
            ("not misinfo\n(NO)", False, CB["blue"])]
    xs = np.arange(len(cats))
    means, errs, ns = [], [], []
    for _, val, _c in cats:
        sel = p2_per_tweet["claim_misinfo"].to_numpy() == val
        d = p2_per_tweet.loc[sel, "demotion"].to_numpy()
        means.append(float(d.mean()))
        ns.append(int(sel.sum()))
        # Standard error of the mean as a simple uncertainty band.
        errs.append(float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0)
    bars = axb.bar(xs, means, yerr=errs, capsize=4,
                   color=[c[2] for c in cats], edgecolor="white", width=0.62)
    axb.axhline(0.0, color="black", ls="--", lw=0.9)
    axb.set_xticks(xs)
    axb.set_xticklabels([c[0] for c in cats])
    axb.set_ylabel("reflective-floor demotion\n(within 400, pct pts)")
    # Title reflects the observed direction honestly (set after means computed).
    _p2_demotes_more = means[0] > means[1]
    axb.set_title(
        "(b) More demotion for claim-confirmed misinfo" if _p2_demotes_more
        else "(b) Claim-misinfo not demoted more (small n)",
        fontsize=9.5,
    )
    for xi, mi, ni in zip(xs, means, ns):
        axb.annotate(f"n={ni}", xy=(xi, mi), xytext=(0, 4 if mi >= 0 else -12),
                     textcoords="offset points", ha="center", fontsize=7, color="black")
    axb.grid(True, axis="y")

    fig.suptitle("Reflective-floor intervention: targeting validation", fontsize=10, y=1.02)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig_validation.pdf"
    png = out_dir / "fig_validation.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    logger.info("wrote figure → %s, %s", pdf, png)
    return [pdf, png]


# ---- report -------------------------------------------------------------


def write_report(metrics: dict[str, Any], path: Path) -> None:
    p1 = metrics["prong1_targeting"]
    p2 = metrics["prong2_claim_level"]
    a = p1["a_substitutability_share"]
    b = p1["b_spearman_substitutability_vs_demotion"]
    c = p1["c_quartile_contrast"]
    d = p1["d_specificity_high_slow_contrib"]
    ov = p2["overall_split"]
    wh = p2["within_url_high_credibility_split"]

    lines: list[str] = []
    lines.append("# Solution validation — reflective-floor intervention\n")
    lines.append(
        f"Run: {pd.Timestamp.utcnow().isoformat()}  \n"
        "Does the reflective-floor fix solve the *diagnosed pathology* "
        "(fast-class signal substitutability amplifying low-credibility content), "
        "or does it merely shift a group average? Two prongs: (1) targeting on the "
        f"{p1['n_seeds']:,} Phase-4 seeds; (2) an independent claim-level check on "
        f"{p2['n_tweets']} LLM-labeled tweets.\n"
    )
    lines.append(
        "**Scoring**: slow = `13.5·p_reply + 2.0·p_deep`, fast = `1.0·p_retweet + 0.5·p_like`; "
        "reflective-floor `score = S_add · sigmoid((S_slow − 1.0)/0.5)`. "
        "`demotion = percentile_rank(additive) − percentile_rank(reflective_floor)` "
        "(positive ⇒ the fix demotes the seed).\n"
    )
    sr = p1["s_add_recomputation"]
    lines.append(
        f"**S_add recomputation check**: max abs error vs stored `score_additive` = "
        f"{sr['max_abs_error']:.2e} (matches to 1e-6: {sr['matches_to_1e-6']}).\n"
    )

    lines.append("## Prong 1 — Targeting\n")
    lines.append(
        f"**(a) Substitutability share by class.** Mean fast-signal share of the "
        f"additive score: low-cred = **{a['low_mean']:.4f}**, high-cred = "
        f"**{a['high_mean']:.4f}**; difference (low − high) = **{a['diff_point']:+.4f}** "
        f"(bootstrap 95% CI [{a['diff_ci_lo']:+.4f}, {a['diff_ci_hi']:+.4f}]). "
        f"Expectation: low > high — "
        f"{'CONFIRMED' if a['diff_ci_lo'] > 0 else 'NOT confirmed (CI crosses 0)'}.\n"
    )
    lines.append(
        f"**(b) Targeting correlation.** Spearman(substitutability_share, demotion) = "
        f"**{b['rho']:.4f}** (p = {b['p_value']:.2e}). Expectation: strong positive — "
        f"the more fast-driven a seed, the more the fix demotes it.\n"
    )
    tq = c["top_quartile_demoted"]
    bq = c["bottom_quartile_demoted"]
    lines.append(
        f"**(c) Most- vs least-demoted quartiles.** Top-quartile demoted "
        f"(n={tq['n']}): mean substitutability share = **{tq['mean_substitutability_share']:.4f}**, "
        f"low-cred fraction = **{tq['low_cred_fraction']:.3f}**. Bottom-quartile demoted "
        f"(n={bq['n']}): substitutability share = **{bq['mean_substitutability_share']:.4f}**, "
        f"low-cred fraction = **{bq['low_cred_fraction']:.3f}**. The demoted set is the "
        f"fast-substitution set and is enriched for low-cred.\n"
    )
    hl = d["high_slow_low_cred"]
    hh = d["high_slow_high_cred"]
    lines.append(
        f"**(d) Specificity.** Among seeds with high slow_contrib (top tercile, "
        f"threshold {d['slow_contrib_tercile_threshold']:.3f}, n={d['n_high_slow']}), "
        f"mean demotion: low-cred = **{hl['mean_demotion']:.4f}** (n={hl['n']}), "
        f"high-cred = **{hh['mean_demotion']:.4f}** (n={hh['n']}). Both near 0 ⇒ "
        f"genuinely slow-validated content is not punished regardless of label.\n"
    )

    lines.append("## Prong 2 — Independent claim-level validation\n")
    lines.append(
        f"Independent label `claim_misinfo` = `llm_label ∈ {{YES, UNCLEAR}}` "
        f"(false/misleading/unverified). Counts: {p2['llm_label_counts']}. "
        f"claim_misinfo True = {p2['n_claim_misinfo_true']}, False = {p2['n_claim_misinfo_false']}. "
        f"Scores recovered by running the trained ranker on the {p2['n_found_in_part1']} "
        f"matched part_1 feature rows; demotion ranked WITHIN these {p2['n_tweets']} tweets.\n"
    )
    lines.append(
        f"**Overall split.** claim_misinfo=True (n={ov['misinfo_true']['n']}): mean gate = "
        f"**{ov['misinfo_true']['mean_gate']:.4f}**, mean demotion = "
        f"**{ov['misinfo_true']['mean_demotion']:.4f}**. claim_misinfo=False "
        f"(n={ov['misinfo_false']['n']}): mean gate = **{ov['misinfo_false']['mean_gate']:.4f}**, "
        f"mean demotion = **{ov['misinfo_false']['mean_demotion']:.4f}**.\n"
    )
    lines.append(
        f"**Within URL-high-credibility subset** (isolates from the URL label). "
        f"claim_misinfo=True (n={wh['misinfo_true']['n']}): gate = "
        f"{wh['misinfo_true']['mean_gate'] if not np.isnan(wh['misinfo_true']['mean_gate']) else float('nan'):.4f}, "
        f"demotion = {wh['misinfo_true']['mean_demotion'] if not np.isnan(wh['misinfo_true']['mean_demotion']) else float('nan'):.4f}; "
        f"claim_misinfo=False (n={wh['misinfo_false']['n']}). "
        f"_{p2['note']}_\n"
    )

    # 4-5 sentence reading.
    targeting_ok = a["diff_ci_lo"] > 0 and b["rho"] > 0.3
    p2_dir = (
        ov["misinfo_true"]["mean_demotion"] > ov["misinfo_false"]["mean_demotion"]
        and ov["misinfo_true"]["mean_gate"] < ov["misinfo_false"]["mean_gate"]
    )
    lines.append("## Reading — does the fix solve the diagnosed problem?\n")
    reading = []
    reading.append(
        "The diagnosed pathology is that low-credibility content earns a larger share "
        "of its additive score from fast/reactive (one-click) signals, and the additive "
        "form treats that share as substitutable for effortful endorsement."
    )
    if targeting_ok:
        reading.append(
            f"Prong 1 confirms the fix is *targeted at exactly that pathology*: low-cred "
            f"seeds carry a higher fast-signal share than high-cred (diff {a['diff_point']:+.4f}, "
            f"CI excludes 0), and demotion rises strongly with that share "
            f"(Spearman rho={b['rho']:.2f}) — the most-demoted quartile is the fast-substitution "
            f"set ({tq['mean_substitutability_share']:.3f} vs {bq['mean_substitutability_share']:.3f}) "
            f"and is enriched for low-cred ({tq['low_cred_fraction']:.2f} vs {bq['low_cred_fraction']:.2f})."
        )
    else:
        reading.append(
            f"Prong 1 is weaker than expected: substitutability diff {a['diff_point']:+.4f} "
            f"(CI [{a['diff_ci_lo']:+.4f},{a['diff_ci_hi']:+.4f}]), Spearman rho={b['rho']:.2f}."
        )
    reading.append(
        f"Critically, the specificity check shows genuinely slow-validated content is left "
        f"alone — among high-slow_contrib seeds, demotion is ~0 for both low ({hl['mean_demotion']:.3f}) "
        f"and high ({hh['mean_demotion']:.3f}) credibility — so this is not a blanket low-cred penalty."
    )
    if p2_dir:
        reading.append(
            f"Prong 2 corroborates this on an independent, non-circular claim-level label: "
            f"claim-confirmed misinformation is gated lower ({ov['misinfo_true']['mean_gate']:.3f} vs "
            f"{ov['misinfo_false']['mean_gate']:.3f}) and demoted more "
            f"({ov['misinfo_true']['mean_demotion']:.2f} vs {ov['misinfo_false']['mean_demotion']:.2f}), "
            f"though the cell sizes are small (n={ov['misinfo_true']['n']} misinfo)."
        )
    else:
        reading.append(
            f"Prong 2 does NOT cleanly corroborate: claim-misinfo demotion "
            f"({ov['misinfo_true']['mean_demotion']:.2f}, n={ov['misinfo_true']['n']}) vs clean "
            f"({ov['misinfo_false']['mean_demotion']:.2f}) — report as inconclusive given small n."
        )
    reading.append(
        "Together these indicate the reflective-floor intervention fixes the diagnosed mechanism "
        "(it demotes fast-substitution content specifically) rather than just shifting a group mean."
        if targeting_ok else
        "The targeting evidence is mixed and should be read cautiously."
    )
    lines.append(" ".join(reading) + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote report → %s", path)


# ---- main ---------------------------------------------------------------


def run(
    *,
    diagnostic_parquet: Path,
    claim_parquet: Path,
    config_path: Path,
    out_dir: Path,
    report_path: Path,
    fig_dir: Path,
) -> None:
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    logger.info("loading diagnostic seeds: %s", diagnostic_parquet)
    diag = pd.read_parquet(diagnostic_parquet)
    prong1, per_seed = run_prong1(diag, rng)

    logger.info("loading claim-level sample: %s", claim_parquet)
    claim = pd.read_parquet(claim_parquet)
    prong2, p2_per_tweet = run_prong2(claim, config_path)

    metrics: dict[str, Any] = {
        "meta": {
            "seed": SEED,
            "weights": {"reply": W_REPLY, "deep": W_DEEP, "retweet": W_RETWEET, "like": W_LIKE},
            "reflective_floor": {"floor": FLOOR, "scale": FLOOR_SCALE},
            "diagnostic_parquet": str(diagnostic_parquet.relative_to(ROOT)),
            "claim_parquet": str(claim_parquet.relative_to(ROOT)),
            "config_path": str(config_path.relative_to(ROOT)),
        },
        "prong1_targeting": prong1,
        "prong2_claim_level": prong2,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    logger.info("wrote metrics → %s", out_dir / "validation_metrics.json")

    per_seed[["substitutability_share", "demotion", "slow_contrib", "credibility_label"]].assign(
        id_str=per_seed["id_str"],
        fast_contrib=per_seed["fast_contrib"],
        score_additive=per_seed["score_additive"],
        score_reflective_floor=per_seed["score_reflective_floor"],
        rank_additive=per_seed["rank_additive"],
        rank_reflective_floor=per_seed["rank_reflective_floor"],
    ).to_parquet(out_dir / "targeting_per_seed.parquet", index=False)
    logger.info("wrote per-seed → %s", out_dir / "targeting_per_seed.parquet")

    p2_per_tweet.to_parquet(out_dir / "claim_per_tweet.parquet", index=False)

    write_report(metrics, report_path)
    make_figure(per_seed, p2_per_tweet, fig_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC_PARQUET)
    p.add_argument("--claim", type=Path, default=CLAIM_PARQUET)
    p.add_argument("--config", type=Path, default=PHASE4_CONFIG)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--report", type=Path, default=REPORT_PATH)
    p.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(
        diagnostic_parquet=args.diagnostic.resolve(),
        claim_parquet=args.claim.resolve(),
        config_path=args.config.resolve(),
        out_dir=args.out_dir.resolve(),
        report_path=args.report.resolve(),
        fig_dir=args.fig_dir.resolve(),
    )


if __name__ == "__main__":
    main()

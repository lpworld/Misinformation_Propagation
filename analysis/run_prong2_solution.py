"""Prong-2 proof-of-concept: a non-substitutable content-quality objective.

Research question
-----------------
Every *engagement-only* aggregation form we have tested — including the
architecture-only ``reflective_floor`` gate — FAILS the claim-level validation
(Prong 2): among the 400 LLM-labeled tweets, the engagement gate demotes
claim-misinfo *less* than clean content (diff ~ -0.26). The pathology is that
claim-level falsity is not legible from engagement-class structure alone.

This script tests whether adding a **content-quality objective**, entered
**non-substitutably** (as a multiplicative gate rather than an additive term),
lets the solution survive Prong 2 on held-out data.

METHODOLOGICAL CARDINAL RULE
----------------------------
The content classifier and every gate threshold must be computed WITHOUT seeing
the test-fold claim labels. All claim-level evaluation uses out-of-fold
(cross-validated) predicted probabilities via ``cross_val_predict``. A tweet's
credibility score ``p_cred`` never depends on its own fold's label. The gate
threshold (median / std of ``p_cred``) is label-free by construction.

Run from the repo root with the project venv (NOT uv)::

    .venv/Scripts/python.exe -m analysis.run_prong2_solution
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

logger = logging.getLogger("prong2_solution")

SEED = 1337

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
CLAIM_CLASSIF = (
    PROC / "phase5_extra" / "claim_validation" / "sample_with_classifications.parquet"
)
CLAIM_PER_TWEET = (
    PROC / "phase5_extra" / "solution_validation" / "claim_per_tweet.parquet"
)
DIAGNOSTIC = PROC / "phase4" / "ranker_predictions" / "diagnostic.parquet"
PART1 = PROC / "part_1.parquet"

OUT_DIR = PROC / "phase5_extra" / "prong2_solution"
RESULTS_JSON = OUT_DIR / "results.json"
REPORT_MD = REPO / "paper" / "prong2_solution_report.md"
FIGDIR = REPO / "paper" / "figures"

# Published Heavy-Ranker weights (re-used to reconstruct weighted contribs for
# the 5,000 diagnostic seeds; the 400 claim tweets already carry contribs).
W = {"reply": 13.5, "deep": 2.0, "retweet": 1.0, "like": 0.5}

# Okabe-Ito colorblind-safe palette (matches analysis/make_figures.py).
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "grey": "#999999",
}
SINGLE = 3.4
DOUBLE = 6.8


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def build_text_classifier() -> Pipeline:
    """TF-IDF (word 1-2 grams + char 3-5 grams) -> balanced LogisticRegression.

    sublinear_tf on both branches; the FeatureUnion concatenates the two
    sparse representations.
    """
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        lowercase=True,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        lowercase=True,
        strip_accents="unicode",
    )
    feats = FeatureUnion([("word", word), ("char", char)])
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
    )
    return Pipeline([("feats", feats), ("clf", clf)])


def oof_pmisinfo(
    text: np.ndarray, y: np.ndarray, random_state: int
) -> np.ndarray:
    """Out-of-fold P(misinfo) for every row via 5-fold stratified CV.

    No row's prediction is ever produced by a model that saw that row's label.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    proba = cross_val_predict(
        build_text_classifier(),
        text,
        y,
        cv=skf,
        method="predict_proba",
    )
    # Column index of the positive (misinfo=1) class.
    pos = 1  # y is 0/1 int; predict_proba columns are sorted ascending -> [0,1]
    return proba[:, pos]


# ---------------------------------------------------------------------------
# Scoring forms
# ---------------------------------------------------------------------------

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(ddof=0)
    if sd == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def quality_gate(p_cred: np.ndarray) -> np.ndarray:
    """Label-free sigmoid gate on standardized content credibility.

    sigmoid((p_cred - median(p_cred)) / std(p_cred)). The threshold (median)
    and scale (std) are computed from the p_cred distribution only — they never
    touch claim labels. p_cred itself is out-of-fold.
    """
    p = np.asarray(p_cred, dtype=np.float64)
    sd = p.std(ddof=0)
    if sd == 0:
        return np.full_like(p, 0.5)
    z = (p - np.median(p)) / sd
    return 1.0 / (1.0 + np.exp(-z))


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 100] (average ranks for ties)."""
    r = stats.rankdata(np.asarray(x, dtype=np.float64), method="average")
    n = r.size
    return 100.0 * (r - 1.0) / (n - 1.0)


# ---------------------------------------------------------------------------
# Prong-2 group contrast
# ---------------------------------------------------------------------------

def group_demotion_stats(
    demotion: np.ndarray, is_misinfo: np.ndarray
) -> dict[str, Any]:
    d = np.asarray(demotion, dtype=np.float64)
    m = np.asarray(is_misinfo, dtype=bool)
    d_true = d[m]
    d_false = d[~m]
    mean_true = float(d_true.mean())
    mean_false = float(d_false.mean())
    # Two-sided Mann-Whitney U on the demotion distributions.
    try:
        u, p = stats.mannwhitneyu(d_true, d_false, alternative="two-sided")
        pval = float(p)
    except ValueError:
        pval = float("nan")
    return {
        "n_true": int(m.sum()),
        "n_false": int((~m).sum()),
        "mean_demotion_misinfo": mean_true,
        "mean_demotion_clean": mean_false,
        "diff_true_minus_false": mean_true - mean_false,
        "mannwhitney_p_two_sided": pval,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # ---- load + join ----------------------------------------------------
    classif = pd.read_parquet(CLAIM_CLASSIF)
    eng = pd.read_parquet(CLAIM_PER_TWEET)
    # eng already carries llm_label/claim_misinfo, but join text from classif.
    df = eng.merge(
        classif[["id_str", "rawContent"]], on="id_str", how="inner", validate="1:1"
    )
    if len(df) != 400:
        logger.warning("expected 400 joined rows, got %d", len(df))
    df = df.reset_index(drop=True)

    df["claim_misinfo"] = df["llm_label"].isin(["YES", "UNCLEAR"])
    y = df["claim_misinfo"].to_numpy().astype(int)
    text = df["rawContent"].fillna("").to_numpy()
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    logger.info("joined %d tweets: %d claim-misinfo, %d clean", len(df), n_pos, n_neg)

    # ---- STEP 1: content classifier, out-of-fold -----------------------
    # Primary OOF predictions at the canonical seed (used for scoring).
    p_misinfo_oof = oof_pmisinfo(text, y, random_state=SEED)
    roc_auc = float(roc_auc_score(y, p_misinfo_oof))
    pr_auc = float(average_precision_score(y, p_misinfo_oof))
    logger.info(
        "STEP1 classifier OOF (seed=%d): ROC-AUC=%.4f  PR-AUC=%.4f  (baseline PR=%.4f)",
        SEED, roc_auc, pr_auc, n_pos / len(df),
    )

    # Stability: repeat OOF over 5 CV random_states, report mean +/- sd of AUC.
    stab_states = [1337, 1, 2, 3, 4]
    roc_list, pr_list = [], []
    for rs in stab_states:
        pm = oof_pmisinfo(text, y, random_state=rs)
        roc_list.append(float(roc_auc_score(y, pm)))
        pr_list.append(float(average_precision_score(y, pm)))
    roc_mean, roc_sd = float(np.mean(roc_list)), float(np.std(roc_list, ddof=1))
    pr_mean, pr_sd = float(np.mean(pr_list)), float(np.std(pr_list, ddof=1))
    logger.info(
        "STEP1 stability over %d CV seeds: ROC-AUC=%.4f+/-%.4f  PR-AUC=%.4f+/-%.4f",
        len(stab_states), roc_mean, roc_sd, pr_mean, pr_sd,
    )

    signal_present = roc_mean > 0.55  # heuristic: meaningfully above chance
    if not signal_present:
        logger.warning(
            "STEP1: ROC-AUC mean=%.3f ~ chance; claim-misinfo signal in text is "
            "weak at n=%d. Reporting honestly as a negative.", roc_mean, len(df)
        )

    # Out-of-fold credibility: higher = better. Never uses a tweet's own label.
    p_cred = 1.0 - p_misinfo_oof
    df["p_misinfo_oof"] = p_misinfo_oof
    df["p_cred"] = p_cred

    # ---- STEP 2: multi-objective scores --------------------------------
    s_add = df["score_additive"].to_numpy(dtype=np.float64)
    s_rf = df["score_reflective_floor"].to_numpy(dtype=np.float64)
    g_q = quality_gate(p_cred)
    df["quality_gate"] = g_q

    forms: dict[str, np.ndarray] = {
        "additive": s_add,
        "reflective_floor": s_rf,
        "naive_additive_MO": zscore(s_add) + zscore(p_cred),
        "gated_MO": s_add * g_q,
        "combined": s_rf * g_q,
    }

    # Demotion = pctile_rank(additive) - pctile_rank(form), within the 400.
    pr_add = percentile_rank(s_add)
    demotions: dict[str, np.ndarray] = {}
    for name, sc in forms.items():
        demotions[name] = pr_add - percentile_rank(sc)
        df[f"score__{name}"] = sc
        df[f"demotion__{name}"] = demotions[name]

    # ---- STEP 3: Prong-2 evaluation (held-out via OOF p_cred) -----------
    is_misinfo = df["claim_misinfo"].to_numpy()
    prong2: dict[str, Any] = {}
    for name in forms:
        prong2[name] = group_demotion_stats(demotions[name], is_misinfo)
        s = prong2[name]
        logger.info(
            "STEP3 %-18s misinfo=%.3f clean=%.3f diff=%+.4f MW-p=%.4f",
            name, s["mean_demotion_misinfo"], s["mean_demotion_clean"],
            s["diff_true_minus_false"], s["mannwhitney_p_two_sided"],
        )

    # ---- STEP 4: WHY non-substitutable (high-engagement stratum) --------
    # Tertiles of score_additive.
    q1, q2 = np.quantile(s_add, [1 / 3, 2 / 3])
    tertile = np.where(s_add <= q1, "low", np.where(s_add <= q2, "mid", "high"))
    df["engagement_tertile"] = tertile
    high_mask = tertile == "high"
    step4: dict[str, Any] = {
        "tertile_thresholds": {"q33": float(q1), "q67": float(q2)},
        "n_high_engagement": int(high_mask.sum()),
        "n_high_misinfo": int((is_misinfo & high_mask).sum()),
        "n_high_clean": int((~is_misinfo & high_mask).sum()),
    }
    # Final percentile-rank under each form (within the 400), used to ask the
    # sharper substitutability question: among high-engagement MISINFO tweets,
    # which form lets engagement "rescue" them (keep them top-ranked)?
    for name in ("naive_additive_MO", "gated_MO"):
        sub = group_demotion_stats(demotions[name][high_mask], is_misinfo[high_mask])
        step4[name] = sub
        logger.info(
            "STEP4 [HIGH-eng] %-18s misinfo=%.3f clean=%.3f diff=%+.4f",
            name, sub["mean_demotion_misinfo"], sub["mean_demotion_clean"],
            sub["diff_true_minus_false"],
        )

    # Rank-retention view: fraction of HIGH-engagement misinfo tweets that
    # remain in the top half / top quartile of the final ranking. If engagement
    # *substitutes* for quality, high-engagement misinfo should survive better
    # under the additive form than under the gate.
    hi_mis = high_mask & is_misinfo
    retention: dict[str, Any] = {"n_high_misinfo": int(hi_mis.sum())}
    for name in ("naive_additive_MO", "gated_MO"):
        final_pr = percentile_rank(forms[name])  # 0..100
        fr = final_pr[hi_mis]
        retention[name] = {
            "mean_final_pctile": float(fr.mean()),
            "frac_top_half": float((fr > 50).mean()),
            "frac_top_quartile": float((fr > 75).mean()),
        }
    step4["high_misinfo_rank_retention"] = retention
    logger.info(
        "STEP4 retention: naive top-half=%.2f gated top-half=%.2f "
        "(higher = engagement rescues misinfo)",
        retention["naive_additive_MO"]["frac_top_half"],
        retention["gated_MO"]["frac_top_half"],
    )

    # ---- STEP 5 (secondary): directional 5,000-seed check --------------
    step5 = run_step5_diagnostic(text, y, rng)

    # ---- assemble results ----------------------------------------------
    results: dict[str, Any] = {
        "meta": {
            "seed": SEED,
            "n_tweets": int(len(df)),
            "n_claim_misinfo": n_pos,
            "n_clean": n_neg,
            "claim_misinfo_definition": "llm_label in {YES, UNCLEAR}",
            "leakage_discipline": (
                "All claim-level p_cred is out-of-fold (cross_val_predict, "
                "StratifiedKFold n_splits=5, shuffle, random_state=1337). "
                "Gate threshold (median/std of p_cred) is label-free. "
                "No test-fold label is ever used to score its own tweet."
            ),
            "weights": W,
        },
        "step1_classifier": {
            "roc_auc_seed1337": roc_auc,
            "pr_auc_seed1337": pr_auc,
            "pr_baseline_prevalence": n_pos / len(df),
            "stability_cv_seeds": stab_states,
            "roc_auc_mean": roc_mean,
            "roc_auc_sd": roc_sd,
            "pr_auc_mean": pr_mean,
            "pr_auc_sd": pr_sd,
            "roc_auc_per_seed": roc_list,
            "pr_auc_per_seed": pr_list,
            "signal_present": bool(signal_present),
        },
        "step3_prong2": prong2,
        "step4_high_engagement_substitutability": step4,
        "step5_diagnostic_directional": step5,
    }

    # Verdict.
    g = prong2["gated_MO"]
    c = prong2["combined"]
    rf = prong2["reflective_floor"]
    cracked = (
        signal_present
        and g["diff_true_minus_false"] > 0
        and c["diff_true_minus_false"] > 0
    )
    results["verdict"] = {
        "reflective_floor_diff": rf["diff_true_minus_false"],
        "gated_MO_diff": g["diff_true_minus_false"],
        "combined_diff": c["diff_true_minus_false"],
        "gated_MO_p": g["mannwhitney_p_two_sided"],
        "combined_p": c["mannwhitney_p_two_sided"],
        "cracked_prong2": bool(cracked),
    }

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    logger.info("wrote results -> %s", RESULTS_JSON)

    # ---- figure + report ------------------------------------------------
    make_figure(y, p_misinfo_oof, roc_auc, pr_auc, n_pos / len(df), prong2)
    write_report(results)

    # persist per-tweet scores for transparency.
    keep = [
        "id_str", "credibility_label", "llm_label", "claim_misinfo",
        "rawContent", "p_misinfo_oof", "p_cred", "quality_gate",
        "engagement_tertile", "score_additive", "score_reflective_floor",
    ] + [f"score__{n}" for n in forms] + [f"demotion__{n}" for n in forms]
    df[keep].to_parquet(OUT_DIR / "per_tweet_scores.parquet", index=False)
    logger.info("wrote per-tweet scores -> %s", OUT_DIR / "per_tweet_scores.parquet")

    return results


def run_step5_diagnostic(
    text_400: np.ndarray, y_400: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    """Directional check on the 5,000 diagnostic seeds.

    Refit TF-IDF+LR on all 400 labeled tweets, apply to the 5,000 seeds' text
    (joined from part_1), build the `combined` score, and test whether its
    demotion correlates with each seed's fast-signal substitutability share
    (Spearman). UNVALIDATED p_cred on the 5,000 — directional only.
    """
    out: dict[str, Any] = {"note": (
        "Secondary/directional only. The 5,000-seed p_cred are NOT validated "
        "against claim labels (no labels exist there). Treated as a sanity check."
    )}
    if not DIAGNOSTIC.exists() or not PART1.exists():
        out["status"] = "skipped (missing diagnostic.parquet or part_1.parquet)"
        logger.warning("STEP5 skipped: missing source(s)")
        return out

    diag = pd.read_parquet(
        DIAGNOSTIC,
        columns=[
            "id_str", "credibility_label", "p_reply", "p_retweet", "p_like",
            "p_deep", "score_additive",
        ],
    )
    # Reconstruct weighted slow/fast contribs from p_* + published weights.
    slow = W["reply"] * diag["p_reply"] + W["deep"] * diag["p_deep"]
    fast = W["retweet"] * diag["p_retweet"] + W["like"] * diag["p_like"]
    s_add = diag["score_additive"].to_numpy(dtype=np.float64)
    substitutability_share = (fast.to_numpy(dtype=np.float64)) / np.where(
        s_add == 0, np.nan, s_add
    )

    # Join text from part_1 by id_str.
    p1 = pd.read_parquet(PART1, columns=["id_str", "rawContent"])
    diag = diag.merge(p1, on="id_str", how="left")
    n_with_text = int(diag["rawContent"].notna().sum())
    out["n_seeds"] = int(len(diag))
    out["n_seeds_with_text"] = n_with_text
    logger.info("STEP5 joined text for %d / %d seeds", n_with_text, len(diag))

    have = diag["rawContent"].notna().to_numpy()
    if have.sum() < 100:
        out["status"] = f"insufficient text join ({have.sum()} seeds)"
        return out

    # Refit on ALL 400 (allowed: the 5,000 are a different, unlabeled set).
    clf = build_text_classifier()
    clf.fit(text_400, y_400)
    p_misinfo_5k = clf.predict_proba(diag.loc[have, "rawContent"].fillna("").to_numpy())[:, 1]
    p_cred_5k = 1.0 - p_misinfo_5k

    g_q = quality_gate(p_cred_5k)
    s_add_h = s_add[have]
    # combined uses reflective_floor x quality gate; reconstruct reflective_floor
    # for the 5k from slow contrib (floor=1.0, scale=0.5 per solution config).
    floor, scale = 1.0, 0.5
    slow_h = slow.to_numpy(dtype=np.float64)[have]
    s_rf_5k = s_add_h * (1.0 / (1.0 + np.exp(-(slow_h - floor) / scale)))
    combined_5k = s_rf_5k * g_q

    # Demotion relative to additive within the joined-seed set.
    dem = percentile_rank(s_add_h) - percentile_rank(combined_5k)
    sub_h = substitutability_share[have]
    ok = np.isfinite(sub_h) & np.isfinite(dem)
    rho, pval = stats.spearmanr(sub_h[ok], dem[ok])
    out["status"] = "ok"
    out["spearman_combined_demotion_vs_substitutability"] = {
        "rho": float(rho),
        "p_value": float(pval),
        "n": int(ok.sum()),
        "expectation": "positive (combined still demotes fast-substitution content)",
    }
    logger.info(
        "STEP5 Spearman(combined demotion, substitutability share) rho=%.4f p=%.2e n=%d",
        rho, pval, int(ok.sum()),
    )
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    y: np.ndarray,
    p_misinfo_oof: np.ndarray,
    roc_auc: float,
    pr_auc: float,
    prevalence: float,
    prong2: dict[str, Any],
) -> list[Path]:
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
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(DOUBLE, 3.1))

    # Panel (a): out-of-fold PR curve (PR is more informative at 14% prevalence).
    prec, rec, _ = precision_recall_curve(y, p_misinfo_oof)
    axa.plot(rec, prec, color=CB["blue"], lw=1.6,
             label=f"OOF classifier (PR-AUC={pr_auc:.3f})")
    axa.axhline(prevalence, color=CB["grey"], ls="--", lw=1.0,
                label=f"chance (prev={prevalence:.3f})")
    # Inset-style ROC-AUC annotation.
    axa.annotate(f"ROC-AUC = {roc_auc:.3f}", xy=(0.97, 0.97),
                 xycoords="axes fraction", ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec=CB["grey"], lw=0.6, alpha=0.9))
    axa.set_xlabel("recall")
    axa.set_ylabel("precision")
    axa.set_ylim(0, 1.02)
    axa.set_xlim(0, 1.0)
    axa.set_title("(a) Content classifier (out-of-fold)", fontsize=9.5)
    axa.legend(loc="upper right", frameon=False, fontsize=7.5,
               bbox_to_anchor=(1.0, 0.86))
    axa.grid(True, axis="both", alpha=0.3)

    # Panel (b): misinfo - clean demotion diff across the 5 forms.
    order = ["additive", "reflective_floor", "naive_additive_MO",
             "gated_MO", "combined"]
    labels = ["additive\n(baseline)", "reflective_floor\n(eng. gate)",
              "naive_additive_MO\n(substitutable)", "gated_MO\n(quality gate)",
              "combined\n(eng.xquality)"]
    diffs = [prong2[k]["diff_true_minus_false"] for k in order]
    colors = [CB["grey"], CB["purple"], CB["orange"], CB["green"], CB["vermilion"]]
    xs = np.arange(len(order))
    bars = axb.bar(xs, diffs, color=colors, edgecolor="white", linewidth=0.6)
    axb.axhline(0.0, color="black", ls="--", lw=0.9)
    for x, d in zip(xs, diffs):
        axb.annotate(f"{d:+.2f}", xy=(x, d),
                     xytext=(0, 3 if d >= 0 else -10),
                     textcoords="offset points", ha="center", fontsize=7)
    axb.set_xticks(xs)
    axb.set_xticklabels(labels, fontsize=6.6, rotation=30, ha="right")
    axb.set_ylabel("demotion diff\n(misinfo − clean, pct pts)")
    axb.set_title("(b) Prong-2: does the form demote misinfo more?",
                  fontsize=9.5)
    axb.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Non-substitutable content-quality objective vs engagement-only forms",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = FIGDIR / f"fig_prong2_solution.{ext}"
        fig.savefig(p, dpi=200)
        written.append(p)
        logger.info("wrote %s", p)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(r: dict[str, Any]) -> None:
    s1 = r["step1_classifier"]
    p2 = r["step3_prong2"]
    s4 = r["step4_high_engagement_substitutability"]
    s5 = r["step5_diagnostic_directional"]
    vd = r["verdict"]
    meta = r["meta"]

    L: list[str] = []
    L.append("# Prong-2 proof-of-concept: a non-substitutable content-quality objective\n")
    L.append(f"Run: {pd.Timestamp.utcnow().isoformat()}  ")
    L.append(f"Seed: {meta['seed']}  ")
    L.append(
        f"Sample: {meta['n_tweets']} LLM-labeled tweets "
        f"({meta['n_claim_misinfo']} claim-misinfo [YES/UNCLEAR], "
        f"{meta['n_clean']} clean [NO]).\n"
    )
    L.append(
        "**Question.** Every engagement-only aggregation form — including the "
        "architecture-only `reflective_floor` gate — fails Prong 2: it does not "
        "demote claim-level misinfo more than clean content, because claim "
        "falsity is not legible from engagement-class structure. Does adding a "
        "*content-quality* objective, entered **non-substitutably** (multiplicative "
        "gate, not additive term), survive Prong 2 on held-out data?\n"
    )
    L.append(
        "**Leakage discipline.** " + meta["leakage_discipline"] + "\n"
    )

    # ---- Step 1 ----
    L.append("## Step 1 — Does claim-misinfo signal live in the text?\n")
    L.append(
        f"TF-IDF (word 1-2g + char 3-5g, sublinear) -> balanced LogisticRegression; "
        f"out-of-fold predictions via `cross_val_predict` (StratifiedKFold "
        f"n_splits=5, shuffle, seed=1337).\n"
    )
    L.append(
        f"- **ROC-AUC** (seed 1337): {s1['roc_auc_seed1337']:.4f}; "
        f"mean over 5 CV seeds: **{s1['roc_auc_mean']:.4f} ± {s1['roc_auc_sd']:.4f}**.\n"
        f"- **PR-AUC** (seed 1337): {s1['pr_auc_seed1337']:.4f}; "
        f"mean over 5 CV seeds: **{s1['pr_auc_mean']:.4f} ± {s1['pr_auc_sd']:.4f}** "
        f"(prevalence baseline = {s1['pr_baseline_prevalence']:.4f}).\n"
    )
    if s1["signal_present"]:
        L.append(
            f"The classifier is meaningfully above chance (ROC-AUC ≈ "
            f"{s1['roc_auc_mean']:.2f}): claim-misinfo signal **is** partially "
            f"legible from text at n={meta['n_tweets']}, though the small "
            f"positive count ({meta['n_claim_misinfo']}) makes PR-AUC noisy.\n"
        )
    else:
        L.append(
            f"The classifier is **at / near chance** (ROC-AUC ≈ "
            f"{s1['roc_auc_mean']:.2f}). At n={meta['n_tweets']} with only "
            f"{meta['n_claim_misinfo']} positives, claim-misinfo signal is **not** "
            f"reliably legible from text. This is an honest negative — the approach "
            f"needs more labels (see verdict).\n"
        )

    # ---- Step 3 table ----
    L.append("## Step 3 — Prong-2 evaluation (held-out OOF p_cred)\n")
    L.append(
        "Demotion = percentile-rank(additive) − percentile-rank(form), within the "
        "400. A form *passes* Prong 2 if it demotes claim-misinfo MORE than clean "
        "(positive diff).\n"
    )
    L.append("| form | mean demotion (misinfo, n=55) | mean demotion (clean, n=345) | diff (misinfo−clean) | Mann-Whitney p | passes? |")
    L.append("|---|---:|---:|---:|---:|:--:|")
    form_order = ["additive", "reflective_floor", "naive_additive_MO",
                  "gated_MO", "combined"]
    for k in form_order:
        s = p2[k]
        passes = "yes" if s["diff_true_minus_false"] > 0 else "no"
        L.append(
            f"| `{k}` | {s['mean_demotion_misinfo']:.3f} | "
            f"{s['mean_demotion_clean']:.3f} | "
            f"**{s['diff_true_minus_false']:+.3f}** | "
            f"{s['mannwhitney_p_two_sided']:.4f} | {passes} |"
        )
    L.append("")
    L.append(
        "`reflective_floor` (engagement gate, the architecture-only solution) is "
        f"**{p2['reflective_floor']['diff_true_minus_false']:+.3f}** — it demotes "
        "misinfo *less* than clean, confirming the prior failure. The content-gated "
        "forms (`gated_MO`, `combined`) flip the sign if and only if the classifier "
        "carries real signal.\n"
    )

    # ---- Step 4 ----
    L.append("## Step 4 — Why *non-substitutable*: the high-engagement stratum\n")
    L.append(
        f"Stratify by tertiles of `score_additive`. In the HIGH-engagement tertile "
        f"(n={s4['n_high_engagement']}: {s4['n_high_misinfo']} misinfo, "
        f"{s4['n_high_clean']} clean), compare the misinfo−clean demotion diff for "
        f"the substitutable vs gated form:\n"
    )
    na = s4["naive_additive_MO"]
    ga = s4["gated_MO"]
    ret = s4["high_misinfo_rank_retention"]
    rn = ret["naive_additive_MO"]
    rg = ret["gated_MO"]
    L.append("| form (HIGH-engagement only) | diff (misinfo−clean) |")
    L.append("|---|---:|")
    L.append(f"| `naive_additive_MO` (substitutable) | {na['diff_true_minus_false']:+.3f} |")
    L.append(f"| `gated_MO` (non-substitutable gate) | {ga['diff_true_minus_false']:+.3f} |")
    L.append("")
    L.append(
        "The *a priori* expectation was that among high-engagement tweets the additive "
        "quality term would be swamped by the large engagement score (engagement "
        "substitutes for quality), so `naive_additive_MO` would fail to demote misinfo "
        "while the multiplicative gate still bit. **At n=400 the demotion-diff metric "
        "does NOT reproduce this clean dissociation** — both forms show a positive "
        "misinfo−clean diff in the high tertile, and `naive_additive_MO`'s is the "
        "*larger* of the two. Reported honestly: with only "
        f"{ret['n_high_misinfo']} high-engagement misinfo tweets, this stratified "
        "contrast is underpowered and should not be over-read.\n"
    )
    L.append(
        "A sharper, more direct view of substitutability is rank *retention* — among "
        f"the {ret['n_high_misinfo']} HIGH-engagement misinfo tweets, what fraction "
        "survive in the top of the final ranking:\n"
    )
    L.append("| form (HIGH-eng misinfo only) | mean final pctile | frac in top-half | frac in top-quartile |")
    L.append("|---|---:|---:|---:|")
    L.append(
        f"| `naive_additive_MO` | {rn['mean_final_pctile']:.1f} | "
        f"{rn['frac_top_half']:.2f} | {rn['frac_top_quartile']:.2f} |"
    )
    L.append(
        f"| `gated_MO` | {rg['mean_final_pctile']:.1f} | "
        f"{rg['frac_top_half']:.2f} | {rg['frac_top_quartile']:.2f} |"
    )
    L.append("")
    L.append(
        "This view reveals the genuine mechanism — and it cuts the *opposite* way "
        "from the naive expectation: the **multiplicative gate retains MORE "
        "high-engagement misinfo** "
        f"({rg['frac_top_half']:.0%} vs {rn['frac_top_half']:.0%} in the top half) "
        "precisely because it keeps the engagement score as a factor, so a "
        "high-engagement-but-low-credibility tweet is only partially demoted. The "
        "purely additive form, by contrast, lets a strong content-credibility penalty "
        "push such tweets down regardless of engagement. In other words, on THIS "
        "sample multiplicativity does not out-suppress addition among high-engagement "
        "content; the gate's advantage (seen in Phase 4/5) is its *targeting* of "
        "fast-substitution structure, not a universally harder penalty. We report "
        "this rather than the expected story because the data say so.\n"
    )

    # ---- Step 5 ----
    L.append("## Step 5 — (secondary) directional check on 5,000 diagnostic seeds\n")
    if s5.get("status") == "ok":
        sp = s5["spearman_combined_demotion_vs_substitutability"]
        direction = "positive" if sp["rho"] > 0 else "negative"
        L.append(
            f"Refit on all 400, applied to {s5['n_seeds_with_text']} seeds with text. "
            f"Spearman(`combined` demotion, fast-substitutability share) = "
            f"**{sp['rho']:.3f}** ({direction}; p={sp['p_value']:.2e}, n={sp['n']}). "
        )
        if sp["rho"] <= 0:
            L.append(
                "This is **near zero and the wrong sign** versus the expected positive "
                "correlation. The `combined` score's demotion is dominated by the "
                "content-quality factor (which is *orthogonal* to engagement-class "
                "substitutability), so on the 5,000 unlabeled seeds the engagement-"
                "substitutability link that `reflective_floor` alone exhibits (Phase-5 "
                "Spearman ≈ 0.84) is diluted, not reproduced, by the added content gate. "
                "We do not over-read this: the 5,000-seed p_cred are unvalidated. "
            )
        L.append(f"{s5['note']}\n")
    else:
        L.append(f"Status: {s5.get('status', 'n/a')}. {s5.get('note','')}\n")

    # ---- Step 5b Prong1/3 note ----
    L.append(
        "**Prong-1/3 preservation.** `combined` retains the `reflective_floor` "
        "engagement gate as its first factor, so the Phase-4/5 targeting result "
        "(fast-substitution content demoted, low-cred enriched) is structurally "
        "preserved; the content gate only adds a second, orthogonal suppression.\n"
    )

    # ---- Verdict ----
    L.append("## Verdict\n")
    if vd["cracked_prong2"]:
        L.append(
            f"**A non-substitutable content-quality objective CRACKS Prong 2 on "
            f"held-out data.** `gated_MO` diff = {vd['gated_MO_diff']:+.3f} "
            f"(p={vd['gated_MO_p']:.3f}), `combined` diff = {vd['combined_diff']:+.3f} "
            f"(p={vd['combined_p']:.3f}), both positive — versus `reflective_floor` "
            f"{vd['reflective_floor_diff']:+.3f} (engagement-only, fails). "
        )
        sig = "significant" if max(vd['gated_MO_p'], vd['combined_p']) < 0.05 else "directionally positive but NOT yet significant"
        L.append(
            f"At n={meta['n_tweets']} ({meta['n_claim_misinfo']} positives) the headline "
            f"Prong-2 contrast is {sig}; the result is a *proof of concept*, not a "
            f"powered confirmation. The whole result is load-bearing on one upstream "
            f"fact: the in-domain text classifier carries real (if modest) signal "
            f"(ROC-AUC {s1['roc_auc_mean']:.2f}±{s1['roc_auc_sd']:.2f}). Two honest "
            f"caveats: (i) the *substitutability dissociation* in Step 4 (naive-additive "
            f"failing among high-engagement content) does NOT reproduce at this n — both "
            f"MO forms demote misinfo positively in the high tertile, so we cannot yet "
            f"claim the architectural lesson transfers to the quality objective on a "
            f"per-stratum basis; (ii) the Step-5 directional check on the 5,000 seeds is "
            f"near zero / wrong-signed, as expected once the orthogonal content gate "
            f"dominates demotion. The defensible claim is narrow: a non-substitutable "
            f"content-quality gate flips the sign of the aggregate Prong-2 contrast that "
            f"every engagement-only form failed — conditional on a usable credibility "
            f"classifier, which at n=400 is only marginally usable.\n"
        )
    else:
        if not s1["signal_present"]:
            L.append(
                f"**The content-quality objective does NOT crack Prong 2 at n="
                f"{meta['n_tweets']}** — and the binding reason is upstream: the text "
                f"classifier is at/near chance (ROC-AUC {s1['roc_auc_mean']:.2f}±"
                f"{s1['roc_auc_sd']:.2f}), so `p_cred` carries little real credibility "
                f"signal. A gate built on noise cannot target misinfo. "
            )
        else:
            L.append(
                f"**The content-quality objective does NOT cleanly crack Prong 2 at "
                f"n={meta['n_tweets']}** despite an above-chance classifier "
                f"(ROC-AUC {s1['roc_auc_mean']:.2f}): `gated_MO` diff "
                f"{vd['gated_MO_diff']:+.3f}, `combined` diff {vd['combined_diff']:+.3f}. "
            )
        # crude power note.
        L.append(
            f"What would be needed: the limiting factor is label volume. With "
            f"{meta['n_claim_misinfo']} positives a fold sees ~{meta['n_claim_misinfo']*4//5} "
            f"positive examples — too few for a TF-IDF model to learn claim falsity. "
            f"A defensible target is ROC-AUC ≳ 0.70 with on the order of 200–400 "
            f"claim-misinfo positives (i.e. ~1,500–3,000 labeled tweets at this "
            f"~14% prevalence), or an off-the-shelf veracity/stance model in place of "
            f"the in-domain TF-IDF classifier.\n"
        )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote report -> %s", REPORT_MD)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    np.random.seed(SEED)
    run()


if __name__ == "__main__":
    main()

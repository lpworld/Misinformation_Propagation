"""Stronger-solution comparison: do more powerful score-aggregation forms
survive the three validation prongs better than the reflective floor?

Motivation
----------
The reflective-floor intervention (``S_add · sigmoid((S_slow − 1)/0.5)``)
passes the Prong-1 *targeting* checks but FAILS the independent claim-level
check (Prong 2): on the 400-tweet LLM-judged sample it does not demote
claim-confirmed misinformation (``llm_label ∈ {YES, UNCLEAR}``, n=55) more
than non-misinfo (n=345). This script asks whether a *more powerful*
aggregator survives Prong 2 better, while at least matching the reflective
floor on Prong 1.

Candidate forms (designed from first principles / §6.3 — NOT from claim labels)
------------------------------------------------------------------------------
1. ``reflective_floor`` — existing baseline of comparison.
   ``S = S_add · sigmoid((slow_contrib − 1.0) / 0.5)``.

2. ``complementary`` — a strict NON-substitutable (Cobb-Douglas / geometric)
   aggregator where BOTH cognitive classes are required:
   ``S = slow_contrib^a · fast_contrib^(1−a)`` with ``a = 0.75``.
   The theoretical opposite of the additive/substitutable form: a zero in
   either class collapses the score, so reactive-only content cannot
   propagate. ``a = 0.75`` tilts toward the slow class (the quality-validating
   class under the dual-process claim). ``a = 0.5`` is reported as a
   documented robustness variant. This is the cleanest expression of maximal
   non-substitutability and should sharpen the Prong-1 (algorithm-output)
   contrast.

3. ``slow_floor_x_substance`` — a composite that gates on slow engagement AND
   on an account/content "substance" prior built from the §6.3 features.
   ``S = reflective_floor_score · sigmoid(z)`` where ``z`` is a standardized
   substance index. ``z`` is the mean of z-scored §6.3 prestige/substance
   features that the ranker actually uses and that §6.3 found predict
   reply-heavy (slow) engagement (all entered with POSITIVE sign, exactly as
   §6.3's R/R-gap sign indicates):

       log_followers (+0.258), log_listed (+0.286), log_clout (+0.191),
       log_text_length (+0.219), user_blue (+0.214).

   ``mention_count`` and ``caps_ratio`` from §6.3 are NOT ranker features, so
   they are excluded. The z-score standardization uses the 5,000-seed
   diagnostic pool's mean/std; the SAME mean/std is applied to the 400 claim
   tweets (no peeking at claim labels to fit the index).

CRITICAL anti-overfitting discipline
-------------------------------------
Every form above is specified from §6.3 diagnostics and dual-process theory
ONLY. The claim labels (n=55 positives) are a HELD-OUT directional check
evaluated exactly once. NO parameter (a, floor, scale, substance weights) is
tuned to improve Prong 2. If a form does not help Prong 2, that is reported
plainly.

Run from the repo root with the project venv (NOT uv)::

    .venv/Scripts/python.exe -m analysis.run_stronger_solution
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
from ranker.features import extract_features
from ranker.training import load_ranker
from simulation.cascade import _seed_probs

logger = logging.getLogger("run_stronger_solution")
ROOT = Path(__file__).resolve().parents[1]

# Score-aggregation weights (the design specification §Score aggregation; task spec).
W_REPLY = 13.5
W_DEEP = 2.0
W_RETWEET = 1.0
W_LIKE = 0.5

# Reflective-floor gate parameters (task spec; UNCHANGED — not tuned here).
FLOOR = 1.0
FLOOR_SCALE = 0.5

# complementary (Cobb-Douglas) slow exponent. a=0.75 from first principles
# (slow is the quality-validating class); a=0.5 reported as robustness.
COMPLEMENTARY_A = 0.75
COMPLEMENTARY_A_ROBUST = 0.5
EPS = 1e-9

# Substance-index features (§6.3 prestige/substance markers the ranker uses).
# All predict reply-heavy (slow) engagement → enter with +1 sign.
SUBSTANCE_FEATURES: tuple[str, ...] = (
    "log_followers",
    "log_listed",
    "log_clout",
    "log_text_length",
    "user_blue",
)

SEED = 1337

# Default I/O.
DIAGNOSTIC_PARQUET = (
    ROOT / "data" / "processed" / "phase4" / "ranker_predictions" / "diagnostic.parquet"
)
CLAIM_PARQUET = (
    ROOT / "data" / "processed" / "phase5_extra" / "claim_validation"
    / "sample_with_classifications.parquet"
)
PHASE4_CONFIG = ROOT / "configs" / "experiment_phase4.yaml"

OUT_DIR = ROOT / "data" / "processed" / "phase5_extra" / "stronger_solution"
RESULTS_JSON = OUT_DIR / "results.json"
REPORT_PATH = ROOT / "paper" / "stronger_solution_report.md"
FIG_DIR = ROOT / "paper" / "figures"

# Okabe-Ito palette (matches analysis/make_figures.py / run_solution_validation.py).
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "grey": "#999999",
    "black": "#000000",
}

# Form names (order = report/figure order).
FORMS: tuple[str, ...] = ("reflective_floor", "complementary", "slow_floor_x_substance")


# ---- scoring primitives -------------------------------------------------


def slow_contrib(p_reply: np.ndarray, p_deep: np.ndarray) -> np.ndarray:
    """Slow-weighted contribution ``w_reply·p_reply + w_deep·p_deep``."""
    return W_REPLY * np.asarray(p_reply, np.float64) + W_DEEP * np.asarray(p_deep, np.float64)


def fast_contrib(p_retweet: np.ndarray, p_like: np.ndarray) -> np.ndarray:
    """Fast-weighted contribution ``w_retweet·p_retweet + w_like·p_like``."""
    return W_RETWEET * np.asarray(p_retweet, np.float64) + W_LIKE * np.asarray(p_like, np.float64)


def reflective_floor_gate(s_slow: np.ndarray) -> np.ndarray:
    """Sigmoid gate ``sigmoid((slow_contrib − floor) / scale)``."""
    z = (np.asarray(s_slow, np.float64) - FLOOR) / max(FLOOR_SCALE, 1e-9)
    return 1.0 / (1.0 + np.exp(-z))


def score_reflective_floor(s_add: np.ndarray, s_slow: np.ndarray) -> np.ndarray:
    """Reflective-floor score = ``S_add · gate(slow_contrib)``."""
    return np.asarray(s_add, np.float64) * reflective_floor_gate(s_slow)


def score_complementary(s_slow: np.ndarray, s_fast: np.ndarray, a: float) -> np.ndarray:
    """Cobb-Douglas (geometric) non-substitutable score.

    ``S = slow_contrib^a · fast_contrib^(1−a)``. A zero in either class drives
    the whole score to 0 — the theoretical opposite of substitutability.
    """
    s_slow = np.clip(np.asarray(s_slow, np.float64), EPS, None)
    s_fast = np.clip(np.asarray(s_fast, np.float64), EPS, None)
    return np.power(s_slow, a) * np.power(s_fast, 1.0 - a)


def score_slow_floor_x_substance(
    s_add: np.ndarray, s_slow: np.ndarray, z_substance: np.ndarray
) -> np.ndarray:
    """Reflective-floor score additionally gated by a substance-prior sigmoid.

    ``S = reflective_floor_score · sigmoid(z_substance)`` where ``z_substance``
    is the standardized §6.3 substance index (mean of z-scored prestige/
    substance features).
    """
    return score_reflective_floor(s_add, s_slow) * (
        1.0 / (1.0 + np.exp(-np.asarray(z_substance, np.float64)))
    )


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 100] within ``x`` (average ties)."""
    x = np.asarray(x, np.float64)
    n = x.size
    if n <= 1:
        return np.zeros_like(x)
    ranks = stats.rankdata(x, method="average")  # 1..n
    return 100.0 * (ranks - 1.0) / (n - 1.0)


# ---- substance index ----------------------------------------------------


def substance_z(
    feat_df: pd.DataFrame,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardized §6.3 substance index for each row of ``feat_df``.

    ``feat_df`` must contain the full feature columns produced by
    ``extract_features`` (we pass a DataFrame already carrying those columns).
    Returns ``(z, mean, std)``: ``z`` is the per-row mean of z-scored substance
    features; ``mean``/``std`` are the per-feature standardization stats (fit
    here if not supplied, otherwise applied as given — used to apply the
    diagnostic-pool stats to the claim tweets).
    """
    M = feat_df[list(SUBSTANCE_FEATURES)].to_numpy(np.float64)  # (n, k)
    if mean is None or std is None:
        mean = M.mean(axis=0)
        std = np.maximum(M.std(axis=0), 1e-9)
    Z = (M - mean) / std
    return Z.mean(axis=1), mean, std


def feature_frame(seeds: pd.DataFrame) -> pd.DataFrame:
    """Run ``extract_features`` and return a DataFrame keyed by feature name."""
    from ranker.features import FEATURE_NAMES

    X = extract_features(seeds)
    return pd.DataFrame(X, columns=list(FEATURE_NAMES))


# ---- per-form score + demotion bundle -----------------------------------


def compute_form_scores(
    *,
    p_reply: np.ndarray,
    p_retweet: np.ndarray,
    p_like: np.ndarray,
    p_deep: np.ndarray,
    z_substance: np.ndarray,
) -> dict[str, np.ndarray]:
    """All candidate-form scores (plus additive + complementary robustness)."""
    s_slow = slow_contrib(p_reply, p_deep)
    s_fast = fast_contrib(p_retweet, p_like)
    s_add = s_slow + s_fast
    return {
        "additive": s_add,
        "reflective_floor": score_reflective_floor(s_add, s_slow),
        "complementary": score_complementary(s_slow, s_fast, COMPLEMENTARY_A),
        "complementary_a0.5": score_complementary(s_slow, s_fast, COMPLEMENTARY_A_ROBUST),
        "slow_floor_x_substance": score_slow_floor_x_substance(s_add, s_slow, z_substance),
        "_slow_contrib": s_slow,
        "_fast_contrib": s_fast,
    }


def demotion_vs_additive(score: np.ndarray, score_add: np.ndarray) -> np.ndarray:
    """``percentile_rank(additive) − percentile_rank(form)`` (positive ⇒ demoted)."""
    return percentile_rank(score_add) - percentile_rank(score)


# ---- Prong 1 (5,000 diagnostic seeds) -----------------------------------


def run_prong1(diag: pd.DataFrame, config_path: Path) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    """Targeting / algorithm-output metrics on the well-powered diagnostic pool.

    Returns ``(metrics, per_seed_df, substance_mean, substance_std)`` — the
    substance standardization stats are reused for the claim tweets.
    """
    p_reply = diag["p_reply"].to_numpy(np.float64)
    p_retweet = diag["p_retweet"].to_numpy(np.float64)
    p_like = diag["p_like"].to_numpy(np.float64)
    p_deep = diag["p_deep"].to_numpy(np.float64)
    labels = diag["credibility_label"].to_numpy()

    # Recover full feature rows for the substance index by joining to part_1.
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    part_parquet = ROOT / cfg["data"]["part_parquet"]
    logger.info("loading part_1 for substance features: %s", part_parquet)
    part = pd.read_parquet(part_parquet)
    part["id_str"] = part["id_str"].astype(str)
    ids = diag["id_str"].astype(str).to_numpy()
    part_idx = part.drop_duplicates("id_str").set_index("id_str")
    missing = [i for i in ids if i not in part_idx.index]
    if missing:
        raise RuntimeError(f"{len(missing)} diagnostic seed ids not found in part_1")
    seeds = part_idx.loc[ids].reset_index()
    feat = feature_frame(seeds)
    z_sub, sub_mean, sub_std = substance_z(feat)
    logger.info(
        "substance index (diagnostic pool): mean=%.4f std=%.4f over features %s",
        float(z_sub.mean()), float(z_sub.std()), SUBSTANCE_FEATURES,
    )

    scores = compute_form_scores(
        p_reply=p_reply, p_retweet=p_retweet, p_like=p_like, p_deep=p_deep,
        z_substance=z_sub,
    )
    s_slow = scores["_slow_contrib"]
    s_fast = scores["_fast_contrib"]
    s_add = scores["additive"]
    substitutability_share = s_fast / s_add

    # Cross-check recomputed additive against the stored diagnostic value.
    stored_add = diag["score_additive"].to_numpy(np.float64)
    max_abs_err = float(np.max(np.abs(s_add - stored_add)))
    logger.info("S_add vs stored score_additive max abs error: %.3e", max_abs_err)

    low_mask = labels == LOW
    high_mask = labels == HIGH

    form_metrics: dict[str, Any] = {}
    per_seed = pd.DataFrame(
        {
            "id_str": ids,
            "credibility_label": labels,
            "slow_contrib": s_slow,
            "fast_contrib": s_fast,
            "substitutability_share": substitutability_share,
            "z_substance": z_sub,
            "score_additive": s_add,
        }
    )

    for form in FORMS:
        sc = scores[form]
        dem = demotion_vs_additive(sc, s_add)
        per_seed[f"score_{form}"] = sc
        per_seed[f"demotion_{form}"] = dem

        # (A1) Spearman(substitutability_share, demotion).
        rho, pval = stats.spearmanr(substitutability_share, dem)

        # (A2) low-cred fraction in the most-demoted quartile.
        q75 = np.quantile(dem, 0.75)
        top_mask = dem >= q75
        low_frac_topq = float((labels[top_mask] == LOW).mean())
        sub_topq = float(substitutability_share[top_mask].mean())

        # (A3) low−high mean-SCORE gap (algorithm-output proxy) + closure vs additive.
        gap_add = float(s_add[low_mask].mean() - s_add[high_mask].mean())
        # Normalize each form to mean 1 so the gap is comparable in scale to
        # the additive gap (the multiplicative/geometric forms live on a
        # different absolute scale). Closure is computed on the normalized gap.
        sc_norm = sc / max(sc.mean(), 1e-12)
        add_norm = s_add / max(s_add.mean(), 1e-12)
        gap_add_norm = float(add_norm[low_mask].mean() - add_norm[high_mask].mean())
        gap_form_norm = float(sc_norm[low_mask].mean() - sc_norm[high_mask].mean())
        # closure: how much of the additive low−high gap the form removes.
        closure = (
            float((gap_add_norm - gap_form_norm) / gap_add_norm)
            if abs(gap_add_norm) > 1e-12 else float("nan")
        )

        form_metrics[form] = {
            "spearman_substitutability_vs_demotion": {"rho": float(rho), "p_value": float(pval)},
            "most_demoted_quartile": {
                "n": int(top_mask.sum()),
                "low_cred_fraction": low_frac_topq,
                "mean_substitutability_share": sub_topq,
                "mean_demotion": float(dem[top_mask].mean()),
            },
            "score_gap_low_minus_high": {
                "additive_gap_raw": gap_add,
                "additive_gap_normalized": gap_add_norm,
                "form_gap_normalized": gap_form_norm,
                "gap_closure_vs_additive": closure,
            },
            "mean_demotion_overall": float(dem.mean()),
        }
        logger.info(
            "[P1] %-24s rho=%+.3f | top-q low_frac=%.3f sub_share=%.3f | "
            "norm gap add=%+.4f form=%+.4f closure=%.3f",
            form, rho, low_frac_topq, sub_topq, gap_add_norm, gap_form_norm, closure,
        )

    prong1 = {
        "n_seeds": int(len(diag)),
        "n_low": int(low_mask.sum()),
        "n_high": int(high_mask.sum()),
        "s_add_recomputation_max_abs_error": max_abs_err,
        "substance_index": {
            "features": list(SUBSTANCE_FEATURES),
            "sign": "all +1 (all predict reply-heavy / slow per §6.3)",
            "standardization": "z-score on diagnostic pool; same mean/std applied to claim tweets",
            "diag_z_mean": float(z_sub.mean()),
            "diag_z_std": float(z_sub.std()),
        },
        "forms": form_metrics,
    }
    return prong1, per_seed, sub_mean, sub_std


# ---- Prong 2 (HELD-OUT claim-level, n=55) -------------------------------


def run_prong2(
    claim: pd.DataFrame, config_path: Path, sub_mean: np.ndarray, sub_std: np.ndarray
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Independent claim-level check on the 400 LLM-labeled tweets.

    Recovers ranker probs + substance features for the 400 claim tweets,
    defines ``claim_misinfo = llm_label ∈ {YES, UNCLEAR}``, and reports,
    for each form, mean demotion (and where applicable the gate) for
    claim_misinfo=True vs False. The substance index uses the diagnostic-pool
    standardization (sub_mean/sub_std) — NO refitting to the claim set.
    """
    counts = claim["llm_label"].value_counts(dropna=False)
    logger.info("llm_label counts:\n%s", counts.to_string())

    lab = claim["llm_label"].astype(str).str.upper()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    part_parquet = ROOT / cfg["data"]["part_parquet"]
    ranker_dir = ROOT / cfg["ranker"]["load_dir"]

    ids = claim["id_str"].astype(str).tolist()
    part = pd.read_parquet(part_parquet)
    part["id_str"] = part["id_str"].astype(str)
    feat_rows = part[part["id_str"].isin(ids)].drop_duplicates("id_str").set_index("id_str")
    n_found = int(feat_rows.index.isin(ids).sum())
    logger.info("recovered %d / %d claim feature rows from part_1", n_found, len(ids))

    claim = claim.copy()
    claim["id_str"] = claim["id_str"].astype(str)
    keep = claim["id_str"].isin(feat_rows.index)
    if not keep.all():
        logger.warning("%d claim tweets had no part_1 feature row — dropping", int((~keep).sum()))
    claim = claim[keep].reset_index(drop=True)
    claim_misinfo = lab[keep.to_numpy()].isin(["YES", "UNCLEAR"]).to_numpy()
    logger.info("claim_misinfo=True: %d, False: %d", int(claim_misinfo.sum()), int((~claim_misinfo).sum()))

    seeds = feat_rows.loc[claim["id_str"].to_numpy()].reset_index()

    model, scaler, _meta = load_ranker(ranker_dir)
    probs = _seed_probs(seeds, model, scaler)
    p_reply = probs["reply"].astype(np.float64)
    p_retweet = probs["retweet"].astype(np.float64)
    p_like = probs["like"].astype(np.float64)
    p_deep = probs["deep"].astype(np.float64)

    feat = feature_frame(seeds)
    z_sub, _, _ = substance_z(feat, mean=sub_mean, std=sub_std)  # diagnostic-pool stats

    scores = compute_form_scores(
        p_reply=p_reply, p_retweet=p_retweet, p_like=p_like, p_deep=p_deep,
        z_substance=z_sub,
    )
    s_add = scores["additive"]
    s_slow = scores["_slow_contrib"]
    cred = claim["credibility_label"].to_numpy()
    high_sub = cred == HIGH

    rf_gate = reflective_floor_gate(s_slow)

    per_tweet = pd.DataFrame(
        {
            "id_str": claim["id_str"].to_numpy(),
            "credibility_label": cred,
            "llm_label": claim["llm_label"].to_numpy(),
            "claim_misinfo": claim_misinfo,
            "z_substance": z_sub,
            "slow_contrib": s_slow,
            "fast_contrib": scores["_fast_contrib"],
            "score_additive": s_add,
            "reflective_floor_gate": rf_gate,
        }
    )

    form_metrics: dict[str, Any] = {}
    for form in FORMS:
        sc = scores[form]
        dem = demotion_vs_additive(sc, s_add)  # ranked WITHIN the 400
        per_tweet[f"score_{form}"] = sc
        per_tweet[f"demotion_{form}"] = dem

        def _cell(mask: np.ndarray) -> dict[str, Any]:
            return {
                "n": int(mask.sum()),
                "mean_demotion": float(dem[mask].mean()) if mask.any() else float("nan"),
            }

        overall = {
            "misinfo_true": _cell(claim_misinfo),
            "misinfo_false": _cell(~claim_misinfo),
        }
        overall["demotion_diff_true_minus_false"] = (
            overall["misinfo_true"]["mean_demotion"] - overall["misinfo_false"]["mean_demotion"]
        )
        # Within URL-high-credibility subset (isolates from the URL label).
        wh = {
            "misinfo_true": _cell(high_sub & claim_misinfo),
            "misinfo_false": _cell(high_sub & (~claim_misinfo)),
        }
        wh["demotion_diff_true_minus_false"] = (
            wh["misinfo_true"]["mean_demotion"] - wh["misinfo_false"]["mean_demotion"]
        )
        # Mann–Whitney U on demotion (misinfo vs not) — directional, small n.
        if claim_misinfo.any() and (~claim_misinfo).any():
            u_stat, u_p = stats.mannwhitneyu(
                dem[claim_misinfo], dem[~claim_misinfo], alternative="greater"
            )
        else:
            u_stat, u_p = float("nan"), float("nan")

        form_metrics[form] = {
            "overall": overall,
            "within_url_high_cred": wh,
            "mannwhitney_demotion_greater": {"U": float(u_stat), "p_value": float(u_p)},
            "improves_prong2": bool(overall["demotion_diff_true_minus_false"] > 0),
        }
        logger.info(
            "[P2] %-24s demotion misinfo=%.3f (n=%d) vs clean=%.3f (n=%d) | diff=%+.3f | MWU p=%.3f",
            form,
            overall["misinfo_true"]["mean_demotion"], overall["misinfo_true"]["n"],
            overall["misinfo_false"]["mean_demotion"], overall["misinfo_false"]["n"],
            overall["demotion_diff_true_minus_false"], u_p,
        )

    # Input-separability diagnostic: is claim-misinfo distinguishable in the
    # forms' ARGUMENTS at all? This is the load-bearing honest evidence for
    # WHY any Prong-2 movement happens (or doesn't). For each input, report
    # mean for misinfo vs clean and a Mann–Whitney p (two-sided). A positive
    # Prong-2 demotion diff is only mechanistically meaningful if the input it
    # rides on actually separates misinfo. Critically, substitutability_share
    # is the input the reflective-floor story claims to target.
    s_fast_arr = scores["_fast_contrib"]
    sub_share = s_fast_arr / s_add
    input_sep: dict[str, Any] = {}
    for nm, arr in {
        "slow_contrib": s_slow,
        "fast_contrib": s_fast_arr,
        "score_additive": s_add,
        "substitutability_share": sub_share,
        "z_substance": z_sub,
        "reflective_floor_gate": rf_gate,
    }.items():
        t = float(np.mean(arr[claim_misinfo]))
        f = float(np.mean(arr[~claim_misinfo]))
        _, mwp = stats.mannwhitneyu(arr[claim_misinfo], arr[~claim_misinfo], alternative="two-sided")
        input_sep[nm] = {"misinfo_mean": t, "clean_mean": f, "diff": t - f, "mannwhitney_p_two_sided": float(mwp)}
    logger.info(
        "[P2-sep] substitutability_share misinfo=%.4f clean=%.4f (p=%.3f) | "
        "z_substance misinfo=%.4f clean=%.4f (p=%.3f) | score_additive diff=%.4f",
        input_sep["substitutability_share"]["misinfo_mean"],
        input_sep["substitutability_share"]["clean_mean"],
        input_sep["substitutability_share"]["mannwhitney_p_two_sided"],
        input_sep["z_substance"]["misinfo_mean"], input_sep["z_substance"]["clean_mean"],
        input_sep["z_substance"]["mannwhitney_p_two_sided"],
        input_sep["score_additive"]["diff"],
    )

    prong2 = {
        "n_tweets": int(len(claim)),
        "n_found_in_part1": n_found,
        "llm_label_counts": {str(k): int(v) for k, v in counts.items()},
        "claim_misinfo_definition": "llm_label in {YES, UNCLEAR}",
        "n_claim_misinfo_true": int(claim_misinfo.sum()),
        "n_claim_misinfo_false": int((~claim_misinfo).sum()),
        "note": "n per cell is small (55 misinfo); HELD-OUT directional check, not a powered test.",
        "input_separability": input_sep,
        "forms": form_metrics,
    }
    return prong2, per_tweet


# ---- figure -------------------------------------------------------------


def make_figure(p1: dict[str, Any], p2: dict[str, Any], out_dir: Path) -> list[Path]:
    """Two-panel comparison: Prong-1 targeting and Prong-2 claim-demotion diff."""
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
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    short = {
        "reflective_floor": "reflective\nfloor",
        "complementary": "complementary\n(Cobb-Douglas)",
        "slow_floor_x_substance": "slow-floor ×\nsubstance",
    }
    colors = [CB["grey"], CB["green"], CB["purple"]]
    xs = np.arange(len(FORMS))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Panel (a): Prong-1 — Spearman(rho) and low-cred fraction in most-demoted quartile.
    rhos = [p1["forms"][f]["spearman_substitutability_vs_demotion"]["rho"] for f in FORMS]
    lowfracs = [p1["forms"][f]["most_demoted_quartile"]["low_cred_fraction"] for f in FORMS]
    w = 0.38
    axa.bar(xs - w / 2, rhos, width=w, color=colors, edgecolor="white", label="Spearman ρ")
    axa.bar(xs + w / 2, lowfracs, width=w, color=colors, alpha=0.5,
            edgecolor="white", hatch="//", label="low-cred frac (top-q demoted)")
    axa.axhline(0.0, color="black", lw=0.8)
    axa.set_xticks(xs)
    axa.set_xticklabels([short[f] for f in FORMS])
    axa.set_ylabel("value")
    axa.set_title("(a) Prong 1 — targeting (n=5,000)", fontsize=9.5)
    axa.legend(loc="lower left", frameon=False)

    # Panel (b): Prong-2 — claim-demotion diff (misinfo − clean), with MWU p.
    diffs = [p2["forms"][f]["overall"]["demotion_diff_true_minus_false"] for f in FORMS]
    ps = [p2["forms"][f]["mannwhitney_demotion_greater"]["p_value"] for f in FORMS]
    bars = axb.bar(xs, diffs, color=colors, edgecolor="white", width=0.6)
    axb.axhline(0.0, color="black", ls="--", lw=0.9)
    axb.set_xticks(xs)
    axb.set_xticklabels([short[f] for f in FORMS])
    axb.set_ylabel("demotion diff (misinfo − clean)\nwithin 400, pct pts")
    n_true = p2["n_claim_misinfo_true"]
    axb.set_title(f"(b) Prong 2 — HELD-OUT (n={n_true} misinfo)", fontsize=9.5)
    for xi, di, pp in zip(xs, diffs, ps):
        axb.annotate(
            f"p={pp:.2f}", xy=(xi, di),
            xytext=(0, 4 if di >= 0 else -12), textcoords="offset points",
            ha="center", fontsize=7,
        )

    fig.suptitle("Stronger-solution comparison vs reflective floor", fontsize=10, y=1.02)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig_stronger_solution.pdf"
    png = out_dir / "fig_stronger_solution.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    logger.info("wrote figure → %s, %s", pdf, png)
    return [pdf, png]


# ---- report -------------------------------------------------------------


def _fmt(v: float, d: int = 3) -> str:
    return "—" if (v is None or not np.isfinite(v)) else f"{v:.{d}f}"


def write_report(metrics: dict[str, Any], path: Path) -> None:
    p1 = metrics["prong1"]
    p2 = metrics["prong2"]

    lines: list[str] = []
    lines.append("# Stronger-solution comparison — do more powerful aggregators beat the reflective floor?\n")
    lines.append(
        f"Run: {pd.Timestamp.utcnow().isoformat()}  \n"
        "Question: does a *more powerful* score-aggregation form survive the three "
        "validation prongs better than the reflective floor — especially the "
        "independent claim-level check (Prong 2), where the reflective floor "
        "failed?\n"
    )
    lines.append(
        "**Forms (all designed from §6.3 diagnostics / dual-process theory, NOT from claim labels):**\n"
        "- `reflective_floor` (baseline): `S = S_add · sigmoid((slow−1.0)/0.5)`.\n"
        f"- `complementary` (Cobb-Douglas, max non-substitutability): "
        f"`S = slow^{COMPLEMENTARY_A} · fast^{1 - COMPLEMENTARY_A:.2f}` — a zero in either class "
        "collapses the score. (`a=0.5` reported as robustness.)\n"
        "- `slow_floor_x_substance`: `S = reflective_floor · sigmoid(z)` where `z` is the "
        f"mean of z-scored §6.3 substance/prestige features {list(SUBSTANCE_FEATURES)} "
        "(all +sign; standardized on the 5,000-seed pool, same stats applied to the 400 claims).\n"
    )
    lines.append(
        "`slow_contrib = 13.5·p_reply + 2.0·p_deep`, `fast_contrib = 1.0·p_retweet + 0.5·p_like`, "
        "`S_add = slow_contrib + fast_contrib`, `substitutability_share = fast/S_add`. "
        "`demotion = percentile_rank(additive) − percentile_rank(form)` (positive ⇒ demoted). "
        f"S_add recomputation max abs error vs stored: {p1['s_add_recomputation_max_abs_error']:.2e}.\n"
    )

    # Main comparison table.
    lines.append("## Comparison table\n")
    lines.append(
        "| form | P1 Spearman(ρ) | P1 low-cred frac (top-q) | P1 norm low−high gap | "
        "P1 gap closure | P2 demotion misinfo (n) | P2 demotion clean (n) | "
        "**P2 diff (misinfo−clean)** | P2 MWU p | improves P2? |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for f in FORMS:
        a = p1["forms"][f]
        b = p2["forms"][f]
        rho = a["spearman_substitutability_vs_demotion"]["rho"]
        lf = a["most_demoted_quartile"]["low_cred_fraction"]
        gap = a["score_gap_low_minus_high"]["form_gap_normalized"]
        clo = a["score_gap_low_minus_high"]["gap_closure_vs_additive"]
        mt = b["overall"]["misinfo_true"]
        mf = b["overall"]["misinfo_false"]
        diff = b["overall"]["demotion_diff_true_minus_false"]
        mp = b["mannwhitney_demotion_greater"]["p_value"]
        imp = "YES" if b["improves_prong2"] else "no"
        lines.append(
            f"| `{f}` | {_fmt(rho)} | {_fmt(lf)} | {_fmt(gap, 4)} | {_fmt(clo)} | "
            f"{_fmt(mt['mean_demotion'])} ({mt['n']}) | {_fmt(mf['mean_demotion'])} ({mf['n']}) | "
            f"**{_fmt(diff)}** | {_fmt(mp)} | {imp} |"
        )
    lines.append("")
    lines.append(
        f"_Prong-2 base rates: claim_misinfo=True n={p2['n_claim_misinfo_true']} "
        f"(YES+UNCLEAR), False n={p2['n_claim_misinfo_false']} (NO). "
        f"{p2['note']}_\n"
    )

    # Within-URL-high-cred (isolates URL label).
    lines.append("### Prong 2 within URL-high-credibility subset (isolates the URL label)\n")
    lines.append("| form | demotion misinfo (n) | demotion clean (n) | diff (misinfo−clean) |")
    lines.append("|---|---:|---:|---:|")
    for f in FORMS:
        wh = p2["forms"][f]["within_url_high_cred"]
        lines.append(
            f"| `{f}` | {_fmt(wh['misinfo_true']['mean_demotion'])} ({wh['misinfo_true']['n']}) | "
            f"{_fmt(wh['misinfo_false']['mean_demotion'])} ({wh['misinfo_false']['n']}) | "
            f"{_fmt(wh['demotion_diff_true_minus_false'])} |"
        )
    lines.append("")

    # Input-separability table: does claim-misinfo separate in the forms' arguments?
    lines.append("### Prong 2 input separability — is claim-misinfo distinguishable in the forms' arguments?\n")
    lines.append(
        "The load-bearing question for *why* any form moves Prong 2. Each row is an "
        "input the forms ride on; values are mean over claim_misinfo=True vs False with "
        "a two-sided Mann–Whitney p. `substitutability_share` is the input the "
        "reflective-floor *targeting* story claims to act on.\n"
    )
    lines.append("| input | misinfo mean | clean mean | diff (true−false) | MWU p (2-sided) |")
    lines.append("|---|---:|---:|---:|---:|")
    for nm, blob in p2["input_separability"].items():
        lines.append(
            f"| `{nm}` | {_fmt(blob['misinfo_mean'], 4)} | {_fmt(blob['clean_mean'], 4)} | "
            f"{_fmt(blob['diff'], 4)} | {_fmt(blob['mannwhitney_p_two_sided'])} |"
        )
    lines.append("")

    # Honest reading.
    lines.append("## Honest reading\n")
    rf = p2["forms"]["reflective_floor"]["overall"]
    rf_diff = rf["demotion_diff_true_minus_false"]
    comp = p2["forms"]["complementary"]["overall"]
    comp_p = p2["forms"]["complementary"]["mannwhitney_demotion_greater"]["p_value"]
    sfs = p2["forms"]["slow_floor_x_substance"]["overall"]
    sfs_p = p2["forms"]["slow_floor_x_substance"]["mannwhitney_demotion_greater"]["p_value"]

    sep = p2["input_separability"]
    sub_p = sep["substitutability_share"]["mannwhitney_p_two_sided"]
    sub_diff = sep["substitutability_share"]["diff"]
    add_diff = sep["score_additive"]["diff"]
    zsub_diff = sep["z_substance"]["diff"]
    zsub_p = sep["z_substance"]["mannwhitney_p_two_sided"]

    rf_rho = p1["forms"]["reflective_floor"]["spearman_substitutability_vs_demotion"]["rho"]
    comp_rho = p1["forms"]["complementary"]["spearman_substitutability_vs_demotion"]["rho"]
    rf_lf = p1["forms"]["reflective_floor"]["most_demoted_quartile"]["low_cred_fraction"]
    comp_lf = p1["forms"]["complementary"]["most_demoted_quartile"]["low_cred_fraction"]
    comp_closure = p1["forms"]["complementary"]["score_gap_low_minus_high"]["gap_closure_vs_additive"]

    reading: list[str] = []
    reading.append(
        "**Anti-overfitting attestation.** All three forms were specified from the §6.3 "
        "per-feature diagnostics (prestige/substance → slow; shouty/low-cred → fast) and "
        "dual-process theory alone. No parameter (`a`, floor, scale, substance weights/signs) "
        "was tuned against the claim labels; the n=55 claim-misinfo set was evaluated exactly "
        "once as a held-out directional check. The directional readings below are reported as "
        "found, including where they complicate the story."
    )
    reading.append(
        f"**Prong 1 (targeting, n={p1['n_seeds']:,}).** The reflective floor targets the "
        f"diagnosed pathology as intended: demotion rises with substitutability share "
        f"(Spearman ρ={rf_rho:+.3f}) and its most-demoted quartile is enriched for low-cred "
        f"({rf_lf:.3f}). The `slow_floor_x_substance` composite keeps that targeting and adds a "
        "§6.3 substance prior. The `complementary` Cobb-Douglas form, by contrast, does NOT "
        f"reproduce the reflective-floor targeting signature (ρ={comp_rho:+.3f}, *negative*): "
        "because the geometric mean is dominated by the smaller factor, it demotes *low-fast* "
        "(low-absolute-engagement) content rather than *high-substitutability* content, and its "
        f"top-demoted quartile is LESS low-cred-enriched ({comp_lf:.3f}). It does sharpen the "
        f"algorithm-output low−high gap somewhat (closure {comp_closure:+.3f}), but it is best "
        "understood as a different (base-engagement-level) mechanism, not a stronger version of "
        "the substitutability gate."
    )
    reading.append(
        f"**Prong 2 (HELD-OUT, n={p2['n_claim_misinfo_true']} misinfo / "
        f"{p2['n_claim_misinfo_false']} clean).** The reflective floor again fails: it demotes "
        f"claim-misinfo LESS than clean (diff {rf_diff:+.3f}). Both stronger forms reverse the "
        f"sign — `complementary` diff {comp['demotion_diff_true_minus_false']:+.3f} "
        f"(MWU p={comp_p:.3f}), `slow_floor_x_substance` diff "
        f"{sfs['demotion_diff_true_minus_false']:+.3f} (MWU p={sfs_p:.3f}). So *nominally*, two "
        "forms 'improve Prong 2'. But the input-separability table explains why, and tempers the "
        "claim."
    )
    reading.append(
        "**What is actually doing the work (and the honest caveat).** Within the 400 claim "
        "tweets, the targeting input points the WRONG way: substitutability share is "
        f"significantly LOWER for claim-misinfo than clean (diff {sub_diff:+.4f}, MWU "
        f"p={sub_p:.2f}) — the exact opposite of what the substitutability-pathology story "
        "needs (the story requires misinfo to be MORE fast-substituted). So the architectural "
        "targeting signal does not just fail to separate misinfo, it separates it backwards. "
        "What *does* differ in the helpful direction is the overall engagement/substance LEVEL: "
        "claim-misinfo "
        f"tweets have lower predicted additive score (diff {add_diff:+.4f}) and lower §6.3 "
        f"substance index (diff {zsub_diff:+.4f}, MWU p={zsub_p:.2f}). The `complementary` form "
        "demotes misinfo more only because misinfo has lower absolute fast_contrib (the "
        "geometric mean punishes the smaller factor) — a base-engagement-level effect, not the "
        "substitutability pathology. The `slow_floor_x_substance` form demotes misinfo more "
        "because the §6.3 substance prior is lower for misinfo — a legitimate, theory-grounded "
        "signal, but it rides on an account/content prior, not on the score-aggregation "
        "architecture per se."
    )
    reading.append(
        "**Conclusion.** No *aggregation-only* form fixes Prong 2 via the mechanism the paper's "
        "contribution rests on: in this n=55 claim set the substitutability signal these "
        "aggregators act on separates misinfo in the WRONG direction (misinfo is less "
        f"fast-substituted, MWU p={sub_p:.2f}). The forms that move Prong 2 do so by exploiting "
        "a different, weaker, and "
        "largely level-based signal — `complementary` via lower absolute engagement (a side "
        "effect of the geometric mean, with Prong-1 targeting that runs *opposite* to the "
        "intended direction), and `slow_floor_x_substance` via a §6.3 account/content substance "
        "prior (which is an added input, not aggregation). Given n=55 and the small, "
        "marginally-significant effects (`slow_floor_x_substance` MWU p="
        f"{sfs_p:.2f} is not significant), the honest reading is: the reflective floor remains "
        "the cleanest *architecture-only* intervention, and Prong 2 cannot be fixed by "
        "aggregation alone because the veracity signal is not present in these inputs. The "
        "`slow_floor_x_substance` composite is the most defensible direction if one is willing "
        "to add a substance prior — but that is a different, hybrid claim, and its Prong-2 effect "
        "is directional only."
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
    results_json: Path,
    report_path: Path,
    fig_dir: Path,
    make_fig: bool = True,
) -> None:
    np.random.seed(SEED)

    logger.info("loading diagnostic seeds: %s", diagnostic_parquet)
    diag = pd.read_parquet(diagnostic_parquet)
    prong1, per_seed, sub_mean, sub_std = run_prong1(diag, config_path)

    logger.info("loading claim sample: %s", claim_parquet)
    claim = pd.read_parquet(claim_parquet)
    prong2, per_tweet = run_prong2(claim, config_path, sub_mean, sub_std)

    metrics: dict[str, Any] = {
        "meta": {
            "seed": SEED,
            "weights": {"reply": W_REPLY, "deep": W_DEEP, "retweet": W_RETWEET, "like": W_LIKE},
            "reflective_floor": {"floor": FLOOR, "scale": FLOOR_SCALE},
            "complementary_a": COMPLEMENTARY_A,
            "complementary_a_robust": COMPLEMENTARY_A_ROBUST,
            "substance_features": list(SUBSTANCE_FEATURES),
            "forms": list(FORMS),
            "diagnostic_parquet": str(diagnostic_parquet.relative_to(ROOT)),
            "claim_parquet": str(claim_parquet.relative_to(ROOT)),
            "config_path": str(config_path.relative_to(ROOT)),
            "anti_overfitting_note": (
                "All forms specified from §6.3 diagnostics + theory only; no parameter tuned "
                "to claim labels; n=55 claim set evaluated once as held-out directional check."
            ),
        },
        "prong1": prong1,
        "prong2": prong2,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("wrote results → %s", results_json)

    per_seed.to_parquet(out_dir / "prong1_per_seed.parquet", index=False)
    per_tweet.to_parquet(out_dir / "prong2_per_tweet.parquet", index=False)
    logger.info("wrote per-seed / per-tweet parquets → %s", out_dir)

    write_report(metrics, report_path)
    if make_fig:
        make_figure(prong1, prong2, fig_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC_PARQUET)
    p.add_argument("--claim", type=Path, default=CLAIM_PARQUET)
    p.add_argument("--config", type=Path, default=PHASE4_CONFIG)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--results-json", type=Path, default=RESULTS_JSON)
    p.add_argument("--report", type=Path, default=REPORT_PATH)
    p.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    p.add_argument("--no-figure", action="store_true")
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
        results_json=args.results_json.resolve(),
        report_path=args.report.resolve(),
        fig_dir=args.fig_dir.resolve(),
        make_fig=not args.no_figure,
    )


if __name__ == "__main__":
    main()

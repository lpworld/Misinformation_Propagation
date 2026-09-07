"""Publication-quality figure generation for the ISR paper.

Produces the full figure set for the "Algorithmic Mechanism of Misinformation
Propagation" paper. Each figure is written as BOTH a vector PDF (for LaTeX)
and a 200-dpi PNG (for review) into ``paper/figures/``.

Run from the repo root with the project venv (NOT uv)::

    .venv/Scripts/python.exe -m analysis.make_figures

Design conventions (ISR-appropriate):
  * 9-10pt fonts, no chartjunk, clear axis labels with units.
  * Bootstrap CIs drawn as error bars / bands.
  * Dashed reference line at 0 wherever a contrast is shown.
  * Colorblind-safe palette (Wong / Okabe-Ito).
  * tight_layout() on every figure.
  * Single-column width ~3.4in, double-column ~6.8in.

The script is modular: one function per figure, a ``main()`` that calls all,
and per-figure error isolation — a missing optional source logs a warning and
skips that figure rather than crashing the run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Stage-1 metric definitions reused verbatim from the analysis module.
from analysis import validation as v

logger = logging.getLogger("make_figures")

# ---- paths --------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "paper" / "figures"
PROC = REPO / "data" / "processed"
P4 = PROC / "phase4"

PHASE4_METRICS = P4 / "phase4_metrics.json"
PER_CASCADE = P4 / "ablation_per_cascade.parquet"
PER_BIN = P4 / "ablation_per_bin.parquet"
PART1 = PROC / "part_1.parquet"
TRUNC = P4 / "truncation_sensitivity" / "truncation_sensitivity_metrics.json"
SUBGROUP = P4 / "subgroup_metrics.json"
DIAGNOSTIC = P4 / "ranker_predictions" / "diagnostic.parquet"
MECHANISM = P4 / "mechanism" / "mechanism_per_seed.parquet"
SWEEP = PROC / "phase5_extra" / "hyperparam_sensitivity" / "sweep_results.csv"

# ---- style --------------------------------------------------------------

# Okabe-Ito colorblind-safe palette.
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

SINGLE = 3.4   # inches, single-column
DOUBLE = 6.8   # inches, double-column

LABEL_LOW = "low_credibility"
LABEL_HIGH = "high_credibility"


def _apply_style() -> None:
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
        "pdf.fonttype": 42,   # editable text in vector PDF
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, name: str) -> list[Path]:
    """Save a figure as PDF + 200-dpi PNG; return written paths."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pdf = FIGDIR / f"{name}.pdf"
    png = FIGDIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    for p in (pdf, png):
        written.append(p)
        logger.info("wrote %s (%d bytes)", p.name, p.stat().st_size)
    return written


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


# ========================================================================
# Figure 1 — Stage-1 validation: simulated vs observed distributions
# ========================================================================

def fig_stage1_validation() -> list[Path]:
    if not (PER_CASCADE.exists() and PART1.exists() and PER_BIN.exists()):
        logger.warning("fig_stage1_validation: missing source(s); skipping")
        return []

    obs = pd.read_parquet(PART1, columns=[
        "id_str", "conversationId", "epoch",
        "replyCount", "retweetCount", "likeCount", "quoteCount",
    ])
    pc = pd.read_parquet(PER_CASCADE)
    sim_pc = pc[pc["regime"] == "additive"].copy()
    pb = pd.read_parquet(PER_BIN)
    sim_pb = pb[pb["regime"] == "additive"].copy()

    # Metric values via the canonical validation.py definitions.
    panels = [
        (
            "Root reply count",
            "root.replyCount (per conversation)",
            v.observed_root_reply_counts(obs),
            v.simulated_root_reply_counts(sim_pc),
            0.057, True,
        ),
        (
            "Aggregate engagement",
            "Σ(reply+retweet+quote) per cascade",
            v.observed_aggregate_engagement(obs),
            v.simulated_aggregate_engagement(sim_pc),
            0.146, True,
        ),
        (
            "Reactive / reflective ratio",
            "reply / (retweet+quote+1)",
            v.observed_reactive_to_reflective(obs),
            v.simulated_reactive_to_reflective(sim_pc),
            0.061, True,
        ),
        (
            "Time to peak",
            "hours from root to peak reply rate",
            v.observed_time_to_peak_hours(obs),
            v.simulated_time_to_peak_hours(sim_pb),
            0.213, False,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.8))
    for ax, (title, xlab, o, s, ks, logx) in zip(axes.ravel(), panels):
        o = np.asarray(o, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        if logx:
            # ECDF on log1p axis: heavy-tailed counts.
            xo, yo = _ecdf(np.log1p(o))
            xs, ys = _ecdf(np.log1p(s))
            ax.plot(xo, yo, color=CB["blue"], label="observed", lw=1.4)
            ax.plot(xs, ys, color=CB["orange"], label="simulated",
                    lw=1.4, ls="--")
            ax.set_xlabel(f"{xlab}\nlog(1+x)")
        else:
            xo, yo = _ecdf(o)
            xs, ys = _ecdf(s)
            ax.plot(xo, yo, color=CB["blue"], label="observed", lw=1.4)
            ax.plot(xs, ys, color=CB["orange"], label="simulated",
                    lw=1.4, ls="--")
            ax.set_xlabel(xlab)
        ax.set_ylabel("ECDF")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="both")
        ax.annotate(
            f"KS = {ks:.3f}",
            xy=(0.96, 0.06), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=CB["grey"], lw=0.6, alpha=0.9),
        )
    axes.ravel()[0].legend(loc="center right", frameon=False)
    fig.suptitle("Stage-1 validation: simulated vs observed cascade signatures",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    return _save(fig, "fig_stage1_validation")


# ========================================================================
# Figure 2 — Headline contrast forest (audience_reach, regime − additive)
# ========================================================================

def fig_headline_contrast() -> list[Path]:
    if not PHASE4_METRICS.exists():
        logger.warning("fig_headline_contrast: missing phase4_metrics.json; skipping")
        return []
    m = json.loads(PHASE4_METRICS.read_text())
    ar = m["stage2_bootstrap_contrasts"]["audience_reach"]

    # From phase4_metrics.json (ablated, additive_retuned); from phase5_report
    # Sweep 1 for ratio_correction / reflective_floor.
    rows = [
        ("ablated", ar["ablated"]["diff_point"],
         ar["ablated"]["diff_ci_lo"], ar["ablated"]["diff_ci_hi"], CB["vermilion"]),
        ("ratio_correction", -2.618, -3.178, -2.043, CB["green"]),
        ("reflective_floor", -37.762, -47.893, -27.686, CB["purple"]),
        ("additive_retuned", ar["additive_retuned"]["diff_point"],
         ar["additive_retuned"]["diff_ci_lo"], ar["additive_retuned"]["diff_ci_hi"],
         CB["blue"]),
    ]

    fig, ax = plt.subplots(figsize=(DOUBLE, 2.9))
    ys = np.arange(len(rows))[::-1]
    for y, (name, pt, lo, hi, c) in zip(ys, rows):
        ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="o",
                    color=c, ecolor=c, elinewidth=1.4, capsize=3,
                    markersize=5)
        ax.annotate(f"{pt:+.1f}", xy=(pt, y), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, color=c)
    ax.axvline(0.0, color="black", ls="--", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("audience-reach contrast  (regime − additive, exposures)")
    ax.set_title("Same weights, different functional form, opposite signs",
                 fontsize=10)
    ax.grid(True, axis="x")
    fig.tight_layout()
    return _save(fig, "fig_headline_contrast")


# ========================================================================
# Figure 3 — Truncation sensitivity (cascade_size contrast vs cap quantile)
# ========================================================================

def fig_truncation() -> list[Path]:
    if not TRUNC.exists():
        logger.warning("fig_truncation: missing truncation_sensitivity_metrics.json; skipping")
        return []
    t = json.loads(TRUNC.read_text())

    base = t["baseline_untruncated"]["cascade_size"]["ablated"]
    points = [("uncapped", base["diff_point"], base["diff_ci_lo"], base["diff_ci_hi"])]
    # cap quantiles p75..p99 (skip p50 cap=0 degenerate at 0).
    qorder = ["75.0", "90.0", "95.0", "99.0"]
    labels = {"75.0": "p75", "90.0": "p90", "95.0": "p95", "99.0": "p99"}
    for q in qorder:
        if q not in t["truncated"]:
            continue
        a = t["truncated"][q]["regimes"]["ablated"]
        points.append((labels[q], a["diff_point"], a["diff_ci_lo"], a["diff_ci_hi"]))

    xs = np.arange(len(points))
    pts = np.array([p[1] for p in points])
    lo = np.array([p[2] for p in points])
    hi = np.array([p[3] for p in points])

    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    ax.fill_between(xs, lo, hi, color=CB["vermilion"], alpha=0.18, lw=0)
    ax.plot(xs, pts, "-o", color=CB["vermilion"], markersize=4)
    ax.axhline(0.0, color="black", ls="--", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in points])
    ax.set_xlabel("cascade-size cap (winsorize quantile)")
    ax.set_ylabel("cascade-size contrast\n(ablated − additive, events)")
    ax.set_title("Contrast collapses toward 0 under truncation", fontsize=9.5)
    ax.grid(True, axis="y")
    fig.tight_layout()
    return _save(fig, "fig_truncation")


# ========================================================================
# Figure 4 — Robustness forest (audience_reach contrast across sweeps)
# ========================================================================

def fig_robustness_forest() -> list[Path]:
    # Values drawn from paper/phase5_report.md and paper/phase5_extra_report.md.
    # (regime/cell label, point, ci_lo, ci_hi, group)
    rows = [
        # Sweep 1 — alternative ablation forms
        ("ablated (F1)", -16.085, -25.436, -6.958, "Check 1: forms"),
        ("ratio_correction (F2)", -2.618, -3.178, -2.043, "Check 1: forms"),
        ("reflective_floor (F3)", -37.762, -47.893, -27.686, "Check 1: forms"),
        # Sweep 3 — ranker training seed
        ("seed 1337", -16.085, -24.463, -7.476, "Check 3: ranker seed"),
        ("seed 42", -16.612, -24.893, -8.729, "Check 3: ranker seed"),
        ("seed 7", -21.330, -29.126, -13.567, "Check 3: ranker seed"),
        ("seed 100", -15.490, -22.795, -8.701, "Check 3: ranker seed"),
        ("seed 2024", -16.510, -24.250, -9.164, "Check 3: ranker seed"),
        # Sweep 4 — data parts
        ("part_1", -16.085, -24.463, -7.476, "Check 4: data part"),
        ("part_2", -16.251, -25.091, -6.196, "Check 4: data part"),
        # RB2 — user pool size
        ("25k users", -16.08, -24.46, -7.48, "Check 6: pool size"),
        ("50k users", -16.08, -24.46, -7.48, "Check 6: pool size"),
        ("100k users", -16.08, -24.46, -7.48, "Check 6: pool size"),
        # RB3 — enlarged training corpus
        ("parts_1_2 (2.0M)", -17.22, -25.14, -9.74, "Check 7: train corpus"),
        ("parts_1_to_5 (5.0M)", -19.28, -27.62, -10.96, "Check 7: train corpus"),
    ]

    group_color = {
        "Check 1: forms": CB["vermilion"],
        "Check 3: ranker seed": CB["blue"],
        "Check 4: data part": CB["green"],
        "Check 6: pool size": CB["orange"],
        "Check 7: train corpus": CB["purple"],
    }

    fig, ax = plt.subplots(figsize=(DOUBLE, 5.0))
    ys = np.arange(len(rows))[::-1]
    seen_groups: set[str] = set()
    for y, (label, pt, lo, hi, grp) in zip(ys, rows):
        c = group_color[grp]
        show_label = grp if grp not in seen_groups else None
        seen_groups.add(grp)
        ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="o", color=c,
                    ecolor=c, elinewidth=1.2, capsize=2.5, markersize=4.5,
                    label=show_label)
    ax.axvline(0.0, color="black", ls="--", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("audience-reach contrast  (ablated − additive, exposures)")
    ax.set_title("Robustness: contrast stays < 0 across every sweep", fontsize=10)
    ax.legend(loc="lower left", frameon=False, ncol=1)
    ax.grid(True, axis="x")
    fig.tight_layout()
    return _save(fig, "fig_robustness_forest")


# ========================================================================
# Figure 5 — Subgroup: audience_reach contrast by follower quintile
# ========================================================================

def fig_subgroup() -> list[Path]:
    if not SUBGROUP.exists():
        logger.warning("fig_subgroup: missing subgroup_metrics.json; skipping")
        return []
    s = json.loads(SUBGROUP.read_text())
    bins = s["cuts"]["follower_count"]["bins"]
    bins = sorted(bins, key=lambda b: b["bin"])

    labels = [f"Q{b['bin'] + 1}" for b in bins]
    pts = np.array([b["ar_ablated"]["diff_point"] for b in bins])
    lo = np.array([b["ar_ablated"]["diff_ci_lo"] for b in bins])
    hi = np.array([b["ar_ablated"]["diff_ci_hi"] for b in bins])
    xs = np.arange(len(bins))

    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    ax.errorbar(xs, pts, yerr=[pts - lo, hi - pts], fmt="-o",
                color=CB["blue"], ecolor=CB["blue"], elinewidth=1.3,
                capsize=3, markersize=4.5)
    ax.axhline(0.0, color="black", ls="--", lw=0.9)
    # Flag Q5's wide CI.
    ax.annotate("Q5: wide CI\n(crosses 0)", xy=(xs[-1], pts[-1]),
                xytext=(xs[-1] - 1.5, pts[-1] - 12),
                fontsize=7, color=CB["grey"],
                arrowprops=dict(arrowstyle="->", color=CB["grey"], lw=0.7))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("follower-count quintile (Q1 low → Q5 high)")
    ax.set_ylabel("audience-reach contrast\n(ablated − additive, exposures)")
    ax.set_title("Suppression strengthens monotonically with reach", fontsize=9.5)
    ax.grid(True, axis="y")
    fig.tight_layout()
    return _save(fig, "fig_subgroup")


# ========================================================================
# Figure 6 — Predicted reactive/reflective ratio by credibility class
# ========================================================================

def fig_predicted_rr() -> list[Path]:
    if not DIAGNOSTIC.exists():
        logger.warning("fig_predicted_rr: missing diagnostic.parquet; skipping")
        return []
    dg = pd.read_parquet(DIAGNOSTIC, columns=[
        "credibility_label", "predicted_reactive_over_reflective"])

    low = dg.loc[dg["credibility_label"] == LABEL_LOW,
                 "predicted_reactive_over_reflective"].to_numpy()
    high = dg.loc[dg["credibility_label"] == LABEL_HIGH,
                  "predicted_reactive_over_reflective"].to_numpy()
    mean_low, mean_high = float(low.mean()), float(high.mean())

    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    lo_x = min(low.min(), high.min())
    hi_x = max(np.percentile(low, 99.5), np.percentile(high, 99.5))
    bins = np.linspace(lo_x, hi_x, 45)
    ax.hist(low, bins=bins, density=True, histtype="stepfilled",
            color=CB["vermilion"], alpha=0.45, label="low-cred.")
    ax.hist(high, bins=bins, density=True, histtype="stepfilled",
            color=CB["blue"], alpha=0.45, label="high-cred.")
    ax.axvline(mean_low, color=CB["vermilion"], ls="--", lw=1.1)
    ax.axvline(mean_high, color=CB["blue"], ls="--", lw=1.1)
    ax.annotate(f"low mean = {mean_low:.3f}", xy=(mean_low, ax.get_ylim()[1] * 0.62),
                xytext=(4, 0), textcoords="offset points",
                color=CB["vermilion"], fontsize=7, ha="left")
    ax.annotate(f"high mean = {mean_high:.3f}", xy=(mean_high, ax.get_ylim()[1] * 0.74),
                xytext=(-4, 0), textcoords="offset points",
                color=CB["blue"], fontsize=7, ha="right")
    ax.set_xlabel("predicted reactive / reflective ratio")
    ax.set_ylabel("density")
    ax.set_title("Low-credibility content skews reactive", fontsize=9.5)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y")
    fig.tight_layout()
    return _save(fig, "fig_predicted_rr")


# ========================================================================
# Figure 7 — Reflective-floor gate value by credibility class
# ========================================================================

def fig_mechanism_gate() -> list[Path]:
    if not MECHANISM.exists():
        logger.warning("fig_mechanism_gate: missing mechanism_per_seed.parquet; skipping")
        return []
    mp = pd.read_parquet(MECHANISM, columns=["credibility_label", "gate"])
    low = mp.loc[mp["credibility_label"] == LABEL_LOW, "gate"].to_numpy()
    high = mp.loc[mp["credibility_label"] == LABEL_HIGH, "gate"].to_numpy()
    mean_low, mean_high = float(low.mean()), float(high.mean())
    supp_low = float((low < 0.5).mean())
    supp_high = float((high < 0.5).mean())

    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    lo_x = min(low.min(), high.min())
    hi_x = max(low.max(), high.max())
    bins = np.linspace(lo_x, hi_x, 45)
    ax.hist(low, bins=bins, density=True, histtype="stepfilled",
            color=CB["vermilion"], alpha=0.45, label="low-cred.")
    ax.hist(high, bins=bins, density=True, histtype="stepfilled",
            color=CB["blue"], alpha=0.45, label="high-cred.")
    ax.axvline(mean_low, color=CB["vermilion"], ls="--", lw=1.1)
    ax.axvline(mean_high, color=CB["blue"], ls="--", lw=1.1)
    ax.axvline(0.5, color=CB["grey"], ls=":", lw=1.0)
    ax.annotate(f"low mean = {mean_low:.3f}", xy=(mean_low, ax.get_ylim()[1] * 0.72),
                xytext=(-4, 0), textcoords="offset points",
                color=CB["vermilion"], fontsize=7, ha="right")
    ax.annotate(f"high mean = {mean_high:.3f}", xy=(mean_high, ax.get_ylim()[1] * 0.86),
                xytext=(4, 0), textcoords="offset points",
                color=CB["blue"], fontsize=7, ha="left")
    ax.set_title("Reflective-floor gate suppresses low-cred. more", fontsize=9.5)
    ax.set_xlabel("reflective-floor gate value  (g)")
    ax.set_ylabel("density")
    ax.legend(loc="upper center", frameon=False)
    ax.text(0.5, ax.get_ylim()[1] * 0.55,
            f"suppressed (g<0.5):\nlow {supp_low*100:.1f}%  vs  high {supp_high*100:.1f}%",
            fontsize=7, ha="center", color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CB["grey"],
                      lw=0.6, alpha=0.9))
    ax.grid(True, axis="y")
    fig.tight_layout()
    return _save(fig, "fig_mechanism_gate")


# ========================================================================
# Figure 8 — Hyperparameter sensitivity (2 panels)
# ========================================================================

def fig_hyperparam_sensitivity() -> list[Path]:
    if not SWEEP.exists():
        logger.warning("fig_hyperparam_sensitivity: missing sweep_results.csv; skipping")
        return []
    sw = pd.read_csv(SWEEP)
    ar = sw[sw["metric"] == "audience_reach"].copy()

    f1 = ar[ar["form"] == "F1"].sort_values("alpha")
    f2 = ar[ar["form"] == "F2"].sort_values("alpha")
    f3 = ar[ar["form"] == "F3"].copy()
    n_f3 = len(f3)
    all_neg = bool((f3["diff_point"] < 0).all())

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(DOUBLE, 3.1))

    # Panel (a): F1 ablated & F2 ratio_correction vs alpha (log-x).
    axa.fill_between(f1["alpha"], f1["ci_lo"], f1["ci_hi"],
                     color=CB["vermilion"], alpha=0.15, lw=0)
    axa.plot(f1["alpha"], f1["diff_point"], "-o", color=CB["vermilion"],
             markersize=4, label="F1 ablated")
    axa.fill_between(f2["alpha"], f2["ci_lo"], f2["ci_hi"],
                     color=CB["green"], alpha=0.15, lw=0)
    axa.plot(f2["alpha"], f2["diff_point"], "-s", color=CB["green"],
             markersize=4, label="F2 ratio_correction")
    axa.axhline(0.0, color="black", ls="--", lw=0.9)
    axa.set_xscale("log")
    axa.set_xlabel(r"$\alpha$ (fast-boost strength)")
    axa.set_ylabel("audience-reach contrast\n(regime − additive)")
    axa.set_title("(a) F1 / F2 vs $\\alpha$", fontsize=9.5)
    axa.legend(loc="lower left", frameon=False)
    axa.grid(True, which="both", axis="both")

    # Panel (b): F3 reflective_floor across floor x scale.
    floors = sorted(f3["floor"].unique())
    scales = sorted(f3["floor_scale"].unique())
    x = np.arange(len(floors))
    width = 0.8 / len(scales)
    scale_colors = [CB["sky"], CB["blue"], CB["purple"]]
    # Clip extreme cells (floor>=1.5 blow-ups) so the plot stays readable.
    clip = -60.0
    for j, sc in enumerate(scales):
        sub = f3[f3["floor_scale"] == sc].set_index("floor").reindex(floors)
        vals = sub["diff_point"].to_numpy()
        clipped = np.clip(vals, clip, 0)
        bars = axb.bar(x + (j - (len(scales) - 1) / 2) * width, clipped,
                       width=width, color=scale_colors[j % len(scale_colors)],
                       label=f"scale={sc:g}", edgecolor="white", linewidth=0.4)
        # Mark clipped (off-scale) bars.
        for xi, raw in zip(x + (j - (len(scales) - 1) / 2) * width, vals):
            if raw < clip:
                axb.annotate(f"{raw:.0f}", xy=(xi, clip), xytext=(0, -2),
                             textcoords="offset points", ha="center",
                             va="top", fontsize=5.5, color=CB["grey"], rotation=90)
    axb.axhline(0.0, color="black", ls="--", lw=0.9)
    axb.set_ylim(clip * 1.05, 4)
    axb.set_xticks(x)
    axb.set_xticklabels([f"{f:g}" for f in floors])
    axb.set_xlabel("reflective floor")
    axb.set_ylabel("audience-reach contrast (clipped)")
    axb.set_title("(b) F3 reflective_floor", fontsize=9.5)
    axb.legend(loc="lower right", frameon=False, fontsize=7)
    axb.grid(True, axis="y")
    axb.text(0.02, 0.04,
             f"all {n_f3} cells < 0\n(direction never flips)"
             if all_neg else f"{(f3['diff_point']<0).sum()}/{n_f3} cells < 0",
             transform=axb.transAxes, fontsize=7, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CB["grey"],
                       lw=0.6, alpha=0.9))
    fig.tight_layout()
    return _save(fig, "fig_hyperparam_sensitivity")


# ========================================================================
# main
# ========================================================================

FIGURES = [
    fig_stage1_validation,
    fig_headline_contrast,
    fig_truncation,
    fig_robustness_forest,
    fig_subgroup,
    fig_predicted_rr,
    fig_mechanism_gate,
    fig_hyperparam_sensitivity,
]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    np.random.seed(1337)
    _apply_style()

    written: list[Path] = []
    skipped: list[str] = []
    for fn in FIGURES:
        try:
            out = fn()
            if out:
                written.extend(out)
            else:
                skipped.append(fn.__name__)
        except Exception as exc:  # isolate per-figure failures
            logger.warning("%s failed: %s — skipping", fn.__name__, exc,
                           exc_info=True)
            skipped.append(fn.__name__)

    logger.info("DONE: %d files written, %d figure(s) skipped",
                len(written), len(skipped))
    if skipped:
        logger.info("skipped: %s", ", ".join(skipped))


if __name__ == "__main__":
    main()

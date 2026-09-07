"""Community Notes cross-match — descriptive predicted-profile analysis.

The feasibility probe (2026-08-30) joined the public Community Notes corpus
(3.0M notes on 2.03M posts; snapshot 2026-08-26) against USC parts 1-5 on
full-precision tweet ids, yielding 851 in-corpus tweets with at least one
note, 805 of them with a note classifying the tweet as misleading and 31
with a misleading note rated CURRENTLY_RATED_HELPFUL. The matched set is
saved in ``data/processed/phase5_extra/community_notes/matches.parquet``.

This driver asks the descriptive question the paper's mechanism motivates:
do note-flagged tweets carry the fast-substitution signature the
reflective floor targets? For each matched tweet we compute, from the
canonical Phase-3 ranker, the predicted reactive-to-reflective ratio, the
substitutability share (fast-class contribution to the additive score),
and the reflective-floor gate value. Because notes concentrate on viral
original posts, the meaningful comparison is against engagement-matched
controls: non-noted tweets from the same parts, matched on originality and
log-decile of total observed engagement (10 controls per noted tweet,
seed 1337). A pooled-corpus random baseline is reported alongside for
orientation.

Note-flagged tweets are an independent, crowd-sourced claim-level signal —
none of the paper's models or labels were fit on them.

Run (repo root)::

    .venv\\Scripts\\python.exe -m analysis.run_community_notes_probe

Outputs:

* ``data/processed/phase5_extra/community_notes/metrics.json``
* ``data/processed/phase5_extra/community_notes/noted_predictions.parquet``
* ``paper/community_notes_report.md``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ranker.scoring import PUBLISHED_WEIGHTS
from ranker.training import load_ranker
from simulation.cascade import _seed_probs

logger = logging.getLogger("community_notes_probe")

ROOT = Path(__file__).resolve().parents[1]
SEED = 1337
N_BOOTSTRAP = 1000
CONTROLS_PER_NOTED = 10
FLOOR, FLOOR_SCALE = 1.0, 0.5

CN_DIR = ROOT / "data" / "processed" / "phase5_extra" / "community_notes"
MATCHES = CN_DIR / "matches.parquet"
RANKER_DIR = ROOT / "data" / "processed" / "phase3_full" / "ranker"
REPORT_MD = ROOT / "paper" / "community_notes_report.md"

USC_COLS = [
    "id", "epoch", "rawContent", "link_urls",
    "replyCount", "retweetCount", "likeCount", "quoteCount",
    "user_followersCount", "user_friendsCount", "user_statusesCount",
    "user_favouritesCount", "user_listedCount", "user_blue",
    "user_created_at", "is_reply", "is_quote", "is_original",
]

SLOW = ("reply", "deep")
FAST = ("retweet", "like")


def _profile_metrics(probs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Predicted R/R ratio, substitutability share, floor gate per item."""
    s_slow = sum(PUBLISHED_WEIGHTS[h] * probs[h] for h in SLOW)
    s_fast = sum(PUBLISHED_WEIGHTS[h] * probs[h] for h in FAST)
    s_add = s_slow + s_fast
    pred_rr = (probs["retweet"] + probs["like"]) / (probs["reply"] + probs["deep"])
    sub_share = s_fast / s_add
    gate = 1.0 / (1.0 + np.exp(-(s_slow - FLOOR) / FLOOR_SCALE))
    return pd.DataFrame({
        "pred_rr": pred_rr, "sub_share": sub_share, "gate": gate,
        "s_slow": s_slow, "s_fast": s_fast, "s_additive": s_add,
    })


def _bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, *, seed: int = SEED, n: int = N_BOOTSTRAP,
) -> tuple[float, float, float]:
    """Point and 95% CI for mean(a) - mean(b), independent resampling."""
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    boots = np.empty(n)
    for i in range(n):
        boots[i] = (
            a[rng.integers(0, a.size, a.size)].mean()
            - b[rng.integers(0, b.size, b.size)].mean()
        )
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _compare(name: str, noted: pd.DataFrame, ctrl: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"n_noted": int(len(noted)), "n_control": int(len(ctrl))}
    for m in ("pred_rr", "sub_share", "gate"):
        a = noted[m].to_numpy()
        b = ctrl[m].to_numpy()
        point, lo, hi = _bootstrap_mean_diff(a, b)
        mwu_p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        out[m] = {
            "noted_mean": float(a.mean()), "control_mean": float(b.mean()),
            "diff": point, "ci": [lo, hi], "mwu_p": mwu_p,
        }
        logger.info(
            "[%s] %-9s noted %.4f vs ctrl %.4f | diff %+.4f [%+.4f, %+.4f] "
            "MWU p=%.2e",
            name, m, a.mean(), b.mean(), point, lo, hi, mwu_p,
        )
    return out


def run() -> dict[str, Any]:
    np.random.seed(SEED)
    matches = pd.read_parquet(MATCHES)
    noted_ids = set(matches["id"].astype(str))
    logger.info("matched noted tweets: %d", len(noted_ids))

    model, scaler, _meta = load_ranker(RANKER_DIR)

    rng = np.random.default_rng(SEED)
    noted_rows: list[pd.DataFrame] = []
    control_rows: list[pd.DataFrame] = []
    baseline_rows: list[pd.DataFrame] = []

    for p in range(1, 6):
        df = pd.read_parquet(
            ROOT / "data" / "processed" / f"part_{p}.parquet", columns=USC_COLS
        )
        df["id"] = df["id"].astype(str)
        df["total_eng"] = (
            df[["replyCount", "retweetCount", "likeCount", "quoteCount"]]
            .apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        )
        df["eng_bin"] = np.floor(np.log10(df["total_eng"] + 1.0)).astype(int)
        is_noted = df["id"].isin(noted_ids)
        noted_p = df[is_noted].copy()
        pool = df[~is_noted]
        noted_rows.append(noted_p)

        # Engagement- and originality-matched controls, sampled per stratum.
        for (obin, orig), grp in noted_p.groupby(["eng_bin", "is_original"]):
            cands = pool[(pool["eng_bin"] == obin) & (pool["is_original"] == orig)]
            need = min(len(cands), CONTROLS_PER_NOTED * len(grp))
            if need == 0:
                # fall back one engagement bin down
                cands = pool[(pool["eng_bin"] == obin - 1) & (pool["is_original"] == orig)]
                need = min(len(cands), CONTROLS_PER_NOTED * len(grp))
            if need > 0:
                control_rows.append(
                    cands.sample(n=need, random_state=int(rng.integers(0, 2**31)))
                )
        baseline_rows.append(
            pool.sample(n=4000, random_state=int(rng.integers(0, 2**31)))
        )
        logger.info("part_%d: %d noted", p, len(noted_p))

    noted = pd.concat(noted_rows, ignore_index=True).drop_duplicates(subset="id")
    controls = pd.concat(control_rows, ignore_index=True).drop_duplicates(subset="id")
    baseline = pd.concat(baseline_rows, ignore_index=True).drop_duplicates(subset="id")
    noted = noted.merge(
        matches[["id", "any_misleading", "any_crh_misleading", "n_notes"]], on="id",
    )
    logger.info(
        "pools: noted=%d controls=%d baseline=%d", len(noted), len(controls), len(baseline)
    )

    frames = {}
    for name, pool_df in (("noted", noted), ("controls", controls), ("baseline", baseline)):
        probs = _seed_probs(pool_df.reset_index(drop=True), model, scaler)
        met = _profile_metrics(probs)
        met.index = pool_df.index
        frames[name] = pd.concat([pool_df.reset_index(drop=True), met.reset_index(drop=True)], axis=1)

    noted_m = frames["noted"]
    ctrl_m = frames["controls"]
    base_m = frames["baseline"]

    results: dict[str, Any] = {
        "meta": {
            "seed": SEED,
            "notes_snapshot": "2026-08-26",
            "controls_per_noted": CONTROLS_PER_NOTED,
            "matching": "is_original x floor(log10(total observed engagement + 1))",
            "ranker": "phase3_full (canonical)",
            "floor": FLOOR, "floor_scale": FLOOR_SCALE,
        },
        "vs_matched_controls": _compare("matched", noted_m, ctrl_m),
        "vs_corpus_baseline": _compare("baseline", noted_m, base_m),
        "misleading_only_vs_matched_controls": _compare(
            "misleading", noted_m[noted_m["any_misleading"]], ctrl_m
        ),
        "crh_misleading_vs_matched_controls": _compare(
            "crh", noted_m[noted_m["any_crh_misleading"]], ctrl_m
        ),
    }

    CN_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["id", "any_misleading", "any_crh_misleading", "n_notes", "total_eng",
            "is_original", "pred_rr", "sub_share", "gate", "s_slow", "s_fast",
            "s_additive"]
    noted_m[keep].to_parquet(CN_DIR / "noted_predictions.parquet", index=False)
    (CN_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results)
    return results


def write_report(results: dict[str, Any]) -> None:
    meta = results["meta"]
    L = [
        "# Community Notes cross-match — predicted-profile descriptives\n",
        f"Run: {pd.Timestamp.utcnow().isoformat()}  ",
        f"Notes snapshot {meta['notes_snapshot']} | controls matched on "
        f"{meta['matching']} ({meta['controls_per_noted']} per noted tweet) | "
        f"seed {meta['seed']}\n",
        "Note-flagged tweets are an independent crowd-sourced claim-level "
        "signal. The question is whether they carry the fast-substitution "
        "signature the reflective floor targets — computed from the canonical "
        "trained ranker's predictions, which never saw note data.\n",
        "| comparison | n | pred R/R (noted vs ctrl) | sub share (noted vs ctrl) | floor gate (noted vs ctrl) |",
        "|---|---|---|---|---|",
    ]

    def row(label: str, blob: dict[str, Any]) -> str:
        def cell(m):
            b = blob[m]
            star = "*" if b["mwu_p"] < 0.05 else ""
            return (
                f"{b['noted_mean']:.3f} vs {b['control_mean']:.3f} "
                f"({b['diff']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]){star}"
            )
        return (
            f"| {label} | {blob['n_noted']} vs {blob['n_control']} | "
            f"{cell('pred_rr')} | {cell('sub_share')} | {cell('gate')} |"
        )

    L.append(row("all noted vs matched controls", results["vs_matched_controls"]))
    L.append(row("all noted vs corpus baseline", results["vs_corpus_baseline"]))
    L.append(row("misleading-note vs matched controls", results["misleading_only_vs_matched_controls"]))
    L.append(row("CRH-misleading vs matched controls", results["crh_misleading_vs_matched_controls"]))
    L.append("")
    L.append("`*` = Mann-Whitney p < 0.05 (two-sided). CIs: 1,000-iteration "
             "bootstrap on the difference in means.\n")
    L.append(
        "**Caveats.** Notes concentrate on viral original posts, so the "
        "matched-control comparison is the meaningful one and the corpus "
        "baseline is orientation only. The matched sample is small (and the "
        "CRH subset very small); read directions and intervals, not point "
        "estimates. Deleted tweets cannot receive notes retroactively and the "
        "USC parts cover May-July 2024 only.\n"
    )
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", REPORT_MD)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run()


if __name__ == "__main__":
    main()

"""
H3 — Carry-over Effects in Referential Chains (XLSX pipeline)
=============================================================
Ported from Youssef's standalone H3.py (which read CSV files) onto the merged
notebook's already-loaded `corpus` DataFrame (XLSX pipeline).

WHY THIS PORT EXISTS
--------------------
Two distinct analyses were both informally called "H3" in the project:

  (1) Carry-over / order effect  ← the OFFICIAL H3 from the project plan:
      "The order of presentation (fictional first vs. real first) modulates
       discourse on the second stimulus, with measurable carry-over effects
       in referential chains."

  (2) "Territorial anchoring vs. uncertainty" (in the per-participant notebook):
      This actually measures the REAL-vs-FICTIONAL contrast, i.e. it is
      evidence for H1, NOT an order effect. It is renamed here as an
      "H1 support analysis" to remove the terminological clash.

This module implements (1) only. It is the canonical H3.

KEY DESIGN CHOICE — ORDER IS INFERRED FROM DATA, NOT FROM THE FILENAME LABEL
---------------------------------------------------------------------------
Youssef's original script infers presentation order from the first condition
actually spoken about in the transcript, rather than trusting an external
'order' label. We keep that choice because it caught a real bug: P9 is labelled
order='B' (fictional-first) in the merged notebook, but the data shows the
participant discusses the REAL map first. Filename + profile doc both say
order A. Inferring from data avoids trusting the wrong label.

Reuses from the notebook: nothing required beyond a `corpus` DataFrame with
columns: participant, condition ('Real'/'Not real'), text, start_time.
All H3 features are recomputed here (they are deliberately INDEPENDENT of the
H1 label counts, because carry-over is about cross-references and deixis, not
about the GEO/DEF/DEM/GEN/UNCERTAINTY taxonomy).
"""

import re
import numpy as np
import pandas as pd
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CFG = {
    "cond_real": "Real",
    "cond_notreal": "Not real",
    "transition_zone_turns": 5,
    "min_turns": 3,
    "alpha": 0.05,
    "n_bootstrap": 2000,
    "seed": 42,
}

# Deictic references (generic pointing)
DEICTIC_PATTERNS = [
    r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b",
    r"\bhere\b", r"\bthere\b", r"\bthe\s+previous\b",
    r"\bthe\s+other\b", r"\bthe\s+same\b",
    r"\bthe\s+first\b", r"\bthe\s+second\b",
]

# Anaphoric references (pronominal pick-up)
ANAPHORIC_PATTERNS = [
    r"\bit\b", r"\bthey\b", r"\bthem\b", r"\btheir\b",
    r"\bits\b", r"\bthis one\b", r"\bthat one\b",
    r"\bthe former\b", r"\bthe latter\b",
]

# Cross-condition references — explicit references to the OTHER stimulus.
# This is the core carry-over signal for H3.
CROSSREF_PATTERNS = [
    r"\bcompared\s+to\b",
    r"\bunlike\b",
    r"\bthe\s+previous\s+(map|one|image|visualization|visualisation)\b",
    r"\bthe\s+other\s+(map|one|image|visualization|visualisation)\b",
    r"\bthe\s+first\s+(map|one|image|visualization|visualisation)\b",
    r"\bthe\s+second\s+(map|one|image|visualization|visualisation)\b",
    r"\blast\s+time\b",
    r"\bsame\s+as\s+(before|last|the\s+previous)\b",
    r"\bjust\s+like\s+before\b",
    r"\bsimilar\s+to\s+(the\s+)?(previous|other|first|last)\b",
    r"\bremember\s+(the|when|that)\b",
    r"\bearlier\s+(map|one|we\s+saw)\b",
    r"\bboth\s+(maps|visualizations|visualisations|images)\b",
]

# Proper-noun proxy: capitalized word mid-sentence (not after . ! ?, not at start).
PROPER_NOUN_PATTERN = r"(?<![.!?]\s)(?<!\A)\b[A-ZÀ-Ö][a-zà-ö]{2,}\b"


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
def _count(text, patterns):
    if not isinstance(text, str) or not text.strip():
        return 0
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


def _ttr(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    toks = re.findall(r"\b\w+\b", text.lower())
    return len(set(toks)) / len(toks) if len(toks) >= 3 else 0.0


def infer_order(corpus):
    """Infer presentation order per participant from the FIRST condition spoken.
    Returns a dict participant -> 'real_first' | 'fictional_first' (or None)."""
    order = {}
    for pid, g in corpus.groupby("participant"):
        conds = g["condition"].dropna().tolist()
        if not conds:
            order[pid] = None
            continue
        order[pid] = "real_first" if conds[0] == CFG["cond_real"] else "fictional_first"
    return order


def build_turn_features(corpus):
    """Build per-turn carry-over features on Speaker-2 turns with a known condition.
    `corpus` must already be Speaker-2-only (as in the merged notebook)."""
    t = corpus[corpus["condition"].notna()].copy().reset_index(drop=True)

    t["n_deictic"]     = t["text"].apply(lambda x: _count(x, DEICTIC_PATTERNS))
    t["n_anaphoric"]   = t["text"].apply(lambda x: _count(x, ANAPHORIC_PATTERNS))
    t["n_crossref"]    = t["text"].apply(lambda x: _count(x, CROSSREF_PATTERNS))
    t["n_proper_noun"] = t["text"].apply(
        lambda x: len(re.findall(PROPER_NOUN_PATTERN, x)) if isinstance(x, str) else 0)
    t["cv_ttr"]        = t["text"].apply(_ttr)
    t["word_count"]    = t["text"].apply(
        lambda x: len(re.findall(r"\b\w+\b", x)) if isinstance(x, str) else 0)

    for f in ["n_deictic", "n_anaphoric", "n_crossref", "n_proper_noun"]:
        t[f"{f}_norm"] = t.apply(
            lambda r, ff=f: r[ff] / r["word_count"] if r["word_count"] > 0 else 0, axis=1)

    # Inferred order + first condition
    order = infer_order(corpus)
    t["presentation_order"] = t["participant"].map(order)
    first_cond = {p: ("Real" if o == "real_first" else "Not real")
                  for p, o in order.items() if o}
    t["first_condition"] = t["participant"].map(first_cond)

    # 2nd stimulus flag
    t["is_second_stimulus"] = t.apply(
        lambda r: (r["first_condition"] is not None)
        and (r["condition"] != r["first_condition"]), axis=1)

    # Turn position within each (participant, stimulus-half)
    t["turn_pos"] = t.groupby(["participant", "is_second_stimulus"]).cumcount()
    tz = CFG["transition_zone_turns"]
    t["in_transition_zone"] = t["is_second_stimulus"] & (t["turn_pos"] < tz)

    return t


# ──────────────────────────────────────────────────────────────────────────────
# STATS HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _bootstrap_ci(values, n_boot=None, ci=0.95):
    n_boot = n_boot or CFG["n_bootstrap"]
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(CFG["seed"])
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    a = (1 - ci) / 2
    return (float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a))))


def _rank_biserial(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return 1 - (2 * u) / (n1 * n2)


def _wilcoxon_r(w, n):
    if n < 2:
        return np.nan
    mean_w = n * (n + 1) / 2 / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mean_w) / std_w if std_w > 0 else 0
    return abs(z) / np.sqrt(n)


# ──────────────────────────────────────────────────────────────────────────────
# THE FOUR H3 TESTS
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = [
    ("n_deictic_norm",     "Deictic references"),
    ("n_proper_noun_norm", "Proper nouns"),
    ("n_anaphoric_norm",   "Anaphoric references"),
    ("n_crossref_norm",    "Cross-condition references"),
    ("cv_ttr",             "Type-Token Ratio"),
]


def run_h3_tests(turns):
    """Returns dict of DataFrames for T1..T4."""
    out = {}
    second = turns[turns["is_second_stimulus"]]
    g_rf = second[second["presentation_order"] == "real_first"]
    g_ff = second[second["presentation_order"] == "fictional_first"]

    # T1 — order effect on 2nd-stimulus discourse (between groups)
    rows = []
    for feat, label in FEATURES:
        rf = g_rf.groupby("participant")[feat].mean().dropna().values
        ff = g_ff.groupby("participant")[feat].mean().dropna().values
        if len(rf) < 2 or len(ff) < 2:
            continue
        u, p = stats.mannwhitneyu(rf, ff, alternative="two-sided")
        rows.append({
            "feature": label,
            "real_first_mean": round(float(rf.mean()), 5),
            "fictional_first_mean": round(float(ff.mean()), 5),
            "U": float(u), "p_value": round(float(p), 4),
            "effect_size_r": round(float(_rank_biserial(rf, ff)), 3),
            "significant": bool(p < CFG["alpha"]),
            "n_rf": int(len(rf)), "n_ff": int(len(ff)),
        })
    out["T1_order_effect_on_2nd_stimulus"] = pd.DataFrame(rows)

    # T2 — 1st vs 2nd stimulus within participant (the core carry-over test)
    rows = []
    for feat, label in FEATURES:
        a, b = [], []
        for pid, g in turns.groupby("participant"):
            f1 = g[~g["is_second_stimulus"]][feat].dropna().values
            f2 = g[g["is_second_stimulus"]][feat].dropna().values
            if len(f1) >= CFG["min_turns"] and len(f2) >= CFG["min_turns"]:
                a.append(f1.mean()); b.append(f2.mean())
        if len(a) < 3 or np.allclose(a, b):
            continue
        a, b = np.array(a), np.array(b)
        w, p = stats.wilcoxon(a, b)
        ci_a, ci_b = _bootstrap_ci(a), _bootstrap_ci(b)
        rows.append({
            "feature": label,
            "first_mean": round(float(a.mean()), 5),
            "first_95CI": f"[{ci_a[0]:.4f}, {ci_a[1]:.4f}]",
            "second_mean": round(float(b.mean()), 5),
            "second_95CI": f"[{ci_b[0]:.4f}, {ci_b[1]:.4f}]",
            "W": float(w), "p_value": round(float(p), 4),
            "effect_size_r": round(float(_wilcoxon_r(w, len(a))), 3),
            "significant": bool(p < CFG["alpha"]),
            "n_participants": int(len(a)),
        })
    out["T2_first_vs_second_stimulus"] = pd.DataFrame(rows)

    # T3 — cross-ref density on 2nd stimulus, by order group
    rows = []
    feat = "n_crossref_norm"
    rf = g_rf.groupby("participant")[feat].mean().dropna().values
    ff = g_ff.groupby("participant")[feat].mean().dropna().values
    if len(rf) >= 2 and len(ff) >= 2:
        u, p = stats.mannwhitneyu(rf, ff, alternative="two-sided")
        rows.append({
            "feature": "Cross-condition references",
            "real_first_mean": round(float(rf.mean()), 5),
            "fictional_first_mean": round(float(ff.mean()), 5),
            "U": float(u), "p_value": round(float(p), 4),
            "effect_size_r": round(float(_rank_biserial(rf, ff)), 3),
            "significant": bool(p < CFG["alpha"]),
        })
    out["T3_crossref_by_order"] = pd.DataFrame(rows)

    # T4 — transition zone (first N turns of 2nd stimulus) vs rest of 2nd stimulus
    rows = []
    for feat, label in FEATURES:
        a, b = [], []
        for pid, g in turns[turns["is_second_stimulus"]].groupby("participant"):
            tz = g[g["in_transition_zone"]][feat].dropna().values
            rest = g[~g["in_transition_zone"]][feat].dropna().values
            if len(tz) >= 1 and len(rest) >= 1:
                a.append(tz.mean()); b.append(rest.mean())
        if len(a) < 3 or np.allclose(a, b):
            continue
        a, b = np.array(a), np.array(b)
        w, p = stats.wilcoxon(a, b)
        rows.append({
            "feature": label,
            f"transition_mean_first{CFG['transition_zone_turns']}": round(float(a.mean()), 5),
            "rest_of_2nd_mean": round(float(b.mean()), 5),
            "W": float(w), "p_value": round(float(p), 4),
            "effect_size_r": round(float(_wilcoxon_r(w, len(a))), 3),
            "significant": bool(p < CFG["alpha"]),
            "n_participants": int(len(a)),
        })
    out["T4_transition_zone_vs_rest"] = pd.DataFrame(rows)

    return out


def order_summary(turns):
    """Small DataFrame: who is real_first vs fictional_first (inferred)."""
    rows = []
    for pid, g in turns.groupby("participant"):
        rows.append({
            "participant": pid,
            "order_label_in_file": g["order"].iloc[0] if "order" in g.columns else "?",
            "inferred_order": g["presentation_order"].iloc[0],
            "first_condition": g["first_condition"].iloc[0],
            "n_turns": len(g),
        })
    return pd.DataFrame(rows).sort_values("participant").reset_index(drop=True)

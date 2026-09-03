"""
Participant metadata loader — familiarity scores + task grades
==============================================================
Implements the FIRST idea suggested by the supervisor:
  "Correct the model using participants' task scores (familiarity, grades
   obtained on the posed questions...)."

Loads the two post-task surveys (results-survey-A/B) and produces a tidy
per-participant table that can be merged onto the turn-level corpus, so that
familiarity and task performance can be used as COVARIATES / MODERATORS in the
H1 mixed-effects model and correlated with the linguistic features.

Surveys are organised by experimental condition-set, not by map:
  - Survey A = participants who saw the REAL map first  (P1,P3,P5,P7,P9)
  - Survey B = participants who saw the FICTIONAL map first (P2,P4,P6,P8,P10)
Both surveys record a grade on BOTH maps (Real /7 and Not real /7), plus
self-reported familiarity with the geography ("this map") and with the
visualization type ("temperature map visualization").

Familiarity is map-condition-specific only for the REAL map (you cannot be
"familiar" with a fictional geography), so:
  - fam_geo   → meaningful for the REAL condition (prior geographic knowledge)
  - fam_viz   → visualization literacy, applies to BOTH conditions
"""

import pandas as pd
import numpy as np


def _find(cols, *needles, startswith=False):
    for c in cols:
        cl = c.lower().strip()
        for n in needles:
            if (cl.startswith(n) if startswith else (n in cl)):
                return c
    return None


def _parse_survey(path):
    df = pd.read_excel(path)
    cols = list(df.columns)
    pcol = _find(cols, "participant", "partcipant")          # note: file B misspells it
    fam_geo = _find(cols, "how familiar are you with this map", startswith=True)
    fam_viz = _find(cols, "type of temperature map")
    grade_real = next((c for c in cols if "Grade" in c and "Not real" not in c), None)
    grade_fict = next((c for c in cols if "Grade" in c and "Not real" in c), None)

    rows = []
    for _, r in df.iterrows():
        pid = str(r[pcol]).strip()
        if not pid.startswith("P") or not pid[1:].isdigit():
            continue  # skip 'Correct answers' footer row
        rows.append({
            "participant": pid,
            "fam_geo": pd.to_numeric(r.get(fam_geo), errors="coerce"),
            "fam_viz": pd.to_numeric(r.get(fam_viz), errors="coerce"),
            "grade_real": pd.to_numeric(r.get(grade_real), errors="coerce"),
            "grade_fictional": pd.to_numeric(r.get(grade_fict), errors="coerce"),
        })
    return pd.DataFrame(rows)


def load_participant_meta(survey_a_path, survey_b_path):
    """Return one tidy row per participant with familiarity + grades.

    Columns:
      participant, fam_geo (1-5), fam_viz (1-5),
      grade_real (/7), grade_fictional (/7),
      grade_first  (grade on the map seen first),
      grade_second (grade on the map seen second),
      learning_gain = grade_second - grade_first  (procedural transfer proxy)
    Order is INFERRED from which survey the participant is in:
      survey A => real first, survey B => fictional first.
    """
    a = _parse_survey(survey_a_path); a["order"] = "real_first"
    b = _parse_survey(survey_b_path); b["order"] = "fictional_first"
    meta = pd.concat([a, b], ignore_index=True)

    def first_grade(r):
        return r["grade_real"] if r["order"] == "real_first" else r["grade_fictional"]

    def second_grade(r):
        return r["grade_fictional"] if r["order"] == "real_first" else r["grade_real"]

    meta["grade_first"] = meta.apply(first_grade, axis=1)
    meta["grade_second"] = meta.apply(second_grade, axis=1)
    meta["learning_gain"] = meta["grade_second"] - meta["grade_first"]
    return meta


def attach_condition_grade(turns, meta):
    """Add a `grade_condition` column to the turn table: the grade the participant
    obtained on the map they are currently talking about. Lets us correlate a
    turn's linguistic features with how well that participant did on that map."""
    g = {}
    for _, r in meta.iterrows():
        g[(r["participant"], "Real")] = r["grade_real"]
        g[(r["participant"], "Not real")] = r["grade_fictional"]
    out = turns.copy()
    out["grade_condition"] = out.apply(
        lambda r: g.get((r["participant"], r["condition"]), np.nan), axis=1)
    # familiarity that applies to the current condition:
    #   real map → geographic familiarity; fictional map → 1 (no prior geo knowledge)
    fam_geo = dict(zip(meta["participant"], meta["fam_geo"]))
    fam_viz = dict(zip(meta["participant"], meta["fam_viz"]))
    out["fam_geo"] = out["participant"].map(fam_geo)
    out["fam_viz"] = out["participant"].map(fam_viz)
    out["fam_condition"] = out.apply(
        lambda r: (r["fam_geo"] if r["condition"] == "Real" else 1.0), axis=1)
    return out

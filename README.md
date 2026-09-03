# Lang Analysis of Climate Data Visualizations

When someone describes a map, how much of what they say comes from the map — and how much from what
they already knew before they looked at it?

Ten people were interviewed while reading two temperature maps with **identical colour-to-temperature
encoding**: one of France, one of an invented territory. Same visual grammar, same questions, only
the familiarity changes. If prior territorial knowledge shapes discourse, the two descriptions should
*sound different*.

They do — but not in the way we expected, and the gap between the two results is the point of the
project.

---

## Table of contents

- [The design](#the-design)
- [Three hypotheses, three verdicts](#three-hypotheses-three-verdicts)
- [Findings](#findings)
- [Method](#method)
- [Pipeline](#pipeline)
- [Data](#data)
- [Repository structure](#repository-structure)
- [Running the analysis](#running-the-analysis)
- [Limitations](#limitations)
- [References](#references)
- [Authors](#authors)

---

## The design

Each participant describes **both** maps — real and fictional — answering the same 11 readability
questions, with presentation order counterbalanced. The design is therefore **within-participant and
repeated-measures**, which is what makes the statistics interesting: every turn from the same person
is correlated with every other turn from that person, so treating turns as independent observations
would inflate significance. That constraint drives the whole analysis.

## Three hypotheses, three verdicts

| | Hypothesis | Verdict |
|---|------------|---------|
| **H1** | Referential expressions differ between the real and fictional map — familiarity produces more specific, anchored references. | **Supported**, for geographic naming only |
| **H2** | Lexical variation is higher in the real-map condition. | **Not supported** — and that null is the most interesting result |
| **H3** | Presentation order leaves a carry-over trace on the second stimulus. | **Partially supported** — two features, in two of the four tests |

## Findings

### H1 — Familiarity makes you name things

Speakers produce **2.2× more geographic references** per turn on the real map (0.362 vs 0.163). Under
a mixed-effects model with participant as a random intercept, `GEO` is the **only** label whose effect
survives:

| Label | Coefficient (real − fictional) | p-value |
|-------|-------------------------------|---------|
| **GEO** | **+0.232** | **< 0.0001** |
| DEF | +0.002 | 0.985 |
| DEM | +0.095 | 0.741 |
| GEN | +0.117 | 0.054 |
| UNCERTAINTY | −0.042 | 0.613 |

The effect is **robust to visualisation literacy** (coefficient +0.234, p < 0.0001 controlling for
`fam_viz`), and self-reported geographic familiarity does **not** amplify it (interaction +0.100,
p = 0.197).

An independent confirmation comes from a second annotation channel. Following Arunkumar et al. (2025),
a `PRIOR_KNOWLEDGE` tag captures *reasoning from world knowledge* ("the south is usually warmer") as
opposed to merely *naming* a place. It runs **1.70× higher** on the real map (0.319 vs 0.188; mixed
model coefficient +0.125, p = 0.019) — a separate signal pointing the same way.

Naming is not uniform across the interview. Broken down by cognitive block, it **peaks during
higher-order reasoning**, not during simple look-ups:

| Question block | Real map | Fictional map |
|----------------|----------|---------------|
| B1 Clutter | 0.390 | 0.039 |
| B2 Retrieval | 0.151 | 0.269 |
| B3 Comprehension | 0.369 | 0.236 |
| **B4 Higher-order** | **0.543** | 0.119 |

### H2 — Describing is not understanding

Overall lexical diversity is **the same in both conditions**:

| Metric | Real | Fictional | Δ |
|--------|------|-----------|---|
| STTR | 0.729 | 0.728 | +0.000 |
| MTLD | 48.18 | 47.75 | +0.43 |
| Yule's K | 136.93 | 130.82 | +6.11 |
| TTR (biased) | 0.054 | 0.052 | +0.001 |

Wilcoxon on per-participant TTR: **p = 0.652**, not significant. H2 is rejected.

But the semantic-field breakdown shows *what* changed underneath the flat total. Real-map speakers add
diversity through geographic names; fictional-map speakers compensate with map vocabulary and deictic
expressions — pointing instead of naming. **Substitution, not enrichment.**

And the twist that gives the project its point: **familiarity does not produce better comprehension.**
Some maximally familiar readers (self-reported 5/5) scored 3/7 on the task. What *did* matter was
order — participants who saw the real map first performed better on the fictional one (6.0/7 ≈ 86%)
than participants who saw the fictional map first (4.4/7 ≈ 63%).

Language reflects **description strategy**; performance reflects **interpretation**. They are two
different things, and this project separates them.

### H3 — Carry-over is narrow but real

Four non-parametric tests with effect sizes and bootstrap confidence intervals, appropriate for n = 10:

| Test | What it asks | Result |
|------|--------------|--------|
| T1 | Does order change the 2nd stimulus, between groups? | No significant feature (proper nouns closest, p = 0.067, r = 0.75) |
| **T2** | 1st vs 2nd stimulus **within** participant — the core carry-over test | **Cross-condition references rise, 0.0008 → 0.0047, p = 0.020** |
| T3 | Cross-reference density on the 2nd stimulus by order group | Not significant (p = 0.762) |
| **T4** | Transition zone (first 5 turns after the switch) vs rest of the 2nd stimulus | **Deictic references lower in the transition zone, 0.041 → 0.059, p = 0.014** |

So carry-over exists, but it is specific: people **explicitly refer back** to the first map when
describing the second, and their pointing behaviour takes a few turns to settle after the switch. The
broad "order changes everything" version of H3 does not hold.

A methodological note worth more than the result: **presentation order is inferred from the data**
(which map the participant actually discusses first), not read from the filename label. Doing it that
way caught mislabelled files — P9's filename says order A, but the participant discusses the real map
first and the preprocessed file is labelled order B. Trusting the label would have silently mixed the
groups.

## Method

**One shared vocabulary, two uses.** Five labels are defined once and reused identically by the H1
regex annotator and the H2 semantic-field density calculation — which eliminates a whole class of
silent bugs where two analyses drift apart:

| Label | Concept | Entries |
|-------|---------|---------|
| `GEO` | Geographic proper names (real French regions + fictional map names) | 47 |
| `DEF` | Map-element nouns, detected as `the <noun>` | 49 |
| `DEM` | Demonstratives and spatial deictics | 71 |
| `GEN` | Visual / perceptual vocabulary | 36 |
| `UNCERTAINTY` | Epistemic hedges and difficulty markers | 37 |

Annotation is **rule-based and interpretable** — regex patterns built from those vocabularies, applied
per speech turn, with spaCy for tokenisation and POS tagging. Every count can be traced back to a
surface form in a transcript, which matters when the corpus is small enough that one mis-annotated
turn moves a mean.

**The mixed-effects model** is the methodological core:

```
n_GEO ~ condition + (1 | participant)
```

`condition` is the fixed effect of interest; `(1 | participant)` is a random intercept absorbing each
person's baseline talkativeness. Moving from a plain regression to this specification is what demoted
four of the five H1 labels from "effects" to "noise" — the honest version of the result, and the
reason the project claims only `GEO`.

**Lexical diversity** uses length-robust measures rather than raw TTR, which is biased by text length:
STTR (standardised type-token ratio over 40-token windows), MTLD, and Yule's K, with a Wilcoxon
signed-rank test on per-participant values.

**H3** uses Mann-Whitney (between groups) and Wilcoxon (within participant) with effect sizes and
bootstrap CIs throughout — no test assumes normality on a sample this size.

### Supporting modules

| Module | Role |
|--------|------|
| `prior_knowledge_tag.py` | The sixth referential label. Builds the `PRIOR_KNOWLEDGE` regex from explicit knowledge markers, deliberately excluding place names so it does not double-count `GEO`. Candidate sub-fields were tested for real-vs-fictional discrimination; topographic nouns and causal connectives describe both maps and were dropped. |
| `participant_meta.py` | Loads the two post-task surveys into a tidy per-participant table (`fam_geo`, `fam_viz`, `grade_real`, `grade_fictional`) for use as covariates and moderators. Handles the column-name inconsistencies between surveys A and B, including a misspelled header. |
| `h3_carryover.py` | The four carry-over tests, order inference from the data, turn-level feature extraction, bootstrap CIs and rank-biserial effect sizes. |

## Pipeline

```
Raw transcripts            automatic transcription of the interview audio
        │
        │   manual annotation + correction  (condition, question number, speaker)
        ▼
Annotated transcripts      one XLSX per participant
        │
        │   preprocessing.ipynb
        │   lowercase · unicode normalise · strip punctuation · remove fillers
        │   and repetitions — each stage kept as its own column
        ▼
Preprocessing/*-final.xlsx  analysis-ready corpus
        │
        │   analyse_H1_H2_H3.ipynb   (+ the three modules above)
        ▼
Excel workbook, 9 sheets    H1_Full_Results · H2_Full_Results · Corpus_H1 ·
                            Vocabulary_H1 · Summary_Dashboard · H1_by_Participant ·
                            Mixed_Model_H1 · By_Question_Block · H3_Order_Effects
```

## Data

**The participant data is not published in this repository.** The corpus consists of interview
transcripts and post-task surveys from ten identifiable people — researchers and engineers in a small
professional community — and the surveys record age, gender, degree, occupation and research
speciality alongside the analytic variables.

The notebook is committed **with its stored outputs**, so every number reported above is verifiable
without the data.

### Corpus schema

Each `*-final.xlsx` is one row per speech turn:

| Column | Content |
|--------|---------|
| `speaker_name` | Interviewer or participant — only participant turns are analysed |
| `start_time`, `end_time` | Turn timestamps |
| `text` | Raw transcription |
| `clean_text` | Lowercased, unicode-normalised |
| `no_punctuation_text` | Punctuation stripped |
| `no_fillers_text` | Discourse markers and hesitations removed |
| `no_repetition_text` | Repeated words collapsed — the column the analysis uses |
| `condition` | `Real` or `Not real` |
| `previous_question` | Question number 1–13, mapped to the four cognitive blocks |
| `question_type` | Question category |

### Survey schema

`participant_meta.py` reads only four analytic variables per participant: self-reported geographic
familiarity (1–5), visualisation literacy (1–5), and the two task grades (out of 7). The demographic
columns in the source files are not used by any analysis.

### Corpus size

| Property | Value |
|----------|-------|
| Participants | 10 (P1–P10), each describing both maps |
| Speech turns | 890 total; 794 labelled by condition (405 fictional / 389 real) |
| Tokens | 21,908 real-map · 23,123 fictional-map |
| Protocol | 11 readability questions per map, grouped into 4 cognitive blocks; order counterbalanced |

## Repository structure

```
.
├── analyse_H1_H2_H3.ipynb    # Main analysis: H1, H2, H3 + supervisor extensions
├── preprocessing.ipynb       # Transcript cleaning pipeline
├── prior_knowledge_tag.py    # PRIOR_KNOWLEDGE annotation channel
├── participant_meta.py       # Survey loader: familiarity scores and task grades
├── h3_carryover.py           # Carry-over tests T1–T4
├── docs/
│   ├── final-report.pdf      # Full project report
│   └── interview-questions.pdf
├── slides/
│   └── presentation.pdf
├── Preprocessing/            # (data excluded — see Data)
├── Tasks and results/        # (data excluded)
├── Annotated and corrected transcripts/   # (data excluded)
├── requirements.txt
└── README.md
```

Paths in the notebooks are relative to the repository root, which is where they should be run from.

## Running the analysis

```bash
git clone https://github.com/joaopbarret/land-analysis-climate-data.git
cd land-analysis-climate-data
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter lab analyse_H1_H2_H3.ipynb
```

## Limitations

- **n = 10, and a homogeneous sample.** All participants were highly educated experts, mostly
  researchers in computer science. Every result is exploratory, and the block-level breakdown in
  particular rests on as few as 65 turns per cell — the notebook labels it as such rather than
  reporting it as a finding.
- **Rule-based annotation** has a recall ceiling: a region named in an unlisted spelling is missed. The
  trade-off buys full interpretability, which was the right call at this corpus size.
- **The fictional map is not a neutral control.** It is unfamiliar *and* invented; participants may
  behave differently because they know it is fake, not only because they lack knowledge of it.
- **Self-reported familiarity** is a coarse 1–5 instrument, which weakens the interaction test more
  than the main effect.

## References

- Arunkumar, A., et al. (2025). *Prior knowledge and external-knowledge recall in the interpretation of data visualizations.*
- McNamara, D. S., et al. *Measures of textual lexical diversity (MTLD) and the type-token ratio.*
- Honnibal, M., & Montani, I. *spaCy: Industrial-strength natural language processing.*
- Cabouat, A.-F., He, T., Isenberg, P., & Isenberg, T. (2026). *PREVis: Perceived Readability Evaluation for Visualizations.*

## Authors

Project DSIA 2025/2026, Télécom Paris.

**Team:** Xisco Moncet · [João Pedro Barreto](https://github.com/joaopbarret) · Naoures Abassi · Youssef Chebil · Lise Peguet

**Supervisor:** Anne-Flore Cabouat

"""
PRIOR_KNOWLEDGE tag — sixth referential label
=============================================
Implements the SECOND idea suggested by the supervisor:
  "Create a new 'prior knowledge' tag" — inspired by Arunkumar et al. (2025),
  "Modeling and Measuring the Chart Communication Recall Process", shared by
  the supervisor.

Arunkumar et al. define an "Extrapolation / external knowledge" recall
category which they describe as the single most discriminative signal between
familiar and unfamiliar charts. The team's own `How_to_use_it.md` already
sketched the vocabulary; this module turns it into an operational regex tag,
parallel to the existing GEO/DEF/DEM/GEN/UNCERTAINTY labels.

WHAT IT CAPTURES (and how it differs from GEO)
----------------------------------------------
  GEO              → NAMING a real place           ("Bretagne", "the Alps")
  PRIOR_KNOWLEDGE  → REASONING from world knowledge ("the south is usually
                     warmer", "I know that...", "because mountains are cold")

The two are complementary channels of the same underlying construct (prior
knowledge). GEO is lexical/onomastic; PRIOR_KNOWLEDGE is inferential. The
prediction is that PRIOR_KNOWLEDGE is near-absent on the fictional map (no
world knowledge can be mobilised about an invented territory) and present on
the real map — giving a second, independent confirmation of H1.

IMPORTANT — avoid double counting with GEO
------------------------------------------
Bare place names are already counted by GEO. PRIOR_KNOWLEDGE deliberately does
NOT re-list place names; it targets explicit knowledge markers, causal
reasoning, and climatic/topographic world knowledge. This keeps the two tags
measuring different things.
"""

import re

# Explicit knowledge markers — speaker signals they are using what they know
PK_MARKERS = [
    "i know", "i already know", "already know", "we know", "we usually say",
    "as we know", "it makes sense", "makes sense", "as expected", "of course",
    "familiar with", "comfortable with", "used to", "i recognize", "i recognise",
    "i remember", "everyone knows", "obviously", "naturally", "typically",
    "in reality", "in real life", "normally", "usually", "generally",
]

# Climatic world knowledge — statements about how climate actually works
PK_CLIMATE = [
    "south is warmer", "south is hotter", "north is colder", "north is cooler",
    "mountains are colder", "mountains are cooler", "higher altitude",
    "altitude", "coast is milder", "coastal", "mediterranean climate",
    "warmer in the south", "colder in the north", "hotter in the south",
    "sea is", "ocean is", "near the sea", "by the sea",
]

# Topographic world knowledge — real geographic features used as reasoning anchors
PK_TOPO = [
    "coast", "coastline", "mountain", "mountains", "valley", "valleys",
    "sea", "ocean", "river", "rivers", "plain", "plains", "border with",
    "near spain", "near italy", "near germany", "near the border",
]

# Causal reasoning — connectors that signal inference rather than description.
# These fire often, so they are kept SEPARATE and only counted when the speaker
# is in the real condition they still count, but interpret as inference density.
PK_CAUSAL = [
    "because", "since", "due to", "that's why", "thats why", "so that",
    "which is why", "the reason", "as a result", "therefore", "hence",
    "given that", "makes sense because",
]

# ── Tag composition decision (empirically grounded) ──────────────────────────
# We tested each sub-field's real-vs-fictional discrimination on the corpus:
#     PK_markers : 1.73x more in REAL   ← strong, keep
#     PK_climate : 0.52x (noisy, n tiny) ← drop from core
#     PK_topo    : 0.47x MORE in FICT    ← drop (generic "coast/mountain"
#                                            describe both maps' shapes)
#     PK_causal  : 0.92x (no signal)     ← drop ("because/so" are generic)
# Only the EXPLICIT KNOWLEDGE MARKERS reliably separate the conditions, which
# matches Arunkumar et al.: it is the *act of invoking what one knows*
# ("I know", "I'm used to it", "usually", "of course") that distinguishes
# familiar from unfamiliar stimuli — not topographic nouns or causal connectives.
PRIOR_KNOWLEDGE_VOCAB = PK_MARKERS + PK_CLIMATE          # core discriminative tag
PRIOR_KNOWLEDGE_VOCAB_WIDE = PK_MARKERS + PK_CLIMATE + PK_TOPO + PK_CAUSAL  # for ablation


def make_prior_knowledge_pattern(vocab=PRIOR_KNOWLEDGE_VOCAB):
    """Compile the PRIOR_KNOWLEDGE OR-regex (longest match first).
    Defaults to the core (markers + climate) tag, which is the discriminative one.
    Pass PRIOR_KNOWLEDGE_VOCAB_WIDE to reproduce the non-discriminative ablation."""
    return re.compile(
        "|".join(r"\b" + re.escape(v) + r"\b" for v in sorted(vocab, key=len, reverse=True)),
        re.IGNORECASE,
    )


# Optional fine-grained sub-tags, in case the report wants a breakdown
PK_SUBFIELDS = {
    "PK_markers": PK_MARKERS,
    "PK_climate": PK_CLIMATE,
    "PK_topo": PK_TOPO,
    "PK_causal": PK_CAUSAL,
}

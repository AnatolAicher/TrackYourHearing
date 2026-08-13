"""Codebook: column layouts, names and human-readable labels for the TYH data.

Track Your Hearing (TYH) is an EMA study. Two raw files ship from the app export:

* ``merged.csv`` -- one row per momentary (EMA) entry. Columns are the daily
  questionnaire (``question1``..``question10``, ``soundlevel``, ``save_date``)
  followed by the participant's three baseline mini-questionnaires *repeated on
  every row*.
* ``mini-questionnaires.csv`` -- one row per participant: ``user_id`` followed by
  the same three baseline mini-questionnaires.

The baseline columns are exported with bare numeric headers (``1``, ``2``,
``25`` ...) and the per-questionnaire submission timestamp appears three times as
``created_at``. Those headers are unusable as-is, so this module defines the
fixed column order and a clean, ``base_``-prefixed name for each, taken from
``data_raw/codebook.xlsx``. The mappings here are the source of
truth used by :mod:`tyh.ingest`; loading the xlsx is optional (:func:`load_codebook`).

Nothing here interprets or recodes values -- it only names columns and records
what each one means and its expected scale, so the diagnostics can flag problems.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Missing-value tokens. The export writes missing cells as the literal string
# "np.nan" (not an empty cell), which pandas would otherwise read as text.
# --------------------------------------------------------------------------- #
NA_STRINGS = ["np.nan", "np.NaN", "NaN", "nan", "NA", "N/A", ""]

# --------------------------------------------------------------------------- #
# EMA / daily questionnaire ("standardanswers"). These keep their original
# names; we only attach labels and record their expected scale.
# --------------------------------------------------------------------------- #
EMA_ID_COLS = ["id", "user_id"]

EMA_LABELS: dict[str, str] = {
    "id": "row id (1 per EMA entry)",
    "user_id": "participant id",
    "question1": "q1 wears hearing aid right now (1=yes, 0=no)",
    "question2": "q2 perceived hearing loss right now (0=none .. 1=total)",
    "question3": "q3 limited by hearing loss right now (0=not at all .. 1=very much)",
    "question4": "q4 hearing loss weighs emotionally right now (0 .. 1)",
    "question5": "q5 mood right now (0=unhappy .. 1=happy)",
    "question6": "q6 stressed right now (0=not .. 1=maximal)",
    "question7": "q7 can concentrate right now (0=not at all .. 1=fully)",
    "question8": "q8 exhausted right now (0=not at all .. 1=very much)",
    "question9a": "q9a tiring to follow conversations, past few hours (0 .. 1)",
    "question9b": "q9b no conversation took place (1=not applicable, 0=applicable)",
    "question10": "q10 ambient noise physically restrictive right now (0 .. 1)",
    "soundlevel": "ambient sound level at the entry (raw app value, ~dB)",
    "save_date": "timestamp the EMA entry was saved",
}

# Continuous sliders that the daily app records on a 0..1 scale; anything
# outside [0, 1] is a data-entry error (the diagnostics report these).
EMA_UNIT_SLIDERS = [
    "question2", "question3", "question4", "question5",
    "question6", "question7", "question8", "question9a", "question10",
]
# Binary switches, expected to be exactly {0, 1}.
EMA_BINARY = ["question1", "question9b"]

# Numeric EMA columns to coerce on load (sliders + binary + soundlevel).
EMA_NUMERIC = EMA_UNIT_SLIDERS + EMA_BINARY + ["soundlevel"]

# --------------------------------------------------------------------------- #
# Baseline mini-questionnaires. The export lays the three questionnaires out in
# a fixed column order, each ending in a ``created_at`` submission timestamp.
# RAW = header tokens exactly as they appear in the file (in order);
# CLEAN = the name we rename each to. The two lists are positional and must
# stay the same length and order.
# --------------------------------------------------------------------------- #
# Mini-Hörverlust-Fragebogen 1 (everyone): demographics + hearing/HA status.
# Mini-Hörverlust-Fragebogen 2 (everyone): hearing-loss experience last month.
# Mini-Hörverlust-Fragebogen 3 (HA owners only): hearing-aid details.
_BASELINE_RAW = [
    # --- questionnaire 1 ---
    "27", "2", "25", "1", "26", "created_at",
    # --- questionnaire 2 ---
    "9", "5", "4", "10", "8", "11", "7", "3", "6", "created_at",
    # --- questionnaire 3 (HA owners) ---
    "19", "20", "14", "12", "15", "18", "13", "16", "17", "21", "22", "24", "23", "created_at",
]
_BASELINE_CLEAN = [
    # --- questionnaire 1 ---
    "base_handedness", "base_wears_ha", "base_birthdate", "base_hearing_problem",
    "base_sex", "base_q1_created_at",
    # --- questionnaire 2 ---
    "base_others_noticed", "base_pct_annoyed", "base_pct_aware", "base_reaction",
    "base_noise_sensitivity", "base_reaction_other", "base_aware_situations",
    "base_restricted", "base_pct_unhappy", "base_q2_created_at",
    # --- questionnaire 3 (HA owners) ---
    "base_ha_changes_right", "base_ha_hours_per_day", "base_ha_type_left",
    "base_ha_duration", "base_ha_type_right", "base_ha_changes_left", "base_ha_side",
    "base_ha_features_left", "base_ha_features_right", "base_ha_satisfaction",
    "base_ha_cleaning", "base_ha_battery_change_freq", "base_ha_battery_type",
    "base_q3_created_at",
]
assert len(_BASELINE_RAW) == len(_BASELINE_CLEAN) == 30

BASELINE_LABELS: dict[str, str] = {
    "base_handedness": "handedness (rechts/links/beidhändig)",
    "base_wears_ha": "owns/wears a hearing aid (ja/nein)",
    "base_birthdate": "date of birth (DD.MM.YYYY)",
    "base_hearing_problem": "hearing problem side (beidseitig/links/rechts/kein Problem)",
    "base_sex": "sex (weiblich/männlich)",
    "base_q1_created_at": "questionnaire 1 submission timestamp",
    "base_others_noticed": "has anyone noticed your hearing is impaired?",
    "base_pct_annoyed": "% of time annoyed by hearing loss last month (scale varies)",
    "base_pct_aware": "% of time aware of hearing loss last month (scale varies)",
    "base_reaction": "usual reaction to noticing hearing loss",
    "base_noise_sensitivity": "sensitive to noise (niemals .. immer)",
    "base_reaction_other": "free text: other reaction",
    "base_aware_situations": "situations where hearing loss is noticed (multi)",
    "base_restricted": "restricted in everyday life by hearing loss (niemals .. immer)",
    "base_pct_unhappy": "% of time unhappy last month due to hearing loss (scale varies)",
    "base_q2_created_at": "questionnaire 2 submission timestamp",
    "base_ha_changes_right": "times changed HA type, right ear",
    "base_ha_hours_per_day": "hours/day wearing HA (last month)",
    "base_ha_type_left": "HA type, left ear (free text)",
    "base_ha_duration": "how long wearing a HA (free text)",
    "base_ha_type_right": "HA type, right ear (free text)",
    "base_ha_changes_left": "times changed HA type, left ear",
    "base_ha_side": "side(s) wearing HA",
    "base_ha_features_left": "left HA features (multi)",
    "base_ha_features_right": "right HA features (multi)",
    "base_ha_satisfaction": "satisfaction with HA (scale varies)",
    "base_ha_cleaning": "cleans/dries HA regularly?",
    "base_ha_battery_change_freq": "how often changes HA batteries (free text)",
    "base_ha_battery_type": "HA battery type (rechargeable/replaceable)",
    "base_q3_created_at": "questionnaire 3 submission timestamp",
}

# Baseline date / numeric columns to coerce on load.
BASELINE_DATE_COLS = ["base_q1_created_at", "base_q2_created_at", "base_q3_created_at"]
BASELINE_BIRTHDATE = "base_birthdate"
# "Percentage" sliders: nominally 0..100 but the export mixes 0..1 and 0..100
# entries, so we coerce to numeric and let the diagnostics show the range.
BASELINE_NUMERIC = ["base_pct_annoyed", "base_pct_aware", "base_pct_unhappy", "base_ha_satisfaction"]

# Population-defining baseline variables (German value translations, for reference
# in later analysis -- not applied at ingestion).
HEARING_PROBLEM_MAP = {
    "beidseitig": "both sides", "links": "left", "rechts": "right", "kein Problem": "no problem",
}
WEARS_HA_MAP = {"ja": "yes", "nein": "no"}
SEX_MAP = {"weiblich": "female", "männlich": "male"}

# --------------------------------------------------------------------------- #
# Full file layouts: RAW header tokens (in order) and the CLEAN names we assign.
# These are verified against the actual file header on load (tyh.ingest), so any
# drift in the export structure fails loudly instead of silently mis-mapping.
# --------------------------------------------------------------------------- #
EMA_HEAD_RAW = [
    "id", "user_id",
    "question1", "question2", "question3", "question4", "question5",
    "question6", "question7", "question8", "question9a", "question9b", "question10",
    "soundlevel", "save_date",
]
MERGED_RAW = EMA_HEAD_RAW + _BASELINE_RAW
MERGED_CLEAN = EMA_HEAD_RAW + _BASELINE_CLEAN          # EMA columns keep their names

MINI_RAW = ["user_id"] + _BASELINE_RAW
MINI_CLEAN = ["user_id"] + _BASELINE_CLEAN

# Combined label lookup used by the diagnostics. ``base_age`` is derived at load
# time (see tyh.ingest), not a raw column, but is labelled here for the report.
COLUMN_LABELS: dict[str, str] = {
    **EMA_LABELS,
    **BASELINE_LABELS,
    "base_age": "age in years (DERIVED from birthdate at registration)",
    # reverse-coded during cleaning (x -> 1 - x), so high = more burden
    "question5_rev": "q5 mood, reverse-coded (0=happy .. 1=unhappy; high = more burden)",
    "question7_rev": "q7 concentration, reverse-coded (0=fully .. 1=not at all; high = more burden)",
}


def label(col: str) -> str:
    """Human-readable label for a (clean) column name, or '' if unknown."""
    return COLUMN_LABELS.get(col, "")


def load_codebook(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the original ``codebook.xlsx`` (optional; for reference/verification).

    Returns a dict of sheet name -> DataFrame ('questions', 'questionnaires').
    """
    return pd.read_excel(path, sheet_name=None)

"""Clean the ingested TYH data into an analysis-ready table.

Steps:

1. **Keep only the needed columns** (drop everything else). The kept set is the
   row id, timestamp, participant id, the two baseline status variables, sex and
   age, and the full daily question battery (``question1``..``question10``,
   including ``question9a``/``question9b``). The two baseline status variables
   are recoded to booleans: ``base_wears_ha`` (ja/nein -> True/False) and
   ``base_hearing_problem`` (any side -> True, 'kein Problem' -> False).
2. **Recode out-of-range momentary answers to missing.** Any ``question*`` value
   outside [0, 1] is a data-entry error (e.g. mood = 37) -> set to ``NaN``. The
   row is kept; only the offending cell is blanked.
3. **Reverse-code positively valenced items.** q5 (mood) and q7 (concentration)
   are recorded high = better; they are inverted (x -> 1 - x) and renamed
   ``question5_rev`` / ``question7_rev`` so that *every* outcome shares a valence
   (higher = more burden / worse).
4. **Drop duplicate submissions.** Two rows are duplicates when they share the
   same ``user_id``, identical values on every other kept column, and a
   ``save_date`` within one minute of each other. One row per duplicate cluster
   is kept (the earliest).
5. **Restrict to the analytic population** with a hearing problem
   (``derived_hearing_problem``), unless ``filter_no_hearing_problem=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .ingest import TYHData

# Full daily question battery. "questions 1-10" in the data = these 11 columns
# (there is no plain "question9"; it splits into 9a slider + 9b skip-flag).
QUESTION_COLS = [
    "question1", "question2", "question3", "question4", "question5",
    "question6", "question7", "question8", "question9a", "question9b", "question10",
]

# Positively valenced items (high = better) are reverse-coded (x -> 1 - x) during
# cleaning so that, for every outcome, higher = more burden / worse. q5 = mood,
# q7 = concentration. The reverse-coded columns are renamed to make this explicit.
REVERSE_CODED = {"question5": "question5_rev", "question7": "question7_rev"}
# Daily battery as it appears in the cleaned output (with q5/q7 renamed).
OUTPUT_QUESTION_COLS = [REVERSE_CODED.get(q, q) for q in QUESTION_COLS]

# Source columns we pull from the EMA table, and the names we expose them under.
_RENAME = {"base_sex": "sex", "base_age": "age"}
_SOURCE_COLS = [
    "id", "save_date", "user_id",
    "base_sex", "base_age", "base_wears_ha", "base_hearing_problem",
] + QUESTION_COLS

# Final column order (after rename). ``derived_hearing_problem`` is added during
# cleaning (it is not a source column); see _add_derived_hearing_problem.
KEEP_COLS = [
    "id", "save_date", "user_id",
    "sex", "age", "base_wears_ha", "base_hearing_problem", "derived_hearing_problem",
] + OUTPUT_QUESTION_COLS

# Columns that must match for two rows to count as the same submission
# (everything kept except the row id and the time-windowed save_date).
_MATCH_COLS = [
    "user_id", "sex", "age", "base_wears_ha", "base_hearing_problem",
] + QUESTION_COLS

DEDUP_WINDOW_SECONDS = 60


@dataclass
class CleaningReport:
    """Counts describing what the cleaning step changed."""

    n_input: int
    n_output: int
    dropped_columns: list[str]
    out_of_range_per_question: dict[str, int]
    duplicate_rows_removed: int
    duplicate_clusters: int
    derived_hp_users: int = 0
    derived_hp_added_users: int = 0
    n_users: int = 0
    hearing_problem_filtered: bool = False
    users_excluded: int = 0
    rows_excluded_filter: int = 0

    @property
    def out_of_range_total(self) -> int:
        return sum(self.out_of_range_per_question.values())

    def summary(self) -> str:
        rule = "=" * 78
        lines = [
            rule,
            "TYH cleaning report",
            rule,
            f"input rows                 : {self.n_input}",
            f"columns dropped            : {len(self.dropped_columns)}",
            f"  {', '.join(self.dropped_columns)}",
            "",
            f"out-of-range cells -> NaN  : {self.out_of_range_total} "
            f"(across {sum(v > 0 for v in self.out_of_range_per_question.values())} questions)",
        ]
        for q, k in self.out_of_range_per_question.items():
            if k:
                lines.append(f"  {q:<12}: {k}")
        lines += [
            "",
            f"reverse-coded (x -> 1-x)   : {', '.join(REVERSE_CODED)} "
            f"-> {', '.join(REVERSE_CODED.values())}  (now high = more burden)",
            "",
            f"duplicate submissions      : {self.duplicate_rows_removed} rows removed "
            f"from {self.duplicate_clusters} clusters "
            f"(same user_id + identical answers + save_date within {DEDUP_WINDOW_SECONDS}s)",
            "",
            f"derived_hearing_problem    : {self.derived_hp_users}/{self.n_users} participants True "
            f"(+{self.derived_hp_added_users} beyond a 'kein Problem' baseline, "
            f"via HA ownership or ever wearing one)",
            "",
            f"hearing-problem filter     : {'ON' if self.hearing_problem_filtered else 'OFF'}"
            + (f"  -> excluded {self.users_excluded} participants / "
               f"{self.rows_excluded_filter} rows (derived_hearing_problem = False)"
               if self.hearing_problem_filtered else ""),
            "",
            f"output rows                : {self.n_output}"
            + (f"  ({self.derived_hp_users} participants)" if self.hearing_problem_filtered else ""),
            rule,
        ]
        return "\n".join(lines)


def _recode_booleans(df: pd.DataFrame) -> None:
    """Recode the two baseline status variables to nullable booleans, in place.

    ``base_wears_ha``       : 'ja' -> True, 'nein' -> False
    ``base_hearing_problem``: 'kein Problem' -> False, any side -> True

    Missing (and any unexpected) values become ``<NA>`` rather than False.
    """
    df["base_wears_ha"] = df["base_wears_ha"].map({"ja": True, "nein": False}).astype("boolean")

    hp = df["base_hearing_problem"]
    has_problem = (hp != "kein Problem").where(hp.notna(), other=pd.NA)
    df["base_hearing_problem"] = has_problem.astype("boolean")


def _add_derived_hearing_problem(df: pd.DataFrame) -> None:
    """Add a participant-level ``derived_hearing_problem`` flag, in place.

    The three hearing-status signals sometimes disagree (e.g. a participant who
    reports 'kein Problem' at baseline but owns a hearing aid or wears one during
    EMA). This flag is True for every row of any participant who, on *any* of the
    three signals, indicates hearing loss / a hearing aid:

    * ``base_wears_ha`` is True (owns a hearing aid), OR
    * ``base_hearing_problem`` is True (reports a hearing problem), OR
    * answered ``question1 == 1`` (wore a hearing aid) on at least one entry.

    It is constant within ``user_id``. Missing baseline values count as not-True.
    """
    wears = df["base_wears_ha"].fillna(False).astype(bool)
    problem = df["base_hearing_problem"].fillna(False).astype(bool)
    ever_wore = df.groupby("user_id")["question1"].transform(lambda s: (s == 1).any()).astype(bool)
    df["derived_hearing_problem"] = (wears | problem | ever_wore).astype("boolean")


def _clamp_unit_range(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """Set values outside [0, 1] to NaN, in place. Return per-column counts."""
    counts: dict[str, int] = {}
    for col in cols:
        oob = (df[col] < 0) | (df[col] > 1)
        counts[col] = int(oob.sum())
        if counts[col]:
            df.loc[oob, col] = pd.NA
    return counts


def _drop_duplicate_submissions(
    df: pd.DataFrame, match_cols: list[str], window_s: int
) -> tuple[pd.DataFrame, int, int]:
    """Collapse near-simultaneous identical submissions to one row each.

    Rows are grouped by identical values on ``match_cols`` (NaN treated as equal);
    within a group, rows are clustered by ``save_date`` proximity (a gap greater
    than ``window_s`` from the previous row starts a new cluster). The earliest
    row of each cluster is kept.
    """
    if df.empty:
        return df, 0, 0

    work = df.sort_values(["user_id", "save_date"], kind="mergesort").copy()

    # Hashable group key over match_cols, with NaN mapped to a sentinel so that
    # two rows that are both missing on a question count as identical.
    keyed = work[match_cols].astype(object).where(work[match_cols].notna(), "\x00NA")
    work["_gkey"] = list(map(tuple, keyed.to_numpy()))

    # Seconds since the previous row within the same identical-answer group.
    gap = work.groupby("_gkey")["save_date"].diff().dt.total_seconds()
    starts_cluster = gap.isna() | (gap > window_s)
    work["_cluster"] = starts_cluster.groupby(work["_gkey"]).cumsum()

    keep = ~work.duplicated(subset=["_gkey", "_cluster"], keep="first")
    removed = int((~keep).sum())
    # A "duplicate cluster" is a (_gkey, _cluster) that contains >1 row.
    cluster_sizes = work.groupby(["_gkey", "_cluster"]).size()
    n_clusters = int((cluster_sizes > 1).sum())

    cleaned = work[keep].drop(columns=["_gkey", "_cluster"])
    cleaned = cleaned.sort_values("id", kind="mergesort").reset_index(drop=True)
    return cleaned, removed, n_clusters


def clean(
    data: TYHData | pd.DataFrame,
    *,
    dedup_window_s: int = DEDUP_WINDOW_SECONDS,
    filter_no_hearing_problem: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return the cleaned, analysis-ready EMA table.

    Parameters
    ----------
    data:
        A :class:`~tyh.ingest.TYHData` or the EMA DataFrame from
        :func:`tyh.ingest.load_ema`.
    dedup_window_s:
        Time window (seconds) within which identical submissions are treated as
        duplicates. Defaults to 60 (one minute).
    filter_no_hearing_problem:
        Drop participants with ``derived_hearing_problem == False`` (no hearing
        problem, no hearing aid, never wore one). Defaults to True -- the
        analytic population is hearing-impaired participants only.
    verbose:
        Print the cleaning report.
    """
    df, report = clean_with_report(
        data, dedup_window_s=dedup_window_s,
        filter_no_hearing_problem=filter_no_hearing_problem,
    )
    if verbose:
        print(report.summary())
    return df


def clean_with_report(
    data: TYHData | pd.DataFrame,
    *,
    dedup_window_s: int = DEDUP_WINDOW_SECONDS,
    filter_no_hearing_problem: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Like :func:`clean` but also return the :class:`CleaningReport` object."""
    ema = data.ema if isinstance(data, TYHData) else data

    missing = [c for c in _SOURCE_COLS if c not in ema.columns]
    if missing:
        raise KeyError(f"EMA table is missing expected columns: {missing}")

    n_input = len(ema)
    dropped = [c for c in ema.columns if c not in _SOURCE_COLS]

    # 1) keep only what we need, with friendly names
    df = ema[_SOURCE_COLS].rename(columns=_RENAME).copy()

    # 1b) recode the baseline status variables to booleans
    _recode_booleans(df)

    # 2) out-of-range momentary answers -> NaN (cell-level, row kept)
    oor = _clamp_unit_range(df, QUESTION_COLS)

    # 3) drop duplicate submissions (within the time window)
    df, removed, n_clusters = _drop_duplicate_submissions(df, _MATCH_COLS, dedup_window_s)

    # 4) participant-level reconciled hearing-status flag
    _add_derived_hearing_problem(df)

    # 4b) reverse-code positively valenced items (q5 mood, q7 concentration) so
    #     every outcome shares valence: higher = more burden. Done on the [0,1]
    #     scale after out-of-range values were already set to NaN.
    for raw in REVERSE_CODED:
        df[raw] = 1.0 - df[raw]
    df = df.rename(columns=REVERSE_CODED)

    df = df[KEEP_COLS]

    # participant-level counts for the derived flag (and how often it overrides
    # a 'kein Problem' baseline), computed BEFORE any population filter
    by_user = df.groupby("user_id")[["derived_hearing_problem", "base_hearing_problem"]].first()
    derived_true = by_user["derived_hearing_problem"].fillna(False)
    base_true = by_user["base_hearing_problem"].fillna(False)
    n_users_all = int(len(by_user))
    derived_hp_users = int(derived_true.sum())

    # 5) restrict to the analytic population: keep only participants with a
    #    (derived) hearing problem
    users_excluded = rows_excluded = 0
    if filter_no_hearing_problem:
        keep = df["derived_hearing_problem"].fillna(False).astype(bool)
        rows_excluded = int((~keep).sum())
        users_excluded = n_users_all - derived_hp_users
        df = df[keep].reset_index(drop=True)

    report = CleaningReport(
        n_input=n_input,
        n_output=len(df),
        dropped_columns=dropped,
        out_of_range_per_question=oor,
        duplicate_rows_removed=removed,
        duplicate_clusters=n_clusters,
        derived_hp_users=derived_hp_users,
        derived_hp_added_users=int((derived_true & ~base_true).sum()),
        n_users=n_users_all,
        hearing_problem_filtered=filter_no_hearing_problem,
        users_excluded=users_excluded,
        rows_excluded_filter=rows_excluded,
    )
    return df, report

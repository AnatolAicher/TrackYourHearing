"""Read the raw TYH CSVs into tidy pandas DataFrames.

Fixes the export's structural quirks (the ``np.nan`` missing-value string, the
unusable numeric/duplicate baseline headers) and applies dtypes. It does not
clean values: out-of-range sliders, duplicate beeps, placeholder birthdates and
test rows are left in place.

Typical use::

    from tyh import load
    data = load()
    data.ema        # long EMA table (baseline repeated per row)
    data.baseline   # one-row-per-participant baseline
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import codebook as cb
from . import paths

# Birthdates are recorded as DD.MM.YYYY; age is computed relative to the start
# of data collection (Oct 2017), which is when participants registered.
STUDY_REFERENCE_DATE = pd.Timestamp("2017-10-01")


@dataclass
class TYHData:
    """The two raw tables, loaded and named.

    Attributes
    ----------
    ema:
        Long EMA table from ``merged.csv`` -- one row per momentary entry, with
        the participant's baseline answers (``base_*``) repeated on every row.
    baseline:
        One row per participant from ``mini-questionnaires.csv``.
    """

    ema: pd.DataFrame
    baseline: pd.DataFrame


def _read_header_tokens(path: Path) -> list[str]:
    """Return the raw header tokens exactly as written in the file.

    We read the header line ourselves (rather than trusting pandas' de-duplicated
    column index) because the baseline ``created_at`` token repeats three times.
    The header contains no quoted commas, so a plain split is safe.
    """
    with open(path, encoding="utf-8") as fh:
        first_line = fh.readline().rstrip("\r\n")
    return first_line.split(",")


def _verify_header(path: Path, expected: list[str]) -> None:
    """Fail loudly if the file's header is not the layout the codebook assumes."""
    actual = _read_header_tokens(path)
    if actual != expected:
        # Pinpoint the first divergence to make drift easy to diagnose.
        diff = next(
            (
                f"position {i}: file has {a!r}, expected {e!r}"
                for i, (a, e) in enumerate(zip(actual, expected))
                if a != e
            ),
            f"length {len(actual)} vs expected {len(expected)}",
        )
        raise ValueError(
            f"Unexpected header layout in {path.name}; the codebook column map is "
            f"out of date with the export.\n  First difference: {diff}\n"
            f"  Actual ({len(actual)}):   {actual}\n"
            f"  Expected ({len(expected)}): {expected}"
        )


def _add_derived_age(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``base_age`` (years) derived from ``base_birthdate``.

    Age is taken at the participant's own questionnaire-1 timestamp when present,
    otherwise at :data:`STUDY_REFERENCE_DATE`. Placeholder/implausible birthdates
    (e.g. 01.01.1980, future years) flow through unchanged.
    """
    if cb.BASELINE_BIRTHDATE not in df.columns:
        return df
    ref = df.get("base_q1_created_at")
    ref = ref.fillna(STUDY_REFERENCE_DATE) if ref is not None else STUDY_REFERENCE_DATE
    age = (ref - df[cb.BASELINE_BIRTHDATE]).dt.days / 365.25
    df["base_age"] = age.round(1)
    return df


def _coerce_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Type the shared baseline (``base_*``) columns: dates, birthdate, sliders."""
    for col in cb.BASELINE_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if cb.BASELINE_BIRTHDATE in df.columns:
        df[cb.BASELINE_BIRTHDATE] = pd.to_datetime(
            df[cb.BASELINE_BIRTHDATE], format="%d.%m.%Y", errors="coerce"
        )
    for col in cb.BASELINE_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return _add_derived_age(df)


def load_ema(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``merged.csv`` -- the long EMA table (one row per momentary entry).

    Baseline answers are merged in and repeated on every row, renamed with a
    ``base_`` prefix.
    """
    path = Path(path) if path is not None else paths.merged_csv()
    _verify_header(path, cb.MERGED_RAW)
    df = pd.read_csv(path, na_values=cb.NA_STRINGS, keep_default_na=True, low_memory=False)
    df.columns = cb.MERGED_CLEAN  # positional rename (header already verified)

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    for col in cb.EMA_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["save_date"] = pd.to_datetime(df["save_date"], errors="coerce")
    return _coerce_baseline(df)


def load_baseline(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``mini-questionnaires.csv`` -- one row per participant."""
    path = Path(path) if path is not None else paths.mini_csv()
    _verify_header(path, cb.MINI_RAW)
    df = pd.read_csv(path, na_values=cb.NA_STRINGS, keep_default_na=True, low_memory=False)
    df.columns = cb.MINI_CLEAN  # positional rename (header already verified)

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    return _coerce_baseline(df)


def load(data_dir: str | Path | None = None) -> TYHData:
    """Load both tables. Optionally point at a directory holding the two CSVs."""
    if data_dir is not None:
        data_dir = Path(data_dir)
        ema = load_ema(data_dir / "merged.csv")
        baseline = load_baseline(data_dir / "mini-questionnaires.csv")
    else:
        ema = load_ema()
        baseline = load_baseline()
    return TYHData(ema=ema, baseline=baseline)

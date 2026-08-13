"""Print human-readable diagnostics on the loaded TYH data.

Everything here is read-only: it summarises what was loaded and surfaces the
known data-quality problems (out-of-range sliders, duplicate beeps, implausible
ages/timestamps, mixed scales, coverage) so they can be inspected before any
cleaning or modelling. Nothing is modified.

Run as ``python run_ingest.py`` (repo) or ``python -m tyh.diagnostics``.
"""

from __future__ import annotations

import pandas as pd

from . import codebook as cb
from .ingest import TYHData, load

# Window in which the study was actually run; entries outside it are suspect.
STUDY_WINDOW = (pd.Timestamp("2017-09-01"), pd.Timestamp("2018-12-31"))


def _rule(char: str = "=") -> str:
    return char * 78


def _header(title: str) -> None:
    print("\n" + _rule())
    print(title)
    print(_rule())


def _pct(n: int, total: int) -> str:
    return f"{(100 * n / total):.1f}%" if total else "n/a"


def _examples(series: pd.Series, k: int = 3) -> str:
    vals = series.dropna().unique()[:k]
    out = ", ".join(str(v) for v in vals)
    return (out[:57] + "...") if len(out) > 60 else out


def schema_table(df: pd.DataFrame, title: str) -> None:
    """Per-column dtype, missingness, cardinality, example values + codebook label."""
    _header(title)
    n = len(df)
    print(f"{n} rows x {df.shape[1]} columns\n")
    print(f"{'column':<30}{'dtype':<14}{'miss':>12}{'uniq':>7}  {'examples / label'}")
    print(_rule("-"))
    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        miss_str = f"{miss} ({_pct(miss, n)})"
        lbl = cb.label(col)
        ex = _examples(s)
        detail = f"{ex}" + (f"   [{lbl}]" if lbl else "")
        print(f"{col:<30}{str(s.dtype):<14}{miss_str:>12}{s.nunique():>7}  {detail[:90]}")


def ema_overview(ema: pd.DataFrame) -> None:
    _header("EMA (merged.csv) -- overview")
    print(f"entries (rows)      : {len(ema)}")
    print(f"unique participants : {ema['user_id'].nunique()}")
    print(f"user_id range       : {ema['user_id'].min()} .. {ema['user_id'].max()}")
    sd = ema["save_date"]
    print(f"save_date span      : {sd.min()}  ->  {sd.max()}")
    bad_dates = int(((sd < STUDY_WINDOW[0]) | (sd > STUDY_WINDOW[1])).sum())
    print(
        f"save_date outside study window {STUDY_WINDOW[0].date()}..{STUDY_WINDOW[1].date()}"
        f" : {bad_dates} ({_pct(bad_dates, len(ema))})"
    )
    print(f"save_date unparseable : {int(sd.isna().sum())}")


def ema_slider_ranges(ema: pd.DataFrame) -> None:
    """Range-check the daily sliders/switches; flag out-of-range data-entry errors."""
    _header("EMA momentary items -- range check (expected scale [0, 1])")
    print(f"{'item':<12}{'min':>8}{'max':>10}{'mean':>9}{'miss':>8}{'out-of-range':>14}  note")
    print(_rule("-"))
    for col in cb.EMA_UNIT_SLIDERS + cb.EMA_BINARY:
        s = ema[col]
        if col in cb.EMA_BINARY:
            oor = int((~s.dropna().isin([0, 1])).sum())
            note = "not in {0,1}" if oor else "ok"
        else:
            oor = int(((s < 0) | (s > 1)).sum())
            note = "<-- data-entry errors (>1)" if oor else "ok"
        print(
            f"{col:<12}{s.min():>8.3g}{s.max():>10.4g}{s.mean():>9.3f}"
            f"{int(s.isna().sum()):>8}{oor:>14}  {note}"
        )
    print("\nnote: out-of-range values (e.g. mood=37, exhaustion=100) are real rows in")
    print("      the export; left in place here, to be recoded to missing when cleaning.")


def ema_soundlevel(ema: pd.DataFrame) -> None:
    _header("EMA soundlevel -- range")
    s = ema["soundlevel"]
    print(f"min={s.min():.1f}  max={s.max():.1f}  mean={s.mean():.1f}  missing={int(s.isna().sum())}")
    implausible = int((s.abs() > 200).sum())  # plausible ambient dB are roughly -120..120
    print(f"implausible (|value| > 200): {implausible}  <-- e.g. {s.max():.0f} is not a real dB level")


def ema_duplicates(ema: pd.DataFrame) -> None:
    _header("EMA duplicates")
    exact = int(ema.drop(columns=["id"]).duplicated().sum())
    beep = int(ema.duplicated(subset=["user_id", "save_date"]).sum())
    print(f"duplicate rows (ignoring the unique 'id' column) : {exact}")
    print(f"repeated beeps (same user_id + save_date)        : {beep} ({_pct(beep, len(ema))})")
    print("note: these are duplicate submissions to drop during cleaning (group-key based).")


def ema_entries_per_user(ema: pd.DataFrame) -> None:
    _header("EMA entries per participant")
    epu = ema.groupby("user_id").size()
    print(epu.describe().round(1).to_string())
    for thr in (2, 3, 5):
        below = int((epu < thr).sum())
        print(f"participants with < {thr} entries : {below}")


def baseline_overview(baseline: pd.DataFrame) -> None:
    _header("Baseline (mini-questionnaires.csv) -- overview")
    print(f"rows                : {len(baseline)}")
    print(f"unique participants : {baseline['user_id'].nunique()}")
    print(f"duplicate user_id   : {int(baseline['user_id'].duplicated().sum())}")


def baseline_population(baseline: pd.DataFrame) -> None:
    """The variables that define the analytic population."""
    _header("Baseline -- population-defining variables")
    for col in ("base_hearing_problem", "base_wears_ha", "base_sex"):
        print(f"\n{col}  [{cb.label(col)}]")
        print(baseline[col].value_counts(dropna=False).to_string())
    print("\ncrosstab: hearing problem (rows) x owns hearing aid (cols)")
    print(
        pd.crosstab(
            baseline["base_hearing_problem"], baseline["base_wears_ha"], dropna=False
        ).to_string()
    )


def baseline_age(baseline: pd.DataFrame) -> None:
    _header("Baseline -- age (derived from birthdate; ref = study start / own reg date)")
    age = baseline["base_age"]
    bd = baseline["base_birthdate"]
    print(f"birthdate unparseable : {int(bd.isna().sum())}")
    print(age.describe().round(1).to_string())
    implausible = int(((age < 18) | (age > 100)).sum())
    print(f"\nages < 18 or > 100 (incl. placeholder/junk birthdates) : {implausible}")
    print(f"  e.g. min age = {age.min()} (future-dated birth year)")


def baseline_scales(baseline: pd.DataFrame) -> None:
    _header("Baseline -- 'percentage' sliders (scale is inconsistent: 0..1 AND 0..100)")
    print(f"{'item':<22}{'min':>8}{'max':>9}{'mean':>9}{'miss':>8}{'>1':>7}")
    print(_rule("-"))
    for col in cb.BASELINE_NUMERIC:
        s = baseline[col]
        gt1 = int((s > 1).sum())
        print(f"{col:<22}{s.min():>8.3g}{s.max():>9.4g}{s.mean():>9.2f}{int(s.isna().sum()):>8}{gt1:>7}")
    print("\nnote: some participants' values are on 0..1, others on 0..100 -> needs a")
    print("      scale decision before these baseline items can be used.")


def linkage(data: TYHData) -> None:
    _header("Linkage between EMA and baseline")
    ema_users = set(data.ema["user_id"].dropna().tolist())
    base_users = set(data.baseline["user_id"].dropna().tolist())
    print(f"participants in EMA      : {len(ema_users)}")
    print(f"participants in baseline : {len(base_users)}")
    print(f"in EMA but no baseline   : {len(ema_users - base_users)}")
    print(f"in baseline but no EMA   : {len(base_users - ema_users)} (registered, never answered EMA)")


def diagnose(data: TYHData | None = None) -> TYHData:
    """Load (if needed) and print the full diagnostic report. Returns the data."""
    if data is None:
        data = load()

    print(_rule("#"))
    print("Track Your Hearing (TYH) -- data ingestion diagnostics")
    print(_rule("#"))

    # EMA
    ema_overview(data.ema)
    schema_table(data.ema, "EMA (merged.csv) -- schema")
    ema_slider_ranges(data.ema)
    ema_soundlevel(data.ema)
    ema_duplicates(data.ema)
    ema_entries_per_user(data.ema)

    # Baseline
    baseline_overview(data.baseline)
    schema_table(data.baseline, "Baseline (mini-questionnaires.csv) -- schema")
    baseline_population(data.baseline)
    baseline_age(data.baseline)
    baseline_scales(data.baseline)

    # Cross-file
    linkage(data)

    _header("Done. Data is loaded faithfully (nothing cleaned).")
    return data


if __name__ == "__main__":
    diagnose()

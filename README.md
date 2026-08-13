# Track Your Hearing (TYH) – analysis codebase

Python pipeline for the Track Your Hearing EMA hearing-aid study: it reads the
raw app export, cleans it into an analysis-ready table, builds and validates the
outcome composites, and estimates the within-person association between
hearing-aid use and momentary burden (directionality, effect size, within vs
between, design/power analysis). Results and their interpretation are reported
in the accompanying paper; this README covers installation and usage.

## Citing

Placeholder

## Layout

```
TrackYourHearing/
├── run_ingest.py        # entry point: load the data and print diagnostics
├── run_clean.py         # entry point: load, clean, summarise (optionally save)
├── run_viz.py           # entry point: diagnostic pair-plots
├── run_rawviz.py        # entry point: raw-data figures (EMA raster + item distributions)
├── run_stats.py         # entry point: directionality (cluster permutation) test
├── run_composites.py    # entry point: build outcome composites + reliability
├── run_validity.py      # entry point: convergent/discriminant validity + figures
├── run_effectsize.py    # entry point: within-person effect size + bootstrap CIs
├── run_resultsviz.py    # entry point: result figures (effect size + directionality)
├── run_signflip.py      # entry point: within- vs between-person association (+ figures)
├── run_power.py         # entry point: minimum detectable effect + power (+ figure)
├── requirements.txt
└── tyh/
    ├── paths.py         # locate the raw files (override with TYH_DATA_DIR)
    ├── codebook.py      # column layouts, clean names, labels, expected scales
    ├── ingest.py        # read merged.csv / mini-questionnaires.csv -> DataFrames
    ├── diagnostics.py   # read-only data-quality report
    ├── clean.py         # column subset + value/duplicate cleaning -> analysis table
    ├── viz.py           # diagnostic visualisations (Altair)
    ├── rawviz.py        # raw-data figures: EMA raster + item distributions
    ├── stats.py         # directionality test (cluster permutation + Type-S)
    ├── composites.py    # unit-weighted standardized outcome composites
    ├── validity.py      # convergent/discriminant validity (HTMT, Fornell-Larcker, MTMM)
    ├── effectsize.py    # within-person effect size (mixed model + cluster bootstrap)
    ├── effectviz.py     # result figures: effect-size forest, bootstrap density, etc.
    ├── withinbetween.py # within- vs between-person association (+ sign-flip figures)
    └── power.py         # design analysis: minimum detectable effect + Monte-Carlo power
```

## Data

The raw data are sensitive and are not part of the repository. The pipeline
expects two files from the app export, plus the machine-readable codebook:

| File | Grain | Contents |
|---|---|---|
| `merged.csv` | one row per momentary EMA entry | daily questionnaire (`question1`–`question10`, `soundlevel`, `save_date`) + the participant's baseline answers repeated on every row |
| `mini-questionnaires.csv` | one row per participant | the three baseline mini-questionnaires |
| `codebook.xlsx` | one row per exported column | maps the numeric baseline export headers to clean `base_` names |

By default all three are looked up in `data_raw/` next to the repository
checkout. `TYH_DATA_DIR` points the two CSVs elsewhere; the codebook path is
overridden separately with `TYH_CODEBOOK`:

```bash
export TYH_DATA_DIR=/path/to/dir/containing/merged.csv
export TYH_CODEBOOK=/path/to/codebook.xlsx
```

## Usage

```bash
pip install -r requirements.txt

python run_ingest.py                 # load + print diagnostics
python run_clean.py                  # load + clean + summarise
python run_clean.py --out clean.csv  # also write the cleaned table
python run_viz.py                    # seaborn-style pairplots (hist diagonal)
python run_viz.py --diag kde         # KDE densities on the diagonal instead
python run_viz.py --style splom      # fast scatter-only SPLOM instead
python run_rawviz.py                 # EMA raster + item distributions
python run_stats.py                  # directionality test (q1 -> q2..q10)
python run_composites.py             # build outcome composites + reliability
python run_validity.py --figures     # convergent/discriminant validity (+ figures)
python run_effectsize.py             # within-person effect size + bootstrap CIs
python run_resultsviz.py             # result figures (forest, density, ...)
python run_signflip.py --figures     # within- vs between-person association (+ figures)
python run_power.py --figures        # minimum detectable effect + power curves (hours; cached)
```

Figure-writing commands save interactive HTML into `figures/` next to the
repository checkout (`--out-dir` to change, `--fmt both` for static PNGs via
vl-convert).

Or from Python:

```python
from tyh import load, diagnose, clean, directional_test

data = load()          # TYHData(ema=<DataFrame>, baseline=<DataFrame>)
diagnose(data)         # print the full data-quality report
cleaned = clean(data)  # analysis-ready table (prints a cleaning report)
result = directional_test(cleaned)   # q1 exposure, q2..q10 outcomes
print(result.summary())
```

## Pipeline

**Ingest** (`tyh/ingest.py` / `run_ingest.py`) – reads the two CSVs into a
`TYHData(ema, baseline)` pair: decodes the literal string `"np.nan"` to real
missing values, renames the numeric baseline export headers to `base_`-prefixed
names from `codebook.xlsx` (the expected header is verified on load, so a
changed export fails loudly instead of silently mis-mapping columns), applies
dtypes and derives `base_age`. Values are not cleaned here. `tyh/diagnostics.py`
prints a read-only data-quality report (out-of-range sliders, duplicate
submissions, implausible timestamps and birthdates, mixed scales, coverage).

**Clean** (`tyh/clean.py` / `run_clean.py`) – `clean(data)` builds the analysis
table: keeps the analysis columns, recodes the two baseline status variables to
nullable booleans, sets out-of-range `question*` values to missing,
reverse-codes q5/q7 into `question5_rev`/`question7_rev` (every outcome oriented
higher = more burden), collapses duplicate submissions (identical answers within
a one-minute gap cluster), derives the participant-level
`derived_hearing_problem` flag, and by default restricts to participants with a
hearing-problem signal (`filter_no_hearing_problem=False` to keep everyone).

**Diagnostic & raw-data figures** (`tyh/viz.py`, `tyh/rawviz.py`) – pairwise
plots of the continuous EMA items coloured by the grouping variables
(seaborn-style pairplot with per-group diagonal distributions, or a fast SPLOM),
an EMA raster (participant × time, coloured by aid wear) and per-item response
distributions.

**Directionality** (`tyh/stats.py` / `run_stats.py`) – within-person
correlations of the exposure (q1, wearing a hearing aid now) with each outcome
item, aggregated across outcomes and tested with a person-level cluster
permutation (mean Fisher-z, sign concordance) plus a cluster-bootstrap Type-S
probability.

**Composites** (`tyh/composites.py` / `run_composites.py`) – two unit-weighted
standardized outcome composites, momentary_burden (q4, q5_rev, q6, q7_rev, q8,
q9a, q10) and hearing_difficulty (q2, q3), with a reliability report
(standardized Cronbach's alpha pooled and within-person, item-rest and
inter-composite correlations).

**Validity** (`tyh/validity.py` / `run_validity.py`) – convergent/discriminant
validity of the composites at the within, between and pooled levels: HTMT with a
participant-cluster bootstrap CI, Fornell-Larcker, Campbell-Fiske own-vs-cross
item correlations and per-item ICC(1) (REML). `--figures` writes a blocked
item-correlation heatmap, HTMT bars and an own-vs-cross dumbbell.

**Effect size** (`tyh/effectsize.py` / `run_effectsize.py`) – the within-person
effect of q1 on each composite in SD units, with q1 Mundlak-split into within-
and between-person parts; random-intercept and random-slope mixed models with
participant-cluster bootstrap CIs. `tyh/effectviz.py` / `run_resultsviz.py`
write the result figures (effect forest, bootstrap density, per-item
directionality dots, within-person slopes).

**Within vs between** (`tyh/withinbetween.py` / `run_signflip.py`) – the
fixed-effects within-person slope alongside the between-person slope (OLS of
person means), both in SD units with cluster-bootstrap CIs; `--figures` writes a
coefficient plot and the per-person scatter.

**Design analysis** (`tyh/power.py` / `run_power.py`) – the minimum detectable
effect implied by the random-slope cluster-bootstrap SE, and prospective
Monte-Carlo power as a function of the number of exposure-varying participants;
simulated datasets mirror the achieved design and are analysed with the same
random-slope estimator. The simulation refits mixed models for hours:
replicates run in a batched process pool, completed replicates are checkpointed
(interrupted runs resume), and final results are cached in `results_cache/`
next to the checkout, keyed by a settings+data hash.

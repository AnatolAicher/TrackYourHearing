"""Locate the raw data files.

The data are sensitive and live outside the repository (and out of version
control): by default the CSVs and the codebook are looked up in ``data_raw/``
next to the repository checkout. The locations can be overridden with the
``TYH_DATA_DIR`` and ``TYH_CODEBOOK`` environment variables::

    export TYH_DATA_DIR=/path/to/dir/containing/merged.csv
"""

from __future__ import annotations

import os
from pathlib import Path

# tyh/paths.py -> tyh -> <checkout> -> the directory containing the checkout,
# where data_raw/, figures/ and results_cache/ live.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where merged.csv / mini-questionnaires.csv are looked up by default.
DEFAULT_DATA_DIR = PROJECT_ROOT / "data_raw"

# The machine-readable codebook (lives alongside the data by default).
DEFAULT_CODEBOOK = PROJECT_ROOT / "data_raw" / "codebook.xlsx"


def data_dir() -> Path:
    """Directory containing the raw CSVs (``TYH_DATA_DIR`` overrides the default)."""
    env = os.environ.get("TYH_DATA_DIR")
    return Path(env).expanduser() if env else DEFAULT_DATA_DIR


def merged_csv() -> Path:
    """Path to the long EMA file (one row per momentary entry)."""
    return data_dir() / "merged.csv"


def mini_csv() -> Path:
    """Path to the baseline file (one row per participant)."""
    return data_dir() / "mini-questionnaires.csv"


def codebook_xlsx() -> Path:
    """Path to ``codebook.xlsx`` (``TYH_CODEBOOK`` overrides the default)."""
    env = os.environ.get("TYH_CODEBOOK")
    return Path(env).expanduser() if env else DEFAULT_CODEBOOK

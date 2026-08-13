#!/usr/bin/env python3
"""Load the TYH data and print diagnostics.

Run from anywhere::

    python run_ingest.py

Point at a different data directory with the TYH_DATA_DIR environment variable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the local `tyh` package importable regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tyh import diagnose  # noqa: E402

if __name__ == "__main__":
    diagnose()

#!/usr/bin/env python3
"""Build the two outcome composites and print reliability diagnostics.

    python run_composites.py

Wellbeing       = q4, q5_rev, q6, q7_rev, q8, q9a, q10
Perceived-hearing = q2, q3

Both are unit-weighted, standardized, and oriented higher = more burden.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from tyh import clean, load  # noqa: E402
from tyh.composites import COMPOSITES, add_composites, composite_report  # noqa: E402


def main() -> None:
    df = clean(load(), verbose=False)
    print(composite_report(df))

    with_comp = add_composites(df)
    cols = ["user_id", "question1", *COMPOSITES]
    print("\nPreview (first 8 rows):")
    with pd.option_context("display.width", 160):
        print(with_comp[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()

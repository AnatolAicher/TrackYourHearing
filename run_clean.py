#!/usr/bin/env python3
"""Load, clean and summarise the TYH data.

    python run_clean.py                 # clean + print report + preview
    python run_clean.py --out clean.csv # also write the cleaned table

Point at a different data directory with the TYH_DATA_DIR environment variable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tyh import clean, load  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="write the cleaned table to this CSV")
    args = parser.parse_args()

    data = load()
    cleaned = clean(data)

    print("\nPreview (first 8 rows):")
    with __import__("pandas").option_context("display.width", 200, "display.max_columns", 30):
        print(cleaned.head(8).to_string(index=False))

    if args.out is not None:
        cleaned.to_csv(args.out, index=False)
        print(f"\nWrote {len(cleaned)} rows x {cleaned.shape[1]} cols -> {args.out}")


if __name__ == "__main__":
    main()

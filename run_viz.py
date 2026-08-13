#!/usr/bin/env python3
"""Generate diagnostic pair-plots of the continuous EMA items.

    python run_viz.py                       # 4 seaborn-style pairplots -> figures/
    python run_viz.py --diag kde            # KDE densities on the diagonal
    python run_viz.py --style splom         # fast scatter-only SPLOM instead
    python run_viz.py --fmt both            # also write PNGs
    python run_viz.py --out-dir some/dir --max-points 2000

Each plot covers question2..question10 (the continuous items), coloured once by
sex, once by base_wears_ha, once by base_hearing_problem and once by
derived_hearing_problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tyh import clean, load  # noqa: E402
from tyh import codebook as cb  # noqa: E402
from tyh.paths import PROJECT_ROOT  # noqa: E402
from tyh.viz import (  # noqa: E402
    CONTINUOUS_QUESTIONS,
    diagnostic_pair_plots,
    diagnostic_seaborn_pairplots,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures")
    parser.add_argument("--style", choices=["seaborn", "splom"], default="seaborn",
                        help="seaborn = distributions on the diagonal; splom = scatter-only")
    parser.add_argument("--diag", choices=["kde", "hist"], default="hist",
                        help="diagonal distribution for --style seaborn (default: hist)")
    parser.add_argument("--fmt", choices=["html", "png", "both"], default="html")
    parser.add_argument("--max-points", type=int, default=None,
                        help="plot a random sample of this many rows (default: all)")
    args = parser.parse_args()

    df = clean(load(), verbose=False)

    print("Continuous items in the matrix:")
    for q in CONTINUOUS_QUESTIONS:
        print(f"  {q.replace('question', 'q'):5s} = {cb.label(q)}")

    if args.style == "seaborn":
        written = diagnostic_seaborn_pairplots(
            df, args.out_dir, fmt=args.fmt, diag=args.diag, max_points=args.max_points)
    else:
        written = diagnostic_pair_plots(
            df, args.out_dir, fmt=args.fmt, max_points=args.max_points)

    print(f"\nWrote {len(written)} file(s) to {args.out_dir}:")
    for p in written:
        print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
    if args.style == "seaborn" and args.diag == "kde":
        print("\nTip: --diag hist is more faithful for the discrete (q5/mood) and "
              "floor-heavy items; KDE smooths those.")
    elif args.style == "splom":
        print("\nNote: SPLOM diagonal is a self-scatter (y = x), not a distribution.")


if __name__ == "__main__":
    main()

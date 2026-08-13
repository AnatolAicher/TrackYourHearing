#!/usr/bin/env python3
"""Convergent & discriminant validity of the outcome composites.

    python run_validity.py              # print report
    python run_validity.py --figures    # also write figures/ heatmap + charts
    python run_validity.py --method spearman   # rank-based sensitivity

Headline level is within-person; pooled/between reported as context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tyh import clean, load  # noqa: E402
from tyh.paths import PROJECT_ROOT  # noqa: E402
from tyh.validity import analyze, diagnostic_validity_plots, report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures", action="store_true", help="also write the validity figures")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures")
    p.add_argument("--fmt", choices=["html", "png", "both"], default="html")
    p.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = clean(load(), verbose=False)
    res = analyze(df, n_boot=args.n_boot, seed=args.seed, method=args.method)
    print(report(res))

    if args.figures:
        written = diagnostic_validity_plots(df, args.out_dir, fmt=args.fmt,
                                            n_boot=args.n_boot, seed=args.seed)
        print(f"\nWrote {len(written)} figure(s) to {args.out_dir}:")
        for w in written:
            print(f"  {w.name}")


if __name__ == "__main__":
    main()

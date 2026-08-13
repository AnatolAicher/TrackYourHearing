#!/usr/bin/env python3
"""Within- vs between-person association of the exposure (the sign-flip analysis).

    python run_signflip.py                 # print report
    python run_signflip.py --figures       # also write figures/
    python run_signflip.py --figures --fmt both

Point at a different data directory with the TYH_DATA_DIR environment variable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tyh import clean, load  # noqa: E402
from tyh.effectsize import DEFAULT_EXPOSURE  # noqa: E402
from tyh.paths import PROJECT_ROOT  # noqa: E402
from tyh.withinbetween import report, within_between, withinbetween_figures  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures", action="store_true", help="also write the figures")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures")
    p.add_argument("--fmt", choices=["html", "png", "both"], default="html")
    p.add_argument("--exposure", default=DEFAULT_EXPOSURE)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = clean(load(), verbose=False)
    print(report(within_between(df, exposure=args.exposure, n_boot=args.n_boot, seed=args.seed)))

    if args.figures:
        written = withinbetween_figures(df, args.out_dir, fmt=args.fmt,
                                        exposure=args.exposure, n_boot=args.n_boot, seed=args.seed)
        print(f"\nWrote {len(written)} figure(s) to {args.out_dir}:")
        for w in written:
            print(f"  {w.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate result figures (effect sizes + directionality) -> figures/.

    python run_resultsviz.py                # 4 result figures (HTML)
    python run_resultsviz.py --fmt both     # also write PNGs
    python run_resultsviz.py --out-dir some/dir

Point at a different data directory with the TYH_DATA_DIR environment variable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tyh import clean, load  # noqa: E402
from tyh.effectsize import DEFAULT_EXPOSURE  # noqa: E402
from tyh.effectviz import results_figures  # noqa: E402
from tyh.paths import PROJECT_ROOT  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures")
    p.add_argument("--fmt", choices=["html", "png", "both"], default="html")
    p.add_argument("--exposure", default=DEFAULT_EXPOSURE)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--rs-n-boot", type=int, default=2000,
                   help="random-slope cluster-bootstrap reps (refits per draw; slower)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = clean(load(), verbose=False)
    written = results_figures(df, args.out_dir, fmt=args.fmt, exposure=args.exposure,
                              n_boot=args.n_boot, rs_n_boot=args.rs_n_boot, seed=args.seed)
    print(f"Wrote {len(written)} figure(s) to {args.out_dir}:")
    for w in written:
        print(f"  {w.name}  ({w.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

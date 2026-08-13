#!/usr/bin/env python3
"""Design analysis: minimum detectable effect + prospective power.

    python run_power.py                     # MDES table + power report (hours; cached)
    python run_power.py --figures           # also write the power-curve figure
    python run_power.py --reps 20 --boot 20 # small smoke run (numbers are not final)

The simulation refits the random-slope model per replicate and bootstrap draw,
so a full run takes hours; results are cached under results_cache/ and reused
when the settings and data are unchanged (--no-cache forces a recompute).
Point at a different data directory with the TYH_DATA_DIR environment variable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tyh import clean, load  # noqa: E402
from tyh.effectsize import DEFAULT_EXPOSURE, effect_sizes  # noqa: E402
from tyh.paths import PROJECT_ROOT  # noqa: E402
from tyh.power import (DEFAULT_CACHE_DIR, analyze_power, power_curve_chart,  # noqa: E402
                       power_report)
from tyh.viz import enable_large_data  # noqa: E402
from tyh.withinbetween import _save_charts  # noqa: E402


def _progress(name: str, done: int, total: int) -> None:
    end = "\n" if done == total else ""
    print(f"\r  [{name}] replicate {done}/{total}", end=end, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures", action="store_true", help="also write the power-curve figure")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures")
    p.add_argument("--fmt", choices=["html", "png", "both"], default="html")
    p.add_argument("--exposure", default=DEFAULT_EXPOSURE)
    p.add_argument("--reps", type=int, default=2000, help="Monte-Carlo replicates per grid cell")
    p.add_argument("--boot", type=int, default=100,
                   help="cluster-bootstrap refits per simulated dataset")
    p.add_argument("--rs-n-boot", type=int, default=2000,
                   help="cluster-bootstrap refits for the MDES (random-slope) SE")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes (default: cores - 2)")
    p.add_argument("--no-cache", action="store_true",
                   help="recompute the simulation even if a matching cache exists")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = clean(load(), verbose=False)
    print("estimating effect sizes (random-slope bootstrap for the MDES) ...")
    eff = effect_sizes(df, exposure=args.exposure, rs_n_boot=args.rs_n_boot, seed=args.seed)
    print("simulating power ...")
    pw = analyze_power(df, exposure=args.exposure, reps=args.reps, boot=args.boot,
                       seed=args.seed, workers=args.workers,
                       cache=None if args.no_cache else DEFAULT_CACHE_DIR,
                       progress=_progress)
    print(power_report(pw, eff))

    if args.figures:
        enable_large_data()
        written = _save_charts({"results_power_curve": power_curve_chart(pw)},
                               args.out_dir, args.fmt)
        print(f"\nWrote {len(written)} figure(s) to {args.out_dir}:")
        for w in written:
            print(f"  {w.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Within-person effect size of the exposure on each outcome composite.

    python run_effectsize.py
    python run_effectsize.py --n-boot 5000

Mixed model per composite (within/between split of q1, random intercept + slope)
with participant-cluster bootstrap CIs and a Type-M caveat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tyh import clean, load  # noqa: E402
from tyh.effectsize import DEFAULT_EXPOSURE, effect_sizes, report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exposure", default=DEFAULT_EXPOSURE)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--rs-n-boot", type=int, default=2000,
                   help="random-slope cluster-bootstrap reps (refits per draw; slower)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = clean(load(), verbose=False)
    results = effect_sizes(df, exposure=args.exposure, n_boot=args.n_boot,
                           rs_n_boot=args.rs_n_boot, seed=args.seed)
    print(report(results, exposure=args.exposure))


if __name__ == "__main__":
    main()

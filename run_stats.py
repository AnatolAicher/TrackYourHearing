#!/usr/bin/env python3
"""Run the directionality (cluster permutation) test on the cleaned data.

    python run_stats.py                      # q1 exposure, q2..q10 outcomes
    python run_stats.py --n-perm 20000

Exposure defaults to q1 (wearing a hearing aid now); outcomes to the q2..q10
continuous items. Valence is already aligned in cleaning (q5/q7 reverse-coded to
question5_rev/question7_rev), so higher = more burden for every outcome. The
cleaned data is restricted to participants with a hearing problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tyh import clean, load  # noqa: E402
from tyh.stats import DEFAULT_EXPOSURE, directional_test  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure", default=DEFAULT_EXPOSURE)
    parser.add_argument("--outcomes", nargs="+", default=None)
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = clean(load(), verbose=False)
    result = directional_test(
        df,
        exposure=args.exposure,
        outcomes=args.outcomes,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(result.summary())


if __name__ == "__main__":
    main()

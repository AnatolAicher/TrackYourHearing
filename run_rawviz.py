#!/usr/bin/env python3
"""Generate raw-data figures (EMA raster + item distributions) -> figures/.

    python run_rawviz.py                # 2 raw-data figures (HTML)
    python run_rawviz.py --fmt both     # also write PNGs
    python run_rawviz.py --out-dir some/dir

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
from tyh.rawviz import rawdata_figures  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures")
    p.add_argument("--fmt", choices=["html", "png", "both"], default="html")
    p.add_argument("--exposure", default=DEFAULT_EXPOSURE)
    args = p.parse_args()

    df = clean(load(), verbose=False)
    written = rawdata_figures(df, args.out_dir, fmt=args.fmt, exposure=args.exposure)
    print(f"Wrote {len(written)} figure(s) to {args.out_dir}:")
    for w in written:
        print(f"  {w.name}  ({w.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

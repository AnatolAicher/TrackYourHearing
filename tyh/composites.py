"""Outcome composites for the effect-size analysis.

Two unit-weighted standardized composites of the burden outcomes, oriented so
that higher = more burden (cleaning already reverse-codes q5/q7).

Construction (per composite):
1. z-score each item over the analytic sample;
2. average the available z-scores per row (mean over non-missing items).

Reliability is reported both pooled (standardized Cronbach's alpha) and
within-person (on person-mean-centred items).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Composites (cleaned column names; all burden-aligned: higher = worse).
COMPOSITES: dict[str, list[str]] = {
    "momentary_burden": [
        "question4", "question5_rev", "question6", "question7_rev",
        "question8", "question9a", "question10",
    ],
    "hearing_difficulty": ["question2", "question3"],
}

# Direction reminder for reports / downstream labelling.
COMPOSITE_DIRECTION = "higher = more burden"


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Column-wise z-score (sample SD), NaN-aware."""
    return (frame - frame.mean()) / frame.std()


def _within_center(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """Person-mean-centre each item (removes between-person differences)."""
    return df.groupby("user_id")[items].transform(lambda s: s - s.mean())


def _std_alpha(corr: pd.DataFrame) -> tuple[float, float]:
    """Standardized Cronbach's alpha and mean inter-item r from a corr matrix."""
    k = corr.shape[0]
    if k < 2:
        return float("nan"), float("nan")
    iu = np.triu_indices(k, 1)
    rbar = float(np.nanmean(corr.to_numpy()[iu]))
    alpha = k * rbar / (1 + (k - 1) * rbar) if not np.isnan(rbar) else float("nan")
    return alpha, rbar


@dataclass
class CompositeReliability:
    name: str
    items: list[str]
    k: int
    n_rows: int                  # rows with >= min_items present
    coverage_mean: float         # mean fraction of items present per scored row
    alpha_pooled: float
    alpha_within: float
    mean_r_pooled: float
    mean_r_within: float
    item_rest: dict[str, float]  # item-rest correlation (pooled, z-scored)

    def lines(self) -> list[str]:
        out = [
            f"[{self.name}]  ({self.k} items, {COMPOSITE_DIRECTION})",
            f"  items            : {', '.join(self.items)}",
            f"  scored rows      : {self.n_rows}   mean item coverage: {self.coverage_mean:.0%}",
            f"  Cronbach alpha   : pooled {self.alpha_pooled:.2f}   within-person {self.alpha_within:.2f}",
            f"  mean inter-item r: pooled {self.mean_r_pooled:.2f}   within-person {self.mean_r_within:.2f}",
            "  item-rest r (pooled, z):",
        ]
        for it, r in self.item_rest.items():
            flag = "   <-- weak fit" if (not np.isnan(r) and r < 0.2) else ""
            out.append(f"    {it:<16}{r:>6.2f}{flag}")
        return out


def reliability(df: pd.DataFrame, name: str, items: list[str], min_items: int = 1) -> CompositeReliability:
    """Reliability diagnostics for one composite."""
    items = [it for it in items if it in df.columns]
    z = _zscore(df[items])
    zw = _zscore(_within_center(df, items))

    alpha_p, rbar_p = _std_alpha(z.corr())
    alpha_w, rbar_w = _std_alpha(zw.corr())

    present = z.notna().sum(axis=1)
    scored = present >= min_items
    coverage = (present[scored] / len(items)).mean() if scored.any() else float("nan")

    # item-rest correlation: each item vs the mean of the others (pooled, z-scored)
    item_rest: dict[str, float] = {}
    for it in items:
        rest = z[[c for c in items if c != it]].mean(axis=1)
        item_rest[it] = float(z[it].corr(rest)) if len(items) > 1 else float("nan")

    return CompositeReliability(
        name=name, items=items, k=len(items),
        n_rows=int(scored.sum()), coverage_mean=float(coverage),
        alpha_pooled=alpha_p, alpha_within=alpha_w,
        mean_r_pooled=rbar_p, mean_r_within=rbar_w,
        item_rest=item_rest,
    )


def add_composites(
    df: pd.DataFrame,
    composites: dict[str, list[str]] | None = None,
    *,
    min_items: int = 1,
) -> pd.DataFrame:
    """Return ``df`` with one unit-weighted standardized composite column per entry.

    Each composite is the mean of its items' z-scores (over available items).
    A row is left missing if fewer than ``min_items`` of the composite's items
    are present.
    """
    composites = composites or COMPOSITES
    out = df.copy()
    for name, items in composites.items():
        items = [it for it in items if it in out.columns]
        z = _zscore(out[items])
        present = z.notna().sum(axis=1)
        comp = z.mean(axis=1)               # mean over available (skips NaN)
        comp[present < min_items] = np.nan
        out[name] = comp
    return out


def composite_report(
    df: pd.DataFrame,
    composites: dict[str, list[str]] | None = None,
    *,
    min_items: int = 1,
) -> str:
    """Build composites and return a printable reliability + correlation report."""
    composites = composites or COMPOSITES
    rule = "=" * 78
    lines = [rule, "TYH outcome composites (unit-weighted, standardized)", rule]

    reports = [reliability(df, n, it, min_items=min_items) for n, it in composites.items()]
    for rep in reports:
        lines += rep.lines()
        lines.append("")

    # correlations between the composites themselves (pooled and within-person)
    with_comp = add_composites(df, composites, min_items=min_items)
    names = list(composites)
    if len(names) >= 2:
        wc = _within_center(with_comp, names)
        lines.append("correlation between composites (pooled / within-person):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                lines.append(
                    f"  {a} vs {b}: {with_comp[a].corr(with_comp[b]):+.2f} / "
                    f"{wc[a].corr(wc[b]):+.2f}"
                )
        lines.append("")
    lines.append(rule)
    return "\n".join(lines)

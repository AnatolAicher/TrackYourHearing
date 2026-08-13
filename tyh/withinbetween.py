"""Within- vs between-person association of the exposure with each composite.

within  = fixed-effects within-person slope (only exposure-varying participants
          contribute).
between = OLS of person-mean composite on person-mean exposure (every participant
          contributes one point).
Both in SD units, with a participant cluster-bootstrap CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd

from .composites import COMPOSITES, add_composites
from .effectsize import DEFAULT_EXPOSURE
from .viz import _save_charts, enable_large_data


def _domain(lo: float, hi: float, pad: float = 0.08) -> list[float]:
    span = hi - lo
    return [lo - pad * span, hi + pad * span]


def _between_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    vx = float(np.var(x))
    return float(np.cov(x, y, bias=True)[0, 1] / vx) if vx > 0 else float("nan")


@dataclass
class WBResult:
    composite: str
    b_within: float
    ci_within: tuple[float, float]
    b_between: float
    ci_between: tuple[float, float]
    n_persons: int
    n_informative: int
    persons: pd.DataFrame   # per-person xbar, ybar, n, informative, composite


def within_between(
    df: pd.DataFrame, composites: dict[str, list[str]] | None = None,
    *, exposure: str = DEFAULT_EXPOSURE, n_boot: int = 2000, seed: int = 0,
) -> list[WBResult]:
    composites = composites or COMPOSITES
    wc = add_composites(df, composites)
    rng = np.random.default_rng(seed)
    out: list[WBResult] = []
    for name in composites:
        sub = wc[["user_id", exposure, name]].dropna()
        cz = ((sub[name] - sub[name].mean()) / sub[name].std()).to_numpy(float)
        x = sub[exposure].to_numpy(float)
        codes, uniq = pd.factorize(sub["user_id"].to_numpy())
        P = len(uniq)

        cnt = np.bincount(codes)
        xbar = np.bincount(codes, x) / cnt
        ybar = np.bincount(codes, cz) / cnt
        xc, yc = x - xbar[codes], cz - ybar[codes]
        Sxc = np.bincount(codes, xc * yc)
        Sxx = np.bincount(codes, xc * xc)
        informative = Sxx > 0

        b_within = float(Sxc.sum() / Sxx.sum()) if Sxx.sum() > 0 else float("nan")
        b_between = _between_slope(xbar, ybar)

        bw, bb = np.empty(n_boot), np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, P, P)
            sxx = Sxx[idx].sum()
            bw[b] = Sxc[idx].sum() / sxx if sxx > 0 else np.nan
            bb[b] = _between_slope(xbar[idx], ybar[idx])
        bw, bb = bw[np.isfinite(bw)], bb[np.isfinite(bb)]

        persons = pd.DataFrame({"xbar": xbar, "ybar": ybar, "n": cnt,
                                "informative": informative, "composite": name})
        out.append(WBResult(
            composite=name, b_within=b_within,
            ci_within=(float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))),
            b_between=b_between,
            ci_between=(float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))),
            n_persons=int(P), n_informative=int(informative.sum()), persons=persons,
        ))
    return out


def report(results: list[WBResult]) -> str:
    rule = "=" * 78
    L = [rule, "TYH within- vs between-person association (SD units)", rule,
         f"  {'composite':<18}{'within b [95% CI]':>26}{'between b [95% CI]':>26}", "  " + "-" * 70]
    for r in results:
        w = f"{r.b_within:+.2f} [{r.ci_within[0]:+.2f}, {r.ci_within[1]:+.2f}]"
        b = f"{r.b_between:+.2f} [{r.ci_between[0]:+.2f}, {r.ci_between[1]:+.2f}]"
        L.append(f"  {r.composite:<18}{w:>26}{b:>26}")
    L += ["",
          "within  = fixed-effects within-person slope (exposure-varying participants only).",
          "between = person-mean composite on person-mean exposure (all participants).",
          "A sign reversal (within < 0 < between) is confounding by indication: aid users",
          "carry more burden between-person, yet wearing lowers burden within-person.", rule]
    return "\n".join(L)


def wb_coeff_chart(results: list[WBResult]) -> alt.LayerChart:
    """Within vs between coefficient per composite, with cluster-bootstrap CIs."""
    rows, order = [], []
    for r in results:
        for level, (b, lo, hi) in (("within", (r.b_within, *r.ci_within)),
                                   ("between", (r.b_between, *r.ci_between))):
            lab = f"{r.composite} · {level}"
            order.append(lab)
            rows.append({"row": lab, "composite": r.composite, "level": level,
                         "b": b, "lo": lo, "hi": hi,
                         "excludes_zero": not (lo < 0.0 < hi)})
    d = pd.DataFrame(rows)
    dom = _domain(min(d.lo.min(), 0.0), max(d.hi.max(), 0.0))
    y = alt.Y("row:N", sort=order, title=None)
    color = alt.Color("level:N", title=None,
                      scale=alt.Scale(domain=["within", "between"], range=["#1f77b4", "#d62728"]))
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(x="x:Q")
    ci = alt.Chart(d).mark_rule(size=2.5).encode(
        y=y, x=alt.X("lo:Q", scale=alt.Scale(domain=dom), title="effect (SD units)"),
        x2="hi:Q", color=color)
    pts = alt.Chart(d).mark_point(size=120, strokeWidth=2.5).encode(
        y=y, x="b:Q", color=color,
        fill=alt.Fill("level:N", legend=None,
                      scale=alt.Scale(domain=["within", "between"], range=["#1f77b4", "#d62728"])),
        fillOpacity=alt.FillOpacity("excludes_zero:N", legend=None,
                                    scale=alt.Scale(domain=[True, False], range=[1.0, 0.0])),
        tooltip=["composite", "level", alt.Tooltip("b:Q", format="+.2f"),
                 alt.Tooltip("lo:Q", format="+.2f"), alt.Tooltip("hi:Q", format="+.2f")])
    # Vega-Lite renders no legend for the fillOpacity channel; this zero-data
    # layer exists solely to draw the filled-vs-hollow legend.
    ci_legend = alt.Chart(pd.DataFrame({"excludes_zero": pd.Series([], dtype=bool)})).mark_point(
        size=120, strokeWidth=2).encode(
        fill=alt.Fill("excludes_zero:N", title="CI excludes 0",
                      scale=alt.Scale(domain=[True, False], range=["#666666", "#ffffff"]),
                      legend=alt.Legend(symbolType="circle", symbolStrokeColor="#666666",
                                        symbolStrokeWidth=2)))
    return (zero + ci + pts + ci_legend).resolve_scale(fill="independent").resolve_legend(
        color="independent", fill="independent").properties(
        width=460, height=alt.Step(38),
        title="Within vs between-person association")


def wb_scatter_chart(results: list[WBResult]) -> alt.HConcatChart:
    """Per-person mean exposure vs mean composite, with the between-person fit."""
    color = alt.Color("exposure type:N", title=None,
                      scale=alt.Scale(domain=["never worn", "always worn", "varies (informative)"],
                                      range=["#999999", "#333333", "#1f77b4"]))
    panels = []
    for r in results:
        d = r.persons.copy()
        d["exposure type"] = np.where(
            d.informative, "varies (informative)",
            np.where(d.xbar >= 0.5, "always worn", "never worn"))
        intercept = float(d.ybar.mean() - r.b_between * d.xbar.mean())
        line_d = pd.DataFrame({"xbar": [0.0, 1.0],
                               "ybar": [intercept, intercept + r.b_between]})
        pts = alt.Chart(d).mark_circle(opacity=0.55).encode(
            x=alt.X("xbar:Q", title="person mean exposure (fraction worn)",
                    scale=alt.Scale(domain=[-0.03, 1.03])),
            y=alt.Y("ybar:Q", title="person mean composite (SD units)"),
            size=alt.Size("n:Q", title="beeps"), color=color,
            tooltip=[alt.Tooltip("xbar:Q", format=".2f"),
                     alt.Tooltip("ybar:Q", format="+.2f"), "n", "exposure type"])
        line = alt.Chart(line_d).mark_line(color="#d62728", size=2).encode(
            x="xbar:Q", y="ybar:Q")
        panels.append((pts + line).properties(width=300, height=300, title=r.composite))
    return alt.hconcat(*panels).properties(
        title=alt.TitleParams(
            text="Between-person view",
            subtitle="red = OLS fit; grey = no within-person exposure variation"))


def withinbetween_figures(df: pd.DataFrame, out_dir, *, fmt: str = "html",
                          exposure: str = DEFAULT_EXPOSURE, n_boot: int = 2000, seed: int = 0) -> list:
    """Build and save the within-vs-between figures."""
    res = within_between(df, exposure=exposure, n_boot=n_boot, seed=seed)
    enable_large_data()
    charts = {
        "results_within_between_coeffs": wb_coeff_chart(res),
        "results_within_between_scatter": wb_scatter_chart(res),
    }
    return _save_charts(charts, out_dir, fmt)

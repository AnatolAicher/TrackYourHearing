"""Result figures (Altair): effect sizes and directionality.

Companion to :mod:`tyh.viz` (diagnostics) and the figures in :mod:`tyh.validity`.
Four charts:

* :func:`effect_forest`        -- per-composite within effect + bootstrap CI;
* :func:`bootstrap_density`    -- cluster-bootstrap draws of each within effect;
* :func:`directionality_dots`  -- per-item within-person correlation (q2..q10);
* :func:`within_slopes`        -- per-participant worn-vs-not composite means.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from .composites import COMPOSITES, add_composites
from .effectsize import DEFAULT_EXPOSURE, EffectResult, effect_sizes
from .stats import DirectionalResult, directional_test
from .viz import _SHORT, _save_charts, enable_large_data


def _domain(lo: float, hi: float, pad: float = 0.08) -> list[float]:
    span = hi - lo
    return [lo - pad * span, hi + pad * span]


def effect_forest(results: list[EffectResult]) -> alt.LayerChart:
    """Forest plot: random-slope within effect per composite, cluster-bootstrap CI."""
    rows = [{"composite": r.composite, "b": r.b_within_rs,
             "lo": r.ci_rs_boot[0], "hi": r.ci_rs_boot[1],
             "excludes_zero": not (r.ci_rs_boot[0] < 0 < r.ci_rs_boot[1])}
            for r in results]
    d = pd.DataFrame(rows)
    los = [r.ci_rs_boot[0] for r in results] + [0.0]
    his = [r.ci_rs_boot[1] for r in results] + [0.0]
    dom = _domain(min(los), max(his))

    y = alt.Y("composite:N", title=None, sort=[r.composite for r in results])
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(x="x:Q")
    ci = alt.Chart(d).mark_rule(size=2.5).encode(
        y=y, x=alt.X("lo:Q", scale=alt.Scale(domain=dom),
                     title="within-person effect (SD units; worn − not)"),
        x2="hi:Q",
    )
    pts = alt.Chart(d).mark_point(filled=True, size=120, opacity=0.95).encode(
        y=y, x="b:Q",
        color=alt.Color("excludes_zero:N", title="CI excludes 0",
                        scale=alt.Scale(domain=[True, False], range=["#1f77b4", "#999999"])),
        tooltip=["composite", alt.Tooltip("b:Q", format="+.2f"),
                 alt.Tooltip("lo:Q", format="+.2f"), alt.Tooltip("hi:Q", format="+.2f")],
    )
    return (zero + ci + pts).properties(
        width=460, height=alt.Step(46),
        title=alt.TitleParams(
            text="Within-person effect of wearing the aid",
            subtitle="random slope; whiskers = 95% cluster-bootstrap CI"),
    )


def bootstrap_density(results: list[EffectResult]) -> alt.LayerChart:
    """Random-slope cluster-bootstrap distribution per composite, with a ±0.1 band."""
    rows = [{"composite": r.composite, "b": float(v)}
            for r in results if r.rs_boot_draws is not None for v in r.rs_boot_draws]
    d = pd.DataFrame(rows)
    pts = pd.DataFrame({"composite": [r.composite for r in results],
                        "b": [r.b_within_rs for r in results]})
    dom = _domain(float(d.b.min()), float(d.b.max()))

    band = alt.Chart(pd.DataFrame({"x": [-0.1], "x2": [0.1]})).mark_rect(
        opacity=0.12, color="gray").encode(x="x:Q", x2="x2:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        strokeDash=[4, 4], color="firebrick").encode(x="x:Q")
    dens = alt.Chart(d).transform_density(
        "b", groupby=["composite"], as_=["b", "density"], extent=dom, steps=200,
    ).mark_area(opacity=0.55).encode(
        x=alt.X("b:Q", scale=alt.Scale(domain=dom),
                title="within-person effect (SD units; worn − not)"),
        y=alt.Y("density:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("composite:N", title="composite"),
    )
    point_rule = alt.Chart(pts).mark_rule(size=2).encode(
        x="b:Q", color=alt.Color("composite:N"),
    )
    return (band + zero + dens + point_rule).properties(
        width=480, height=240,
        title="Random-slope cluster-bootstrap distribution of the within effect",
    )


def directionality_dots(res: DirectionalResult) -> alt.LayerChart:
    """Per-item within-person correlation with the exposure, plus the aggregate."""
    rows = [{"outcome": _SHORT.get(o.outcome, o.outcome), "r": o.r_within,
             "p1": o.p_one_sided, "boot_pos": o.boot_prob_positive, "n": o.n_obs}
            for o in res.per_outcome]
    d = pd.DataFrame(rows)
    order = list(d.sort_values("r")["outcome"])
    dom = _domain(min(d.r.min(), res.mean_r), max(d.r.max(), res.mean_r))

    y = alt.Y("outcome:N", sort=order, title=None)
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(x="x:Q")
    agg = alt.Chart(pd.DataFrame({"x": [res.mean_r]})).mark_rule(
        color="black", size=1.5).encode(x="x:Q")
    pts = alt.Chart(d).mark_point(filled=True, size=120).encode(
        y=y, x=alt.X("r:Q", scale=alt.Scale(domain=dom), title="within-person r with exposure"),
        color=alt.Color("boot_pos:Q", title="boot P(r>0)",
                        scale=alt.Scale(scheme="redblue", domain=[1, 0])),
        tooltip=["outcome", alt.Tooltip("r:Q", format="+.3f"),
                 alt.Tooltip("p1:Q", format=".3f"), alt.Tooltip("boot_pos:Q", format=".2f"), "n"],
    )
    return (zero + agg + pts).properties(
        width=460, height=alt.Step(26),
        title=alt.TitleParams(
            text="Per-item within-person direction",
            subtitle=f"black line = aggregate mean r = {res.mean_r:+.3f}"),
    )


def within_slopes(df: pd.DataFrame, *, exposure: str = DEFAULT_EXPOSURE,
                  composites: dict[str, list[str]] | None = None) -> alt.LayerChart:
    """Per-participant worn-vs-not composite means (informative participants only)."""
    composites = composites or COMPOSITES
    wc = add_composites(df, composites)
    ind_rows, agg_rows = [], []
    for name in composites:
        sub = wc[["user_id", exposure, name]].dropna().copy()
        sub[name] = (sub[name] - sub[name].mean()) / sub[name].std()
        per = sub.groupby(["user_id", exposure])[name].mean().reset_index()
        both = per.groupby("user_id")[exposure].nunique()
        per = per[per["user_id"].isin(both[both == 2].index)]
        per["condition"] = per[exposure].map({0.0: "not worn", 1.0: "worn"})
        per["composite"] = name
        ind_rows.append(per.rename(columns={name: "value"})[
            ["user_id", "condition", "value", "composite"]])
        agg = per.groupby("condition")[name].mean().reset_index()
        agg["composite"] = name
        agg_rows.append(agg.rename(columns={name: "value"}))
    ind = pd.concat(ind_rows, ignore_index=True)
    agg = pd.concat(agg_rows, ignore_index=True)
    ind["series"], agg["series"] = "participant", "mean"

    # Fold the per-panel informative-participant count into the facet header.
    n_per = ind.groupby("composite")["user_id"].nunique().to_dict()
    panel = {c: f"{c} (n = {n_per.get(c, 0)})" for c in composites}
    ind["panel"] = ind["composite"].map(panel)
    agg["panel"] = agg["composite"].map(panel)

    cond = alt.X("condition:N", sort=["not worn", "worn"], title=None)
    yv = alt.Y("value:Q", title="composite (SD units)")
    color = alt.Color("series:N", title=None,
                      scale=alt.Scale(domain=["participant", "mean"], range=["#888", "firebrick"]),
                      legend=alt.Legend(orient="right"))
    lines = alt.Chart(ind).mark_line(opacity=0.3).encode(
        x=cond, y=yv, detail="user_id:N", color=color)
    dots = alt.Chart(ind).mark_point(opacity=0.3, size=18).encode(
        x=cond, y=yv, detail="user_id:N", color=color)
    avg_line = alt.Chart(agg).mark_line(size=3).encode(x=cond, y=yv, color=color)
    avg_pts = alt.Chart(agg).mark_point(filled=True, size=90).encode(
        x=cond, y=yv, color=color,
        tooltip=["composite", "condition", alt.Tooltip("value:Q", format="+.2f")])
    return (lines + dots + avg_line + avg_pts).properties(width=180, height=300).facet(
        column=alt.Column("panel:N", title=None, sort=[panel[c] for c in composites]),
    ).resolve_scale(y="independent").properties(
        title="Within-person worn vs not",
    )


def results_figures(df: pd.DataFrame, out_dir, *, fmt: str = "html",
                    exposure: str = DEFAULT_EXPOSURE, n_boot: int = 2000,
                    rs_n_boot: int = 2000, seed: int = 0) -> list:
    """Build and save the four result figures."""
    eff = effect_sizes(df, exposure=exposure, n_boot=n_boot, rs_n_boot=rs_n_boot, seed=seed)
    dirres = directional_test(df, exposure=exposure, n_boot=n_boot, seed=seed)
    enable_large_data()
    charts = {
        "results_effect_forest": effect_forest(eff),
        "results_bootstrap_density": bootstrap_density(eff),
        "results_directionality_dots": directionality_dots(dirres),
        "results_within_slopes": within_slopes(df, exposure=exposure),
    }
    return _save_charts(charts, out_dir, fmt)

"""Raw-data figures (Altair): what was collected, before any modelling.

Companion to :mod:`tyh.effectviz` (results) and :mod:`tyh.viz` (diagnostics).
Two descriptive charts for the cohort / data-quality narrative:

* :func:`ema_raster`         -- participant x time beep raster, coloured by aid use;
* :func:`item_distributions` -- raw response distribution of each EMA item.

Both read the cleaned table from :func:`tyh.clean.clean`; the items appear under
their cleaned names (q5/q7 reverse-coded), so every outcome is burden-aligned
(higher = more burden). Saved via the shared :func:`tyh.viz._save_charts`.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from .composites import COMPOSITES
from .effectsize import DEFAULT_EXPOSURE
from .viz import CONTINUOUS_QUESTIONS, _SHORT, _save_charts, enable_large_data

# House palette (matches effectviz / withinbetween).
_AID = {"worn": "#1f77b4", "not worn": "#999999"}
_COMPOSITE_RANGE = ["#1f77b4", "#ff7f0e"]  # momentary_burden, hearing_difficulty
# Faint band backgrounds echoing each band's dominant tick colour (the
# exposure-varying band is left unshaded so it reads as the focus).
_BAND_BG = {"never": "#999999", "always": "#1f77b4"}


def ema_raster(df: pd.DataFrame, *, exposure: str = DEFAULT_EXPOSURE) -> alt.LayerChart:
    """Per-participant beep raster over time, each beep coloured by aid use.

    One row per participant, sorted by the fraction of beeps with the aid worn,
    so the three exposure regimes separate into bands: a never-worn block (top),
    the exposure-varying / informative participants (middle), and an always-worn
    block (bottom). Time is days since each participant's first beep on a √ axis,
    so the dense early days stay legible despite the long enrolment tail.
    """
    d = df[["user_id", "save_date", exposure]].dropna(subset=["save_date", exposure]).copy()
    d["save_date"] = pd.to_datetime(d["save_date"])
    first = d.groupby("user_id")["save_date"].transform("min")
    d["rel_day"] = (d["save_date"] - first).dt.total_seconds() / 86400.0
    d["aid"] = d[exposure].map({1.0: "worn", 0.0: "not worn"})

    # Order participants by fraction worn (then volume) -> never / varies / always bands.
    per = d.groupby("user_id").agg(frac_worn=(exposure, "mean"), n=(exposure, "size"))
    order = [str(u) for u in per.sort_values(["frac_worn", "n"]).index]
    d["user_id"] = d["user_id"].astype(str)

    # One row per participant, tagged with its band (only never/always are shaded).
    per = per.reset_index()
    per["user_id"] = per["user_id"].astype(str)
    per["band"] = np.where(per.frac_worn == 0, "never",
                           np.where(per.frac_worn == 1, "always", "varies"))
    band_df = per[per.band.isin(_BAND_BG)][["user_id", "band"]]

    yscale = alt.Scale(paddingInner=0, paddingOuter=0)  # seamless background bands
    bg = alt.Chart(band_df).mark_rect(opacity=0.13).encode(
        y=alt.Y("user_id:N", sort=order, scale=yscale, axis=None),
        color=alt.Color("band:N", legend=None,
                        scale=alt.Scale(domain=list(_BAND_BG), range=list(_BAND_BG.values()))),
    )

    color = alt.Color("aid:N", title="hearing aid",
                      scale=alt.Scale(domain=list(_AID), range=list(_AID.values())))
    ticks = [0, 1, 7, 30, 90, 180, 365]
    raster = alt.Chart(d).mark_tick(thickness=1, opacity=0.7).encode(
        y=alt.Y("user_id:N", sort=order, scale=yscale,
                title="participants (sorted by fraction of beeps worn)",
                axis=alt.Axis(labels=False, ticks=False, domain=False)),
        x=alt.X("rel_day:Q", title="days since first beep (√ scale)",
                scale=alt.Scale(type="sqrt", domainMin=0),
                axis=alt.Axis(values=ticks)),
        color=color,
        tooltip=[alt.Tooltip("user_id:N", title="participant"),
                 alt.Tooltip("save_date:T", title="time", format="%Y-%m-%d %H:%M"),
                 alt.Tooltip("rel_day:Q", title="day", format=".1f"),
                 alt.Tooltip("aid:N", title="aid")],
    )
    return alt.layer(bg, raster).resolve_scale(color="independent").properties(
        width=520, height=alt.Step(5),
        title=alt.TitleParams(
            text="EMA sampling and hearing-aid use over time",
            subtitle="One tick per beep; rows sorted by fraction worn",
        ),
    )


def item_distributions(df: pd.DataFrame,
                       composites: dict[str, list[str]] | None = None) -> alt.FacetChart:
    """Raw response distribution of each continuous EMA item (0-1 sliders).

    One histogram per item, coloured by the composite it belongs to. Panel
    headers carry each item's momentary-missingness rate. All items are
    burden-aligned (q5/q7 reverse-coded), so higher = more burden throughout.
    """
    composites = composites or COMPOSITES
    of_composite = {it: name for name, items in composites.items() for it in items}

    rows = []
    for it in CONTINUOUS_QUESTIONS:
        s = df[it]
        label = f"{_SHORT[it]}  ({s.isna().mean() * 100:.0f}% NA)"
        for v in s.dropna():
            rows.append({"item": label, "value": float(v),
                         "composite": of_composite.get(it, "—")})
    long = pd.DataFrame(rows)
    labels = [f"{_SHORT[it]}  ({df[it].isna().mean() * 100:.0f}% NA)"
              for it in CONTINUOUS_QUESTIONS]

    bars = alt.Chart(long).mark_bar(opacity=0.85).encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=22, extent=[0, 1]),
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(values=[0, 0.5, 1], title=None)),
        y=alt.Y("count():Q", title="beeps"),
        color=alt.Color("composite:N", title="composite",
                        scale=alt.Scale(domain=list(composites), range=_COMPOSITE_RANGE)),
    ).properties(width=150, height=110)

    return bars.facet(
        facet=alt.Facet("item:N", title=None, sort=labels, header=alt.Header(labelFontWeight="bold")),
        columns=3,
    ).resolve_scale(y="independent").properties(
        title=alt.TitleParams(
            text="Raw EMA item responses",
            subtitle="higher = more burden",
        ),
    )


def rawdata_figures(df: pd.DataFrame, out_dir, *, fmt: str = "html",
                    exposure: str = DEFAULT_EXPOSURE) -> list:
    """Build and save the two raw-data figures."""
    enable_large_data()
    charts = {
        "data_ema_raster": ema_raster(df, exposure=exposure),
        "data_item_distributions": item_distributions(df),
    }
    return _save_charts(charts, out_dir, fmt)

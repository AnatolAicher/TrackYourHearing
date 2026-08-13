"""Diagnostic visualisations (Altair).

Pairwise plots of the continuous EMA items, one per grouping variable (sex,
hearing-aid ownership, baseline hearing problem, derived hearing problem). Two
styles:

* :func:`pair_plot` / :func:`diagnostic_pair_plots` -- a fast ``repeat``-based
  scatterplot matrix (SPLOM); scatter in every cell.
* :func:`seaborn_pair_plot` / :func:`diagnostic_seaborn_pairplots` -- a
  seaborn-style pairplot: scatter off-diagonal, per-group **distribution
  (KDE or histogram) on the diagonal**, hand-built as a concatenated grid.

Built on the cleaned table from :func:`tyh.clean.clean`. Charts are returned as
Altair objects and can be saved to standalone HTML (interactive) or PNG (static,
via vl-convert).

Notes / Altair specifics
------------------------
* Altair's ``repeat`` operator applies the same mark to every cell, so a SPLOM's
  **diagonal is a self-scatter (y = x line)**, not a histogram -- which is why
  the seaborn-style version is built by hand from a concatenated grid instead.
* The hand-built grid shares one colour scale (``resolve_scale(color="shared")``)
  so a single legend is hoisted for the whole figure.
* Altair caps inline data at 5000 rows by default; we disable that
  (:func:`enable_large_data`). Identical inline data is consolidated to a single
  embedded copy, so the files stay reasonable (~1.6-2 MB).
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

from .clean import OUTPUT_QUESTION_COLS

# Continuous slider items only (drop the binary switches: q1 wearing-now, q9b skip-flag).
# Uses the cleaned output names, so q5/q7 appear as their reverse-coded versions.
BINARY_QUESTIONS = ["question1", "question9b"]
CONTINUOUS_QUESTIONS = [q for q in OUTPUT_QUESTION_COLS if q not in BINARY_QUESTIONS]

# Compact axis labels for the matrix (question2 -> q2, question9a -> q9a).
_SHORT = {q: q.replace("question", "q") for q in CONTINUOUS_QUESTIONS}

# Grouping (colour) variables -> (legend title, value relabelling for the legend).
COLOR_SPECS: dict[str, tuple[str, dict]] = {
    "sex": ("Sex", {"weiblich": "female", "männlich": "male"}),
    "base_wears_ha": ("Owns hearing aid", {True: "yes", False: "no"}),
    "base_hearing_problem": ("Hearing problem (baseline)", {True: "yes", False: "no"}),
    "derived_hearing_problem": ("Hearing problem (derived)", {True: "yes", False: "no"}),
}


def enable_large_data() -> None:
    """Allow Altair to embed more than its default 5000 rows."""
    alt.data_transformers.disable_max_rows()


def _prep(df: pd.DataFrame, color_col: str) -> pd.DataFrame:
    """Select the columns we plot and relabel the colour column for the legend."""
    cols = ["user_id"] + CONTINUOUS_QUESTIONS + [color_col]
    p = df[cols].copy()
    _, mapping = COLOR_SPECS[color_col]
    p[color_col] = p[color_col].map(
        lambda v: None if pd.isna(v) else mapping.get(v, v)
    ).astype("string")
    return p.rename(columns=_SHORT)


def pair_plot(
    df: pd.DataFrame,
    color_col: str,
    *,
    point_size: int = 6,
    opacity: float = 0.25,
    cell_px: int = 90,
) -> alt.RepeatChart:
    """Scatterplot matrix of the continuous EMA items, coloured by ``color_col``.

    ``color_col`` must be one of :data:`COLOR_SPECS`
    (``sex``, ``base_wears_ha``, ``base_hearing_problem``, ``derived_hearing_problem``).
    """
    if color_col not in COLOR_SPECS:
        raise ValueError(f"color_col must be one of {list(COLOR_SPECS)}, got {color_col!r}")

    p = _prep(df, color_col)
    title, _ = COLOR_SPECS[color_col]
    fields = [_SHORT[q] for q in CONTINUOUS_QUESTIONS]

    base = (
        alt.Chart(p)
        .mark_circle(size=point_size, opacity=opacity)
        .encode(
            x=alt.X(
                alt.repeat("column"),
                type="quantitative",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(values=[0, 0.5, 1], grid=False),
            ),
            y=alt.Y(
                alt.repeat("row"),
                type="quantitative",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(values=[0, 0.5, 1], grid=False),
            ),
            color=alt.Color(f"{color_col}:N", title=title, scale=alt.Scale(scheme="set1")),
            tooltip=[alt.Tooltip("user_id:N", title="participant"),
                     alt.Tooltip(f"{color_col}:N", title=title)],
        )
        .properties(width=cell_px, height=cell_px)
    )

    return (
        base.repeat(row=fields, column=fields)
        .properties(title=f"TYH continuous EMA items – pairwise scatter, coloured by {title}")
        .configure_axis(labelFontSize=7, titleFontSize=9)
        .configure_legend(titleFontSize=12, labelFontSize=11, symbolOpacity=0.9)
        .configure_title(fontSize=14)
        .configure_view(strokeOpacity=0.15)
    )


# --------------------------------------------------------------------------- #
# Seaborn-style pairplot: scatter off-diagonal, distribution on the diagonal.
# Hand-built grid (Altair's `repeat` can't mix mark types per cell).
# --------------------------------------------------------------------------- #


def _color_enc(p: pd.DataFrame, color_col: str, title: str) -> alt.Color:
    """Colour encoding with a pinned domain so colours match across every panel.

    Every panel carries the same legend definition; the builder shares the colour
    scale across the grid (``resolve_scale(color="shared")``) so Vega-Lite hoists
    a single legend to the top of the composition.
    """
    domain = sorted(v for v in p[color_col].dropna().unique())
    legend = alt.Legend(title=title, orient="right", symbolOpacity=0.9)
    return alt.Color(f"{color_col}:N", scale=alt.Scale(domain=domain, scheme="set1"), legend=legend)


def _edge_axis(show: bool, title: str | None) -> alt.Axis | None:
    """A clean [0,1] axis on grid edges only (interior panels get no axis)."""
    if not show:
        return None
    return alt.Axis(title=title, values=[0, 0.5, 1], grid=False)


def _scatter_panel(p, xf, yf, color_col, title, *, size, opacity, cell, x_axis, y_axis):
    return (
        alt.Chart(p)
        .mark_circle(size=size, opacity=opacity)
        .encode(
            x=alt.X(f"{xf}:Q", scale=alt.Scale(domain=[0, 1]), axis=x_axis),
            y=alt.Y(f"{yf}:Q", scale=alt.Scale(domain=[0, 1]), axis=y_axis),
            color=_color_enc(p, color_col, title),
            tooltip=[alt.Tooltip("user_id:N", title="participant"),
                     alt.Tooltip(f"{color_col}:N", title=title)],
        )
        .properties(width=cell, height=cell)
    )


def _diag_panel(p, f, color_col, title, *, diag, cell, x_axis, y_title):
    """Diagonal cell: per-group KDE (area) or layered histogram of ``f``."""
    color = _color_enc(p, color_col, title)
    # the density/count axis carries no meaning across cells -> hide it (keep an
    # optional title only on the top-left cell, to label row 0's variable)
    y_axis = alt.Axis(title=y_title, labels=False, ticks=False, grid=False) if y_title else None
    if diag == "hist":
        return (
            alt.Chart(p)
            .mark_bar(opacity=0.5)
            .encode(
                x=alt.X(f"{f}:Q", bin=alt.Bin(maxbins=22, extent=[0, 1]),
                        scale=alt.Scale(domain=[0, 1]), axis=x_axis),
                y=alt.Y("count():Q", stack=None, axis=y_axis),
                color=color,
            )
            .properties(width=cell, height=cell)
        )
    return (
        alt.Chart(p)
        .transform_density(f, groupby=[color_col], extent=[0, 1], steps=80, as_=[f, "density"])
        .mark_area(opacity=0.4)
        .encode(
            x=alt.X(f"{f}:Q", scale=alt.Scale(domain=[0, 1]), axis=x_axis),
            y=alt.Y("density:Q", stack=None, axis=y_axis),
            color=color,
        )
        .properties(width=cell, height=cell)
    )


def seaborn_pair_plot(
    df: pd.DataFrame,
    color_col: str,
    *,
    diag: str = "hist",
    point_size: int = 6,
    opacity: float = 0.22,
    cell_px: int = 78,
) -> alt.VConcatChart:
    """Seaborn-style pairplot of the continuous EMA items, coloured by ``color_col``.

    Off-diagonal cells are scatter plots; diagonal cells show each item's
    per-group distribution (``diag="hist"`` for layered histograms, the default
    and more faithful for the discrete/floor-heavy items; ``diag="kde"`` for
    smoothed densities). ``color_col`` must be one of :data:`COLOR_SPECS`.
    """
    if color_col not in COLOR_SPECS:
        raise ValueError(f"color_col must be one of {list(COLOR_SPECS)}, got {color_col!r}")
    if diag not in {"kde", "hist"}:
        raise ValueError(f"diag must be 'kde' or 'hist', got {diag!r}")

    p = _prep(df, color_col)
    title, _ = COLOR_SPECS[color_col]
    fields = [_SHORT[q] for q in CONTINUOUS_QUESTIONS]
    n = len(fields)

    rows = []
    for i in range(n):
        cells = []
        for j in range(n):
            on_bottom, on_left = (i == n - 1), (j == 0)
            x_axis = _edge_axis(on_bottom, fields[j])
            if i == j:
                cells.append(_diag_panel(
                    p, fields[i], color_col, title, diag=diag, cell=cell_px,
                    x_axis=x_axis, y_title=(fields[i] if on_left else None),
                ))
            else:
                cells.append(_scatter_panel(
                    p, fields[j], fields[i], color_col, title,
                    size=point_size, opacity=opacity, cell=cell_px,
                    x_axis=x_axis, y_axis=_edge_axis(on_left, fields[i]),
                ))
        rows.append(alt.hconcat(*cells, spacing=4))

    diag_label = "KDE" if diag == "kde" else "histogram"
    return (
        alt.vconcat(*rows, spacing=4)
        .resolve_scale(color="shared")  # -> a single hoisted legend for the grid
        .properties(title=f"TYH continuous EMA items – pairplot ({diag_label} diagonal), "
                          f"coloured by {title}")
        .configure_axis(labelFontSize=7, titleFontSize=9, domainColor="#999")
        .configure_view(strokeOpacity=0.15)
        .configure_legend(titleFontSize=12, labelFontSize=11)
        .configure_title(fontSize=14)
    )


def _save_charts(charts: dict[str, alt.TopLevelMixin], out_dir: str | Path, fmt: str) -> list[Path]:
    """Write each named chart to HTML and/or PNG; return the paths written."""
    if fmt not in {"html", "png", "both"}:
        raise ValueError(f"fmt must be 'html', 'png' or 'both', got {fmt!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    enable_large_data()
    written: list[Path] = []
    for stem, chart in charts.items():
        if fmt in {"html", "both"}:
            path = out / f"{stem}.html"
            # self-contained (vendored JS) + canvas renderer for many points
            chart.save(str(path), inline=True, embed_options={"renderer": "canvas"})
            written.append(path)
        if fmt in {"png", "both"}:
            path = out / f"{stem}.png"
            chart.save(str(path), ppi=144)
            written.append(path)
    return written


def diagnostic_pair_plots(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    fmt: str = "html",
    max_points: int | None = None,
) -> list[Path]:
    """Build and save one SPLOM (repeat-based, scatter-only) per grouping variable."""
    if max_points is not None and len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    charts = {f"pairplot_by_{c}": pair_plot(df, c) for c in COLOR_SPECS}
    return _save_charts(charts, out_dir, fmt)


def diagnostic_seaborn_pairplots(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    fmt: str = "html",
    diag: str = "hist",
    max_points: int | None = None,
) -> list[Path]:
    """Build and save one seaborn-style pairplot per grouping variable."""
    if max_points is not None and len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    charts = {f"pairplot_seaborn_{diag}_by_{c}": seaborn_pair_plot(df, c, diag=diag)
              for c in COLOR_SPECS}
    return _save_charts(charts, out_dir, fmt)

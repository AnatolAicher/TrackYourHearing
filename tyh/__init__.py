"""Track Your Hearing (TYH) -- EMA hearing-aid analysis.

Public API::

    from tyh import load, diagnose
    data = load()        # -> TYHData(ema=..., baseline=...)
    diagnose(data)       # print data-quality diagnostics
"""

from __future__ import annotations

from .ingest import TYHData, load, load_baseline, load_ema
from .diagnostics import diagnose
from .clean import CleaningReport, clean, clean_with_report
from .viz import (
    diagnostic_pair_plots,
    diagnostic_seaborn_pairplots,
    pair_plot,
    seaborn_pair_plot,
)
from .stats import DirectionalResult, directional_test
from .composites import COMPOSITES, add_composites, composite_report, reliability
from .validity import ValidityResult, analyze, diagnostic_validity_plots, report as validity_report
from .effectsize import EffectResult, effect_size, effect_sizes, report as effectsize_report
from .effectviz import (
    bootstrap_density,
    directionality_dots,
    effect_forest,
    results_figures,
    within_slopes,
)
from .withinbetween import WBResult, within_between, withinbetween_figures
from .power import PowerResult, analyze_power, mdes, power_figures, power_report
from .rawviz import ema_raster, item_distributions, rawdata_figures

__all__ = [
    "TYHData", "load", "load_ema", "load_baseline", "diagnose",
    "clean", "clean_with_report", "CleaningReport",
    "pair_plot", "diagnostic_pair_plots",
    "seaborn_pair_plot", "diagnostic_seaborn_pairplots",
    "directional_test", "DirectionalResult",
    "COMPOSITES", "add_composites", "composite_report", "reliability",
    "analyze", "validity_report", "diagnostic_validity_plots", "ValidityResult",
    "effect_size", "effect_sizes", "effectsize_report", "EffectResult",
    "effect_forest", "bootstrap_density", "directionality_dots", "within_slopes",
    "results_figures",
    "within_between", "withinbetween_figures", "WBResult",
    "analyze_power", "mdes", "power_report", "power_figures", "PowerResult",
    "ema_raster", "item_distributions", "rawdata_figures",
]

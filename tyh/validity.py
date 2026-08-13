"""Convergent & discriminant validity of the outcome composites.

Convergent validity = items within a construct correlate with each other;
discriminant validity = they do not correlate too strongly with the other
constructs' items.

Computed at three levels:

* ``within``  -- person-mean-centred items (headline)
* ``between`` -- one row per participant (person means)
* ``pooled``  -- z-scored raw items

Metrics (all read off an item correlation matrix):

* mean within-block vs between-block correlation, and the gap;
* **HTMT** (heterotrait-monotrait ratio), with a participant-cluster bootstrap CI;
* Fornell-Larcker: sqrt(AVE) vs inter-composite correlation;
* item-level own- vs cross-construct correlations (Campbell-Fiske MTMM);
* per-item ICC(1).

Edge cases handled: for a two-item construct (hearing_difficulty) standardized
alpha == Spearman-Brown (one number); a single-item construct has no internal
convergent metric (discriminant evidence only); AVE here is a descriptive PC
proxy, not a fitted CFA estimate; correlations are pairwise-complete (q9a is
~33% skip-missing) and per-cell n is for coverage only -- all inference comes
from the person-cluster bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd

from .composites import COMPOSITES, _within_center, _zscore
from .viz import _SHORT, _save_charts, enable_large_data

LEVELS = ("within", "between", "pooled")

# Items in fixed block order (flat), and the block -> items map.
BLOCKS: dict[str, list[str]] = {name: list(items) for name, items in COMPOSITES.items()}
ITEMS: list[str] = [it for items in BLOCKS.values() for it in items]


# --------------------------------------------------------------------------- #
# Correlation matrices at each level
# --------------------------------------------------------------------------- #
def _items_at_level(df: pd.DataFrame, items: list[str], level: str) -> pd.DataFrame:
    if level == "within":
        return _within_center(df, items)
    if level == "between":
        return df.groupby("user_id")[items].mean()
    if level == "pooled":
        return _zscore(df[items])
    raise ValueError(f"level must be one of {LEVELS}, got {level!r}")


def corr_matrix(df: pd.DataFrame, items: list[str], level: str, method: str = "pearson"):
    """Pairwise-complete correlation matrix + per-cell n at the given level."""
    frame = _items_at_level(df, items, level)
    R = frame.corr(method=method)               # pairwise-complete
    present = frame.notna().astype(int)
    N = present.T @ present                      # n per pair (coverage only)
    return R, N


# --------------------------------------------------------------------------- #
# Block-level convergent / discriminant statistics
# --------------------------------------------------------------------------- #
def _within_mean(R: pd.DataFrame, items: list[str], absolute: bool = False) -> float:
    if len(items) < 2:
        return float("nan")
    sub = R.loc[items, items].to_numpy()
    iu = np.triu_indices(len(items), 1)
    v = np.abs(sub[iu]) if absolute else sub[iu]
    return float(np.nanmean(v))


def _between_mean(R: pd.DataFrame, a: list[str], b: list[str], absolute: bool = False) -> float:
    sub = R.loc[a, b].to_numpy()
    v = np.abs(sub) if absolute else sub
    return float(np.nanmean(v))


def _max_abs(R: pd.DataFrame, a: list[str], b: list[str]) -> float:
    return float(np.nanmax(np.abs(R.loc[a, b].to_numpy())))


def htmt(R: pd.DataFrame, a: list[str], b: list[str]) -> float:
    """Heterotrait-monotrait ratio; NaN if either block has <2 items."""
    wa, wb = _within_mean(R, a, absolute=True), _within_mean(R, b, absolute=True)
    if np.isnan(wa) or np.isnan(wb) or wa <= 0 or wb <= 0:
        return float("nan")
    return _between_mean(R, a, b, absolute=True) / np.sqrt(wa * wb)


def ave_proxy(R: pd.DataFrame, items: list[str]) -> float:
    """Descriptive AVE proxy = variance share on PC1 of the block submatrix.

    PSD-guarded (clip negative eigenvalues from pairwise-complete matrices).
    NOT a fitted-CFA AVE; for k=2 it is just-identified (treat as approximate).
    """
    if len(items) < 2:
        return float("nan")
    sub = R.loc[items, items].to_numpy()
    w = np.clip(np.linalg.eigvalsh(sub), 0, None)
    return float(w.max() / len(items))


def std_alpha(R: pd.DataFrame, items: list[str]) -> float:
    """Standardized Cronbach's alpha (== Spearman-Brown when k==2)."""
    rbar = _within_mean(R, items)
    k = len(items)
    if k < 2 or np.isnan(rbar):
        return float("nan")
    return k * rbar / (1 + (k - 1) * rbar)


# --------------------------------------------------------------------------- #
# Item-level own- vs cross-construct correlations (Campbell-Fiske)
# --------------------------------------------------------------------------- #
@dataclass
class ItemLoading:
    item: str
    block: str
    r_own: float                 # corrected item-rest within its own block
    cross: dict[str, float]      # correlation with each other block's composite
    max_cross: float
    gap: float                   # r_own - max_cross (>0 = clean)


def item_loadings(df: pd.DataFrame, level: str = "within", method: str = "pearson") -> list[ItemLoading]:
    frame = _items_at_level(df, ITEMS, level)
    comps = {name: frame[items].mean(axis=1) for name, items in BLOCKS.items()}
    out: list[ItemLoading] = []
    for name, items in BLOCKS.items():
        for it in items:
            others = [o for o in items if o != it]
            r_own = float(frame[it].corr(frame[others].mean(axis=1), method=method)) if others else float("nan")
            cross = {c: float(frame[it].corr(comps[c], method=method)) for c in BLOCKS if c != name}
            mx = max(cross.values()) if cross else float("nan")
            out.append(ItemLoading(it, name, r_own, cross, mx, r_own - mx))
    return out


# --------------------------------------------------------------------------- #
# ICC(1) per item -- between-person share of variance. REML variance components
# (statsmodels) with a one-way ANOVA method-of-moments fallback; the estimator
# used is reported alongside.
# --------------------------------------------------------------------------- #
def _icc1_anova(d: pd.DataFrame, it: str) -> float:
    g = d.groupby("user_id")[it]
    ng = g.size().to_numpy()
    k, N = len(ng), len(d)
    if k < 2 or N <= k:
        return float("nan")
    means = g.mean()
    grand = d[it].mean()
    ssb = float((ng * (means.to_numpy() - grand) ** 2).sum())
    ssw = float(((d[it] - d["user_id"].map(means)) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (ng ** 2).sum() / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den else float("nan")


def icc1(df: pd.DataFrame, items: list[str] | None = None) -> tuple[dict[str, float], str]:
    """Per-item ICC(1); returns (values, estimator_label)."""
    items = items or ITEMS
    try:
        import warnings
        import statsmodels.formula.api as smf
        method = "REML variance components"
    except Exception:
        smf = None
        method = "one-way ANOVA"

    out: dict[str, float] = {}
    for it in items:
        d = df[["user_id", it]].dropna()
        if d["user_id"].nunique() < 2 or len(d) <= d["user_id"].nunique():
            out[it] = float("nan")
            continue
        if smf is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dd = d.rename(columns={it: "y"})
                    m = smf.mixedlm("y ~ 1", dd, groups=dd["user_id"]).fit(reml=True, method="lbfgs")
                vg, ve = float(m.cov_re.iloc[0, 0]), float(m.scale)
                out[it] = vg / (vg + ve) if (vg + ve) > 0 else float("nan")
                continue
            except Exception:
                pass
        out[it] = _icc1_anova(d, it)
    return out, method


# --------------------------------------------------------------------------- #
# Person-cluster bootstrap of HTMT (within level)
# --------------------------------------------------------------------------- #
def _pairwise_abs_corr(M: np.ndarray) -> np.ndarray:
    """|Pearson r| matrix with pairwise-complete deletion (M may contain NaN)."""
    p = M.shape[1]
    R = np.ones((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            a, b = M[:, i], M[:, j]
            m = ~np.isnan(a) & ~np.isnan(b)
            if m.sum() < 3:
                R[i, j] = R[j, i] = np.nan
                continue
            ai, bi = a[m] - a[m].mean(), b[m] - b[m].mean()
            den = np.sqrt((ai * ai).sum() * (bi * bi).sum())
            r = (ai * bi).sum() / den if den > 0 else np.nan
            R[i, j] = R[j, i] = abs(r)
    return R


def htmt_bootstrap(
    df: pd.DataFrame, block_a: list[str], block_b: list[str], *, n_boot: int = 2000, seed: int = 0
) -> np.ndarray:
    """Cluster-bootstrap distribution of HTMT(block_a, block_b) at the within level.

    Resamples participants with replacement. Within-person centring is per
    participant, hence invariant to resampling, so we centre once and stack the
    drawn participants' centred rows each replicate.
    """
    items = list(block_a) + list(block_b)
    ka = len(block_a)
    centred = _within_center(df, items).to_numpy()
    person = df["user_id"].to_numpy()
    uniq = pd.unique(person)
    per_person = [centred[person == p] for p in uniq]
    P = len(uniq)
    rng = np.random.default_rng(seed)

    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, P, P)
        M = np.vstack([per_person[i] for i in idx])
        R = _pairwise_abs_corr(M)
        wa = np.nanmean(R[:ka, :ka][np.triu_indices(ka, 1)]) if ka > 1 else np.nan
        wb = np.nanmean(R[ka:, ka:][np.triu_indices(len(block_b), 1)]) if len(block_b) > 1 else np.nan
        cross = np.nanmean(R[:ka, ka:])
        vals[b] = cross / np.sqrt(wa * wb) if (wa > 0 and wb > 0) else np.nan
    return vals[~np.isnan(vals)]


# --------------------------------------------------------------------------- #
# Text report
# --------------------------------------------------------------------------- #
@dataclass
class ValidityResult:
    level: str
    R: dict[str, pd.DataFrame]            # per-level correlation matrices
    within_rbar: dict[str, dict[str, float]]   # level -> block -> within mean r
    between_rbar: dict[str, dict[str, float]]  # level -> "B|C" -> between mean r
    htmt: dict[str, dict[str, float]]          # level -> "B|C" -> HTMT (or maxabs for k=1)
    htmt_ci: dict[str, tuple[float, float]]    # "B|C" -> (lo, hi) within bootstrap
    ave: dict[str, dict[str, float]]
    composite_corr: dict[str, dict[str, float]]
    alpha: dict[str, dict[str, float]]
    icc: dict[str, float]
    icc_method: str
    loadings: list[ItemLoading]
    n_boot: int = 0


def analyze(
    df: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0, method: str = "pearson"
) -> ValidityResult:
    pair_keys = [(a, b) for i, a in enumerate(BLOCKS) for b in list(BLOCKS)[i + 1:]]
    R = {lv: corr_matrix(df, ITEMS, lv, method)[0] for lv in LEVELS}

    within_rbar = {lv: {n: _within_mean(R[lv], it) for n, it in BLOCKS.items()} for lv in LEVELS}
    between_rbar = {lv: {f"{a}|{b}": _between_mean(R[lv], BLOCKS[a], BLOCKS[b]) for a, b in pair_keys}
                    for lv in LEVELS}
    htmt_v = {lv: {} for lv in LEVELS}
    for lv in LEVELS:
        for a, b in pair_keys:
            v = htmt(R[lv], BLOCKS[a], BLOCKS[b])
            htmt_v[lv][f"{a}|{b}"] = v if not np.isnan(v) else _max_abs(R[lv], BLOCKS[a], BLOCKS[b])
    ave = {lv: {n: ave_proxy(R[lv], it) for n, it in BLOCKS.items()} for lv in LEVELS}
    alpha = {lv: {n: std_alpha(R[lv], it) for n, it in BLOCKS.items()} for lv in LEVELS}

    # composite (phi) correlations per level
    comp_corr = {lv: {} for lv in LEVELS}
    for lv in LEVELS:
        frame = _items_at_level(df, ITEMS, lv)
        comps = pd.DataFrame({n: frame[it].mean(axis=1) for n, it in BLOCKS.items()})
        cc = comps.corr(method=method)
        for a, b in pair_keys:
            comp_corr[lv][f"{a}|{b}"] = float(cc.loc[a, b])

    # bootstrap CI for HTMT (within) only for pairs where both blocks have k>=2
    htmt_ci: dict[str, tuple[float, float]] = {}
    for a, b in pair_keys:
        if len(BLOCKS[a]) >= 2 and len(BLOCKS[b]) >= 2:
            boot = htmt_bootstrap(df, BLOCKS[a], BLOCKS[b], n_boot=n_boot, seed=seed)
            if boot.size:
                htmt_ci[f"{a}|{b}"] = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    icc_vals, icc_method = icc1(df)
    return ValidityResult(
        level="within", R=R, within_rbar=within_rbar, between_rbar=between_rbar,
        htmt=htmt_v, htmt_ci=htmt_ci, ave=ave, composite_corr=comp_corr, alpha=alpha,
        icc=icc_vals, icc_method=icc_method, loadings=item_loadings(df, "within", method),
        n_boot=n_boot,
    )


def report(res: ValidityResult) -> str:
    rule = "=" * 78
    L = [rule, "TYH composite validity (convergent & discriminant)", rule,
         "Levels: within-person = headline; pooled/between = context.", ""]

    L.append("CONVERGENT  (within-block mean r; standardized alpha; AVE proxy)")
    L.append(f"  {'construct':<18}{'k':>2}  {'r-bar W/B/pool':>18}  {'alpha W':>8}{'AVE W':>7}  note")
    for name, items in BLOCKS.items():
        k = len(items)
        rb = [res.within_rbar[lv][name] for lv in LEVELS]
        rbs = "  ".join(f"{x:+.2f}" if not np.isnan(x) else "  -- " for x in rb)
        aw = res.alpha["within"][name]
        av = res.ave["within"][name]
        note = ("single item (no internal metric)" if k == 1
                else "alpha = Spearman-Brown (k=2)" if k == 2 else "")
        L.append(f"  {name:<18}{k:>2}  {rbs:>18}  "
                 f"{aw:>8.2f}" .replace("nan", "  -- ") + f"{av:>7.2f}".replace("nan", "  -- ")
                 + f"  {note}")

    L += ["", "DISCRIMINANT  (between-block mean r; HTMT within/pooled; Fornell-Larcker)"]
    for key in res.htmt["within"]:
        a, b = key.split("|")
        single = len(BLOCKS[a]) < 2 or len(BLOCKS[b]) < 2
        hw, hp = res.htmt["within"][key], res.htmt["pooled"][key]
        rbw, rbp = res.between_rbar["within"][key], res.between_rbar["pooled"][key]
        L.append(f"  [{a} vs {b}]")
        L.append(f"     between r-bar : within {rbw:+.2f}   pooled {rbp:+.2f}")
        if single:
            L.append(f"     max|r|       : within {hw:.2f}   pooled {hp:.2f}   (HTMT undefined, k=1)")
        else:
            ci = res.htmt_ci.get(key)
            cis = f"  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
            L.append(f"     HTMT         : within {hw:.2f}{cis}   pooled {hp:.2f}   "
                     f"(<0.85 distinct){'  <-- borderline' if hp >= 0.83 else ''}")
            # Fornell-Larcker (within)
            phi = res.composite_corr["within"][key]
            sa, sb = np.sqrt(res.ave['within'][a]), np.sqrt(res.ave['within'][b])
            ok = sa > phi and sb > phi
            L.append(f"     Fornell-Larcker (within): sqrt(AVE) {sa:.2f}/{sb:.2f} vs phi {phi:.2f}"
                     f"  -> {'pass' if ok else 'FAIL'}")

    L += ["", "ITEM CROSS-LOADINGS (within: own-construct item-rest vs strongest other construct)"]
    L.append(f"  {'item':<14}{'block':<18}{'r_own':>7}{'max_cross':>11}{'gap':>7}  flag")
    for it in res.loadings:
        own = f"{it.r_own:+.2f}" if not np.isnan(it.r_own) else "  -- "
        flag = ""
        if np.isnan(it.r_own):
            flag = "single-item: discriminant only"
        elif it.gap < 0:
            flag = "Campbell-Fiske VIOLATION (cross > own)"
        elif it.r_own < 0.20:
            flag = "weak own-construct fit"
        L.append(f"  {_SHORT.get(it.item, it.item):<14}{it.block:<18}{own:>7}"
                 f"{it.max_cross:>11.2f}{it.gap:>7.2f}  {flag}")

    L += ["", f"ICC(1) per item (between-person share of variance; {res.icc_method}):"]
    L.append("  " + "  ".join(f"{_SHORT.get(it, it)}={v:.2f}" for it, v in res.icc.items()))
    L.append("  (these items carry substantial between-person/trait variance; the within level")
    L.append("   is the headline because the research question and models are within-person.)")

    L += ["", rule]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Visualisations (Altair)
# --------------------------------------------------------------------------- #
_ORDER = [_SHORT.get(it, it) for it in ITEMS]
_BLOCK_OF = {_SHORT.get(it, it): name for name, items in BLOCKS.items() for it in items}


def corr_heatmap(df: pd.DataFrame, *, method: str = "pearson") -> alt.LayerChart:
    """Item correlation heatmap, items blocked by construct.

    Upper triangle = within-person r; lower triangle = between-person r;
    diagonal = 1.
    """
    Rw = corr_matrix(df, ITEMS, "within", method)[0]
    Rb = corr_matrix(df, ITEMS, "between", method)[0]
    short = [_SHORT.get(it, it) for it in ITEMS]
    rows = []
    for i, ri in enumerate(short):
        for j, cj in enumerate(short):
            if i < j:
                r, lvl = Rw.iloc[i, j], "within (upper)"
            elif i > j:
                r, lvl = Rb.iloc[i, j], "between (lower)"
            else:
                r, lvl = 1.0, "diag"
            rows.append({"row": ri, "col": cj, "r": float(r), "level": lvl,
                         "same_block": _BLOCK_OF[ri] == _BLOCK_OF[cj]})
    d = pd.DataFrame(rows)
    base = alt.Chart(d).encode(
        x=alt.X("col:N", sort=_ORDER, title=None),
        y=alt.Y("row:N", sort=_ORDER, title=None),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
                        legend=alt.Legend(title="r")),
        tooltip=["row", "col", alt.Tooltip("r:Q", format=".2f"), "level"],
    )
    text = base.mark_text(fontSize=9).encode(
        text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.55", alt.value("white"), alt.value("black")),
    )
    return (rect + text).properties(
        width=360, height=360,
        title="Item correlations by construct – upper=within, lower=between (q5/q7 reverse-coded)",
    ).configure_axis(labelFontSize=10)


def htmt_chart(res: ValidityResult) -> alt.LayerChart:
    """HTMT (within & pooled) for each defined (k>=2) pair, with bootstrap CI + cutoffs."""
    rows = []
    for key in res.htmt["within"]:
        a, b = key.split("|")
        if len(BLOCKS[a]) < 2 or len(BLOCKS[b]) < 2:
            continue
        ci = res.htmt_ci.get(key, (None, None))
        x = f"{a} vs {b}\n(within)"
        rows.append({"x": x, "level": "within", "htmt": res.htmt["within"][key],
                     "lo": ci[0], "hi": ci[1]})
        rows.append({"x": f"{a} vs {b}\n(pooled)", "level": "pooled",
                     "htmt": res.htmt["pooled"][key], "lo": None, "hi": None})
    d = pd.DataFrame(rows)
    enc_x = alt.X("x:N", title=None)
    bars = alt.Chart(d).mark_bar().encode(
        x=enc_x, y=alt.Y("htmt:Q", scale=alt.Scale(domain=[0, 1]), title="HTMT"),
        color=alt.Color("level:N", title="level"),
        tooltip=["x", alt.Tooltip("htmt:Q", format=".3f")],
    )
    err = alt.Chart(d).mark_errorbar(color="black").encode(x=enc_x, y="lo:Q", y2="hi:Q")
    cutoff = alt.Chart(pd.DataFrame({"y": [0.85, 0.90]})).mark_rule(
        strokeDash=[4, 4], color="firebrick").encode(y="y:Q")
    return (bars + err + cutoff).properties(
        width=alt.Step(70),
        title="HTMT discriminant validity (< 0.85 = distinct; whiskers = 95% cluster-bootstrap CI)",
    )


def item_loading_chart(res: ValidityResult) -> alt.LayerChart:
    """Per-item own-construct vs strongest other-construct correlation (within)."""
    rows = []
    for it in res.loadings:
        lbl = _SHORT.get(it.item, it.item)
        if not np.isnan(it.r_own):
            rows.append({"item": lbl, "kind": "own construct", "r": it.r_own,
                         "block": it.block, "gap": it.gap})
        rows.append({"item": lbl, "kind": "max other construct", "r": it.max_cross,
                     "block": it.block, "gap": it.gap})
    d = pd.DataFrame(rows)
    order = [_SHORT.get(it.item, it.item) for it in
             sorted(res.loadings, key=lambda x: (x.block, -(x.gap if not np.isnan(x.gap) else -9)))]
    conn = alt.Chart(d).mark_line().encode(
        x=alt.X("r:Q", scale=alt.Scale(domain=[-0.3, 1]), title="within-person correlation"),
        y=alt.Y("item:N", sort=order, title=None),
        detail="item:N",
        color=alt.condition("datum.gap < 0", alt.value("firebrick"), alt.value("seagreen")),
    )
    pts = alt.Chart(d).mark_point(filled=True, size=70).encode(
        x="r:Q", y=alt.Y("item:N", sort=order),
        shape=alt.Shape("kind:N", title=None),
        color=alt.Color("kind:N", scale=alt.Scale(range=["#1f77b4", "#d62728"]), title=None),
        tooltip=["item", "kind", alt.Tooltip("r:Q", format=".2f")],
    )
    rules = alt.Chart(pd.DataFrame({"x": [0.20, 0.30]})).mark_rule(
        strokeDash=[3, 3], color="gray").encode(x="x:Q")
    return (rules + conn + pts).properties(
        width=420, height=18 * len(res.loadings),
        title="Item own- vs other-construct correlation (red = cross > own: Campbell-Fiske violation)",
    )


def diagnostic_validity_plots(df: pd.DataFrame, out_dir, *, fmt: str = "html",
                              n_boot: int = 2000, seed: int = 0) -> list:
    """Build and save the three validity figures."""
    res = analyze(df, n_boot=n_boot, seed=seed)
    enable_large_data()
    charts = {
        "validity_corr_heatmap": corr_heatmap(df),
        "validity_htmt": htmt_chart(res),
        "validity_item_loadings": item_loading_chart(res),
    }
    return _save_charts(charts, out_dir, fmt)

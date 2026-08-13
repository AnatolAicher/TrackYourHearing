"""Within-person effect size of the exposure on each outcome composite.

Model (per composite C, z-scored to SD units):

    C_ij = b0 + b_within * q1_within_ij + b_between * q1_mean_p + u_p (+ slope) + e_ij

q1 is Mundlak-split into its person-mean-centred within part and the person-mean
between part. ``b_within`` is the estimand: with C in SD units and q1 binary, it
is the within-person standardized mean difference between worn and not-worn
moments.

Reported per composite:

* random-slope within slope with a participant-cluster bootstrap 95% CI, the
  reported within-person magnitude;
* fixed-effects within slope (equals the random-intercept estimate), reported as
  context alongside the between-person slope;
* unweighted mean of per-participant worn-vs-not differences.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .composites import COMPOSITES, add_composites

DEFAULT_EXPOSURE = "question1"
TRIVIAL = 0.10  # |b_within| below this SD threshold is treated as practically negligible


@dataclass
class EffectResult:
    composite: str
    n_obs: int
    n_persons: int
    n_informative: int            # participants with within-person exposure variation
    b_within: float               # fixed-effects within slope (== random-intercept estimate)
    ci_boot: tuple[float, float]  # participant-cluster bootstrap 95% CI for b_within
    boot_se: float
    prob_trivial: float           # bootstrap P(|b_within| < TRIVIAL)
    ci_ri_wald: tuple[float, float]    # random-intercept model's own Wald CI (matches ci_boot closely)
    b_within_rs: float            # random-slope within slope (the reported magnitude)
    ci_rs_wald: tuple[float, float]
    rs_ok: bool                   # random-slope model converged
    mean_person_diff: float       # unweighted mean per-participant worn-vs-not diff
    b_between: float              # between-person slope (context; confounded)
    var_intercept: float
    var_resid: float
    boot_draws: np.ndarray | None = None   # finite cluster-bootstrap b_within draws
    ci_rs_boot: tuple[float, float] = (float("nan"), float("nan"))  # random-slope cluster bootstrap CI
    rs_boot_se: float = float("nan")
    prob_trivial_rs: float = float("nan")  # bootstrap P(|b_within_rs| < TRIVIAL)
    rs_boot_draws: np.ndarray | None = None   # finite cluster-bootstrap b_within_rs draws

    @property
    def direction(self) -> str:
        return "lower burden" if self.b_within < 0 else "higher burden"

    @property
    def ci_crosses_zero(self) -> bool:
        return self.ci_boot[0] < 0 < self.ci_boot[1]


def _within_components(df: pd.DataFrame, exposure: str) -> pd.DataFrame:
    """Mundlak split of the exposure into within and between parts."""
    g = df.groupby("user_id")[exposure]
    out = pd.DataFrame({
        "user_id": df["user_id"].to_numpy(),
        "x_between": g.transform("mean").to_numpy(),
    })
    out["x_within"] = df[exposure].to_numpy() - out["x_between"]
    return out


def _fe_within_slope(c: np.ndarray, xw: np.ndarray, person: np.ndarray):
    """Fixed-effects within slope + per-person sufficient stats for bootstrapping.

    Person-mean-centres the composite and regresses on the (already within)
    exposure; equals the within-person association. Returns (slope, Sxc, Sxx)
    where Sxc/Sxx are per-person sums so the cluster bootstrap just re-sums them.
    """
    d = pd.DataFrame({"c": c, "xw": xw, "p": person}).dropna()
    cc = (d["c"] - d.groupby("p")["c"].transform("mean")).to_numpy()
    xx = d["xw"].to_numpy()  # already person-centred (x_within)
    # re-centre xw within person on the non-missing-composite support
    xx = xx - d.groupby("p")["xw"].transform("mean").to_numpy()
    codes, _ = pd.factorize(d["p"].to_numpy())
    P = codes.max() + 1
    Sxc = np.bincount(codes, weights=cc * xx, minlength=P)
    Sxx = np.bincount(codes, weights=xx * xx, minlength=P)
    slope = Sxc.sum() / Sxx.sum() if Sxx.sum() > 0 else float("nan")
    return slope, Sxc, Sxx


def _cluster_bootstrap(Sxc: np.ndarray, Sxx: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    P = len(Sxc)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P, size=(n_boot, P))
    num = Sxc[idx].sum(axis=1)
    den = Sxx[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = num / den
    return b[np.isfinite(b)]


def _fit_one(smf, d, re_formula):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.mixedlm("c ~ x_within + x_between", d, groups=d["user_id"],
                          re_formula=re_formula).fit(reml=True, method="lbfgs")
    bw = float(res.params["x_within"])
    if not np.isfinite(bw):
        raise ValueError("non-finite within slope")
    ci = res.conf_int().loc["x_within"]
    return res, bw, (float(ci[0]), float(ci[1]))


def _fit_mixed(d: pd.DataFrame) -> dict:
    """Fit the random-intercept and random-slope forms of the within-person model."""
    out = {"b_ri": float("nan"), "ci_ri": (float("nan"), float("nan")),
           "b_between": float("nan"), "var_intercept": float("nan"), "var_resid": float("nan"),
           "b_rs": float("nan"), "ci_rs": (float("nan"), float("nan")), "rs_ok": False}
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return out
    try:
        res, bw, ci = _fit_one(smf, d, None)         # random intercept
        out.update(b_ri=bw, ci_ri=ci, b_between=float(res.params["x_between"]),
                   var_intercept=float(res.cov_re.iloc[0, 0]) if res.cov_re.size else float("nan"),
                   var_resid=float(res.scale))
    except Exception:
        pass
    try:
        _, bw, ci = _fit_one(smf, d, "~x_within")    # random slope
        out.update(b_rs=bw, ci_rs=ci, rs_ok=True)
    except Exception:
        pass
    return out


def _rs_cluster_bootstrap(d: pd.DataFrame, n_boot: int, seed: int) -> np.ndarray:
    """Cluster-bootstrap draws of the random-slope within coefficient.

    Resamples participants with replacement and refits the random-slope mixed
    model per draw. Each drawn participant gets a unique group label, so a person
    sampled twice forms two clusters. Non-converging draws are dropped.
    """
    if n_boot <= 0:
        return np.empty(0)
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return np.empty(0)
    cols = ["c", "x_within", "x_between"]
    uniq = pd.unique(d["user_id"].to_numpy())
    groups = [d.loc[d["user_id"] == p, cols].reset_index(drop=True) for p in uniq]
    P = len(uniq)
    rng = np.random.default_rng(seed)
    out: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, P, P)
        parts = []
        for newid, i in enumerate(idx):
            g = groups[i].copy()
            g["gid"] = newid
            parts.append(g)
        dd = pd.concat(parts, ignore_index=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm("c ~ x_within + x_between", dd, groups=dd["gid"],
                                  re_formula="~x_within").fit(reml=True, method="lbfgs")
            bw = float(res.params["x_within"])
            if np.isfinite(bw):
                out.append(bw)
        except Exception:
            continue
    return np.asarray(out)


def effect_size(
    df: pd.DataFrame, composite: str, items: list[str],
    *, exposure: str = DEFAULT_EXPOSURE, n_boot: int = 2000, rs_n_boot: int = 2000, seed: int = 0,
) -> EffectResult:
    wc = add_composites(df, {composite: items})
    comp = wc[composite]
    cz = (comp - comp.mean()) / comp.std()          # SD units
    parts = _within_components(df, exposure)

    d = pd.DataFrame({
        "c": cz.to_numpy(), "x_within": parts["x_within"].to_numpy(),
        "x_between": parts["x_between"].to_numpy(), "user_id": df["user_id"].to_numpy(),
    }).dropna()

    person = d["user_id"].to_numpy()
    b_fe, Sxc, Sxx = _fe_within_slope(d["c"].to_numpy(), d["x_within"].to_numpy(), person)
    boot = _cluster_bootstrap(Sxc, Sxx, n_boot, seed)
    ci_boot = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    prob_trivial = float(np.mean(np.abs(boot) < TRIVIAL))

    # informative clusters + unweighted mean per-participant worn-vs-not difference
    raw = pd.DataFrame({"c": d["c"].to_numpy(), "x": (d["x_within"] + d["x_between"]).to_numpy(),
                        "p": person})

    def _pdiff(g):
        return g.loc[g.x == 1, "c"].mean() - g.loc[g.x == 0, "c"].mean() if g.x.nunique() > 1 else np.nan
    pdiffs = raw.groupby("p").apply(_pdiff, include_groups=False).dropna()
    n_inform = int(len(pdiffs))

    mixed = _fit_mixed(d)

    rs_boot = _rs_cluster_bootstrap(d, rs_n_boot, seed) if mixed["rs_ok"] else np.empty(0)
    if rs_boot.size:
        ci_rs_boot = (float(np.percentile(rs_boot, 2.5)), float(np.percentile(rs_boot, 97.5)))
        rs_boot_se = float(np.std(rs_boot, ddof=1))
        prob_trivial_rs = float(np.mean(np.abs(rs_boot) < TRIVIAL))
    else:
        ci_rs_boot, rs_boot_se = (float("nan"), float("nan")), float("nan")
        prob_trivial_rs = float("nan")

    return EffectResult(
        composite=composite, n_obs=len(d), n_persons=int(d["user_id"].nunique()),
        n_informative=n_inform, b_within=b_fe, ci_boot=ci_boot,
        boot_se=float(np.std(boot, ddof=1)), prob_trivial=prob_trivial,
        ci_ri_wald=mixed["ci_ri"], b_within_rs=mixed["b_rs"], ci_rs_wald=mixed["ci_rs"],
        rs_ok=mixed["rs_ok"], mean_person_diff=float(pdiffs.mean()),
        b_between=mixed["b_between"], var_intercept=mixed["var_intercept"],
        var_resid=mixed["var_resid"], boot_draws=boot,
        ci_rs_boot=ci_rs_boot, rs_boot_se=rs_boot_se, prob_trivial_rs=prob_trivial_rs,
        rs_boot_draws=rs_boot,
    )


def effect_sizes(
    df: pd.DataFrame, composites: dict[str, list[str]] | None = None,
    *, exposure: str = DEFAULT_EXPOSURE, n_boot: int = 2000, rs_n_boot: int = 2000, seed: int = 0,
) -> list[EffectResult]:
    composites = composites or COMPOSITES
    return [effect_size(df, n, it, exposure=exposure, n_boot=n_boot, rs_n_boot=rs_n_boot, seed=seed)
            for n, it in composites.items()]


def report(results: list[EffectResult], *, exposure: str = DEFAULT_EXPOSURE) -> str:
    rule = "=" * 78
    L = [rule, "TYH within-person effect size: exposure -> burden composites", rule,
         f"exposure: {exposure} (wearing a hearing aid now). Composites z-scored (SD units).",
         "b_within = within-person standardized mean difference (worn vs not), random-intercept",
         "mixed model (== fixed-effects within estimator), CI = participant-cluster bootstrap.",
         "The random-slope estimate below is the reported magnitude; direction was tested",
         "first -> Type-M (see caveat).", ""]
    L.append(f"  {'composite':<18}{'b_within':>9}{'boot 95% CI':>20}{'P(|b|<.1)':>11}{'rand-slope':>12}")
    L.append("  " + "-" * 72)
    for r in results:
        ci = f"[{r.ci_boot[0]:+.2f}, {r.ci_boot[1]:+.2f}]"
        star = " *" if r.ci_crosses_zero else "  "
        rs = f"{r.b_within_rs:+.2f}" if r.rs_ok else "  -- "
        L.append(f"  {r.composite:<18}{r.b_within:>+9.3f}{ci:>20}{star}{r.prob_trivial:>9.2f}{rs:>12}")
    L += ["", "per composite:"]
    for r in results:
        L.append(f"  [{r.composite}]  n={r.n_obs} rows, {r.n_persons} participants "
                 f"({r.n_informative} with within-person exposure variation)")
        L.append(f"     within b (random intercept) = {r.b_within:+.3f}   "
                 f"bootstrap 95% CI [{r.ci_boot[0]:+.2f}, {r.ci_boot[1]:+.2f}]  (SE {r.boot_se:.3f})")
        L.append(f"        model Wald CI [{r.ci_ri_wald[0]:+.2f}, {r.ci_ri_wald[1]:+.2f}];  "
                 f"P(|b|<{TRIVIAL:.1f}) = {r.prob_trivial:.2f}")
        rs = (f"{r.b_within_rs:+.3f}  Wald CI [{r.ci_rs_wald[0]:+.2f}, {r.ci_rs_wald[1]:+.2f}]"
              if r.rs_ok else "did not converge")
        L.append(f"     random-slope within b (reported magnitude) = {rs}")
        if not np.isnan(r.ci_rs_boot[0]):
            L.append(f"        random-slope bootstrap 95% CI "
                     f"[{r.ci_rs_boot[0]:+.2f}, {r.ci_rs_boot[1]:+.2f}]  (SE {r.rs_boot_se:.3f}, "
                     f"P(|b|<{TRIVIAL:.1f}) = {r.prob_trivial_rs:.2f})")
        L.append(f"        typical-participant (unweighted mean diff) = {r.mean_person_diff:+.3f};  "
                 f"between b = {r.b_between:+.3f} (context, confounded)")
        L.append(f"     -> wearing the aid: {r.direction}, "
                 f"{'CI crosses 0' if r.ci_crosses_zero else 'CI excludes 0'}")
    L += ["",
          "Type-M caveat: magnitudes are estimated on data already selected for a significant",
          "direction, so point estimates are optimistically biased (true effects plausibly",
          "smaller). Interpret via the bootstrap CIs, not the points. '*' = CI includes zero.",
          "Random-intercept (data-weighted) vs random-slope/typical-participant estimates differ",
          "most for hearing_difficulty -> a few high-volume participants pull the pooled estimate.",
          rule]
    return "\n".join(L)

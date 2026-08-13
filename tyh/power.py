"""Design analysis: minimum detectable effect and prospective power.

* :func:`mdes`           -- minimum detectable effect from an estimator SE.
* :func:`analyze_power`  -- Monte-Carlo power for a within-person design vs the
  number of exposure-varying ("informative") participants.

Simulated datasets mirror the achieved design: each synthetic participant's
prompt count and wear probability are resampled from the observed informative
participants (exposure redrawn until it varies, so every simulated participant
is informative), and outcomes are generated from the REML variance components.
Every dataset is analysed with the same estimator as the effect-size analysis –
the random-slope mixed model with a participant-cluster bootstrap – rejecting
when |b| exceeds 1.96 bootstrap SEs. Replicates whose point fit fails, or where
fewer than 80% of bootstrap refits converge, are excluded and reported as an
invalid share per grid cell.

The simulation refits the mixed model hundreds of thousands of times, so it
runs replicates in parallel (``workers``), checkpoints completed replicates to
disk (an interrupted or crashed run resumes where it left off), and caches
final results keyed by a hash of the settings and a data fingerprint
(``cache``).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats

from .composites import COMPOSITES, add_composites
from .effectsize import (DEFAULT_EXPOSURE, EffectResult, _fit_one,
                         _rs_cluster_bootstrap, _within_components)
from .paths import PROJECT_ROOT
from .viz import _save_charts, enable_large_data

DEFAULT_N_GRID = (20, 34, 50, 75, 100, 150, 200, 300)
DEFAULT_EFFECTS = (0.15, 0.25, 0.40)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results_cache"
MIN_VALID_BOOT = 0.80  # a replicate needs >= this share of converged bootstrap refits
_Z_CRIT = float(stats.norm.ppf(0.975))


def mdes(se: float, *, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable effect at the given power/two-sided alpha for an SE."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * se


def _prep(df: pd.DataFrame, composite: str, items: list[str], exposure: str) -> pd.DataFrame:
    wc = add_composites(df, {composite: items})
    comp = wc[composite]
    cz = (comp - comp.mean()) / comp.std()
    parts = _within_components(df, exposure)
    d = pd.DataFrame({
        "c": cz.to_numpy(), "x_within": parts["x_within"].to_numpy(),
        "x_between": parts["x_between"].to_numpy(), "user_id": df["user_id"].to_numpy(),
    }).dropna()
    d["x"] = d["x_within"] + d["x_between"]
    return d


def _variance_components(d: pd.DataFrame) -> tuple[float, float, float]:
    """Random-slope (intercept var, slope var, residual var) in SD-unit space."""
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.mixedlm("c ~ x_within + x_between", d, groups=d["user_id"],
                          re_formula="~x_within").fit(reml=True, method="lbfgs")
    cov = res.cov_re
    var_int = float(cov.iloc[0, 0])
    var_slope = float(cov.iloc[1, 1]) if cov.shape[0] > 1 else 0.0
    return var_int, var_slope, float(res.scale)


def _design_profiles(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-participant (prompt count, wear probability) of the informative participants."""
    g = d.groupby("user_id")["x"]
    inform = g.nunique()[lambda s: s > 1].index
    gi = d[d["user_id"].isin(inform)].groupby("user_id")["x"]
    return gi.size().to_numpy(dtype=int), gi.mean().to_numpy(dtype=float)


def _sim_replicate(args: tuple) -> list[tuple[int, float, int, int]]:
    """One Monte-Carlo replicate: (n, b_true, valid, reject) per grid cell.

    Draws one pool of max(n_grid) participants; each grid N reuses the first N
    (common random numbers, so the power curve is smooth in N), and the same
    exposure/noise draws serve every true effect. Seeded by (seed, composite
    index, replicate index), so a replicate is reproducible in isolation.
    """
    (seed, ci, rep_idx, n_grid, effects, m_arr, p_arr,
     var_int, var_slope, var_resid, boot) = args
    import statsmodels.formula.api as smf
    rng = np.random.default_rng(np.random.SeedSequence([seed, ci, rep_idx]))
    sig_i, sig_s, sig_e = np.sqrt(var_int), np.sqrt(var_slope), np.sqrt(var_resid)
    n_max = max(n_grid)

    pick = rng.integers(0, len(m_arr), n_max)
    ms, ps = m_arr[pick], p_arr[pick]
    xs: list[np.ndarray] = []
    for m_i, p_i in zip(ms, ps):
        while True:
            x = rng.binomial(1, p_i, int(m_i)).astype(float)
            if 0.0 < x.sum() < m_i:
                break
        xs.append(x)
    u = rng.normal(0.0, sig_i, n_max)
    s = rng.normal(0.0, sig_s, n_max)
    base = [u[i] + s[i] * xs[i] + rng.normal(0.0, sig_e, int(ms[i])) for i in range(n_max)]
    xw = [x - x.mean() for x in xs]
    xb = [np.full(x.size, x.mean()) for x in xs]

    out: list[tuple[int, float, int, int]] = []
    for n in n_grid:
        d0 = pd.DataFrame({
            "user_id": np.repeat(np.arange(n), ms[:n]),
            "x_within": np.concatenate(xw[:n]),
            "x_between": np.concatenate(xb[:n]),
        })
        x_cat = np.concatenate(xs[:n])
        base_cat = np.concatenate(base[:n])
        for b_true in effects:
            d = d0.assign(c=base_cat + b_true * x_cat)
            cell_seed = int(rng.integers(2**31 - 1))
            try:
                _, b, _ = _fit_one(smf, d, "~x_within")
            except Exception:
                out.append((n, b_true, 0, 0))
                continue
            draws = _rs_cluster_bootstrap(d, boot, cell_seed)
            se = float(np.std(draws, ddof=1)) if draws.size > 1 else 0.0
            if draws.size < MIN_VALID_BOOT * boot or se <= 0.0:
                out.append((n, b_true, 0, 0))
                continue
            out.append((n, b_true, 1, int(abs(b / se) > _Z_CRIT)))
    return out


CHECKPOINT_EVERY = 25       # completed replicates between checkpoint writes
MAX_POOL_DEATHS = 5         # give up after this many worker-pool breakages
BATCH_TASKS_PER_WORKER = 10  # replicates per worker before the pool is rebuilt


def _write_checkpoint(path: Path, done: dict[int, list]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"done": {str(k): v for k, v in done.items()}}),
                   encoding="utf-8")
    tmp.replace(path)


def _simulate(n_grid: tuple[int, ...], effects: tuple[float, ...],
              m_arr: np.ndarray, p_arr: np.ndarray,
              var_int: float, var_slope: float, var_resid: float,
              *, reps: int, boot: int, seed: int, ci: int,
              workers: int | None, checkpoint: Path | None,
              progress: Callable[[int, int], None] | None) -> pd.DataFrame:
    """Grid of (n_informative, b_true) -> power + invalid share, over ``reps`` replicates.

    Replicates run in a ProcessPoolExecutor: a killed worker (e.g. by memory
    pressure) raises BrokenProcessPool instead of hanging the run; the pool is
    then rebuilt and the unfinished replicates resubmitted. The pool is also
    rebuilt after every batch of ``workers * BATCH_TASKS_PER_WORKER`` replicates,
    which bounds per-worker memory growth without max_tasks_per_child (whose
    worker-replacement path can deadlock when many workers retire at once).
    Completed tallies are checkpointed to disk so an interrupted run resumes
    where it left off.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    done: dict[int, list] = {}
    if checkpoint and checkpoint.exists():
        blob = json.loads(checkpoint.read_text(encoding="utf-8"))
        done = {int(k): v for k, v in blob["done"].items() if int(k) < reps}

    def task(i: int) -> tuple:
        return (seed, ci, i, n_grid, effects, m_arr, p_arr,
                var_int, var_slope, var_resid, boot)

    pending = [i for i in range(reps) if i not in done]
    workers = workers or max(1, mp.cpu_count() - 2)
    if progress and done:
        progress(len(done), reps)

    def on_complete(i: int, cells: list) -> None:
        done[i] = cells
        if checkpoint and len(done) % CHECKPOINT_EVERY == 0:
            _write_checkpoint(checkpoint, done)
        if progress:
            progress(len(done), reps)

    if workers <= 1 or len(pending) <= 1:
        for i in list(pending):
            on_complete(i, _sim_replicate(task(i)))
        pending = []

    deaths = 0
    while pending:
        batch = pending[: workers * BATCH_TASKS_PER_WORKER]
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(batch)),
                                     mp_context=mp.get_context("spawn")) as ex:
                futs = {ex.submit(_sim_replicate, task(i)): i for i in batch}
                for fut in as_completed(futs):
                    i = futs[fut]
                    on_complete(i, fut.result())
                    pending.remove(i)
        except BrokenProcessPool:
            deaths += 1
            if checkpoint:
                _write_checkpoint(checkpoint, done)
            if deaths >= MAX_POOL_DEATHS:
                raise RuntimeError(
                    f"worker pool died {deaths} times; giving up with "
                    f"{len(done)}/{reps} replicates done (checkpoint kept)")

    if checkpoint:
        _write_checkpoint(checkpoint, done)

    hits: dict[tuple[int, float], int] = {}
    valid: dict[tuple[int, float], int] = {}
    total: dict[tuple[int, float], int] = {}
    for cells in done.values():
        for n, b_true, ok, rej in cells:
            key = (int(n), float(b_true))
            total[key] = total.get(key, 0) + 1
            valid[key] = valid.get(key, 0) + ok
            hits[key] = hits.get(key, 0) + rej
    rows = [{"n_informative": n, "b_true": b,
             "power": hits[n, b] / valid[n, b] if valid[n, b] else float("nan"),
             "invalid_share": 1.0 - valid[n, b] / total[n, b]}
            for n, b in sorted(total)]
    return pd.DataFrame(rows)


@dataclass
class PowerResult:
    composite: str
    n_informative_obs: int
    m_median: int          # median prompt count of the observed informative participants
    p_mean: float          # mean per-person wear probability of the same group
    boot: int              # cluster-bootstrap refits per simulated dataset
    var_int: float
    var_slope: float
    var_resid: float
    grid: pd.DataFrame     # n_informative, b_true, power, invalid_share

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "grid"}
        d["grid"] = self.grid.to_dict(orient="records")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PowerResult":
        d = dict(d)
        d["grid"] = pd.DataFrame(d["grid"])
        return cls(**d)


def _cache_path(cache_dir: Path, settings: dict, fingerprint: dict) -> Path:
    key = json.dumps({"settings": settings, "fingerprint": fingerprint}, sort_keys=True)
    return cache_dir / f"power_sim-{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def analyze_power(
    df: pd.DataFrame, composites: dict[str, list[str]] | None = None,
    *, exposure: str = DEFAULT_EXPOSURE,
    n_grid: tuple[int, ...] = DEFAULT_N_GRID,
    effects: tuple[float, ...] = DEFAULT_EFFECTS,
    reps: int = 500, boot: int = 100, seed: int = 0,
    workers: int | None = None,
    cache: Path | None = DEFAULT_CACHE_DIR,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[PowerResult]:
    """Prospective power per composite; loads from ``cache`` when settings and data match."""
    composites = composites or COMPOSITES
    settings = {"scheme": 2, "exposure": exposure, "n_grid": list(n_grid),
                "effects": [float(b) for b in effects],
                "reps": reps, "boot": boot, "seed": seed}
    preps = {name: _prep(df, name, items, exposure) for name, items in composites.items()}
    profiles = {name: _design_profiles(d) for name, d in preps.items()}
    fingerprint = {name: {"n_obs": int(len(preps[name])),
                          "n_informative": int(len(profiles[name][0]))}
                   for name in preps}

    path = _cache_path(cache, settings, fingerprint) if cache else None
    if path and path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        return [PowerResult.from_dict(r) for r in blob["results"]]

    out: list[PowerResult] = []
    checkpoints: list[Path] = []
    for ci, (name, d) in enumerate(preps.items()):
        var_int, var_slope, var_resid = _variance_components(d)
        m_arr, p_arr = profiles[name]
        cb = (lambda done, total, _name=name: progress(_name, done, total)) if progress else None
        ckpt = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            ckpt = path.parent / f"{path.stem}-{name}.ckpt.json"
            checkpoints.append(ckpt)
        grid = _simulate(tuple(n_grid), tuple(effects), m_arr, p_arr,
                         var_int, var_slope, var_resid, reps=reps, boot=boot,
                         seed=seed, ci=ci, workers=workers, checkpoint=ckpt,
                         progress=cb)
        out.append(PowerResult(
            composite=name, n_informative_obs=len(m_arr),
            m_median=int(np.median(m_arr)), p_mean=float(p_arr.mean()), boot=boot,
            var_int=var_int, var_slope=var_slope, var_resid=var_resid, grid=grid,
        ))

    if path:
        blob = {"settings": settings, "fingerprint": fingerprint,
                "results": [r.to_dict() for r in out]}
        path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
        for ckpt in checkpoints:
            ckpt.unlink(missing_ok=True)
    return out


def power_report(power_results: list[PowerResult],
                 effect_results: list[EffectResult] | None = None,
                 *, power: float = 0.80) -> str:
    rule = "=" * 78
    L = [rule, "TYH design analysis: minimum detectable effect & prospective power", rule]

    if effect_results:
        L += ["MINIMUM DETECTABLE EFFECT (achieved design; 80% power, two-sided .05;",
              " SE = random-slope participant-cluster bootstrap)",
              f"  {'composite':<18}{'rs boot SE':>11}{'MDES':>8}{'observed b':>13}{'  detectable?':>14}"]
        for r in effect_results:
            if not np.isfinite(r.rs_boot_se):
                L.append(f"  {r.composite:<18}{'--':>11}{'--':>8}{r.b_within_rs:>+13.2f}"
                         f"{'(no rs bootstrap)':>14}")
                continue
            md = mdes(r.rs_boot_se, power=power)
            ok = "yes" if abs(r.b_within_rs) >= md else "no (b < MDES)"
            L.append(f"  {r.composite:<18}{r.rs_boot_se:>11.3f}{md:>8.2f}"
                     f"{r.b_within_rs:>+13.2f}{ok:>14}")
        L += ["  -> the design could only resolve effects >= MDES; smaller true effects",
              "     would be expected to yield wide, zero-spanning intervals.", ""]

    L += ["PROSPECTIVE POWER (Monte-Carlo; same estimator as the effect-size analysis:",
          " random-slope REML, reject when |b| > 1.96 participant-cluster-bootstrap SEs)"]
    for pr in power_results:
        L.append(f"  [{pr.composite}]  observed informative n = {pr.n_informative_obs}; "
                 f"per-participant (beeps, P(worn)) resampled from them "
                 f"(median beeps = {pr.m_median}, mean P(worn) = {pr.p_mean:.2f}); "
                 f"B = {pr.boot} bootstrap refits per dataset")
        L.append(f"     variance components (SD units): intercept {pr.var_int:.2f}, "
                 f"slope {pr.var_slope:.2f}, residual {pr.var_resid:.2f}")
        for b_true in sorted(pr.grid["b_true"].unique()):
            g = pr.grid[pr.grid.b_true == b_true].sort_values("n_informative")
            target = g[g.power >= 0.80]["n_informative"]
            need = int(target.min()) if len(target) else None
            cur = float(np.interp(pr.n_informative_obs, g["n_informative"], g["power"]))
            cur_s = f"{cur:.0%}"
            need_s = f"~{need} informative participants" if need else ">300 informative participants"
            L.append(f"     true b = {b_true:+.2f}: power at current n ~ {cur_s}; "
                     f"80% power needs {need_s}")
        worst = float(pr.grid["invalid_share"].max())
        if worst > 0:
            w = pr.grid.loc[pr.grid["invalid_share"].idxmax()]
            L.append(f"     non-convergent replicates excluded: worst cell "
                     f"{worst:.1%} (n = {int(w.n_informative)}, b = {w.b_true:+.2f})")
        else:
            L.append("     all replicates converged")
    L += ["",
          "DESIGN LESSON",
          "  Within-person power is governed by the number of participants who VARY their",
          "  exposure, not by total N or total beeps. Here most participants always or never",
          "  wore the aid, contributing no within-person information. Future EMA studies should",
          "  induce exposure variation by design (e.g. a micro-randomized aid on/off schedule)",
          "  rather than relying on observational variation.", rule]
    return "\n".join(L)


def power_curve_chart(power_results: list[PowerResult]) -> alt.HConcatChart:
    """Power vs number of informative participants, per assumed true effect."""
    ref = alt.Chart(pd.DataFrame({"y": [0.80]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(y="y:Q")
    panels = []
    for pr in power_results:
        g = pr.grid.copy()
        g["true effect (SD)"] = g["b_true"].map(lambda v: f"{v:.2f}")
        line = alt.Chart(g).mark_line(point=True).encode(
            x=alt.X("n_informative:Q", title="informative (exposure-varying) participants"),
            y=alt.Y("power:Q", scale=alt.Scale(domain=[0, 1]), title="power",
                    axis=alt.Axis(format="%")),
            color=alt.Color("true effect (SD):N", title="true effect (SD)"))
        cur = alt.Chart(pd.DataFrame({"n": [pr.n_informative_obs]})).mark_rule(
            strokeDash=[2, 2], color="firebrick").encode(x="n:Q")
        panels.append((ref + cur + line).properties(width=320, height=260, title=pr.composite))
    return alt.hconcat(*panels).properties(
        title=alt.TitleParams(
            text="Prospective power vs number of informative participants",
            subtitle="grey = 80%; red = current sample"))


def power_figures(df: pd.DataFrame, out_dir, *, fmt: str = "html",
                  exposure: str = DEFAULT_EXPOSURE, reps: int = 500, boot: int = 100,
                  seed: int = 0, workers: int | None = None,
                  cache: Path | None = DEFAULT_CACHE_DIR) -> list:
    """Build and save the power-curve figure."""
    res = analyze_power(df, exposure=exposure, reps=reps, boot=boot, seed=seed,
                        workers=workers, cache=cache)
    enable_large_data()
    return _save_charts({"results_power_curve": power_curve_chart(res)}, out_dir, fmt)

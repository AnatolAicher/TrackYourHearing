"""Directionality test for the exposure vs the burden outcomes.

Aggregates the direction (sign) of the within-person association across outcomes.

Design
------
* **Exposure** (default ``question1``) and **outcomes** (default
  ``question2``..``question10`` continuous items) are each person-mean-centred;
  the per-outcome statistic is the within-person correlation r_m.
* **Cluster permutation test.** The centred exposure is shuffled within each
  participant (the whole outcome block held fixed) and every r_m recomputed, B
  times. Directional statistics: the mean Fisher-z and the sign-concordance
  count (how many r_m share a direction).
* **Cluster bootstrap.** Resample participants with replacement; the reported
  Type-S value is the bootstrap probability that the aggregate effect has the
  opposite sign to the observed one.

Cleaning reverse-codes q5/q7 to ``question5_rev``/``question7_rev`` upstream, so
every outcome shares valence (higher = more burden).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clean import OUTPUT_QUESTION_COLS

DEFAULT_EXPOSURE = "question1"
# q2..q10 continuous outcomes (exclude exposure q1 and binary skip-flag q9b);
# cleaned output names, so q5/q7 appear reverse-coded.
DEFAULT_OUTCOMES = [q for q in OUTPUT_QUESTION_COLS if q not in ("question1", "question9b")]


@dataclass
class OutcomeResult:
    outcome: str
    r_within: float          # observed within-person correlation
    n_obs: int               # rows with exposure & outcome both present
    p_one_sided: float       # permutation p in the observed direction
    p_two_sided: float
    boot_prob_positive: float  # cluster-bootstrap P(r > 0)


@dataclass
class DirectionalResult:
    exposure: str
    outcomes: list[str]
    n_participants: int
    n_informative: int       # participants with within-person exposure variation
    n_rows: int
    per_outcome: list[OutcomeResult]
    mean_r: float
    mean_z: float
    direction: str           # 'positive' or 'negative' (of the aggregate)
    p_one_sided: float       # PRIMARY directional p-value (mean Fisher-z)
    p_two_sided: float
    n_positive: int          # sign-concordance count
    k: int
    concordance_p: float     # permutation p for the observed concordance (either direction)
    boot_prob_same: float    # bootstrap P(aggregate same sign as observed)
    type_s: float            # bootstrap P(aggregate opposite sign) -- Type-S error
    boot_se_mean_z: float
    n_perm: int
    n_boot: int

    def summary(self) -> str:
        rule = "=" * 78
        L = [
            rule,
            "TYH directionality test (person-level cluster permutation)",
            rule,
            f"exposure              : {self.exposure}",
            f"outcomes (k={self.k})         : {', '.join(self.outcomes)}",
            f"participants          : {self.n_participants} "
            f"({self.n_informative} with within-person exposure variation)",
            f"rows                  : {self.n_rows}",
            f"permutations / boots  : {self.n_perm} / {self.n_boot}",
            "",
            "per-outcome within-person correlation:",
            f"  {'outcome':<12}{'n':>6}{'r':>9}{'p(1-sided)':>12}{'p(2-sided)':>12}"
            f"{'boot P(r>0)':>13}",
            "  " + "-" * 64,
        ]
        for o in self.per_outcome:
            L.append(
                f"  {o.outcome:<12}{o.n_obs:>6}{o.r_within:>9.3f}"
                f"{o.p_one_sided:>12.3f}{o.p_two_sided:>12.3f}{o.boot_prob_positive:>13.2f}"
            )
        L += [
            "",
            f"sign concordance      : {self.n_positive}/{self.k} correlations positive "
            f"(permutation p = {self.concordance_p:.4f})",
            f"aggregate effect      : mean r = {self.mean_r:+.3f}, "
            f"mean Fisher-z = {self.mean_z:+.3f}  ({self.direction})",
            "",
            f">>> DIRECTIONAL p-value : {self.p_one_sided:.4f}  (one-sided, mean Fisher-z, "
            f"in the observed direction)",
            f"    two-sided p        : {self.p_two_sided:.4f}",
            "",
            "Type-S framing (cluster bootstrap):",
            f"  P(aggregate is {self.direction:<8}) = {self.boot_prob_same:.3f}",
            f"  Type-S error  P(opposite sign) = {self.type_s:.3f}   "
            f"(chance the direction is wrong)",
            f"  bootstrap SE of mean Fisher-z  = {self.boot_se_mean_z:.3f}",
            "",
            self._interpretation(),
            rule,
        ]
        return "\n".join(L)

    def _interpretation(self) -> str:
        conf = 1 - self.type_s
        return (
            f"Read: {self.n_positive}/{self.k} outcomes point {self.direction}; the aggregate "
            f"direction is ~{conf:.0%} stable under resampling.\n"
            f"      The permutation p reflects directional consistency, not effect size; "
            f"a non-trivial p with\n      low Type-S is the underpowered-but-directionally-"
            f"informative case (vs. a strong magnitude claim)."
        )


def _fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def directional_test(
    df: pd.DataFrame,
    *,
    exposure: str = DEFAULT_EXPOSURE,
    outcomes: list[str] | None = None,
    n_perm: int = 10_000,
    n_boot: int = 2_000,
    seed: int = 0,
    chunk: int = 1_000,
) -> DirectionalResult:
    """Person-level cluster permutation test for a consistent direction of effect.

    Parameters
    ----------
    df:
        Cleaned EMA table (must contain ``user_id``, the exposure and outcomes).
        Outcomes are assumed already valence-aligned (cleaning reverse-codes
        q5/q7), so higher = more burden throughout.
    exposure, outcomes:
        Column names. Defaults: q1 exposure, q2..q10 continuous outcomes
        (with q5/q7 as their reverse-coded versions).
    n_perm, n_boot:
        Permutation and cluster-bootstrap replicate counts.
    seed:
        RNG seed for reproducibility.
    """
    outcomes = list(outcomes) if outcomes is not None else list(DEFAULT_OUTCOMES)
    rng = np.random.default_rng(seed)

    needed = ["user_id", exposure, *outcomes]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"df is missing columns: {missing}")

    # --- restrict to rows with the exposure present ("active" rows) ----------
    d = df[needed].copy()
    d = d[d[exposure].notna()]
    if d.empty:
        raise ValueError("No rows with the exposure present.")

    # person-centre the exposure (mean over the participant's active rows)
    xc = (d[exposure] - d.groupby("user_id")[exposure].transform("mean")).to_numpy(float)
    person = d["user_id"].to_numpy()

    # person-centre each outcome on its support (exposure & outcome both present);
    # zero outside support so masked sums work
    A = len(d)
    k = len(outcomes)
    Y = np.zeros((A, k))
    SUPP = np.zeros((A, k))
    n_obs = np.zeros(k, dtype=int)
    for j, m in enumerate(outcomes):
        col = d[m]
        mask = col.notna().to_numpy()
        yc = (col - d.groupby("user_id")[m].transform("mean")).to_numpy(float)
        Y[mask, j] = yc[mask]
        SUPP[mask, j] = 1.0
        n_obs[j] = int(mask.sum())

    # sort rows by participant so each person's rows are contiguous (for the
    # within-cluster shuffle and per-person reductions)
    order = np.argsort(person, kind="mergesort")
    xc_s = xc[order]
    Y_s = Y[order]
    SUPP_s = SUPP[order]
    _, block_id = np.unique(person[order], return_inverse=True)
    P = block_id.max() + 1
    block_f = block_id.astype(float)
    deny = (Y_s * Y_s).sum(axis=0)  # per-outcome Σ yc² (fixed across permutations)

    def _r_from_xc(xc_mat: np.ndarray) -> np.ndarray:
        """r per outcome for one or many centred-exposure vectors (rows = reps)."""
        xc_mat = np.atleast_2d(xc_mat)
        num = xc_mat @ Y_s                      # reps × k
        denx = (xc_mat * xc_mat) @ SUPP_s       # reps × k
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num / np.sqrt(denx * deny[None, :])
        r[~np.isfinite(r)] = 0.0
        return r

    # --- observed ------------------------------------------------------------
    r_obs = _r_from_xc(xc_s)[0]
    mean_z_obs = _fisher_z(r_obs).mean()
    mean_r_obs = float(r_obs.mean())
    n_pos_obs = int((r_obs > 0).sum())
    concord_obs = max(n_pos_obs, k - n_pos_obs)
    direction = "positive" if mean_z_obs >= 0 else "negative"

    informative = int(d.groupby("user_id")[exposure].nunique().gt(1).sum())

    # --- permutation null (within-person shuffle of the exposure) ------------
    mz_perm = np.empty(n_perm)
    npos_perm = np.empty(n_perm, dtype=int)
    concord_perm = np.empty(n_perm, dtype=int)
    r_perm_oneside = np.zeros(k)   # counts for per-outcome one-sided p
    r_perm_twoside = np.zeros(k)
    for start in range(0, n_perm, chunk):
        nb = min(chunk, n_perm - start)
        keys = block_f[None, :] + rng.random((nb, A))
        perm = np.argsort(keys, axis=1)         # within-block shuffle
        r = _r_from_xc(xc_s[perm])              # nb × k
        mz_perm[start:start + nb] = _fisher_z(r).mean(axis=1)
        pos = (r > 0).sum(axis=1)
        npos_perm[start:start + nb] = pos
        concord_perm[start:start + nb] = np.maximum(pos, k - pos)
        # per-outcome tallies vs observed
        if direction == "positive":
            r_perm_oneside += (r >= r_obs[None, :]).sum(axis=0)
        else:
            r_perm_oneside += (r <= r_obs[None, :]).sum(axis=0)
        r_perm_twoside += (np.abs(r) >= np.abs(r_obs)[None, :]).sum(axis=0)

    def _p(count: float) -> float:
        return (1.0 + count) / (n_perm + 1.0)

    if direction == "positive":
        agg_one = _p((mz_perm >= mean_z_obs).sum())
    else:
        agg_one = _p((mz_perm <= mean_z_obs).sum())
    agg_two = _p((np.abs(mz_perm) >= abs(mean_z_obs)).sum())
    concord_p = _p((concord_perm >= concord_obs).sum())

    # --- cluster bootstrap (Type-S) via per-person sufficient statistics -----
    XY = xc_s[:, None] * Y_s
    XX = (xc_s * xc_s)[:, None] * SUPP_s
    YY = Y_s * Y_s
    starts = np.searchsorted(block_id, np.arange(P))
    Sxy = np.add.reduceat(XY, starts, axis=0)   # P × k
    Sxx = np.add.reduceat(XX, starts, axis=0)
    Syy = np.add.reduceat(YY, starts, axis=0)

    idx = rng.integers(0, P, size=(n_boot, P))
    txy = Sxy[idx].sum(axis=1)
    txx = Sxx[idx].sum(axis=1)
    tyy = Syy[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_boot = txy / np.sqrt(txx * tyy)
    r_boot[~np.isfinite(r_boot)] = 0.0
    mz_boot = _fisher_z(r_boot).mean(axis=1)
    boot_pos = float(np.mean(r_boot > 0, axis=0).mean())  # not used directly
    if direction == "positive":
        boot_same = float(np.mean(mz_boot > 0))
    else:
        boot_same = float(np.mean(mz_boot < 0))
    type_s = 1.0 - boot_same
    boot_se = float(np.std(mz_boot, ddof=1))
    boot_prob_pos_per = np.mean(r_boot > 0, axis=0)

    per_outcome = [
        OutcomeResult(
            outcome=outcomes[j],
            r_within=float(r_obs[j]),
            n_obs=int(n_obs[j]),
            p_one_sided=_p(r_perm_oneside[j]),
            p_two_sided=_p(r_perm_twoside[j]),
            boot_prob_positive=float(boot_prob_pos_per[j]),
        )
        for j in range(k)
    ]

    return DirectionalResult(
        exposure=exposure,
        outcomes=outcomes,
        n_participants=int(P),
        n_informative=informative,
        n_rows=A,
        per_outcome=per_outcome,
        mean_r=mean_r_obs,
        mean_z=float(mean_z_obs),
        direction=direction,
        p_one_sided=agg_one,
        p_two_sided=agg_two,
        n_positive=n_pos_obs,
        k=k,
        concordance_p=concord_p,
        boot_prob_same=boot_same,
        type_s=type_s,
        boot_se_mean_z=boot_se,
        n_perm=n_perm,
        n_boot=n_boot,
    )

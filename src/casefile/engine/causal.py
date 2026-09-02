"""
Estimating whether reversing a candidate cause would have moved the metric.

The audit's warning was blunt: synthetic difference-in-differences with a single treated
unit has no valid standard error, and eight donors cannot reach a placebo p of 0.05. So:

  * donors are region x category cells, which gives 24 units, not 4;
  * weights come from non-negative least squares on the pre-period, constrained to the
    simplex, which is plain synthetic control rather than an optimisation we cannot defend;
  * inference is by IN-SPACE PLACEBO. Every donor is assigned the treatment in turn, the
    effect is re-estimated, and the treated unit's effect is ranked against that null
    distribution. With 24 donors the smallest attainable p is 1/25 = 0.04, which is exactly
    what we report, floor and all. Nothing here pretends to more precision than the design
    supports.
  * a pre-trend placebo runs on a fake intervention date before the real one. If that comes
    back significant, parallel trends is not credible and the estimate is DEMOTED to R2 with
    the failure shown in the case file rather than dropped.

Rungs are assigned here and nowhere else.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from datetime import date, timedelta
from scipy.optimize import nnls


@dataclass
class CausalEstimate:
    hypothesis_id: str
    method: str
    effect: float
    effect_pct: float
    ci_low: float
    ci_high: float
    placebo_p: float
    placebo_p_floor: float
    pretrend_p: float
    n_donors: int
    rung: str
    refutations: dict
    note: str
    derived_from: list = field(default_factory=list)


def _sc_weights(Y_pre_treated: np.ndarray, Y_pre_donors: np.ndarray) -> np.ndarray:
    """Simplex-constrained synthetic control weights via NNLS + renormalisation."""
    A = np.vstack([Y_pre_donors, np.ones((1, Y_pre_donors.shape[1])) * 1e3])
    b = np.concatenate([Y_pre_treated, [1e3]])
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else np.full(Y_pre_donors.shape[1], 1.0 / Y_pre_donors.shape[1])


def synthetic_control(panel: pd.DataFrame, unit_col: str, time_col: str, value_col: str,
                      treated_unit: str, t0: date, t1: date,
                      pre_days: int = 56, exclude: set | None = None,
                      pre_end: date | None = None):
    """Returns (effect, weights, donors, gaps, pre_mspe, post_mspe).

    pre_end: last day usable for fitting. Defaults to t0, but when the change-point puts
    the onset before the analysis window it must be the onset, or the fit is contaminated."""
    p = panel.copy(); p[time_col] = pd.to_datetime(p[time_col]).dt.date
    fit_end = pre_end or t0
    pre_lo = fit_end - timedelta(days=pre_days)
    wide = p.pivot_table(index=time_col, columns=unit_col, values=value_col, aggfunc="mean")
    wide = wide.sort_index()
    span = wide[(wide.index >= pre_lo) & (wide.index <= t1)]
    cover = span.notna().mean()
    keep = cover[cover >= 0.8].index                 # drop units without real history
    wide = wide[keep].ffill().bfill()
    pre = wide[(wide.index >= pre_lo) & (wide.index < fit_end)]
    post = wide[(wide.index >= t0) & (wide.index <= t1)]
    if treated_unit not in wide.columns or pre.empty or post.empty:
        return float("nan"), np.array([]), [], np.array([]), np.nan, np.nan
    ex = (exclude or set()) | {treated_unit}
    donors = [c for c in wide.columns if c not in ex]
    if not donors:
        return float("nan"), np.array([]), [], np.array([]), np.nan, np.nan
    w = _sc_weights(pre[treated_unit].to_numpy(), pre[donors].to_numpy())
    synth_pre  = pre[donors].to_numpy() @ w
    synth_post = post[donors].to_numpy() @ w
    gaps_pre  = pre[treated_unit].to_numpy() - synth_pre
    gaps_post = post[treated_unit].to_numpy() - synth_post
    return (float(np.mean(gaps_post)), w, donors, gaps_post,
            float(np.mean(gaps_pre ** 2)), float(np.mean(gaps_post ** 2)))


def estimate(panel: pd.DataFrame, hypothesis_id: str, treated_unit: str,
             t0: date, t1: date, unit_col="unit", time_col="d", value_col="v",
             control_available: bool = True, treated_pool: set | None = None,
             onset: date | None = None, pretrend_buffer_days: int = 10) -> CausalEstimate:
    if not control_available:
        return CausalEstimate(hypothesis_id, "none", float("nan"), float("nan"),
                              float("nan"), float("nan"), 1.0, 1.0, 1.0, 0, "R2",
                              {"reason": "no untreated population exists for this change"},
                              "The change rolled out to every unit at once, so no control "
                              "group exists. An effect cannot be identified from this data; "
                              "capped at R2 (temporal and mechanism only).")

    treated_pool = treated_pool or set()
    fit_end = (onset - timedelta(days=2)) if onset and onset < t0 else t0
    eff, w, donors, gaps, pre_mspe, post_mspe = synthetic_control(
        panel, unit_col, time_col, value_col, treated_unit, t0, t1,
        exclude=treated_pool, pre_end=fit_end)
    if not np.isfinite(eff) or len(donors) == 0:
        return CausalEstimate(hypothesis_id, "synthetic_control", float("nan"), float("nan"),
                              float("nan"), float("nan"), 1.0, 1.0, 1.0, 0, "R1",
                              {"reason": "estimation failed"}, "insufficient donor pool")

    # Abadie in-space placebo on the MSPE RATIO (post fit / pre fit). Using the raw effect
    # would let a small noisy donor cell dominate the null; the ratio normalises for how
    # well each unit could be fitted in the first place.
    treated_ratio = post_mspe / pre_mspe if pre_mspe and pre_mspe > 0 else np.inf
    ratios, effs = [], []
    for d in donors:
        e, _, _, _, prem, postm = synthetic_control(panel, unit_col, time_col, value_col,
                                                    d, t0, t1, exclude=treated_pool,
                                                    pre_end=fit_end)
        if np.isfinite(e) and prem and prem > 0:
            ratios.append(postm / prem); effs.append(e)
    ratios = np.array(ratios); placebo = np.array(effs)
    n = len(ratios)
    p_floor = 1.0 / (n + 1)
    p = float((1 + np.sum(ratios >= treated_ratio)) / (n + 1)) if n else 1.0

    # pre-trend placebo, offset by a buffer so it cannot overlap the intervention's ramp-in
    fake_t1 = (onset or t0) - timedelta(days=pretrend_buffer_days)
    fake_t0 = fake_t1 - timedelta(days=21)
    e_pre, _, _, _, prem2, postm2 = synthetic_control(
        panel, unit_col, time_col, value_col, treated_unit, fake_t0,
        fake_t1, exclude=treated_pool, pre_end=fake_t0)
    pre_ratio = postm2 / prem2 if prem2 and prem2 > 0 else np.inf
    pre_p = float((1 + np.sum(ratios >= pre_ratio)) / (n + 1)) if n else 1.0

    lo, hi = (np.percentile(placebo, [2.5, 97.5]) if n >= 10 else (np.nan, np.nan))
    base = abs(np.nanmean(panel[panel[unit_col] == treated_unit][value_col])) or 1.0

    refut = {"in_space_placebo_p": round(p, 4),
             "placebo_p_floor": round(p_floor, 4),
             "pretrend_placebo_p": round(pre_p, 4),
             "pretrend_window": [str(fake_t0), str(fake_t1)],
             "fitting_period_ends": str(fit_end),
             "pretrend_passes": bool(pre_p > 0.10),
             "n_donors": n,
             "inference": "Abadie in-space placebo on the MSPE ratio",
             "treated_mspe_ratio": round(float(treated_ratio), 2) if np.isfinite(treated_ratio) else None,
             "donor_weight_concentration": round(float(np.max(w)) if len(w) else 1.0, 3)}

    if p <= 0.10 and pre_p > 0.10:
        rung, note = "R3", (f"Synthetic control on {n} donor units. Placebo p={p:.3f} "
                            f"(floor {p_floor:.3f}); pre-trend placebo p={pre_p:.2f} passes.")
    elif pre_p <= 0.10:
        rung, note = "R2", (f"DEMOTED: pre-trend placebo p={pre_p:.3f} indicates the treated "
                            f"unit was already diverging before the change. Parallel trends "
                            f"is not credible, so no causal claim is made.")
    else:
        rung, note = "R2", (f"Effect estimated but placebo p={p:.3f} does not clear 0.10 on "
                            f"{n} donors. Consistent with, but not demonstrating, a causal role.")

    return CausalEstimate(hypothesis_id, "synthetic_control", eff, eff / base,
                          float(lo), float(hi), p, p_floor, pre_p, n, rung, refut, note)

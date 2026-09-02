"""Three gates, a multiplicity correction, and a priority score.

The FDR procedure runs on the full family; materiality enters as p-value weights rather
than filtering the family first. A change-point then dates the regime break.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, asdict, field
from datetime import date
from .baseline import Baseline


@dataclass
class Monitor:
    monitor_id: str
    kpi: str
    slice: dict
    baseline_method: str
    n_calib: int
    actual: float
    expected: float
    delta_abs: float
    delta_pct: float
    p_raw: float
    p_conformal_rank: float
    p_conformal_floor: float
    effect_sd: float
    weight: float
    p_weighted: float
    q_value: float = np.nan
    gate_statistical: bool = False
    gate_materiality: bool = False
    gate_persistence: bool = False
    persistence_days: int = 0
    material_value: float = 0.0
    priority_score: float = 0.0
    fired: bool = False
    note: str = ""


def _bh(p: np.ndarray, q: float, method: str = "benjamini_yekutieli"):
    m = len(p); order = np.argsort(p); ranked = p[order]
    c = np.sum(1.0 / np.arange(1, m + 1)) if method == "benjamini_yekutieli" else 1.0
    thresh = (np.arange(1, m + 1) / m) * q / c
    below = ranked <= thresh
    k = np.max(np.where(below)[0]) + 1 if below.any() else 0
    rej = np.zeros(m, bool)
    if k: rej[order[:k]] = True
    qv = np.minimum.accumulate((ranked * m * c / np.arange(1, m + 1))[::-1])[::-1]
    out_q = np.empty(m); out_q[order] = np.clip(qv, 0, 1)
    return rej, out_q


def change_point(resid_z: np.ndarray, dates, n_boot: int = 500, seed: int = 7):
    """Binary segmentation on the residual series, with an onset interval obtained by
        resampling residuals WITHIN each segment while holding the step fixed.
    """
    r = np.nan_to_num(np.asarray(resid_z, dtype=float))
    n = len(r)
    if n < 20: return None, None, 0.0

    def best_split(a):
        m = len(a); k = np.arange(4, m - 4)
        if len(k) == 0: return None, 0.0
        cs = np.cumsum(a); tot = cs[-1]
        left = cs[k - 1] / k; right = (tot - cs[k - 1]) / (m - k)
        stat = np.abs(left - right) * np.sqrt(k * (m - k) / m)
        i = int(np.argmax(stat)); return int(k[i]), float(stat[i])

    idx, stat = best_split(r)
    if idx is None: return None, None, 0.0

    m1, m2 = r[:idx].mean(), r[idx:].mean()
    e1, e2 = r[:idx] - m1, r[idx:] - m2          # centred residuals per segment
    rng = np.random.default_rng(seed); locs = []
    for _ in range(n_boot):
        a = np.concatenate([m1 + rng.choice(e1, idx, replace=True),
                            m2 + rng.choice(e2, n - idx, replace=True)])
        j, _ = best_split(a)
        if j is not None: locs.append(j)
    lo, hi = (np.percentile(locs, [5, 95]) if locs else (idx, idx))
    d = pd.DatetimeIndex(dates)
    return d[idx].date(), (d[int(max(0, lo))].date(), d[int(min(n - 1, hi))].date()), stat


def evaluate(monitors: list[tuple[str, str, dict, Baseline, slice, float]],
             contract: dict) -> tuple[list[Monitor], dict]:
    """monitors: (monitor_id, kpi, slice, baseline, test_index, rupee_per_unit_deviation)"""
    rows: list[Monitor] = []
    for mid, kpi, slc, bl, test_idx, rupee in monitors:
        kcfg = contract["kpis"][kpi]; th = kcfg["thresholds"]
        if bl.method == "INSUFFICIENT_HISTORY":
            rows.append(Monitor(mid, kpi, slc, bl.method, 0, np.nan, np.nan, np.nan, np.nan,
                                1.0, 1.0, 1.0, 0.0, 1.0, 1.0, note=bl.note)); continue
        idx = np.arange(len(bl.actual))[test_idx]
        act = float(np.nanmean(bl.actual[idx])); exp = float(np.nanmean(bl.expected[idx]))
        p_pt = bl.rank_p(idx)
        p_win, eff_sd = bl.window_rank_p(idx)
        d_abs = act - exp; d_pct = d_abs / exp if exp else np.nan
        mat = th["materiality"]
        mat_val = abs(d_abs) * rupee
        gate_mat = (mat_val >= mat.get("abs_currency", 0)
                    if "abs_currency" in mat else abs(d_abs) >= mat.get("abs_pp", 0)) \
                   or abs(d_pct) >= mat["pct_of_baseline"]
        persist = int(np.sum(p_pt < 0.05))
        gate_per = persist >= int(th["persistence_days"])
        w = max(0.05, min(20.0, mat_val / max(mat.get("abs_currency", 1) or 1, 1))) \
            if "abs_currency" in mat else max(0.05, min(20.0, abs(d_abs) / max(mat.get("abs_pp", 1), 1e-9)))
        rows.append(Monitor(mid, kpi, slc, bl.method, bl.n_calib, act, exp, d_abs, d_pct,
                            p_win, bl.last_p_rank, bl.last_p_floor, bl.last_effect_sd, w, p_win, gate_materiality=gate_mat,
                            gate_persistence=gate_per, persistence_days=persist,
                            material_value=mat_val, note=bl.note + f" | window effect {eff_sd:+.2f} sd"))

    live = [r for r in rows if np.isfinite(r.p_raw)]
    if live:
        w = np.array([r.weight for r in live]); w = w / w.mean()      # GRW weights, mean 1
        pw = np.clip(np.array([r.p_raw for r in live]) / w, 0, 1)
        q = float(contract["kpis"]["net_revenue"]["thresholds"]["statistical"]["fdr_q"])
        rej, qv = _bh(pw, q, contract["kpis"]["net_revenue"]["thresholds"]["statistical"]["fdr_method"])
        for r, a, b, c in zip(live, pw, rej, qv):
            r.p_weighted = float(a); r.gate_statistical = bool(b); r.q_value = float(c)

    for r in rows:
        r.fired = bool(r.gate_statistical and r.gate_materiality and r.gate_persistence)
        r.priority_score = float(r.material_value * (1 - min(r.q_value, 1) if np.isfinite(r.q_value) else 0)
                           * (1 + 0.1 * r.persistence_days)) if r.fired else 0.0
    rows.sort(key=lambda x: -x.priority_score)
    fam = {"family_size": len(rows), "evaluated": len(live),
           "fired": sum(1 for r in rows if r.fired),
           "suppressed_by_fdr": sum(1 for r in live if not r.gate_statistical),
           "suppressed_by_materiality": sum(1 for r in live if r.gate_statistical and not r.gate_materiality),
           "suppressed_by_persistence": sum(1 for r in live if r.gate_statistical and r.gate_materiality and not r.gate_persistence),
           "fdr_method": contract["kpis"]["net_revenue"]["thresholds"]["statistical"]["fdr_method"],
           "fdr_q": contract["kpis"]["net_revenue"]["thresholds"]["statistical"]["fdr_q"]}
    return rows, fam

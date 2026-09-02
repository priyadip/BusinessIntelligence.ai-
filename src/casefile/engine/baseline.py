"""Expectation, not history: three baseline rungs selected by available history.

MSTL is fitted on training data only and projected forward; calibration uses
rolling-origin forecast errors at the same horizon, not in-sample residuals.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field

RATE_KPIS = {"conversion_rate", "fulfilment_ontime_pct", "gross_margin_pct"}


def _fwd(y: np.ndarray, kpi: str) -> np.ndarray:
    if kpi in RATE_KPIS:
        p = np.clip(y, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
    return np.log(np.clip(y, 1e-9, None))


def _inv(z: np.ndarray, kpi: str) -> np.ndarray:
    if kpi in RATE_KPIS: return 1.0 / (1.0 + np.exp(-z))
    return np.exp(z)


@dataclass
class Baseline:
    kpi: str
    method: str                       # MSTL | POOLED_COHORT | INSUFFICIENT_HISTORY
    dates: pd.DatetimeIndex
    actual: np.ndarray
    expected: np.ndarray
    resid_z: np.ndarray               # on the transformed scale
    calib_scores: np.ndarray
    n_calib: int
    widen: float = 1.0
    note: str = ""
    causal_claims_allowed: bool = True
    deseasonalised: np.ndarray = field(default_factory=lambda: np.array([]))
    oos_window_stats: np.ndarray = field(default_factory=lambda: np.array([]))
    last_p_rank: float = 1.0
    last_p_param: float = 1.0
    last_p_floor: float = 1.0
    last_effect_sd: float = 0.0

    def rank_p(self, idx) -> np.ndarray:
        """Split-conformal rank p-value per point. Exactly valid under exchangeability."""
        s = np.abs(self.resid_z[idx]) / self.widen
        c = self.calib_scores
        return (1.0 + np.sum(c[None, :] >= s[:, None], axis=1)) / (len(c) + 1.0)

    def window_rank_p(self, idx) -> tuple[float, float]:
        """Conformal p-value for the WINDOW, not for a point."""
        idx = np.asarray(idx)
        if idx.dtype == bool: idx = np.where(idx)[0]
        w = len(idx)
        if w == 0 or self.n_calib == 0: return 1.0, 0.0
        exp_t = _fwd(self.expected[idx], self.kpi)
        act_t = _fwd(self.actual[idx], self.kpi)
        stat = float(np.nanmean(act_t - exp_t))
        sw = self.oos_window_stats
        if sw.size < 8: return 1.0, stat
        sw = sw / self.widen
        if stat <= 0:
            p_rank = (1.0 + np.sum(sw <= stat)) / (len(sw) + 1.0)
        else:
            p_rank = (1.0 + np.sum(sw >= stat)) / (len(sw) + 1.0)
        self.last_p_rank = float(np.clip(p_rank, 1e-9, 1.0))
        self.last_p_floor = 1.0 / (len(sw) + 1.0)
        mu, sd = float(np.mean(sw)), float(np.std(sw, ddof=1))
        if sd <= 0: return self.last_p_rank, stat
        from scipy.stats import t as _t
        dfree = max(3, len(sw) - 1)
        zt = (stat - mu) / sd
        p_par = float(_t.cdf(zt, dfree)) if stat <= mu else float(_t.sf(zt, dfree))
        self.last_p_param = max(p_par, 1e-12)
        self.last_effect_sd = zt
        return self.last_p_param, stat

    def interval(self, idx, alpha=0.01):
        q = np.quantile(self.calib_scores, min(1.0, 1 - alpha)) * self.widen
        exp_t = _fwd(self.expected[idx], self.kpi)
        return _inv(exp_t - q, self.kpi), _inv(exp_t + q, self.kpi)


def _mstl_expected(z: np.ndarray, periods=(7, 365)) -> np.ndarray:
    from statsmodels.tsa.seasonal import MSTL
    per = tuple(p for p in periods if len(z) >= 2 * p)
    if not per: raise ValueError("no usable period")
    r = MSTL(pd.Series(z), periods=per if len(per) > 1 else per[0]).fit()
    seas = r.seasonal.sum(axis=1).to_numpy() if hasattr(r.seasonal, "sum") and r.seasonal.ndim > 1 \
           else np.asarray(r.seasonal)
    return np.asarray(r.trend) + seas


def _project(train_z: np.ndarray, n_ahead: int, periods=(7, 365)):
    periods = tuple(p for p in periods if p)
    """Fit decomposition on the TRAIN block only, then project forward.

    Fitting on the full series would let the trend component absorb the very anomaly we
    are trying to detect, and the expectation would track the actual. The baseline must
    never see the test window."""
    from statsmodels.tsa.seasonal import MSTL
    per = tuple(p for p in periods if len(train_z) >= 2 * p)
    if not per:
        raise ValueError("insufficient history for any seasonal period")
    stl_kw = {"trend": 181, "robust": True, "seasonal_deg": 0, "trend_deg": 1}
    try:
        res = MSTL(pd.Series(train_z), periods=per if len(per) > 1 else per[0],
                   stl_kwargs=stl_kw).fit()
    except Exception:
        res = MSTL(pd.Series(train_z), periods=per if len(per) > 1 else per[0]).fit()
    seas = res.seasonal.to_numpy()
    if seas.ndim == 1: seas = seas[:, None]
    trend = np.asarray(res.trend)

    # level from non-anomalous days only: a prior incident in the trend tail would
    # depress the projection and make the next window look healthy.
    resid_in = train_z - fitted_pre if (fitted_pre := trend + seas.sum(axis=1)) is not None else None
    sc = np.median(np.abs(resid_in - np.median(resid_in))) * 1.4826 + 1e-9
    normal = np.abs(resid_in - np.median(resid_in)) < 2.5 * sc
    K = min(60, len(trend))
    tail_idx = np.arange(len(trend) - K, len(trend))
    good = tail_idx[normal[tail_idx]]
    lvl = float(np.median(trend[good])) if len(good) >= 10 else float(np.median(trend[-14:]))
    level_note = (f"level from {len(good)}/{K} non-anomalous days"
                  if len(good) >= 10 else "level from last 14 days (too few clean days)")
    k = min(90, len(trend))
    if k >= 20:
        x = np.arange(k); yv = trend[-k:]
        sl = np.median([(yv[j] - yv[i]) / (j - i)
                        for i in range(0, k, 5) for j in range(i + 5, k, 5)]) if k > 10 else 0.0
    else:
        sl = 0.0
    drift = 0.5 * float(np.nan_to_num(sl))          # damped: extrapolating slope is risky

    seas_sum = seas.sum(axis=1)
    fitted = trend + seas_sum

    def project_from(origin: int, horizon: int) -> np.ndarray:
        """Expectation for horizon days after `origin`, using only information up to it."""
        lv = float(np.median(trend[max(0, origin - 14):origin]))
        out = np.zeros(horizon)
        for i in range(horizon):
            sv = 0.0
            for c, pr in enumerate(per):
                sv += seas[(origin - pr + i) % len(seas), c]
            out[i] = lv + drift * (i + 1) + sv
        return out

    fut = np.zeros(n_ahead)
    for i in range(n_ahead):
        sv = 0.0
        for c, pr in enumerate(per):
            sv += seas[len(seas) - pr + (i % pr), c]      # same phase, last full cycle
        fut[i] = lvl + drift * (i + 1) + sv
    def seasonal_at(i: int) -> float:
        if i < len(seas_sum): return float(seas_sum[i])
        return float(sum(seas[len(seas) - pr + ((i - len(seas)) % pr), c]
                         for c, pr in enumerate(per)))
    return (fitted, fut, f"MSTL periods={per}, {level_note}, damped drift {drift:+.2e}/day",
            project_from, seasonal_at)


def fit(kpi: str, s: pd.Series, contract: dict, test_start, cohort: pd.DataFrame | None = None,
        calib_frac: float = 0.4) -> Baseline:
    """s: full series (pre-period + test window). The baseline is fit ONLY on data strictly
    before test_start, then projected across the window under test."""
    s = s.sort_index()
    ts = pd.Timestamp(test_start)
    y = s.to_numpy(dtype=float); n = len(y)
    kcfg = contract["kpis"][kpi]; min_h = int(kcfg["min_history_days"])
    sp = contract["decision_policy"]["sparse_history"]
    z = _fwd(y, kpi)
    is_test = s.index >= ts
    n_tr = int((~is_test).sum()); n_te = int(is_test.sum())

    if n_tr >= max(2 * 7, min_h):
        try:
            seas_cfg = kcfg.get("seasonality", {}) or {}
            periods = tuple(p for p, on in ((7, seas_cfg.get("weekly", True)),
                                            (365, seas_cfg.get("yearly", True))) if on)
            fitted, fut, note, projector, seasonal_at = _project(z[:n_tr], n_te, periods or (7,))
            note = f"{note}; contract declares weekly={seas_cfg.get('weekly')} yearly={seas_cfg.get('yearly')}"
            exp_t = np.concatenate([fitted, fut])
            method, widen, allow = "MSTL_PROJECTED", 1.0, True
            note = f"{note}; trained on {n_tr} days, projected {n_te}"
        except Exception as e:
            base = float(np.median(z[max(0, n_tr - 28):n_tr]))
            exp_t = np.concatenate([pd.Series(z[:n_tr]).rolling(28, min_periods=7, center=True)
                                    .median().bfill().ffill().to_numpy(), np.full(n_te, base)])
            method, widen, allow, projector, seasonal_at = "ROLLING_MEDIAN", 1.25, True, None, None
            note = f"MSTL unavailable ({type(e).__name__}); rolling median on {n_tr} days"
    elif cohort is not None and len(cohort) > 0 and n_tr >= 8:
        cz = _fwd(cohort.to_numpy(dtype=float).ravel(), kpi); cz = cz[np.isfinite(cz)]
        shape = np.resize(cz - cz.mean(), n)
        own_level = float(np.mean(z[:n_tr])); own_var = float(np.var(z[:n_tr], ddof=1)) if n_tr > 1 else 1.0
        tau2 = max(float(np.var(cz, ddof=1)), 1e-6)
        w = tau2 / (tau2 + own_var / max(n_tr, 1))
        exp_t = w * (own_level + shape) + (1 - w) * (float(cz.mean()) + shape)
        method, widen, projector, seasonal_at = "POOLED_COHORT", float(sp["widen_interval_factor"]), None, None
        allow = bool(sp["causal_claims_allowed"])
        note = (f"n={n_tr} days of own history < min_history {min_h}. Level from own data, "
                f"seasonal shape pooled from {len(cz)} sibling observations, shrinkage w={w:.2f}. "
                f"Intervals widened {widen}x; causal claims disabled.")
    else:
        return Baseline(kpi, "INSUFFICIENT_HISTORY", s.index, y, np.full(n, np.nan),
                        np.full(n, np.nan), np.array([np.nan]), 0, 1.0,
                        f"n={n_tr} days before the window; below the floor for any baseline. "
                        f"No verdict is offered.", False)

    resid = z - exp_t
    # rolling-origin forecast-error calibration at the SAME horizon as the test window
    oos_stats = []
    if projector is not None and n_te > 0:
        step = 1
        for origin in range(max(60, 2 * n_te), n_tr - n_te, step):
            pe_ = projector(origin, n_te)
            oos_stats.append(float(np.mean(z[origin:origin + n_te] - pe_)))
    mad = pd.Series(resid[:n_tr]).rolling(28, min_periods=7, center=True).apply(
        lambda a: np.median(np.abs(a - np.median(a))), raw=True).bfill().ffill()
    sigma_tr = np.maximum(1.4826 * mad.to_numpy(), 1e-6)
    sigma = np.concatenate([sigma_tr, np.full(n_te, float(np.median(sigma_tr[-28:])))])
    rz = resid / sigma
    n_cal = max(20, int(n_tr * calib_frac))
    calib = np.abs(rz[max(0, n_tr - n_cal):n_tr])          # held-out block, disjoint from test
    calib = np.sort(calib[np.isfinite(calib)])
    b = Baseline(kpi, method, s.index, y, _inv(exp_t, kpi), rz, calib, len(calib),
                 widen, note, allow)
    b.oos_window_stats = np.sort(np.array(oos_stats)) if oos_stats else np.array([])
    des = (z - np.array([seasonal_at(i) for i in range(n)])
           if seasonal_at is not None else z.copy())
    slow = pd.Series(des).rolling(121, min_periods=30, center=True).median().bfill().ffill().to_numpy()
    b.deseasonalised = des - slow
    return b


def cohort_series(panel: pd.DataFrame, kpi_num: str, kpi_den: str,
                  exclude_category: str) -> pd.DataFrame:
    """Sibling observations used to borrow seasonal shape for a new product."""
    sib = panel[panel.category != exclude_category]
    g = sib.groupby("d").agg(num=(kpi_num, "sum"), den=(kpi_den, "sum"))
    return (g.num / g.den.replace(0, np.nan)).dropna().to_frame("v")

#!/usr/bin/env python3
"""Tier-2 pass: run the engine's statistics against real public data.

Everything the engine is scored on elsewhere comes from a simulator written by the same
author as the engine, so a good result there is a statement about internal consistency.
This pass runs the same machinery over UCI Online Retail II, 1,067,371 real transactions
that nobody here generated, and asks four questions whose answers were not designed in:

  T2-B  Do the exact decompositions still close on real segments?
  T2-C  On windows where nothing is known to have happened, how often does the detector
        fire? A calibrated procedure should fire at about its nominal rate.
  T2-D  On random dates with no intervention at all, how often does synthetic control
        report a causal effect? A calibrated placebo test should say R3 about 10% of the
        time, because 0.10 is the threshold.
  T2-E  With a known shock injected into a real series, is it recovered, and at what size?

T2-C and T2-D are negative controls. They can fail. That is the point of running them.
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casefile.adapters import uci_retail as U
from casefile.engine import baseline as B, causal as CA, contribution as C_
from casefile.paths import contract as _contract, out as _out

WINDOW_DAYS = 14
MIN_TRAIN = 220          # trading days of history required before a window may be tested
SC_UNITS = 40            # units per synthetic-control trial: 39 donors, floor 1/40 = 0.025
SC_MIN_DAYS = 483        # 80% of 603 trading days: causal.py drops anything sparser, so a
                         # looser filter here silently empties the donor pool
STABLE_MONTHS = (2, 3, 4, 5, 6, 7, 8)   # outside the Christmas ramp

_S = None
_C = None
_PANEL = None
_UNITS = None


def _init(series, contract, panel):
    global _S, _C, _PANEL, _UNITS
    _S, _C, _PANEL = series, contract, panel
    if panel is not None:
        _UNITS = sorted(panel["unit"].unique())


def _fit_window(series, start_pos, contract):
    ts = series.index[start_pos]
    sub = series.iloc[:start_pos + WINDOW_DAYS]
    bl = B.fit("net_revenue", sub, contract, test_start=ts)
    if bl.method != "MSTL_PROJECTED":
        return None
    idx = np.where(bl.dates >= ts)[0]
    if len(idx) == 0:
        return None
    p_stud, eff_sd = bl.window_rank_p(idx)
    actual = float(np.nanmean(bl.actual[idx]))
    expected = float(np.nanmean(bl.expected[idx]))
    return {"window_start": str(ts.date()), "p_studentised": float(p_stud),
            "p_conformal_rank": float(bl.last_p_rank),
            "p_conformal_floor": float(bl.last_p_floor),
            "effect_sd": float(eff_sd), "actual": actual, "expected": expected,
            "delta_pct": (actual - expected) / expected if expected else None,
            "n_calibration_windows": int(bl.oos_window_stats.size)}


def _null_one(start_pos):
    try:
        return _fit_window(_S, start_pos, _C)
    except Exception as e:
        return {"window_start": None, "error": f"{type(e).__name__}: {e}"}


def _power_one(arg):
    start_pos, delta = arg
    try:
        s = _S.copy()
        s.iloc[start_pos:start_pos + WINDOW_DAYS] *= (1.0 - delta)
        r = _fit_window(s, start_pos, _C)
        if r is None:
            return None
        r["injected_delta"] = float(delta)
        return r
    except Exception as e:
        return {"injected_delta": float(delta), "error": f"{type(e).__name__}: {e}"}


def _sc_one(seed):
    try:
        rng = np.random.default_rng(seed)
        days = sorted(_PANEL["d"].unique())
        lo, hi = 90, len(days) - WINDOW_DAYS - 2
        if hi <= lo:
            return None
        pos = int(rng.integers(lo, hi))
        t0, t1 = days[pos], days[pos + WINDOW_DAYS - 1]
        units = list(rng.choice(_UNITS, size=min(SC_UNITS, len(_UNITS)), replace=False))
        treated = units[0]
        sub = _PANEL[_PANEL["unit"].isin(units)]
        est = CA.estimate(sub, "H_NEGATIVE_CONTROL", treated, t0, t1,
                          unit_col="unit", time_col="d", value_col="v", onset=t0)
        return {"seed": seed, "onset": str(t0), "treated_unit": str(treated),
                "rung": est.rung, "placebo_p": float(est.placebo_p),
                "placebo_p_floor": float(est.placebo_p_floor),
                "pretrend_p": float(est.pretrend_p), "n_donors": int(est.n_donors),
                "effect_pct": (float(est.effect_pct) if np.isfinite(est.effect_pct) else None)}
    except Exception as e:
        return {"seed": seed, "error": f"{type(e).__name__}: {e}"}


def t2b_decomposition(clean_df):
    """The ratio-of-sums identity must close on real segments, not just simulated ones."""
    panel = U.country_panel(clean_df)
    panel["d"] = pd.to_datetime(panel["d"])
    days = sorted(panel["d"].unique())
    mid = len(days) // 2
    pre = panel[panel["d"].isin(days[mid - 60:mid - 3])]
    post = panel[panel["d"].isin(days[mid:mid + 14])]
    segs, tot, resid = C_.ratio_segments(pre, post, "region")
    drill = C_.hierarchical_drill(pre, post, ["region"])
    return {"segments": len(segs), "total_move": float(tot), "residual": float(resid),
            "residual_relative": float(abs(resid) / max(abs(tot), 1e-12)),
            "identity_closes": bool(abs(resid) < 1e-9 * max(1.0, abs(tot))),
            "drilled_segments": len(drill),
            "all_drilled_carry_rank": all("rank" in d for d in drill),
            "top3": [{"segment": d["segment"], "share_of_move": round(d["share_of_move"], 4),
                      "rank": d["rank"]} for d in drill[:3]],
            "note": "Countries as segments, on the real invoice ledger. The identity is "
                    "algebraic, so a failure here would mean the adapter, not the method, "
                    "is wrong."}


def _rate(ps, a):
    return float(np.mean([p <= a for p in ps])) if ps else float("nan")


def main(n_null=200, n_sc=120, n_power=40, workers=16, out_dir=None):
    t_start = time.time()
    out = Path(out_dir or (_out() / "tier2"))
    out.mkdir(parents=True, exist_ok=True)

    if not U.available():
        raise SystemExit("data/online_retail_ii.parquet not found. "
                         "Run: python3 scripts/fetch_public_data.py")

    base = yaml.safe_load(open(_contract()))
    contract = U.contract_overlay(base)

    print("T2-A  ingesting the real source ...")
    raw = U.load()
    clean, q = U.clean(raw)
    series = U.daily_series(clean, q, country="United Kingdom", measure="net_revenue")
    quality = U.report_dict(q)
    print(f"      {quality['rows_kept']:,} of {quality['rows_raw']:,} rows kept "
          f"({quality['rows_kept_pct']}%), {quality['trading_days_imputed']} days imputed")

    print("T2-B  exact decomposition on real segments ...")
    t2b = t2b_decomposition(clean)
    print(f"      residual {t2b['residual']:.3e} over {t2b['segments']} country segments "
          f"-> closes = {t2b['identity_closes']}")

    valid = list(range(MIN_TRAIN, len(series) - WINDOW_DAYS))
    rng = np.random.default_rng(20260901)

    print(f"T2-C  null calibration of the detector, {n_null} windows ...")
    starts = sorted(rng.choice(valid, size=min(n_null, len(valid)), replace=False).tolist())
    with Pool(workers, initializer=_init, initargs=(series, contract, None)) as p:
        null_rows = [r for r in p.map(_null_one, starts) if r and "error" not in r]
    ps_stud = [r["p_studentised"] for r in null_rows]
    ps_conf = [r["p_conformal_rank"] for r in null_rows]
    t2c = {
        "windows_tested": len(null_rows), "window_days": WINDOW_DAYS,
        "nominal_vs_empirical": {
            str(a): {"nominal": a,
                     "empirical_studentised": round(_rate(ps_stud, a), 4),
                     "empirical_conformal": round(_rate(ps_conf, a), 4)}
            for a in (0.01, 0.05, 0.10)},
        "p_studentised_quartiles": [round(float(x), 4) for x in np.percentile(ps_stud, [25, 50, 75])],
        "p_conformal_quartiles": [round(float(x), 4) for x in np.percentile(ps_conf, [25, 50, 75])],
        "mean_conformal_floor": round(float(np.mean([r["p_conformal_floor"] for r in null_rows])), 5),
        "caveat": "Real windows are not a true null: genuine shocks occur in this data and "
                  "are not labelled. The empirical rate is therefore an upper bound on the "
                  "false-positive rate, not an unbiased estimate of it.",
    }
    months = [pd.Timestamp(r["window_start"]).month for r in null_rows]
    t2c["by_month_fire_rate_at_0.01"] = {
        str(m): round(float(np.mean([p_ <= 0.01 for p_, mm in zip(ps_stud, months) if mm == m])), 3)
        for m in sorted(set(months))}
    stable = [(p_, pc) for p_, pc, m in zip(ps_stud, ps_conf, months) if m in STABLE_MONTHS]
    t2c["seasonally_stable_subset"] = {
        "months": list(STABLE_MONTHS),
        "windows": len(stable),
        "empirical_at_0.01": round(_rate([x[0] for x in stable], 0.01), 4),
        "empirical_at_0.05": round(_rate([x[0] for x in stable], 0.05), 4),
        "empirical_at_0.10": round(_rate([x[0] for x in stable], 0.10), 4),
        "why": "February to August contains no Christmas ramp. If the aggregate failure is "
               "caused by an unavailable annual cycle, this subset should be calibrated."}
    for a, v in t2c["nominal_vs_empirical"].items():
        print(f"      alpha={a}: studentised {v['empirical_studentised']:.3f}  "
              f"conformal {v['empirical_conformal']:.3f}")

    print(f"T2-D  synthetic-control negative control, {n_sc} trials ...")
    panel = U.sku_panel(clean, country="United Kingdom", min_days=SC_MIN_DAYS).copy()
    panel["d"] = pd.to_datetime(panel["d"]).dt.date
    with Pool(workers, initializer=_init, initargs=(series, contract, panel)) as p:
        sc_rows = [r for r in p.map(_sc_one, list(range(7000, 7000 + n_sc)))
                   if r and "error" not in r]
    rungs = [r["rung"] for r in sc_rows]
    sc_ps = [r["placebo_p"] for r in sc_rows]
    t2d = {
        "trials": len(sc_rows), "units_per_trial": SC_UNITS,
        "mean_donors": round(float(np.mean([r["n_donors"] for r in sc_rows])), 1) if sc_rows else None,
        "mean_placebo_floor": round(float(np.mean([r["placebo_p_floor"] for r in sc_rows])), 4) if sc_rows else None,
        "spurious_R3_rate_all_trials": round(float(np.mean([x == "R3" for x in rungs])), 4) if rungs else None,
        "expected_R3_rate": 0.10,
        "rung_counts": {k: int(rungs.count(k)) for k in sorted(set(rungs))},
        "placebo_p_quartiles": [round(float(x), 4) for x in np.percentile(sc_ps, [25, 50, 75])] if sc_ps else None,
        "note": "No intervention exists at these dates. Every R3 here is a false positive. "
                "The threshold is p <= 0.10, so a calibrated test yields about 10%.",
    }
    # A trial whose donor pool gives a placebo floor above 0.10 cannot reach R3 whatever the
    # data says, so including it would flatter the false-positive rate. Report both.
    permits = [r for r in sc_rows if r["placebo_p_floor"] <= 0.10]
    blocked = [r for r in sc_rows if r["placebo_p_floor"] > 0.10]
    t2d["trials_where_R3_was_reachable"] = len(permits)
    t2d["trials_blocked_by_placebo_floor"] = len(blocked)
    t2d["spurious_R3_rate_reachable_only"] = (
        round(float(np.mean([r["rung"] == "R3" for r in permits])), 4) if permits else None)
    t2d["placebo_p_quartiles_reachable"] = (
        [round(float(x), 4) for x in np.percentile([r["placebo_p"] for r in permits], [25, 50, 75])]
        if permits else None)
    t2d["verdict"] = (
        "PASS, conservative" if (t2d["spurious_R3_rate_reachable_only"] or 1) <= 0.10 else
        "FAIL, over-claims causality")
    print(f"      spurious R3: {t2d['spurious_R3_rate_reachable_only']} on the "
          f"{len(permits)} trials where the floor permits it, against an expected 0.10 "
          f"-> {t2d['verdict']}")

    print(f"T2-E  recovery of an injected shock, {n_power} windows x 4 sizes ...")
    stable_valid = [i for i in valid if series.index[i].month in STABLE_MONTHS]
    pstarts = sorted(rng.choice(stable_valid, size=min(n_power, len(stable_valid)),
                                replace=False).tolist())
    jobs = [(s, d) for d in (0.02, 0.05, 0.10, 0.20) for s in pstarts]
    with Pool(workers, initializer=_init, initargs=(series, contract, None)) as p:
        pw = [r for r in p.map(_power_one, jobs) if r and "error" not in r]
    by_delta = {}
    for d in (0.02, 0.05, 0.10, 0.20):
        rows = [r for r in pw if abs(r["injected_delta"] - d) < 1e-9]
        if not rows:
            continue
        est = [r["delta_pct"] for r in rows if r["delta_pct"] is not None]
        by_delta[f"{d:.2f}"] = {
            "windows": len(rows),
            "detected_at_0.05": round(_rate([r["p_studentised"] for r in rows], 0.05), 3),
            "detected_at_0.01": round(_rate([r["p_studentised"] for r in rows], 0.01), 3),
            "median_estimated_delta": round(float(np.median(est)), 4) if est else None,
            "injected_delta": -d,
            "median_bias": round(float(np.median(est)) + d, 4) if est else None}
        m = by_delta[f"{d:.2f}"]
        print(f"      injected -{100*d:.0f}%: detected {m['detected_at_0.05']:.0%} at "
              f"alpha=0.05, median estimate {100*(m['median_estimated_delta'] or 0):.2f}%")
    t2e = {"by_injected_delta": by_delta,
           "windows_drawn_from": f"months {list(STABLE_MONTHS)} only",
           "why_restricted": "A power measurement taken across the Christmas ramp measures "
                             "baseline bias, not power. The windows are drawn from the "
                             "months where T2-C shows the baseline is sound.",
           "note": "The shock is synthetic but the noise, seasonality and calendar are real. "
                   "Bias is the median estimate minus the injected size."}

    summary = {
        "pass": "Tier-2, real public data",
        "source": {"name": "UCI Online Retail II",
                   "url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
                   "rows": quality["rows_raw"], "span": "2009-12-01 to 2011-12-09",
                   "series_used": "United Kingdom daily net revenue, trading calendar",
                   "series_length": len(series)},
        "T2A_data_quality": quality,
        "T2B_decomposition": t2b,
        "T2C_null_calibration": t2c,
        "T2D_causal_negative_control": t2d,
        "T2E_injected_recovery": t2e,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(null_rows).to_parquet(out / "null_windows.parquet")
    pd.DataFrame(sc_rows).to_parquet(out / "sc_negative_control.parquet")
    pd.DataFrame(pw).to_parquet(out / "injected_recovery.parquet")
    print(f"\nwrote {out}/summary.json  ({summary['wall_seconds']}s)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-sc", type=int, default=120)
    ap.add_argument("--n-power", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    main(a.n_null, a.n_sc, a.n_power, a.workers)

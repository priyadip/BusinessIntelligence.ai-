"""The Tier-2 pass runs on real public data, so its tests skip when the file is absent."""
import json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
for c in (ROOT, ROOT / "src"):
    if (c / "casefile").exists(): sys.path.insert(0, str(c)); break
from casefile.adapters import uci_retail as U
from casefile.paths import out as _out

pytestmark = pytest.mark.skipif(not U.available(),
                                reason="data/online_retail_ii.parquet not fetched")


def test_cleaning_is_accounted_for():
    """Every dropped row must be attributed to a named rule, not vanish."""
    raw = U.load()
    clean, q = U.clean(raw)
    named = (q.cancellations_dropped + q.non_positive_quantity_dropped
             + q.non_positive_price_dropped)
    assert q.rows_dropped == named, (
        f"{q.rows_dropped} rows dropped but only {named} attributed to a rule")
    assert q.rows_kept > 0 and q.rows_raw > q.rows_kept
    assert len(q.decisions) >= 3, "each cleaning rule must state why"


def test_trading_calendar_excludes_saturday():
    """The source is closed on Saturdays; a 7-day cycle would be the wrong period."""
    import pandas as pd
    idx = U.trading_index(pd.Timestamp("2010-01-01"), pd.Timestamp("2010-03-31"))
    assert (idx.dayofweek == 5).sum() == 0
    ov = U.contract_overlay({"kpis": {"net_revenue": {}, "avg_order_value": {}}})
    assert ov["kpis"]["net_revenue"]["seasonality"]["weekly_period_days"] == 6


def test_series_is_regular_and_imputation_is_reported():
    raw = U.load()
    clean, q = U.clean(raw)
    s = U.daily_series(clean, q)
    assert len(s) == q.trading_days_expected
    assert q.trading_days_imputed == q.trading_days_expected - q.trading_days_present
    assert s.notna().all()


def test_declared_period_reaches_the_baseline():
    """A contract that declares a 6-day cycle must produce a 6-day decomposition."""
    import yaml, numpy as np, pandas as pd
    from casefile.engine import baseline as B
    from casefile.paths import contract as _c
    raw = U.load(); clean, q = U.clean(raw)
    s = U.daily_series(clean, q)
    C = U.contract_overlay(yaml.safe_load(open(_c())))
    bl = B.fit("net_revenue", s.iloc[:414], C, test_start=s.index[400])
    assert bl.method == "MSTL_PROJECTED"
    assert "periods=(6,)" in bl.note, bl.note


def test_tier2_results_hold_their_claims():
    """If the pass has been run, its stored conclusions must match its stored numbers."""
    f = _out() / "tier2" / "summary.json"
    if not f.exists():
        pytest.skip("tier-2 pass has not been run")
    d = json.loads(f.read_text())

    # the decomposition is algebraic: it must close on real data too
    assert d["T2B_decomposition"]["identity_closes"] is True
    assert abs(d["T2B_decomposition"]["residual"]) < 1e-9

    # the seasonally stable subset is the claim that the failure is seasonal, not general
    st = d["T2C_null_calibration"]["seasonally_stable_subset"]
    assert st["empirical_at_0.01"] <= 0.05, (
        "the detector is claimed to be calibrated outside the Christmas ramp; it is not")

    # the negative control must not over-claim causality
    t2d = d["T2D_causal_negative_control"]
    assert t2d["spurious_R3_rate_reachable_only"] <= 0.10, (
        "synthetic control reported more spurious causal effects than its own threshold allows")
    assert t2d["verdict"].startswith("PASS")

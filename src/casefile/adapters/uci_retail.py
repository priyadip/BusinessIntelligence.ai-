"""Adapter for UCI Online Retail II: real transactions, mapped to the engine's shape.

This source is real, public and messy in ways the synthetic world is not. It carries
cancellations, negative quantities, non-positive prices, missing customers, and a trading
calendar that is closed on Saturdays. Every cleaning decision here is declared and counted
rather than applied silently, because the count is the interesting part.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

SATURDAY = 5


@dataclass
class QualityReport:
    """What the adapter had to do to the raw file, and how much it touched."""
    source: str
    rows_raw: int = 0
    rows_kept: int = 0
    cancellations_dropped: int = 0
    non_positive_quantity_dropped: int = 0
    non_positive_price_dropped: int = 0
    missing_customer_id_kept: int = 0
    calendar_days_spanned: int = 0
    trading_days_expected: int = 0
    trading_days_present: int = 0
    trading_days_imputed: int = 0
    saturdays_observed: int = 0
    decisions: list = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_raw - self.rows_kept


def path(root: Path | None = None) -> Path:
    from ..paths import ROOT
    r = root or ROOT
    for c in (r / "data/online_retail_ii.parquet", r.parent / "data/online_retail_ii.parquet"):
        if c.exists():
            return c
    return r / "data/online_retail_ii.parquet"


def available() -> bool:
    return path().exists()


def load(p: Path | None = None) -> pd.DataFrame:
    f = p or path()
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found. Run: python3 scripts/fetch_public_data.py")
    return pd.read_parquet(f)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Apply the declared cleaning rules and count every row removed."""
    q = QualityReport(source="UCI Online Retail II", rows_raw=len(raw))
    d = raw.copy()

    inv = d["invoice"].astype(str)
    is_cancel = inv.str.startswith("C")
    q.cancellations_dropped = int(is_cancel.sum())
    q.decisions.append(
        "Invoices prefixed 'C' are cancellations. A cancellation is a reversal of an "
        "earlier sale, not a sale, so it is removed rather than netted: netting would "
        "attribute a return to the day it was processed instead of the day it was sold.")

    neg_q = (~is_cancel) & (d["quantity"] <= 0)
    q.non_positive_quantity_dropped = int(neg_q.sum())
    q.decisions.append(
        "Non-positive quantity outside a cancellation is an adjustment or a data error. "
        "Removed; it cannot represent demand.")

    bad_p = (~is_cancel) & (~neg_q) & (d["price"] <= 0)
    q.non_positive_price_dropped = int(bad_p.sum())
    q.decisions.append(
        "Non-positive price rows are samples, write-offs and manual corrections. Removed "
        "from revenue, since a zero-priced line is not a sale at any value.")

    d = d[~(is_cancel | neg_q | bad_p)].copy()
    q.missing_customer_id_kept = int(d["customer_id"].isna().sum())
    q.decisions.append(
        "Rows without a customer id are retained. They are guest checkouts and dropping "
        "them would bias revenue downward by roughly a fifth.")

    d["revenue"] = d["quantity"] * d["price"]
    d["d"] = d["invoicedate"].dt.normalize()
    q.rows_kept = len(d)
    return d, q


def trading_index(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    """Every calendar day except Saturday. The source is closed on Saturdays."""
    all_days = pd.date_range(first, last, freq="D")
    return all_days[all_days.dayofweek != SATURDAY]


def country_panel(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Day x country, with a numerator and a denominator, as the engine expects."""
    g = (clean_df.groupby(["d", "country"], as_index=False)
         .agg(numerator=("revenue", "sum"), denominator=("invoice", "nunique")))
    g["region"] = g["country"]
    return g


def sku_panel(clean_df: pd.DataFrame, country: str = "United Kingdom",
              min_days: int = 300) -> pd.DataFrame:
    """Day x stockcode revenue, restricted to stockcodes with real coverage.

    Stockcodes are the unit of analysis for synthetic control here: countries are too few
    to give a donor pool that can express a small p-value.
    """
    c = clean_df[clean_df["country"] == country]
    c = c[c["d"].dt.dayofweek != SATURDAY]
    cover = c.groupby("stockcode")["d"].nunique()
    keep = cover[cover >= min_days].index
    p = (c[c["stockcode"].isin(keep)]
         .groupby(["d", "stockcode"], as_index=False)["revenue"].sum())
    return p.rename(columns={"stockcode": "unit", "revenue": "v"})


def daily_series(clean_df: pd.DataFrame, q: QualityReport,
                 country: str | None = "United Kingdom",
                 measure: str = "net_revenue") -> pd.Series:
    """A regular series on the trading calendar, with holidays linearly imputed.

    MSTL needs a regular index. The source closes on Saturdays and on public holidays. The
    Saturdays are removed from the calendar entirely, which is the honest treatment of a
    day the business never trades. The remaining gaps are holidays inside a trading week
    and are interpolated, with the count reported so a reader can weigh it.
    """
    c = clean_df if country is None else clean_df[clean_df["country"] == country]
    if measure == "net_revenue":
        g = c.groupby("d")["revenue"].sum()
    elif measure == "avg_order_value":
        agg = c.groupby("d").agg(rev=("revenue", "sum"), inv=("invoice", "nunique"))
        g = agg["rev"] / agg["inv"].replace(0, np.nan)
    else:
        raise KeyError(f"{measure} has no analogue in this source")

    first, last = g.index.min(), g.index.max()
    q.calendar_days_spanned = int((last - first).days + 1)
    q.saturdays_observed = int((g.index.dayofweek == SATURDAY).sum())

    idx = trading_index(first, last)
    q.trading_days_expected = len(idx)
    s = g.reindex(idx)
    q.trading_days_present = int(s.notna().sum())
    q.trading_days_imputed = int(s.isna().sum())
    if q.trading_days_imputed:
        q.decisions.append(
            f"{q.trading_days_imputed} trading days are absent (public holidays inside a "
            f"trading week). Linearly interpolated so the seasonal decomposition has a "
            f"regular index; they are a {100*q.trading_days_imputed/max(len(idx),1):.1f}% "
            f"share of the series.")
    return s.interpolate(limit_direction="both").dropna()


def contract_overlay(base_contract: dict) -> dict:
    """The contract entries this source needs, declared rather than assumed.

    The trading week is six days, not seven, so the weekly cycle is declared explicitly.
    Two years of history is not enough for a yearly cycle at this cadence, so it is off.
    """
    c = {k: v for k, v in base_contract.items()}
    kpis = dict(c["kpis"])
    kpis["net_revenue"] = {
        **base_contract["kpis"]["net_revenue"],
        "from": "uci_retail.online_retail_ii",
        "min_history_days": 180,
        "seasonality": {"weekly": True, "weekly_period_days": 6,
                        "yearly": False, "holiday_set": "UK_RETAIL"},
        "lineage": ["uci_retail.quantity", "uci_retail.price"],
    }
    kpis["avg_order_value"] = {
        **base_contract["kpis"]["avg_order_value"],
        "from": "uci_retail.online_retail_ii",
        "min_history_days": 180,
        "seasonality": {"weekly": True, "weekly_period_days": 6,
                        "yearly": False, "holiday_set": "UK_RETAIL"},
        "lineage": ["uci_retail.quantity", "uci_retail.price", "uci_retail.invoice"],
    }
    c["kpis"] = kpis
    return c


def report_dict(q: QualityReport) -> dict:
    d = asdict(q)
    d["rows_dropped"] = q.rows_dropped
    d["rows_kept_pct"] = round(100.0 * q.rows_kept / max(q.rows_raw, 1), 2)
    return d

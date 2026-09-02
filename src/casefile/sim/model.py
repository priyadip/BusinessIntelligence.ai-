"""Structural simulator for a fictional retailer. Deterministic given a seed."""
from __future__ import annotations
import zlib
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from datetime import date, timedelta


def stable_hash(*parts) -> int:
    """Deterministic across processes."""
    return zlib.crc32("\x1f".join(str(p) for p in parts).encode()) & 0xFFFFFFFF

REGIONS    = ["WEST", "NORTH", "SOUTH", "EAST"]
CHANNELS   = ["WEB", "APP", "STORE", "MARKETPLACE"]
CATEGORIES = ["KITCHEN", "DECOR", "FURNITURE", "LIGHTING", "OUTDOOR", "SMART_HOME"]
CARRIERS   = ["C_SWIFT", "C_METRO", "C_BLUEDART_LIKE", "C_NEWCO"]

START = date(2024, 1, 1)
END   = date(2026, 8, 31)
SMART_HOME_LAUNCH = date(2026, 8, 10)          # sparse-history scenario
OPS_COVERAGE_FROM = date(2024, 6, 1)           # ops source has shorter history, on purpose

HOLIDAYS = {date(2026,1,26),date(2026,3,4),date(2026,3,21),date(2026,4,14),date(2026,5,1),
            date(2026,8,15),date(2026,10,2),date(2026,10,20),date(2026,11,8),date(2026,12,25),
            date(2025,1,26),date(2025,3,14),date(2025,4,14),date(2025,5,1),date(2025,8,15),
            date(2025,10,2),date(2025,10,21),date(2025,11,1),date(2025,12,25),
            date(2024,1,26),date(2024,3,25),date(2024,4,14),date(2024,5,1),date(2024,8,15),
            date(2024,10,2),date(2024,11,1),date(2024,12,25)}

REGION_SCALE  = {"WEST": 1.00, "NORTH": 0.86, "SOUTH": 0.79, "EAST": 0.58}
CHANNEL_SHARE = {"WEB": 0.41, "APP": 0.33, "STORE": 0.18, "MARKETPLACE": 0.08}
CAT_SHARE     = {"KITCHEN":0.27,"DECOR":0.21,"FURNITURE":0.19,"LIGHTING":0.16,"OUTDOOR":0.14,"SMART_HOME":0.03}
BASE_CVR      = {"WEB": 0.0305, "APP": 0.0412, "STORE": 0.1850, "MARKETPLACE": 0.0244}
BASE_AOV      = {"KITCHEN":2150,"DECOR":1580,"FURNITURE":8400,"LIGHTING":2650,"OUTDOOR":3100,"SMART_HOME":4750}
CARRIER_BASE_ONTIME = {"C_SWIFT":0.951,"C_METRO":0.938,"C_BLUEDART_LIKE":0.962,"C_NEWCO":0.906}


@dataclass
class Intervention:
    """One injected real-world cause. This IS the ground truth."""
    id: str
    hypothesis_id: str          # must resolve to contracts/kpi_contract.yaml hypothesis_library
    label: str
    start: date
    end: date | None = None
    regions: list[str] | None = None
    channels: list[str] | None = None
    categories: list[str] | None = None
    units: list[tuple] | None = None       # explicit (region, category) assignment
    # latent driver deltas
    carrier_shift_to: str | None = None       # move shipment share to this carrier
    carrier_shift_frac: float = 0.0
    price_index_mult: float = 1.0
    promo_depth_delta: float = 0.0            # absolute change in promo depth (fraction)
    checkout_error_delta: float = 0.0         # absolute add to checkout error rate
    stockout_delta: float = 0.0               # absolute add to stockout rate
    competitor_gap_delta: float = 0.0         # absolute add to competitor price gap
    marketing_mult: float = 1.0
    cvr_mult: float = 1.0                     # direct conversion effect (external shocks)
    ramp_days: int = 1                        # linear ramp-in, models real rollouts
    visible_in_release_log: bool = False
    visible_in_incident_report: bool = False
    generates_reviews: str | None = None      # review theme keyword
    generates_tickets: str | None = None
    generates_supplier_email: bool = False
    generates_competitor_intel: bool = False

    def active(self, d: date) -> float:
        """Ramp factor in [0,1] for a given day."""
        if d < self.start: return 0.0
        if self.end is not None and d > self.end: return 0.0
        if self.ramp_days <= 1: return 1.0
        return min(1.0, (d - self.start).days / float(self.ramp_days) + 1.0 / self.ramp_days)

    def matches(self, region: str, channel: str, category: str) -> bool:
        if self.units is not None:
            return (region, category) in self.units
        if self.regions    is not None and region   not in self.regions:    return False
        if self.channels   is not None and channel  not in self.channels:   return False
        if self.categories is not None and category not in self.categories: return False
        return True


def _dates(start: date, end: date) -> list[date]:
    n = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n)]


def _weekly(d: date) -> float:
    # Sat/Sun lift, Tue trough. Retail-typical.
    return {0:0.97, 1:0.93, 2:0.96, 3:1.01, 4:1.09, 5:1.18, 6:1.06}[d.weekday()]


def _yearly(d: date) -> float:
    doy = d.timetuple().tm_yday
    return (1.0
            + 0.13 * np.sin(2*np.pi*(doy-80)/365.25)      # spring/autumn wave
            + 0.07 * np.sin(4*np.pi*(doy-40)/365.25))     # secondary


def _holiday(d: date) -> float:
    if d in HOLIDAYS: return 1.42
    for h in HOLIDAYS:
        delta = (h - d).days
        if 1 <= delta <= 4:  return 1.0 + 0.11 * (5 - delta) / 4.0   # pre-holiday build
        if -2 <= delta <= -1: return 0.88                             # post-holiday slump
    return 1.0


class World:
    """Deterministic given (seed, interventions). Re-runnable for counterfactuals."""

    def __init__(self, interventions: list[Intervention], seed: int = 20260901,
                 start: date = START, end: date = END):
        self.iv = interventions
        self.seed = seed
        self.start, self.end = start, end
        self.days = _dates(start, end)

    # ---------------------------------------------------------------- latent drivers
    def _weather(self, region: str) -> dict[date, float]:
        """Shared exogenous covariate. Observable to the engine, so it belongs in the baseline."""
        rng = np.random.default_rng(self.seed + 991 + stable_hash("weather", region) % 100000)
        n = len(self.days)
        # AR(1) rainfall-ish index, monsoon-weighted
        eps = rng.normal(0, 1, n); z = np.zeros(n)
        for i in range(1, n): z[i] = 0.82*z[i-1] + eps[i]
        z = (z - z.mean()) / (z.std() + 1e-9)
        seas = np.array([1.0 + 0.9*np.exp(-((d.timetuple().tm_yday-200)/38.0)**2) for d in self.days])
        reg_amp = {"WEST":1.0,"NORTH":0.6,"SOUTH":0.8,"EAST":1.15}[region]
        idx = np.clip(seas * (1 + 0.35*z) * reg_amp, 0, None)
        return dict(zip(self.days, idx))

    def _marketing(self, region: str) -> dict[date, float]:
        rng = np.random.default_rng(self.seed + 555 + stable_hash("marketing", region) % 100000)
        n = len(self.days)
        base = 1.0 + 0.06*np.sin(2*np.pi*np.arange(n)/28.0) + rng.normal(0, 0.035, n)
        return dict(zip(self.days, np.clip(base, 0.6, 1.5)))

    # ---------------------------------------------------------------- main panel
    def simulate(self, drop_interventions: set[str] | None = None) -> pd.DataFrame:
        """Daily panel at region x channel x category grain.

        drop_interventions: switch these OFF. Used for counterfactual ground truth."""
        drop = drop_interventions or set()
        active_iv = [v for v in self.iv if v.id not in drop]
        rng = np.random.default_rng(self.seed)

        weather   = {r: self._weather(r)   for r in REGIONS}
        marketing = {r: self._marketing(r) for r in REGIONS}

        rows = []
        # deterministic per-cell noise streams so counterfactuals differ ONLY by intervention
        for r in REGIONS:
            for ch in CHANNELS:
                for cat in CATEGORIES:
                    cell_rng = np.random.default_rng(
                        self.seed + stable_hash("cell", r, ch, cat) % 1000000)
                    n = len(self.days)
                    sess_noise = cell_rng.normal(1.0, 0.055, n)
                    cvr_noise  = cell_rng.normal(1.0, 0.042, n)
                    aov_noise  = cell_rng.normal(1.0, 0.030, n)

                    base_sessions = (14200 * REGION_SCALE[r] * CHANNEL_SHARE[ch] * CAT_SHARE[cat])
                    for i, d in enumerate(self.days):
                        if cat == "SMART_HOME" and d < SMART_HOME_LAUNCH:
                            continue                                   # sparse history by design

                        # ---- accumulate intervention effects on latent drivers
                        price_mult, promo_delta = 1.0, 0.0
                        ck_err, oos, comp_gap, mkt_mult = 0.004, 0.021, 0.0, 1.0
                        cvr_ext = 1.0
                        carrier_shift = {}
                        for v in active_iv:
                            if not v.matches(r, ch, cat): continue
                            a = v.active(d)
                            if a <= 0: continue
                            price_mult *= (1.0 + (v.price_index_mult - 1.0) * a)
                            promo_delta += v.promo_depth_delta * a
                            ck_err      += v.checkout_error_delta * a
                            oos         += v.stockout_delta * a
                            comp_gap    += v.competitor_gap_delta * a
                            mkt_mult    *= (1.0 + (v.marketing_mult - 1.0) * a)
                            cvr_ext     *= (1.0 + (v.cvr_mult - 1.0) * a)
                            if v.carrier_shift_to:
                                carrier_shift[v.carrier_shift_to] = \
                                    carrier_shift.get(v.carrier_shift_to, 0.0) + v.carrier_shift_frac * a

                        # ---- fulfilment on-time, driven by carrier mix
                        mix = {"C_SWIFT":0.34,"C_METRO":0.30,"C_BLUEDART_LIKE":0.29,"C_NEWCO":0.07}
                        for cname, frac in carrier_shift.items():
                            take = min(frac, 0.92)
                            for k in list(mix):
                                if k != cname: mix[k] *= (1 - take)
                            mix[cname] = mix.get(cname, 0.0) + take
                        tot = sum(mix.values()); mix = {k: v/tot for k, v in mix.items()}
                        ontime = sum(mix[c] * CARRIER_BASE_ONTIME[c] for c in mix)
                        wx = weather[r][d]
                        ontime *= (1.0 - 0.028 * max(0.0, wx - 1.0))       # weather hurts delivery
                        ontime = float(np.clip(ontime + cell_rng.normal(0, 0.006), 0.55, 0.995))

                        # ---- sessions
                        launch_ramp = 1.0
                        if cat == "SMART_HOME":
                            launch_ramp = min(1.0, 0.25 + 0.75*((d - SMART_HOME_LAUNCH).days/45.0))
                        trend = 1.0 + 0.00022 * i
                        sessions = (base_sessions * _weekly(d) * _yearly(d) * _holiday(d) * trend
                                    * marketing[r][d] * mkt_mult * launch_ramp
                                    * (1.0 - 0.020*max(0.0, wx-1.0)) * sess_noise[i])
                        sessions = max(0.0, sessions)

                        # ---- conversion  (the KPI graph, made real)
                        cvr = BASE_CVR[ch]
                        cvr *= (1.0 - 1.55 * (price_mult - 1.0))          # price elasticity
                        cvr *= (1.0 + 2.20 * promo_delta)                 # promo lift (retail-calibrated)
                        cvr *= (1.0 - 2.20 * max(0.0, 0.945 - ontime))    # fulfilment -> conversion
                        cvr *= (1.0 - 1.35 * max(0.0, oos - 0.021))       # stock-outs
                        cvr *= (1.0 - 6.50 * max(0.0, ck_err - 0.004))    # checkout defects
                        cvr *= (1.0 - 0.65 * comp_gap)                    # competitor price gap
                        cvr *= cvr_ext                                    # external shocks
                        if cat == "SMART_HOME": cvr *= 0.82
                        cvr = float(np.clip(cvr * cvr_noise[i], 0.0008, 0.60))

                        # ---- AOV
                        aov = BASE_AOV[cat] * price_mult * (1.0 - 0.55*promo_delta) * aov_noise[i]

                        orders  = sessions * cvr
                        gross   = orders * aov
                        disc    = gross * (0.055 + promo_delta)
                        returns = gross * (0.031 + 0.42*max(0.0, 0.945 - ontime))
                        unit_cost_ratio = 0.615 + 0.035*np.sin(2*np.pi*i/365.25)
                        cogs = (gross - disc) * unit_cost_ratio

                        rows.append((d, r, ch, cat, sessions, cvr, orders, aov, gross, disc,
                                     returns, cogs, ontime, price_mult, promo_delta, ck_err,
                                     oos, comp_gap, wx, mkt_mult,
                                     max(mix, key=mix.get)))

        df = pd.DataFrame(rows, columns=[
            "d","region","channel","category","sessions","cvr","orders","aov","gross_amount",
            "discount_amount","returns_amount","cogs","ontime_pct","price_index","promo_depth",
            "checkout_error_rate","stockout_rate","competitor_gap","weather_idx","marketing_mult",
            "dominant_carrier"])
        df["net_revenue"] = df.gross_amount - df.discount_amount - df.returns_amount
        return df

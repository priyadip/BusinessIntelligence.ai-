"""
Injected incidents = the ground truth the engine is scored against.

Ground truth contribution is computed by EXACT SHAPLEY VALUES over counterfactual
re-simulations: for every subset of interventions we re-run the world with only that
subset active, then average each intervention's marginal contribution over all orderings.
With n<=6 interventions that is 2^n runs of a 1.7s simulator, which is affordable and is
the correct decomposition rather than an approximation. It is also the honest one: the
interventions interact (a checkout defect and a carrier failure are not additive), and
leave-one-out would mis-state that.
"""
from __future__ import annotations
import itertools, json, math
from dataclasses import asdict
from datetime import date
import numpy as np, pandas as pd
from .model import World, Intervention


# --------------------------------------------------------------------- scenarios
def incident_interventions() -> list[Intervention]:
    """All causes live in the demo world. Concurrent incidents, as in a real enterprise."""
    return [
        # ---- INC-001 : multi-factor, RESOLVABLE (valid controls exist) --------------
        Intervention(
            id="IV_CARRIER", hypothesis_id="H_CARRIER_DEGRADE",
            label="Fulfilment re-contracted to C_NEWCO in WEST",
            start=date(2026, 7, 15), end=date(2026, 8, 5), regions=["WEST"],   # fixed after INC-001 action
            carrier_shift_to="C_NEWCO", carrier_shift_frac=0.62, ramp_days=6,
            visible_in_release_log=False, visible_in_incident_report=True,
            generates_reviews="delivery_late", generates_tickets="delivery"),
        Intervention(
            id="IV_CHECKOUT", hypothesis_id="H_CHECKOUT_DEFECT",
            label="App release 8.4.1 broke UPI retry on checkout",
            start=date(2026, 7, 18), end=date(2026, 8, 6), channels=["APP"],   # flag reverted after INC-001
            checkout_error_delta=0.019, ramp_days=2,
            visible_in_release_log=True,
            generates_reviews="payment_fail", generates_tickets="payment"),
        Intervention(
            id="IV_WEATHER_TAIL", hypothesis_id="H_EXTERNAL",
            label="Extended monsoon depression, EAST and WEST",
            start=date(2026, 7, 20), end=date(2026, 8, 12), regions=["WEST", "EAST"],
            marketing_mult=0.965, cvr_mult=0.974, ramp_days=4,
            visible_in_incident_report=True),

        # ---- INC-002 : COLLINEAR, unidentifiable (both national, 1 day apart) -------
        Intervention(
            id="IV_PRICE", hypothesis_id="H_PRICE_RISE",
            label="List price +6.5% on 214 SKUs, KITCHEN and DECOR",
            start=date(2026, 8, 3), categories=["KITCHEN", "DECOR"],
            price_index_mult=1.065, ramp_days=1,
            visible_in_release_log=True, generates_reviews="price"),
        Intervention(
            id="IV_PROMO", hypothesis_id="H_PROMO_WITHDRAWAL",
            label="Monsoon Sale ended, promo depth -4.0pp, KITCHEN and DECOR",
            start=date(2026, 8, 4), categories=["KITCHEN", "DECOR"],
            promo_depth_delta=-0.040, ramp_days=1,
            visible_in_release_log=True,
            # A shopper sees the price they pay, not its decomposition. An ended discount and
            # a list-price rise are indistinguishable at the shelf, so both must generate the
            # same complaint theme. Letting only the price rise generate text made a genuinely
            # confounded pair look separable, which is a modelling error, not a demo choice.
            generates_reviews="price"),

        # ---- background noise cause, gives the engine something to correctly reject -
        Intervention(
            id="IV_COMPETITOR", hypothesis_id="H_COMPETITOR",
            label="Competitor festive combo in SOUTH catchments",
            start=date(2026, 8, 8), regions=["SOUTH"],
            competitor_gap_delta=0.055, ramp_days=3,
            generates_competitor_intel=True),
    ]


INCIDENTS = {
    "INC-001": dict(
        kpi="conversion_rate", window=(date(2026, 7, 22), date(2026, 8, 2)),
        slice={"region": "WEST"},
        expect="resolvable_multifactor",
        true_causes=["IV_CARRIER", "IV_CHECKOUT", "IV_WEATHER_TAIL"]),
    "INC-002": dict(
        kpi="net_revenue", window=(date(2026, 8, 10), date(2026, 8, 31)),
        slice={"category_in": ["KITCHEN", "DECOR"]},
        expect="ambiguous_collinear",
        true_causes=["IV_PRICE", "IV_PROMO"]),
    "INC-003": dict(
        kpi="avg_order_value", window=(date(2026, 8, 18), date(2026, 8, 31)),
        slice={"category": "SMART_HOME"},
        expect="sparse_history_abstain",
        true_causes=[]),
}


# --------------------------------------------------------------------- metrics
def metric_on(df: pd.DataFrame, kpi: str, window, slc: dict) -> float:
    lo, hi = window
    m = (df.d >= lo) & (df.d <= hi)
    if "region" in slc:       m &= df.region == slc["region"]
    if "channel" in slc:      m &= df.channel == slc["channel"]
    if "category" in slc:     m &= df.category == slc["category"]
    if "category_in" in slc:  m &= df.category.isin(slc["category_in"])
    s = df[m]
    if s.empty: return float("nan")
    if kpi == "net_revenue":           return float(s.net_revenue.sum())
    if kpi == "conversion_rate":       return float(s.orders.sum() / max(s.sessions.sum(), 1e-9))
    if kpi == "avg_order_value":       return float(s.net_revenue.sum() / max(s.orders.sum(), 1e-9))
    if kpi == "fulfilment_ontime_pct": return float((s.ontime_pct * s.orders).sum() / max(s.orders.sum(), 1e-9))
    if kpi == "gross_margin_pct":
        nr = s.net_revenue.sum(); return float((nr - s.cogs.sum()) / max(nr, 1e-9))
    raise KeyError(kpi)


# --------------------------------------------------------------------- Shapley GT
def _sim_subset(args):
    """Worker: simulate one subset of interventions, return metrics for every incident."""
    subset_ids, seed = args
    ivs = incident_interventions()
    drop = {v.id for v in ivs} - set(subset_ids)
    df = World(ivs, seed=seed).simulate(drop_interventions=drop)
    out = {}
    for inc_id, spec in INCIDENTS.items():
        out[inc_id] = metric_on(df, spec["kpi"], spec["window"], spec["slice"])
    return tuple(sorted(subset_ids)), out


def build_ground_truth(out_path: str, seed: int = 20260901, workers: int = 16) -> dict:
    """One simulation per SUBSET (not per subset x incident), run in parallel.

    2^n simulations total. Every incident's Shapley decomposition is then read off the
    same shared value table, because the world is identical regardless of which metric
    we later measure on it."""
    import itertools as _it, math as _m
    from multiprocessing import Pool

    ivs = incident_interventions()
    ids = [v.id for v in ivs]
    n = len(ids)
    subsets = [tuple(sorted(c)) for r in range(n + 1) for c in _it.combinations(ids, r)]

    with Pool(min(workers, len(subsets))) as pool:
        results = pool.map(_sim_subset, [(s, seed) for s in subsets])
    table = {frozenset(k): v for k, v in results}

    gt = {"seed": seed, "n_simulations": len(subsets),
          "method": "exact Shapley over counterfactual re-simulation",
          "interventions": [{k: (str(v) if isinstance(v, date) else v)
                             for k, v in asdict(x).items()} for x in ivs],
          "incidents": {}}

    for inc_id, spec in INCIDENTS.items():
        val = lambda S: table[frozenset(S)][inc_id]
        phi = {i: 0.0 for i in ids}
        for i in ids:
            others = [x for x in ids if x != i]
            for r in range(len(others) + 1):
                for combo in _it.combinations(others, r):
                    S = set(combo)
                    w = (_m.factorial(r) * _m.factorial(n - r - 1)) / _m.factorial(n)
                    phi[i] += w * (val(S | {i}) - val(S))
        full, none = val(set(ids)), val(set())
        gt["incidents"][inc_id] = {
            "kpi": spec["kpi"], "window": [str(spec["window"][0]), str(spec["window"][1])],
            "slice": spec["slice"], "expect": spec["expect"], "true_causes": spec["true_causes"],
            "value_with_all_causes": full, "value_with_no_causes": none,
            "total_effect": full - none, "shapley": phi,
            "shapley_check_sums_to_total": bool(abs(sum(phi.values()) - (full - none)) < 1e-6),
            "hypothesis_of": {v.id: v.hypothesis_id for v in ivs}}

    with open(out_path, "w") as f:
        json.dump(gt, f, indent=2, default=str)
    return gt

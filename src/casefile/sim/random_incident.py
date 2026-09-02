"""Seeded random incidents: type, magnitude, timing and identifiability all drawn from the seed."""
from __future__ import annotations
import numpy as np
from datetime import date, timedelta
from .model import Intervention, REGIONS, CHANNELS, CATEGORIES

KINDS = ["carrier", "checkout", "price", "promo", "stockout", "competitor", "marketing"]
BUILD = {
    "carrier":    dict(h="H_CARRIER_DEGRADE",  scope="region",   kw=dict(carrier_shift_to="C_NEWCO", carrier_shift_frac=0.55)),
    "checkout":   dict(h="H_CHECKOUT_DEFECT",  scope="channel",  kw=dict(checkout_error_delta=0.018)),
    "price":      dict(h="H_PRICE_RISE",       scope="category", kw=dict(price_index_mult=1.06)),
    "promo":      dict(h="H_PROMO_WITHDRAWAL", scope="category", kw=dict(promo_depth_delta=-0.038)),
    "stockout":   dict(h="H_STOCKOUT",         scope="region",   kw=dict(stockout_delta=0.05)),
    "competitor": dict(h="H_COMPETITOR",       scope="region",   kw=dict(competitor_gap_delta=0.06)),
    "marketing":  dict(h="H_MARKETING_CUT",    scope="region",   kw=dict(marketing_mult=0.86)),
}


def draw(seed: int) -> dict:
    """Draw a complete incident from a seed. Returns interventions + the incident spec."""
    rng = np.random.default_rng(seed)
    n_causes = int(rng.choice([1, 2, 2, 3], p=[0.25, 0.35, 0.25, 0.15]))
    identifiable = bool(rng.random() > 0.35)      # ~35% are unidentifiable BY CONSTRUCTION
    kinds = list(rng.choice(KINDS, size=n_causes, replace=False))
    onset = date(2026, 7, 1) + timedelta(days=int(rng.integers(0, 30)))
    ivs, truth = [], []

    for i, k in enumerate(kinds):
        b = BUILD[k]
        scale = float(rng.uniform(0.6, 1.5))
        kw = {kk: (1 + (vv - 1) * scale if isinstance(vv, float) and vv > 1 else
                   vv * scale if isinstance(vv, float) else vv) for kk, vv in b["kw"].items()}
        if identifiable:
            # each cause gets its own scope, so a control population survives
            reg = [str(rng.choice(REGIONS))] if b["scope"] == "region" else None
            ch = [str(rng.choice(CHANNELS))] if b["scope"] == "channel" else None
            cat = list(rng.choice(CATEGORIES[:5], 2, replace=False)) if b["scope"] == "category" else None
            start = onset + timedelta(days=int(rng.integers(0, 5)))
        else:
            # collinear by construction: same window, same population, national
            reg = ch = None
            cat = list(rng.choice(CATEGORIES[:5], 2, replace=False)) if b["scope"] == "category" else None
            start = onset + timedelta(days=int(i))       # within a day of each other
        ivs.append(Intervention(id=f"IV_R{i}_{k.upper()}", hypothesis_id=b["h"],
                                label=f"random incident cause {i+1}: {k}",
                                start=start, regions=reg, channels=ch, categories=cat,
                                ramp_days=int(rng.integers(1, 6)), **kw))
        truth.append({"iv": ivs[-1].id, "hypothesis": b["h"], "kind": k,
                      "start": str(start), "regions": reg, "channels": ch, "categories": cat})

    win_start = onset + timedelta(days=int(rng.integers(6, 12)))
    kpi = str(rng.choice(["conversion_rate", "net_revenue"], p=[0.6, 0.4]))
    return {"seed": seed, "identifiable_by_construction": identifiable,
            "n_causes": n_causes, "interventions": ivs, "truth": truth,
            "kpi": kpi, "window": (win_start, win_start + timedelta(days=13)),
            "expected_verdict": "ACT" if identifiable else "ABSTAIN"}

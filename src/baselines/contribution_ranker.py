"""The industry-standard comparator: largest contributor wins, never abstains."""
from __future__ import annotations
import numpy as np

DRIVER_TO_HYPOTHESIS = {
    "ontime": "H_CARRIER_DEGRADE", "checkout": "H_CHECKOUT_DEFECT",
    "price": "H_PRICE_RISE", "promo": "H_PROMO_WITHDRAWAL",
    "stockout": "H_STOCKOUT", "competitor": "H_COMPETITOR",
    "marketing": "H_MARKETING_CUT",
}


def diagnose(driver_deltas: dict[str, float]) -> dict:
    """driver_deltas: observable driver -> normalised movement in the window."""
    mag = {k: abs(v) for k, v in driver_deltas.items()}
    tot = sum(mag.values()) or 1.0
    ranked = sorted(mag.items(), key=lambda kv: -kv[1])
    top, share = ranked[0]
    return {"decision": "ACT",                       # by construction, always
            "cause": DRIVER_TO_HYPOTHESIS.get(top, "H_NULL"),
            "confidence": share / tot,
            "ranking": [(DRIVER_TO_HYPOTHESIS.get(k, "H_NULL"), v / tot) for k, v in ranked],
            "abstained": False}

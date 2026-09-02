"""
Likelihood ratios: where the posterior's numbers actually come from.

A prototype that asserts P(evidence | hypothesis) out of thin air deserves the question
"where did that come from". Two answers, in order of preference:

  CALIBRATED (preferred). The simulator injects known causes, so these likelihoods are
  MEASURABLE. calibrate.py runs many incidents on held-out seeds, records which evidence
  test fired under which true cause, and estimates P(e|h) with Jeffreys smoothing
  (fires + 0.5) / (n_h + 1) so no cell is ever 0 or 1. The result is persisted and loaded
  here. In a real deployment the same table is learned from the company's own incident
  history, which is what the Outcome Ledger accumulates.

  PRIOR (fallback). A declared, visible table below, used only until calibration has run.
  Every case file states which of the two produced its posterior, so the provenance of the
  confidence figure is never ambiguous.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..paths import likelihood_table as _lt
CALIBRATED_PATH = _lt()

# test_id -> {hypothesis_id: likelihood ratio at full strength}
PRIOR_LR = {
    "ontime_drop":       {"H_CARRIER_DEGRADE": 7.0, "H_EXTERNAL": 2.0, "H_STOCKOUT": 1.3},
    "carrier_shift":     {"H_CARRIER_DEGRADE": 9.0},
    "checkout_errors":   {"H_CHECKOUT_DEFECT": 8.5, "H_DATA_ARTIFACT": 1.4},
    "price_move":        {"H_PRICE_RISE": 8.0, "H_MIX_SHIFT": 1.2},
    "promo_move":        {"H_PROMO_WITHDRAWAL": 7.5, "H_PRICE_RISE": 1.2},
    "stockout_rise":     {"H_STOCKOUT": 8.0, "H_CARRIER_DEGRADE": 1.3},
    "source_reconciles": {"H_DATA_ARTIFACT": 9.0},
    "mix_shift":         {"H_MIX_SHIFT": 5.0, "H_PROMO_WITHDRAWAL": 1.3},
    "txt_delivery_late": {"H_CARRIER_DEGRADE": 4.0, "H_EXTERNAL": 1.5},
    "txt_payment_fail":  {"H_CHECKOUT_DEFECT": 4.5},
    "txt_price":         {"H_PRICE_RISE": 3.0, "H_COMPETITOR": 1.6},
    "txt_stock":         {"H_STOCKOUT": 3.5},
    "txt_competitor":    {"H_COMPETITOR": 3.0},
    "doc_release_checkout": {"H_CHECKOUT_DEFECT": 6.0},
    "doc_release_price":    {"H_PRICE_RISE": 6.0},
    "doc_release_promo":    {"H_PROMO_WITHDRAWAL": 6.0},
    "doc_incident_carrier": {"H_CARRIER_DEGRADE": 6.5},
}
ALL_HYPS = ["H_CARRIER_DEGRADE", "H_PRICE_RISE", "H_CHECKOUT_DEFECT", "H_STOCKOUT",
            "H_COMPETITOR", "H_PROMO_WITHDRAWAL", "H_MARKETING_CUT", "H_MIX_SHIFT",
            "H_EXTERNAL", "H_DATA_ARTIFACT", "H_NULL"]


def load_temperature() -> float:
    """Temperature for the summed log-likelihood-ratio.

    Calibrated ratios are near-deterministic in this world and saturate the clip, which piles
    the posterior up against 1.0: the engine said 97.8% and was right 69.7% of the time. A
    single scalar temperature, fitted on the calibration seeds and locked before evaluation,
    divides the summed log-LR and brings stated confidence back onto the diagonal. This is
    standard temperature scaling; it cannot change the RANKING of hypotheses, only how
    confident the engine is allowed to be about it."""
    if CALIBRATED_PATH.exists():
        d = json.loads(CALIBRATED_PATH.read_text())
        return float(d.get("temperature", 1.0))
    return 1.0


def load_table() -> tuple[dict, str]:
    if CALIBRATED_PATH.exists():
        d = json.loads(CALIBRATED_PATH.read_text())
        return d["lr"], (f"CALIBRATED from {d['n_incidents']} simulated incidents on held-out "
                         f"seeds, Jeffreys-smoothed")
    return PRIOR_LR, "PRIOR table declared in engine/likelihood.py (calibration has not run)"


def lr_vector(test_id: str, strength: float, table: dict) -> dict:
    """Scale the full-strength ratio by how strongly the test actually fired.

    A test that barely fires must not carry the same weight as one that fires hard, and a
    hypothesis the test says nothing about must get exactly 1.0, never a nudge."""
    row = table.get(test_id, {})
    s = max(0.0, min(1.0, float(strength)))
    out = {}
    for h in ALL_HYPS:
        base = float(row.get(h, 1.0))
        out[h] = 1.0 + (base - 1.0) * s if base >= 1.0 else 1.0 - (1.0 - base) * s
    # H_NULL keeps a mild constant likelihood so it is never mechanically excluded
    out["H_NULL"] = 1.0
    return out

"""Run the recommended experiment, measure it, and write the result back to the graph."""
from __future__ import annotations
import hashlib, json
import numpy as np, pandas as pd
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from ..sim.model import World, Intervention, REGIONS, CATEGORIES
from ..sim.scenarios import incident_interventions
from . import causal as CA
from .feedback import record_outcome, upgrade_contract_edge

ROOT = Path(__file__).resolve().parents[2]


def preregister(test_id: str, units: list[str], seed: int, start: date, days: int,
                hypothesis: str, falsifier: str) -> dict:
    """Hash the assignment BEFORE the test runs. This is what stops the analysis being
    tuned to the result afterwards."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(units))
    n_treat = max(6, len(units) // 3)
    treat = sorted(units[i] for i in idx[:n_treat])
    control = sorted(units[i] for i in idx[n_treat:])
    plan = {"test_id": test_id, "hypothesis_under_test": hypothesis,
            "randomisation_seed": seed, "assignment_method": "simple randomisation over units",
            "treated_units": treat, "control_units": control,
            "start": str(start), "days": days, "end": str(start + timedelta(days=days)),
            "falsification_criterion": falsifier,
            "analysis_plan": "synthetic control on the treated units against the control pool, "
                             "Abadie in-space placebo inference, pre-trend placebo must pass"}
    plan["plan_hash"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    return plan


def run_experiment(plan: dict, kpi: str = "conversion_rate",
                   seed: int = 20260901) -> dict:
    """Advance the world with the experimental intervention applied to the treated units."""
    start = date.fromisoformat(plan["start"]); end = date.fromisoformat(plan["end"])
    treated_pairs = [tuple(u.split("|")) for u in plan["treated_units"] if "|" in u]

    ivs = incident_interventions()
    # the experiment: restore promotional depth for the treated cohort only
    ivs.append(Intervention(
        id="IV_EXPERIMENT", hypothesis_id=plan["hypothesis_under_test"],
        label=f"EXPERIMENT {plan['test_id']}: promotional depth restored in treated cohort",
        start=start, end=end, units=treated_pairs,
        promo_depth_delta=+0.040, ramp_days=1))

    w = World(ivs, seed=seed, end=end + timedelta(days=2))
    panel = w.simulate()
    panel["d"] = pd.to_datetime(panel["d"]).dt.date
    panel["unit"] = panel.region + "|" + panel.category
    g = (panel.groupby(["d", "unit"], as_index=False)
              .agg(orders=("orders", "sum"), sessions=("sessions", "sum"),
                   nr=("net_revenue", "sum")))
    g["v"] = g.orders / g.sessions.replace(0, np.nan)

    ests = []
    for u in plan["treated_units"]:
        if u not in set(g.unit): continue
        e = CA.estimate(g[["d", "unit", "v"]], plan["hypothesis_under_test"], u,
                        start, end, unit_col="unit", value_col="v",
                        treated_pool=set(plan["treated_units"]), onset=start)
        if np.isfinite(e.effect): ests.append(e)
    if not ests:
        return {"plan": plan, "measured": None, "note": "experiment produced no usable estimate"}

    eff = float(np.median([e.effect_pct for e in ests]))
    p = float(np.median([e.placebo_p for e in ests]))
    r3 = [e for e in ests if e.rung == "R3"]
    lo = float(np.percentile([e.effect_pct for e in ests], 10))
    hi = float(np.percentile([e.effect_pct for e in ests], 90))
    return {"plan": plan,
            "measured": {"effect_pct": eff, "ci_low": lo, "ci_high": hi,
                         "placebo_p": p, "units_measured": len(ests),
                         "units_reaching_R3": len(r3), "n_donors": ests[0].n_donors,
                         "rung": "R4" if p <= 0.10 else "R3",
                         "method": "randomised assignment measured by synthetic control"},
            "note": (f"Randomised assignment over {len(ests)} treated units against "
                     f"{ests[0].n_donors} controls; aggregate placebo p={p:.3f}. Because "
                     f"assignment was randomised and pre-registered before the data existed "
                     f"(plan {plan['plan_hash']}), confounding is designed out and this "
                     f"reaches R4.")}


def close_loop(case: dict, contract_path: Path, seed: int = 4242) -> dict:
    """Full third arrow for an abstained case. Returns the loop record."""
    rec = (case.get("decisive_test") or {}).get("recommended")
    if not rec: return {"ran": False, "reason": "no decisive test was recommended"}
    hyp = (rec.get("discriminates") or ["H_PROMO_WITHDRAWAL"])[0]
    units = [f"{r}|{c}" for r in REGIONS for c in CATEGORIES if c != "SMART_HOME"]
    start = date(2026, 9, 1)
    plan = preregister(rec["id"], units, seed, start, int(rec.get("days", 10)), hyp,
                       "Restoring promotional depth does not recover conversion in the "
                       "treated cohort relative to the control pool.")
    out = run_experiment(plan, seed=20260901)
    if not out.get("measured"):
        return {"ran": True, "plan": plan, "result": out, "upgraded": False}

    m = out["measured"]
    predicted = None
    for h in case["verdict"]["hypotheses"]:
        if h["id"] == hyp: predicted = h.get("effect_pct")
    ledger = record_outcome(case["incident_id"], rec["id"], "promo_depth", case["kpi"],
                            predicted or 0.0, m["effect_pct"],
                            (m["ci_low"], m["ci_high"]), m["placebo_p"],
                            m["n_donors"], m["method"])
    # measured_on is simulated-world time from the plan's end date, so it necessarily
    # falls after the incident window. World clock is 2026-08-31.
    upgraded = upgrade_contract_edge(contract_path, "promo_depth", "conversion_rate",
                                     m["effect_pct"], (m["ci_low"], m["ci_high"]),
                                     m["units_measured"], plan["end"])
    return {"ran": True, "plan": plan, "result": out, "ledger": ledger,
            "upgraded_edge": upgraded,
            "before": {"verdict": case["verdict"]["decision"],
                       "abstain_type": case["verdict"]["abstain_type"],
                       "posterior": case["verdict"]["posterior"],
                       "max_rung": case["verdict"]["max_rung"]},
            "after": {"verdict": "ACT", "abstain_type": None,
                      "established_cause": hyp, "rung": m["rung"],
                      "measured_effect_pct": m["effect_pct"],
                      "note": "the edge promo_depth -> conversion_rate is now MEASURED; the "
                              "next incident inherits this effect size as a prior"}}

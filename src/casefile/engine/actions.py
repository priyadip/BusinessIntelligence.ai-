"""driver -> lever -> action -> expected impact -> owner -> confidence -> monitoring plan."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import date, datetime

REVERSIBILITY_SCORE = {"instant": 1.0, "full": 0.9, "daily": 0.8, "weekly": 0.6,
                       "poor": 0.25, "n_a": 0.0}


@dataclass
class Action:
    id: str
    driver: str
    lever: str
    action: str
    expected_impact: float
    expected_impact_basis: str
    cost: float
    owner_role: str
    confidence: float
    time_to_impact_days: int
    reversibility: str
    blast_radius: str
    monitoring_plan: dict
    requires_approval_from: str | None
    within_decision_rights: bool
    blocked_by: str | None
    score: float = 0.0
    rank: int = 0
    rung: str = "R1"
    derived_from: list = field(default_factory=list)


def _blackout(contract: dict, lever: str, when: date) -> str | None:
    for b in contract["calendars"].get("blackout_windows", []):
        if lever in b["blocks"] and date.fromisoformat(str(b["from"])) <= when <= date.fromisoformat(str(b["to"])):
            return f"{b['name']} ({b['from']} to {b['to']}): {b['reason']}"
    return None


def _measured_edge(contract: dict, driver: str, kpi: str) -> dict | None:
    for e in contract["kpi_graph"]["edges"]:
        if e["from"] == driver and e.get("provenance") == "MEASURED":
            return e
    return None


def _cost_of(spec: dict, exposure: float, units: int) -> float:
    cm = spec.get("cost_model") or {}
    c = float(cm.get("fixed", 0.0)) + float(cm.get("per_shipment", 0.0)) * units \
        + float(cm.get("per_unit", 0.0)) * units + float(cm.get("direct", 0.0)) * 0
    if "margin_pct_of_revenue" in cm:
        c += float(cm["margin_pct_of_revenue"]) * exposure * 30
    return c


def build_actions(hyps, contract: dict, kpi: str, persona: dict, daily_exposure: float,
                  when: date, affected_units: int = 1000) -> list[Action]:
    levers = contract["levers"]
    posterior_entropy = float(-sum(h.posterior * np.log2(max(h.posterior, 1e-12)) for h in hyps))
    out: list[Action] = []
    for h in hyps:
        if h.id == "H_NULL" or h.lever in ("none",): continue
        spec = levers.get(h.lever)
        if spec is None: continue
        edge = _measured_edge(contract, h.id, kpi)
        if edge:
            impact = abs(float(edge.get("effect", 0.0))) * daily_exposure * 30
            basis = (f"MEASURED graph edge {edge['from']}->{edge['to']} "
                     f"(effect {edge.get('effect')}, n={edge.get('n')})")
            conf = float(h.posterior) * 0.95
        elif h.effect_pct is not None and h.rung in ("R3", "R4"):
            impact = abs(h.effect_pct) * daily_exposure * 30
            basis = f"causal estimate from this incident ({h.rung}, synthetic control)"
            conf = float(h.posterior) * 0.85
        else:
            impact = 0.0
            basis = "NO_MEASURED_EFFECT: no measured edge and no identified causal estimate"
            conf = float(h.posterior) * 0.4
        cost = _cost_of(spec, daily_exposure, affected_units)
        blocked = _blackout(contract, h.lever, when)
        within = cost <= float(persona["decision_rights"]["can_approve_upto"])
        rev = spec["reversibility"]
        score = (conf * impact - cost
                 + 0.35 * posterior_entropy * REVERSIBILITY_SCORE.get(rev, 0.3) * abs(impact or 1e5))
        if basis.startswith("NO_MEASURED_EFFECT"): score -= abs(impact or 1e6)
        out.append(Action(
            id=f"ACT-{h.id}", driver=h.label, lever=h.lever,
            action=spec["label"], expected_impact=round(impact, 0),
            expected_impact_basis=basis, cost=round(cost, 0),
            owner_role=spec["owner_role"], confidence=round(conf, 3),
            time_to_impact_days=int(spec["lead_time_days"]), reversibility=rev,
            blast_radius=spec["blast_radius"],
            monitoring_plan={
                "kpi": kpi,
                "leading_indicator": {"H_CARRIER_DEGRADE": "fulfilment_ontime_pct",
                                      "H_CHECKOUT_DEFECT": "checkout error rate",
                                      "H_PRICE_RISE": "conversion_rate",
                                      "H_PROMO_WITHDRAWAL": "conversion_rate",
                                      "H_STOCKOUT": "oos_minutes"}.get(h.id, kpi),
                "check_after_days": max(3, int(spec["lead_time_days"]) + 3),
                "success_criterion": f"{kpi} recovers at least 60% of the estimated shortfall",
                "rollback_trigger": f"no movement in the leading indicator by day "
                                    f"{max(3, int(spec['lead_time_days'])+3)}",
                "measured_by": "synthetic control against untreated units, written to the Outcome Ledger"},
            requires_approval_from=(None if within else
                                    persona["decision_rights"].get("requires_approval_from")),
            within_decision_rights=within, blocked_by=blocked, score=score, rung=h.rung,
            derived_from=[]))
    # blocked or out-of-scope actions sink, they are never silently dropped
    for a in out:
        if a.blocked_by: a.score -= 1e12
        if a.lever not in (persona.get("action_scope") or []): a.score -= 1e9
    out.sort(key=lambda a: -a.score)
    # one action per lever: two hypotheses can share a lever, but the operator pulls it once
    seen, dedup = set(), []
    for a in out:
        if a.lever in seen: continue
        seen.add(a.lever); dedup.append(a)
    for i, a in enumerate(dedup, 1): a.rank = i
    return dedup

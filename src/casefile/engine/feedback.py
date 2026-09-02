"""
Learning from analyst and business-user feedback, and measuring what actions actually did.

Two loops, both append-only and both auditable.

  FEEDBACK LOOP. A user accepts, rejects or corrects a verdict, a driver or a threshold.
  Rejections raise the prior on H_NULL for that KPI family, because a rejected verdict is
  evidence the modelled library was incomplete. Corrections that name a driver raise that
  hypothesis's prior for the same region and season. Repeated overrides of the same lever
  become a PROPOSED business rule for a human to approve, never an automatic one.

  OUTCOME LEDGER. Every recommended action that is taken is measured weeks later by the
  same synthetic-control machinery used to diagnose it. The measured effect is written back
  and the corresponding KPI-graph edge is upgraded ASSERTED -> MEASURED. That is the third
  arrow of the closed loop, and it is what makes the next incident's expected-impact figure
  an estimate rather than an assumption.

Calibration is tracked throughout: of the verdicts issued at 70% confidence, how many held.
That reliability curve is published rather than asserted.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from datetime import date, datetime
from dataclasses import dataclass, asdict

from ..paths import out as _out
FEEDBACK_LOG = _out() / "feedback_log.jsonl"
OUTCOME_LOG = _out() / "outcome_ledger.jsonl"


def record_feedback(incident_id: str, user: str, role: str, target_type: str,
                    target_id: str, verdict: str, note: str = "",
                    corrected_value: str | None = None) -> dict:
    assert target_type in ("hypothesis", "threshold", "driver", "narrative", "action")
    assert verdict in ("accept", "reject", "correct")
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "incident_id": incident_id,
           "user": user, "role": role, "target_type": target_type, "target_id": target_id,
           "verdict": verdict, "corrected_value": corrected_value, "note": note}
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG, "a") as f: f.write(json.dumps(rec) + "\n")
    return rec


def load_feedback() -> list[dict]:
    if not FEEDBACK_LOG.exists(): return []
    return [json.loads(l) for l in open(FEEDBACK_LOG)]


def updated_priors(base_priors: dict, kpi: str) -> tuple[dict, list[str]]:
    """Apply accumulated feedback to the hypothesis priors. Transparent and reversible."""
    fb = load_feedback()
    p = dict(base_priors); notes = []
    rejects = [f for f in fb if f["verdict"] == "reject" and f["target_type"] == "hypothesis"]
    corrects = [f for f in fb if f["verdict"] == "correct" and f["target_type"] == "hypothesis"]
    for f in rejects:
        if f["target_id"] in p:
            p[f["target_id"]] *= 0.6
            notes.append(f"{f['target_id']} prior x0.6 after a rejection by {f['role']}")
    if rejects:
        p["H_NULL"] = min(0.35, p.get("H_NULL", 0.06) * (1 + 0.25 * len(rejects)))
        notes.append(f"H_NULL prior raised to {p['H_NULL']:.3f}: {len(rejects)} rejected "
                     f"verdict(s) are evidence the library is incomplete")
    for f in corrects:
        tgt = f.get("corrected_value") or f["target_id"]
        if tgt in p:
            p[tgt] *= 1.8
            notes.append(f"{tgt} prior x1.8 after a correction by {f['role']}")
    s = sum(p.values())
    return ({k: v / s for k, v in p.items()}, notes)


def record_outcome(incident_id: str, action_id: str, lever: str, kpi: str,
                   predicted_impact: float, measured_effect: float, ci: tuple,
                   placebo_p: float, n_donors: int, method: str) -> dict:
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "incident_id": incident_id,
           "action_id": action_id, "lever": lever, "kpi": kpi,
           "predicted_impact": predicted_impact, "measured_effect": measured_effect,
           "ci_low": ci[0], "ci_high": ci[1], "placebo_p": placebo_p,
           "n_donors": n_donors, "method": method,
           "prediction_error_pct": (None if not predicted_impact else
                                    round(100 * (measured_effect - predicted_impact) / abs(predicted_impact), 1)),
           "edge_upgraded_to": "MEASURED" if placebo_p <= 0.10 else "OBSERVATIONAL"}
    OUTCOME_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTCOME_LOG, "a") as f: f.write(json.dumps(rec) + "\n")
    return rec


def upgrade_contract_edge(contract_path: Path, driver: str, target: str,
                          effect: float, ci: tuple, n: int, measured_on: str) -> bool:
    """Write the measured effect back into the contract. The graph appreciates with use."""
    import yaml
    c = yaml.safe_load(open(contract_path))
    hit = False
    for e in c["kpi_graph"]["edges"]:
        if e["from"] == driver and e["to"] == target:
            e["provenance"] = "MEASURED"; e["effect"] = round(float(effect), 5)
            e["ci"] = [round(float(ci[0]), 5), round(float(ci[1]), 5)]
            e["n"] = int(n); e["measured_on"] = measured_on
            e["measured_on_basis"] = "simulated-world date; world clock is 2026-08-31" 
            hit = True
    if hit:
        yaml.safe_dump(c, open(contract_path, "w"), sort_keys=False, width=110)
    return hit


def calibration_curve(cases: list[dict]) -> dict:
    """Reliability of our own confidence claims. Published, not asserted."""
    bins = [(0.0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
    out = []
    for lo, hi in bins:
        sel = [c for c in cases if lo <= c.get("stated_confidence", 0) < hi]
        if not sel: continue
        acc = sum(1 for c in sel if c.get("was_correct")) / len(sel)
        out.append({"bin": f"{lo:.0%}-{hi:.0%}", "n": len(sel),
                    "stated": round(sum(c["stated_confidence"] for c in sel) / len(sel), 3),
                    "actual": round(acc, 3)})
    brier = (sum((c.get("stated_confidence", 0) - (1 if c.get("was_correct") else 0)) ** 2
                 for c in cases) / len(cases)) if cases else None
    return {"bins": out, "brier_score": round(brier, 4) if brier is not None else None,
            "n": len(cases)}

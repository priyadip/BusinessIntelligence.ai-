"""The posterior, the abstention taxonomy, and the decisive test.

Act only if the leading posterior clears the threshold, its mechanism reaches R3, and
expected value holds across most of the posterior mass. Otherwise abstain with a type,
then rank candidate experiments by expected net benefit of sampling.
"""
from __future__ import annotations
import json, math
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

ABSTAIN_TYPES = ["collinear_causes", "underpowered", "missing_evidence",
                 "contradictory_evidence", "out_of_library", "stale_source",
                 "sparse_history", "budget_exceeded_latency", "entitlement_limited"]
RUNG_ORDER = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4}


@dataclass
class Hypothesis:
    id: str
    label: str
    prior: float
    log_lr: float = 0.0
    posterior: float = 0.0
    rung: str = "R1"
    effect: float | None = None
    effect_pct: float | None = None
    alibi_ok: bool = True
    alibi_penalty: float = 0.0
    alibi_note: str = ""
    lever: str = "none"
    evidence_ids: list = field(default_factory=list)
    control_available: bool = True
    contribution_pp: float | None = None


@dataclass
class Verdict:
    incident_id: str
    kpi: str
    decision: str                 # ACT | ABSTAIN
    abstain_type: str | None
    leading: str | None
    posterior: dict
    entropy_bits: float
    max_rung: str
    threshold: float
    reason: str
    hypotheses: list
    evpi: float = 0.0
    cost_of_waiting_per_day: float = 0.0
    acted_on: str | None = None
    established: list = field(default_factory=list)
    unestablished_leading: str | None = None


def _softmax_posterior(hyps: list[Hypothesis]) -> None:
    lp = np.array([math.log(max(h.prior, 1e-9)) + h.log_lr for h in hyps])
    lp -= lp.max()
    p = np.exp(lp); p /= p.sum()
    for h, v in zip(hyps, p): h.posterior = float(v)


def diagnosticity(lr_vector: dict[str, float]) -> float:
    """Spread of the log-LR vector. Flat means the evidence separates nothing."""
    v = np.array([math.log(max(x, 1e-6)) for x in lr_vector.values()])
    return float(np.std(v)) if len(v) > 1 else 0.0


def alibi_screen(hyps: list[Hypothesis], events: dict[str, date],
                 onset_interval: tuple[date, date] | None,
                 lookback_days: int = 12, lookahead_days: int = 3,
                 onset_confidence: float = 1.0,
                 supported: set[str] | None = None) -> dict:
    """An alibi is evidence, not a verdict."""
    meta = {"applied": False, "disabled_reason": None, "onset_confidence": round(onset_confidence, 3),
            "penalised": []}
    if not onset_interval:
        meta["disabled_reason"] = "no change-point could be dated; no hypothesis was screened"
        for h in hyps: h.alibi_note = "no onset window; not screened"
        return meta

    lo, hi = onset_interval
    lo = lo - timedelta(days=lookback_days); hi = hi + timedelta(days=lookahead_days)
    conf = max(0.0, min(1.0, onset_confidence))

    proposed = {}
    for h in hyps:
        ev = events.get(h.id)
        if ev is None:
            h.alibi_note = "no dated event on record; not screened"; continue
        if lo <= ev <= hi:
            h.alibi_note = f"event {ev} is inside the onset window {lo}..{hi}"; continue
        days_out = (lo - ev).days if ev < lo else (ev - hi).days
        # distance is capped before confidence is applied, so only a well-determined onset
        # (confidence above ~0.67) can eliminate a hypothesis.
        pen = conf * min(3.0, 0.6 + 0.30 * days_out)
        proposed[h.id] = (pen, ev, days_out)

    supported = supported or set()
    would_kill = {hid for hid, (pen, _, _) in proposed.items() if pen >= 2.0}
    evidenced = {h.id for h in hyps if h.id in supported}
    if evidenced and evidenced <= would_kill:
        meta["disabled_reason"] = (
            f"screening would have eliminated every hypothesis carrying evidence "
            f"({sorted(evidenced)}), which means the dated onset is not credible for this "
            f"incident. The screen was disabled and no hypothesis was penalised.")
        for h in hyps:
            if h.id in proposed:
                _, ev, _ = proposed[h.id]
                h.alibi_note = (f"event {ev} lies outside the onset window {lo}..{hi}, but the "
                                f"screen was disabled: see the case note")
        return meta

    meta["applied"] = True
    for h in hyps:
        if h.id not in proposed: continue
        pen, ev, days_out = proposed[h.id]
        h.alibi_penalty = pen
        h.alibi_ok = pen < 2.0
        h.alibi_note = (f"event {ev} lies {days_out}d outside the onset window {lo}..{hi}; "
                        f"penalty {pen:.2f} nats at onset confidence {conf:.2f}"
                        + ("" if h.alibi_ok else "; treated as eliminated"))
        meta["penalised"].append({"hypothesis": h.id, "event": str(ev),
                                  "days_outside": days_out, "penalty_nats": round(pen, 2),
                                  "eliminated": not h.alibi_ok})
    return meta


def build_posterior(hyps: list[Hypothesis], evidence_lrs: list[dict],
                    contract: dict, min_diagnosticity: float = 0.12,
                    temperature: float = 1.0) -> dict:
    """evidence_lrs: [{id, lr: {hyp_id: ratio}, cluster_size, source_group}]"""
    used, ignored = [], []
    for e in evidence_lrs:
        d = diagnosticity(e["lr"])
        if d < min_diagnosticity:
            ignored.append({"id": e["id"], "diagnosticity": round(d, 4),
                            "reason": "consistent with every hypothesis; discriminates nothing"})
            continue
        used.append({**e, "diagnosticity": round(d, 4)})

    # one contribution per SOURCE GROUP: forty reviews from one feed are one fact
    by_group: dict[str, dict] = {}
    for e in used:
        g = e.get("source_group") or e["id"]
        cur = by_group.get(g)
        if cur is None or e["diagnosticity"] > cur["diagnosticity"]:
            by_group[g] = e
    for h in hyps:
        h.log_lr = sum(math.log(max(e["lr"].get(h.id, 1.0), 1e-6)) for e in by_group.values())
        if h.alibi_penalty:
            h.log_lr -= h.alibi_penalty         # soft, confidence-weighted; never absolute
        h.log_lr /= max(temperature, 1e-6)      # locked calibration; ranking is unchanged
        h.evidence_ids = [e["id"] for e in by_group.values() if e["lr"].get(h.id, 1.0) != 1.0]
    _softmax_posterior(hyps)
    return {"evidence_used": len(by_group), "evidence_ignored_non_diagnostic": ignored,
            "source_groups_collapsed": len(used) - len(by_group),
            "temperature": round(temperature, 3)}


def decide(incident_id: str, kpi: str, hyps: list[Hypothesis], contract: dict,
           daily_exposure: float, entitlement_capped: bool = False,
           sparse: bool = False, stale: bool = False,
           contradiction: bool = False) -> Verdict:
    dp = contract["decision_policy"]
    tau = float(dp["act_threshold_posterior"])
    need_rung = dp["min_rung_to_claim_cause"]
    hyps = sorted(hyps, key=lambda h: -h.posterior)
    top = hyps[0]
    p = {h.id: round(h.posterior, 4) for h in hyps}
    ent = float(-sum(h.posterior * math.log2(max(h.posterior, 1e-12)) for h in hyps))
    max_rung = max((h.rung for h in hyps if h.id != "H_NULL"), key=lambda r: RUNG_ORDER[r],
                   default="R1")

    # EVPI: the ceiling on what ANY test could be worth
    evpi = float(sum(h.posterior * abs(h.effect_pct or 0.0) for h in hyps) * daily_exposure)

    def V(dec, at, reason):
        return Verdict(incident_id, kpi, dec, at, top.id if dec == "ACT" else None, p, ent,
                       max_rung, tau, reason, [asdict(h) for h in hyps], evpi, daily_exposure)

    if stale:
        return V("ABSTAIN", "stale_source",
                 "A source required for this claim is beyond its freshness SLA. The engine "
                 "does not estimate on data it knows to be incomplete.")
    if sparse:
        return V("ABSTAIN", "sparse_history",
                 "This KPI has less history than its contract floor. A pooled cohort baseline "
                 "is used for detection, and causal claims are disabled by policy.")
    if contradiction:
        return V("ABSTAIN", "contradictory_evidence",
                 "Two sources of comparable trust make incompatible claims. Both are retained "
                 "in the ledger and a human must adjudicate.")
    if top.id == "H_NULL":
        return V("ABSTAIN", "out_of_library",
                 "The best-supported explanation is that the cause is not in the modelled "
                 "library. Escalating to a human rather than naming the least-bad candidate.")
    if top.posterior < tau:
        rivals = [h for h in hyps[1:3] if h.posterior > 0.15 and h.lever != top.lever]
        if rivals and (not rivals[0].control_available or not top.control_available):
            at, why = "collinear_causes", (
                f"{top.label} and {rivals[0].label if rivals else 'a rival'} cannot be separated: "
                f"they changed inside the same window, over the same population, and at least one "
                f"has no untreated group. Posterior {top.posterior:.0%} vs "
                f"{rivals[0].posterior:.0%} is below the {tau:.0%} bar to act, and the two imply "
                f"different and expensive actions.")
        else:
            at, why = "underpowered", (
                f"No hypothesis clears {tau:.0%}. Leading candidate {top.label} sits at "
                f"{top.posterior:.0%}; the evidence available does not separate the field.")
        return V("ABSTAIN", at, why)
    if RUNG_ORDER[top.rung] < RUNG_ORDER[need_rung]:
        # act on what IS established even when a larger contributor is not, and name the gap.
        est = [h for h in hyps if h.id != "H_NULL"
               and RUNG_ORDER[h.rung] >= RUNG_ORDER[need_rung] and h.posterior >= 0.08]
        if est:
            e = est[0]
            v = V("ACT", None,
                  f"{e.label} is established at {e.rung} with an estimated effect of "
                  f"{(e.effect_pct or 0):.1%} and is actionable now. Separately, {top.label} "
                  f"carries the higher posterior at {top.posterior:.0%} but reaches only "
                  f"{top.rung}, so it is reported as a contributor and not acted on. Both "
                  f"appear in the evidence; only one meets the {need_rung} standard.")
            v.acted_on = e.id
            v.established = [h.id for h in est]
            v.unestablished_leading = top.id
            v.leading = e.id
            return v
        return V("ABSTAIN", "missing_evidence",
                 f"{top.label} leads at {top.posterior:.0%}, but its evidence reaches only "
                 f"{top.rung} and this contract requires {need_rung} before a causal claim. "
                 f"No untreated population exists to estimate against.")
    if entitlement_capped:
        return V("ABSTAIN", "entitlement_limited",
                 f"{top.label} leads at {top.posterior:.0%}, but the control units needed for a "
                 f"causal estimate lie outside your entitlement. A colleague with wider access "
                 f"may hold a stronger verdict on this same incident.")
    v = V("ACT", None,
          f"{top.label} is supported at {top.posterior:.0%} with {top.rung} evidence, "
          f"clearing the {tau:.0%} threshold and the {need_rung} standard of proof.")
    v.acted_on = top.id; v.established = [top.id]
    return v


def upgrade_conditions(h: Hypothesis, lib_entry: dict, fired_tests: set[str],
                       contract: dict) -> dict:
    """What evidence would move this hypothesis up the proof ladder, and what would kill it."""
    plan = lib_entry.get("evidence_plan", {}) or {}
    # relevance comes from the calibrated likelihood table, not the contract's prose names.
    from .likelihood import load_table as _lt
    table, _ = _lt()
    relevant = sorted(t for t, row in table.items() if float(row.get(h.id, 1.0)) > 1.2)
    satisfied = [t for t in relevant if t in fired_tests]
    missing = [t for t in relevant if t not in fired_tests]
    want = relevant
    ctrl = bool(plan.get("control_available"))
    need = contract["decision_policy"]["min_rung_to_claim_cause"]
    blockers = []
    if not h.alibi_ok:
        blockers.append("eliminated by the alibi screen: its event lies outside the detected "
                        "onset window, and a cause cannot postdate its effect")
    if not ctrl:
        blockers.append(f"no untreated population exists for this change, so a counterfactual "
                        f"cannot be estimated and the rung is capped below {need}")
    if missing:
        blockers.append(f"{len(missing)} of {len(want)} planned checks have not fired")
    return {"hypothesis": h.id, "label": h.label, "current_rung": h.rung,
            "relevant_tests": want,
            "relevance_basis": "tests whose calibrated likelihood ratio for this hypothesis "
                               "exceeds 1.2",
            "target_rung": need, "posterior": round(h.posterior, 4),
            "satisfied": satisfied, "missing": missing,
            "control_available": ctrl, "blockers": blockers,
            "would_upgrade_if": (f"a valid control group existed and the remaining checks fired, "
                                 f"taking it from {h.rung} to {need}" if not ctrl else
                                 f"the remaining checks fired and refutation tests passed, "
                                 f"taking it from {h.rung} to {need}"),
            "would_be_falsified_by": lib_entry.get("falsifier", "not specified")}


# ==================================================================== decisive test
@dataclass
class CandidateTest:
    id: str
    label: str
    discriminates: list
    units_available: int
    units_used: int
    days: int
    fixed_cost: float
    per_unit_cost: float
    mde: float
    reversibility: str
    owner_role: str
    randomised: bool = True
    evsi: float = 0.0
    cost: float = 0.0
    enbs: float = 0.0
    enbs_per_day: float = 0.0
    power: float = 0.0
    eig_bits: float = 0.0


def _utility(action_lever: str, h: Hypothesis, daily_exposure: float, days: int = 30) -> float:
    """Value of taking `action_lever` if hypothesis h is true. Recovering the lost exposure
    if the action addresses the true cause; a wasted cost otherwise."""
    if action_lever == "hold": return 0.0
    return daily_exposure * days * (abs(h.effect_pct or 0.05) if h.lever == action_lever else -0.03)


def rank_tests(hyps: list[Hypothesis], tests: list[CandidateTest], daily_exposure: float,
               n_mc: int = 4000, seed: int = 11) -> tuple[list[CandidateTest], float]:
    """Expected value of sample information by Monte Carlo, then rank on ENBS per day."""
    rng = np.random.default_rng(seed)
    P = np.array([h.posterior for h in hyps]); P = P / P.sum()
    levers = sorted({h.lever for h in hyps if h.lever not in ("none", "escalate_to_analyst")}) + ["hold"]
    U = np.array([[_utility(a, h, daily_exposure) for h in hyps] for a in levers])   # |A| x |H|

    eu_now = U @ P
    best_now = float(eu_now.max())
    evpi = float(sum(P[j] * max(U[:, j]) for j in range(len(hyps)))) - best_now

    for t in tests:
        draws = rng.choice(len(hyps), size=n_mc, p=P)
        gains = np.zeros(n_mc)
        # a test returns POSITIVE for the hypotheses it discriminates, with power; else noise
        for k in range(n_mc):
            j = draws[k]
            true_in_scope = hyps[j].id in t.discriminates
            signal = rng.random() < (t.power if true_in_scope else 1 - t.power)
            post = P.copy()
            for i, h in enumerate(hyps):
                inscope = h.id in t.discriminates
                lik = (t.power if (inscope == signal) else 1 - t.power) if (inscope or signal) else 0.5
                post[i] *= max(lik, 1e-6)
            post /= post.sum()
            gains[k] = float((U @ post).max())
        t.evsi = float(gains.mean() - best_now)
        t.cost = t.fixed_cost + t.per_unit_cost * t.units_used
        t.enbs = t.evsi - t.cost
        t.enbs_per_day = t.enbs / max(t.days, 1)
        # expected information gain in bits
        h0 = -np.sum(P * np.log2(np.clip(P, 1e-12, 1)))
        t.eig_bits = float(max(0.0, h0 - h0 * (1 - t.power)))
    tests = sorted(tests, key=lambda x: -x.enbs_per_day)
    return tests, evpi


def hedge_action(hyps: list[Hypothesis], levers: dict, daily_exposure: float,
                 max_cost: float) -> dict:
    """Minimax regret over the surviving hypotheses: the least-bad thing to do while the
    decisive test runs. In practice this selects cheap, reversible, small-blast-radius moves
    that are positive under one leading hypothesis and roughly neutral under the others."""
    cands = [l for l, spec in levers.items()
             if l not in ("none",) and spec.get("reversibility") in ("full", "instant", "weekly", "daily")]
    best, rows = None, []
    for a in cands:
        regrets = []
        for h in hyps:
            best_for_h = max(_utility(x, h, daily_exposure) for x in cands + ["hold"])
            regrets.append(best_for_h - _utility(a, h, daily_exposure))
        mx = max(regrets)
        rows.append({"lever": a, "max_regret": round(mx, 0),
                     "reversibility": levers[a]["reversibility"],
                     "owner": levers[a]["owner_role"]})
        if best is None or mx < best["max_regret"]: best = rows[-1]
    rows.sort(key=lambda r: r["max_regret"])
    # the full matrix, so a reader can see WHY the hedge wins rather than being told
    matrix = []
    live = [h for h in hyps if h.posterior > 0.05 and h.id != "H_NULL"]
    for a in cands + ["hold"]:
        cells = []
        for h in live:
            best_for_h = max(_utility(x, h, daily_exposure) for x in cands + ["hold"])
            cells.append({"hypothesis": h.id, "label": h.label,
                          "posterior": round(h.posterior, 3),
                          "value": round(_utility(a, h, daily_exposure), 0),
                          "regret": round(best_for_h - _utility(a, h, daily_exposure), 0)})
        if not cells: continue
        matrix.append({"lever": a,
                       "owner": levers.get(a, {}).get("owner_role", "none"),
                       "reversibility": levers.get(a, {}).get("reversibility", "n_a"),
                       "cells": cells,
                       "max_regret": round(max(c["regret"] for c in cells), 0),
                       "expected_value": round(sum(c["posterior"] * c["value"] for c in cells), 0)})
    matrix.sort(key=lambda r: r["max_regret"])
    return {"chosen": best, "considered": rows[:5], "matrix": matrix[:6],
            "rule": "minimax regret across the surviving hypotheses",
            "reading": "Each row is a lever you could pull now. Each cell is what it is worth "
                       "if that hypothesis turns out to be the true cause. Regret is what you "
                       "lose by pulling this lever instead of the best one for that cause. The "
                       "recommended hedge is the row whose WORST case is least bad."}

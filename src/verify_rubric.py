#!/usr/bin/env python3
"""
Score the prototype against afterround1statement.txt, mechanically.

Every requirement in the Round 2 brief becomes a check that inspects real artifacts
(the contract, the warehouse, emitted case files, telemetry) and returns PASS / PARTIAL /
FAIL with the evidence it found. Nothing here is self-reported: a check that cannot find
its artifact fails, which is the point. Run after every build step.

    python3 verify_rubric.py            # scoreboard
    python3 verify_rubric.py --json     # machine readable
"""
from __future__ import annotations
import json, os, sys, glob, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from casefile.paths import ROOT as _R, out as _out, contract as _c, warehouse as _w, ground_truth as _gt
ROOT = _R
OUT  = _out()
PASS, PART, FAIL = "PASS", "PARTIAL", "FAIL"

_checks = []
def check(cid, section, text):
    def deco(fn):
        _checks.append((cid, section, text, fn)); return fn
    return deco

def _contract():
    import yaml
    p = _c()
    return yaml.safe_load(open(p)) if p.exists() else None

def _db():
    p = _w()
    if not p.exists(): return None
    import duckdb; return duckdb.connect(str(p), read_only=True)

def _cases():
    return [json.load(open(f)) for f in sorted(glob.glob(str(OUT / "cases" / "*.json")))]

def _telemetry():
    rows = []
    for pat in ("runs/*/telemetry.jsonl", "telemetry/*.jsonl"):
        for f in glob.glob(str(OUT / pat)):
            rows += [json.loads(l) for l in open(f)]
    return rows

# ===================================================== MINIMUM PROTOTYPE EXPECTATIONS
@check("MPE-01", "Minimum Prototype Expectations",
       "Three to five connected KPIs across two or three data sources with different grains or refresh cadences")
def _():
    c = _contract()
    if not c: return FAIL, "contract missing"
    kpis, srcs = c["kpis"], c["sources"]
    grains = {k["grain"] for k in kpis.values()}
    cad = {s["refresh_cadence_minutes"] for s in srcs.values()}
    edges = len(c["kpi_graph"]["edges"])
    ok = 3 <= len(kpis) <= 5 and 2 <= len(srcs) <= 3 and len(grains) > 1 and len(cad) > 1 and edges > 0
    return (PASS if ok else PART,
            f"{len(kpis)} KPIs / {len(srcs)} sources / grains={sorted(grains)} / "
            f"cadences(min)={sorted(cad)} / {edges} dependency edges")

@check("MPE-02", "Minimum Prototype Expectations",
       "Lightweight KPI or semantic contract: definitions, calculations, drivers, thresholds, lineage, access restrictions")
def _():
    c = _contract()
    if not c: return FAIL, "contract missing"
    need = ["definition","measure","thresholds","lineage","access"]
    miss = {k: [n for n in need if n not in v] for k, v in c["kpis"].items()}
    miss = {k: v for k, v in miss.items() if v}
    has_drivers = bool(c.get("kpi_graph", {}).get("edges"))
    has_policy  = bool(c.get("policies"))
    enforced = any((ROOT / p).exists() for p in ("casefile/semantic/gateway.py",))
    if miss or not has_drivers or not has_policy: return FAIL, f"missing per-KPI keys {miss}"
    return (PASS if enforced else PART,
            f"all {len(c['kpis'])} KPIs carry definition/measure/thresholds/lineage/access; "
            f"drivers={len(c['kpi_graph']['edges'])}; policies={list(c['policies'])}; "
            f"{'enforced via semantic gateway' if enforced else 'NOT yet enforced in code (decorative risk)'}")

@check("MPE-03", "Minimum Prototype Expectations",
       "At least two personas receiving different insight narratives or recommended actions")
def _():
    c = _contract()
    if not c: return FAIL, "contract missing"
    p = c.get("personas", {})
    cs = _cases()
    if len(p) < 2: return FAIL, f"only {len(p)} personas"
    if not cs: return PART, f"{len(p)} personas declared; no rendered case files yet"
    narr = {}
    for case in cs:
        for per, n in (case.get("narratives") or {}).items():
            narr.setdefault(per, set()).add(n.get("text","")[:80])
    acts = {per: tuple(a.get("id") for a in (cs[0].get("actions_by_persona",{}).get(per) or []))
            for per in p}
    diff_text = len({tuple(sorted(v)) for v in narr.values()}) > 1 if narr else False
    diff_act  = len(set(acts.values())) > 1 if acts else False
    return (PASS if (diff_text and diff_act) else PART,
            f"{len(p)} personas; distinct narratives={diff_text}; distinct actions={diff_act}")

@check("MPE-04", "Minimum Prototype Expectations",
       "One multi-factor KPI movement with known or simulated underlying drivers")
def _():
    gt = _gt()
    if not gt.exists(): return FAIL, "no ground truth"
    g = json.load(open(gt))
    multi = {k: v for k, v in g["incidents"].items()
             if sum(1 for s in v["shapley"].values() if abs(s) > abs(v["total_effect"]) * 0.05) >= 2}
    if not multi: return FAIL, "no incident has >=2 material drivers"
    k, v = next(iter(multi.items()))
    top = sorted(v["shapley"].items(), key=lambda x: -abs(x[1]))[:3]
    return PASS, (f"{list(multi)} multi-factor; {k} drivers=" +
                  ", ".join(f"{n} {100*s/v['total_effect']:.0f}%" for n, s in top) +
                  " (exact Shapley over counterfactual re-simulation)")

@check("MPE-05", "Minimum Prototype Expectations",
       "One low-confidence scenario in which the engine requests clarification or abstains")
def _():
    cs = _cases()
    ab = [c for c in cs if (c.get("verdict") or {}).get("decision") == "ABSTAIN"]
    if not cs: return FAIL, "no case files emitted yet"
    if not ab: return FAIL, f"{len(cs)} cases, none abstained"
    types = {(c["verdict"].get("abstain_type")) for c in ab}
    has_test = any((c.get("decisive_test") or {}).get("recommended") for c in ab)
    return (PASS if has_test else PART,
            f"{len(ab)} abstained, types={sorted(t for t in types if t)}; "
            f"{'decisive test recommended' if has_test else 'NO decisive test attached'}")

@check("MPE-06", "Minimum Prototype Expectations",
       "One sparse-history or newly launched KPI scenario")
def _():
    c = _contract(); db = _db()
    if not db: return FAIL, "no warehouse"
    n = db.execute("SELECT COUNT(DISTINCT d) FROM meta.daily_panel_truth WHERE category='SMART_HOME'").fetchone()[0]
    minh = c["kpis"]["avg_order_value"]["min_history_days"] if c else None
    cs = [x for x in _cases() if (x.get("verdict") or {}).get("abstain_type") == "sparse_history"]
    return (PASS if cs else PART,
            f"SMART_HOME has {n} days history vs min_history_days={minh}; "
            f"{'engine abstained with sparse_history' if cs else 'engine has not yet handled it'}")

@check("MPE-07", "Minimum Prototype Expectations",
       "One role-based security or entitlement scenario")
def _():
    cs = _cases()
    if not cs: return FAIL, "no case files emitted yet"
    withheld = [c for c in cs if any((v or {}).get("withheld_count", 0) > 0
                                     for v in (c.get("entitlement") or {}).values())]
    rung_diff = [c for c in cs
                 if len({(v or {}).get("max_rung") for v in (c.get("entitlement") or {}).values()}) > 1]
    if not withheld: return FAIL, "no case shows evidence withheld by policy"
    return (PASS if rung_diff else PART,
            f"{len(withheld)} cases withhold evidence by policy; "
            f"{'entitlement changes the achievable proof rung' if rung_diff else 'entitlement changes view only, not verdict'}")

@check("MPE-08", "Minimum Prototype Expectations",
       "Evidence showing source freshness, analytical method, contribution, confidence and lineage")
def _():
    cs = _cases()
    if not cs: return FAIL, "no case files emitted yet"
    need = ["freshness","method","contribution","confidence","lineage"]
    ev = [e for c in cs for e in (c.get("evidence") or [])]
    if not ev: return FAIL, "cases carry no evidence items"
    miss = [n for n in need if not all(n in e for e in ev)]
    return (PASS if not miss else PART,
            f"{len(ev)} evidence items across {len(cs)} cases; missing fields: {miss or 'none'}")

@check("MPE-09", "Minimum Prototype Expectations",
       "A clear breakdown of LLM versus non-LLM processing")
def _():
    rows = _telemetry()
    if not rows: return FAIL, "no telemetry emitted yet"
    classes = {r.get("method_class") for r in rows}
    brief8 = {"deterministic logic","SQL","business rules","statistics","traditional ML",
              "causal inference","retrieval","LLM"}
    llm = [r for r in rows if r.get("backend") in ("anthropic","local")]
    quant = [r for r in rows if r.get("on_quantitative_path")]
    llm_quant = [r for r in quant if r.get("backend") in ("anthropic","local")]
    return (PASS if (classes & brief8) and not llm_quant else PART,
            f"method classes used: {sorted(c for c in classes if c)}; "
            f"{len(llm)} LLM calls, {len(llm_quant)} of them on the quantitative path "
            f"(must be 0)")

@check("MPE-10", "Minimum Prototype Expectations",
       "Runtime telemetry covering latency, model calls, token usage and estimated cost")
def _():
    rows = _telemetry()
    if not rows: return FAIL, "no telemetry emitted yet"
    need = ["wall_ms","model_calls","tokens_in","tokens_out","cost_usd","token_source"]
    miss = [n for n in need if not any(n in r for r in rows)]
    labelled = all(r.get("token_source") in ("exact_api","local_tokenizer","none")
                   for r in rows if r.get("tokens_in"))
    return (PASS if not miss and labelled else PART,
            f"{len(rows)} telemetry spans; missing={miss or 'none'}; "
            f"token provenance labelled={labelled}")

# ===================================================== ROUND 2 OBJECTIVES
OBJ = [
 ("OBJ-1","Detects and prioritises material KPI movements",
  lambda: _obj_files(["casefile/engine/signal_gate.py"], "priority_score")),
 ("OBJ-2","Reconciles data and business context across heterogeneous sources",
  lambda: _obj_files(["casefile/engine/reconcile.py"], "reconciliation")),
 ("OBJ-3","Identifies and ranks explanatory drivers using appropriate analytical methods",
  lambda: _obj_files(["casefile/engine/contribution.py","casefile/engine/causal.py"], "rank")),
 ("OBJ-4","Generates persona-specific narratives supported by traceable evidence",
  lambda: _obj_files(["casefile/engine/narrative.py"], "persona")),
 ("OBJ-5","Communicates uncertainty and abstains when evidence is insufficient or contradictory",
  lambda: _obj_files(["casefile/engine/verdict.py"], "ABSTAIN")),
 ("OBJ-6","Recommends practical actions grounded in business levers, constraints and decision rights",
  lambda: _obj_files(["casefile/engine/actions.py"], "decision_rights")),
 ("OBJ-7","Mechanism to learn from analyst and business-user feedback",
  lambda: _obj_files(["casefile/engine/feedback.py"], "feedback")),
 ("OBJ-8","Operates within realistic security, cost, latency and scalability constraints",
  lambda: _obj_files(["casefile/security/policy.py","casefile/telemetry/spans.py"], "budget")),
]
def _obj_files(paths, token):
    have = [p for p in paths if (ROOT / p).exists()]
    if not have: return FAIL, f"not built: {paths}"
    hit = any(token.lower() in (ROOT / p).read_text().lower() for p in have)
    return (PASS if (len(have) == len(paths) and hit) else PART,
            f"built {have}; key concept '{token}' {'present' if hit else 'ABSENT'}")
for cid, text, fn in OBJ:
    check(cid, "Round 2 Objectives", text)(fn)

# ===================================================== extra: brief's explicit warnings
@check("X-01", "Explicit brief requirements",
       "LLM is not the source of quantitative truth (numbers identical with LLM off)")
def _():
    p = OUT / "llm_invariance.json"
    if not p.exists(): return FAIL, "no llm_invariance.json (run the off-vs-on regression)"
    d = json.load(open(p))
    return (PASS if d.get("identical") else FAIL,
            f"compared {d.get('fields_compared','?')} computed fields across llm-mode off vs on: "
            f"identical={d.get('identical')}")

@check("X-02", "Explicit brief requirements",
       "Distinguish native / configured / custom / externally integrated capabilities")
def _():
    cands = [ROOT / "CAPABILITY_PROVENANCE.md", ROOT.parent / "docs/CAPABILITY_PROVENANCE.md",
             ROOT / "docs/CAPABILITY_PROVENANCE.md"]
    p = next((c for c in cands if c.exists()), None)
    if p is None: return FAIL, "CAPABILITY_PROVENANCE.md missing"
    t = p.read_text().lower()
    have = [w for w in ["native","configured","custom","integrated"] if w in t]
    return (PASS if len(have) == 4 else PART, f"classifies: {have}")

@check("X-03", "Explicit brief requirements",
       "Sparse history for new products handled without fabricating confidence")
def _():
    p = ROOT / "casefile/engine/baseline.py"
    if not p.exists(): return FAIL, "baseline.py not built"
    t = p.read_text()
    return (PASS if "INSUFFICIENT_HISTORY" in t and "shrink" in t.lower() else PART,
            "baseline selector present" if "baseline_selector" in t else "no explicit selector")

def main():
    res = []
    for cid, sec, text, fn in _checks:
        try: st, ev = fn()
        except Exception as e: st, ev = FAIL, f"check error: {type(e).__name__}: {e}"
        res.append(dict(id=cid, section=sec, requirement=text, status=st, evidence=ev))
    if "--json" in sys.argv:
        print(json.dumps(res, indent=2)); return
    ico = {PASS: "PASS  ", PART: "PARTIAL", FAIL: "FAIL  "}
    cur = None
    for r in res:
        if r["section"] != cur:
            cur = r["section"]; print(f"\n\033[1m{cur}\033[0m")
        c = {"PASS":"\033[32m","PARTIAL":"\033[33m","FAIL":"\033[31m"}[r["status"]]
        print(f"  {c}{ico[r['status']]}\033[0m {r['id']}  {r['requirement'][:78]}")
        print(f"          {r['evidence'][:150]}")
    n = {s: sum(1 for r in res if r["status"] == s) for s in (PASS, PART, FAIL)}
    tot = len(res); score = (n[PASS] + 0.5 * n[PART]) / tot
    print(f"\n\033[1mSCORE {n[PASS]} pass / {n[PART]} partial / {n[FAIL]} fail "
          f"= {score:.0%} of {tot} brief requirements\033[0m")
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / "rubric.json", "w"), indent=2)

if __name__ == "__main__":
    main()

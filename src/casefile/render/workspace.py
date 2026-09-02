"""
The Decision Workspace: one self-contained HTML file, no server, no network.

Answer-first. A judge meets the verdict, the standard of proof and the next move in the
first forty words, before any architecture. Everything below that exists to let them audit
the claim: the evidence ledger with source and freshness per item, the proof ladder, the
posterior, the abstention type, the ranked experiments, the actions per persona, and the
method provenance showing which stages used a model and which are forbidden from doing so.
"""
from __future__ import annotations
import html, json, math
from pathlib import Path

RUNG_C = {"R0": "var(--r0)", "R1": "var(--r1)", "R2": "var(--r2)", "R3": "var(--r3)", "R4": "var(--r4)"}
E = html.escape


def _bar(frac, color, w=170, h=9):
    frac = max(0.0, min(1.0, float(frac or 0)))
    return (f'<svg class="bar" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">'
            f'<rect width="{w}" height="{h}" rx="2" fill="var(--surface-2)"/>'
            f'<rect width="{max(2,frac*w):.0f}" height="{h}" rx="2" fill="{color}"/></svg>')


def _pill(txt, cls=""):
    return f'<span class="pill {cls}">{E(str(txt))}</span>'


def _rung(r):
    return f'<span class="rung" style="color:{RUNG_C.get(r,"var(--r0)")}">{E(r)}</span>'


def _decision_card(c):
    """The verdict rendered as a decision someone can act on, not a status label."""
    v = c["verdict"]; d = c["detection"]; dt = c.get("decisive_test") or {}
    rec = dt.get("recommended") or {}
    delta = d.get("delta_pct")
    dstr = f"{100*delta:+.2f}%" if delta is not None else "not computable"
    act = v["decision"] == "ACT"
    head = "ACT" if act else "DON'T ACT YET"
    top = sorted(v["hypotheses"], key=lambda h: -h["posterior"])
    live = [h for h in top if h["posterior"] > 0.05 and h["id"] != "H_NULL"][:3]
    why = "".join(
        f"<tr><td>{E(h['label'][:46])}</td><td class='num'>{100*h['posterior']:.0f}%</td>"
        f"<td>{_bar(h['posterior'], RUNG_C.get(h['rung'],'var(--r0)'), 120)}</td>"
        f"<td>{_rung(h['rung'])}</td></tr>" for h in live) or \
        "<tr><td colspan='4' class='n'>no hypothesis carries material posterior</td></tr>"
    nxt = ""
    if rec:
        nxt = (f"<div class='resolve'><div class='n' style='margin-bottom:6px'>"
               f"WHAT WILL RESOLVE IT</div>"
               f"<div class='rec'><b>{E(rec['label'])}</b></div>"
               f"<div class='recmeta'>cost <b>{rec['cost']:,.0f}</b> &middot; "
               f"<b>{rec['days']}</b> days &middot; {rec['units_used']} units &middot; "
               f"reversible <b>{E(rec['reversibility'])}</b> &middot; owner {E(rec['owner_role'])}"
               f"</div>"
               f"<div class='n' style='margin-top:6px'>Chosen by expected net benefit of "
               f"sampling per day, not by how conclusive it sounds. EVPI ceiling "
               f"{dt.get('evpi',0):,.0f}; waiting costs "
               f"{dt.get('cost_of_waiting_per_day',0):,.0f} per day.</div></div>")
    elif act:
        nxt = ("<div class='resolve'><div class='n'>No test required: the evidence already "
               "meets the contract's standard of proof.</div></div>")
    return (f"<section class='decision {'ok' if act else 'warn'}'>"
            f"<div class='dhead'><span class='dlabel'>CASEFILE DECISION</span>"
            f"{_rung(v['max_rung'])}"
            f"{_pill(v['abstain_type'] or 'cause established', 'muted')}</div>"
            f"<div class='dtop'><div><div class='n'>{E(c['label'])}</div>"
            f"<div class='dnum'>{E(dstr)}</div>"
            f"<div class='n'>against a seasonality and covariate adjusted baseline</div></div>"
            f"<div class='dverdict {'ok' if act else 'warn'}'>{head}</div></div>"
            f"<p class='sub'>{E(v['reason'])}</p>"
            f"<div class='dgrid'><div><div class='n' style='margin-bottom:4px'>WHY</div>"
            f"<div class='tw'><table>{why}</table></div></div>{nxt}</div></section>")


def _hero(c):
    v = c["verdict"]; d = c["detection"]
    delta = d.get("delta_pct")
    dstr = f"{100*delta:+.2f}%" if delta is not None else "not computable"
    if v["decision"] == "ACT":
        head = f"{c['label']} is {dstr} against expectation. We can name a cause."
        sub = v["reason"]
        cls = "ok"
    else:
        head = f"{c['label']} is {dstr} against expectation. We cannot yet say why."
        sub = v["reason"]
        cls = "warn"
    rec = (c.get("decisive_test") or {}).get("recommended")
    nxt = ""
    if rec:
        nxt = (f"<p class='next'><b>Cheapest decisive test:</b> {E(rec['label'])} &middot; "
               f"{rec['days']} days &middot; {rec['units_used']} units &middot; "
               f"cost {rec['cost']:,.0f} &middot; ENBS/day {rec['enbs_per_day']:,.0f}</p>")
    elif v["decision"] == "ACT":
        nxt = "<p class='next'><b>No test required</b>: the evidence meets the standard of proof.</p>"
    return (f'<section class="hero {cls}"><div class="verdict-line">'
            f'{_pill(v["decision"], cls)}'
            f'{_pill(v["abstain_type"] or "cause established", "muted") if True else ""}'
            f'{_rung(v["max_rung"])}</div>'
            f'<h1>{E(head)}</h1><p class="sub">{E(sub)}</p>{nxt}</section>')


def _detection(c):
    d = c["detection"]; sg = c.get("signal_gate", {}).get("family", {})
    cp = c.get("change_point", {})
    rows = [
        ("Baseline", d.get("baseline_method"), d.get("baseline_note", "")[:160]),
        ("Actual vs expected",
         f"{d.get('actual')!s:.12} vs {d.get('expected')!s:.12}" if d.get("actual") is not None else "n/a",
         f"studentised p={d.get('p_studentised'):.2e}, conformal rank p={d.get('p_conformal_rank'):.4f} "
         f"(floor {d.get('p_conformal_floor'):.4f})" if d.get("p_studentised") else ""),
        ("Multiplicity",
         f"{sg.get('fired','?')} fired of {sg.get('family_size','?')} monitors",
         f"{sg.get('fdr_method','')} at q={sg.get('fdr_q','')}; suppressed "
         f"{sg.get('suppressed_by_fdr',0)} by FDR, {sg.get('suppressed_by_materiality',0)} by "
         f"materiality, {sg.get('suppressed_by_persistence',0)} by persistence"),
        ("Regime break", cp.get("onset") or "not dated",
         f"90% onset interval {cp.get('onset_interval_90')}; this window is the alibi test "
         f"every hypothesis must survive"),
        ("Source freshness", f"{d['freshness']['state']}",
         f"{d['freshness']['source']}.{d['freshness']['table']} lag "
         f"{d['freshness']['lag_minutes']:.0f}m against a {d['freshness']['sla_minutes']:.0f}m SLA"),
    ]
    body = "".join(f"<tr><td class='k'>{E(k)}</td><td class='v'>{E(str(v))}</td>"
                   f"<td class='n'>{E(str(n))}</td></tr>" for k, v, n in rows)
    return f'<h2>Is the change real?</h2><div class="tw"><table>{body}</table></div>'


def _posterior(c):
    hs = sorted(c["verdict"]["hypotheses"], key=lambda h: -h["posterior"])
    rows = ""
    for h in hs:
        if h["posterior"] < 0.005 and h["alibi_ok"]: continue
        col = RUNG_C.get(h["rung"], "var(--r0)")
        eff = f"{100*h['effect_pct']:+.2f}%" if h.get("effect_pct") is not None else "not identified"
        alibi = "" if h["alibi_ok"] else _pill("ALIBI", "stop")
        rows += (f"<tr><td>{E(h['label'])} {alibi}</td>"
                 f"<td class='num'>{100*h['posterior']:.1f}%</td>"
                 f"<td>{_bar(h['posterior'], col)}</td>"
                 f"<td>{_rung(h['rung'])}</td><td class='num'>{E(eff)}</td>"
                 f"<td class='n'>{E((h.get('alibi_note') or '')[:90])}</td></tr>")
    meta = c.get("posterior_meta", {})
    ign = meta.get("evidence_ignored_non_diagnostic") or []
    note = (f"<p class='n'>{meta.get('evidence_used',0)} evidence groups used. "
            f"{len(ign)} discarded as non-diagnostic (consistent with every hypothesis, so they "
            f"discriminate nothing): {', '.join(E(i['id']) for i in ign[:4]) or 'none'}. "
            f"Likelihood ratios: {E(c.get('likelihood_source',''))}.</p>")
    return (f"<h2>Competing explanations</h2><div class='tw'><table>"
            f"<thead><tr><th>Hypothesis</th><th>Posterior</th><th></th><th>Rung</th>"
            f"<th>Effect</th><th>Alibi screen</th></tr></thead>{rows}</table></div>{note}")


def _change_my_mind(c):
    rows = c.get("what_would_change_my_mind") or []
    if not rows: return ""
    out = ""
    for w in rows[:4]:
        sat = "".join(f"<li class='yes'>{E(t)}</li>" for t in w["satisfied"])
        mis = "".join(f"<li class='no'>{E(t)}</li>" for t in w["missing"])
        blk = "".join(f"<li>{E(b)}</li>" for b in w["blockers"]) or "<li>none</li>"
        out += (f"<div class='cmm'><div class='cmmhead'>{E(w['label'])} "
                f"{_rung(w['current_rung'])} <span class='n'>&rarr;</span> "
                f"{_rung(w['target_rung'])} <span class='num'>{100*w['posterior']:.0f}%</span></div>"
                f"<div class='cmmcols'>"
                f"<div><div class='n'>SATISFIED</div><ul class='chk'>{sat or '<li class=no>none</li>'}</ul></div>"
                f"<div><div class='n'>STILL MISSING</div><ul class='chk'>{mis or '<li class=yes>none</li>'}</ul></div>"
                f"<div><div class='n'>WHAT IS BLOCKING THE UPGRADE</div><ul class='chk plain'>{blk}</ul></div>"
                f"</div>"
                f"<div class='n' style='margin-top:6px'><b>Would upgrade if:</b> "
                f"{E(w['would_upgrade_if'])}<br><b>Would be falsified by:</b> "
                f"{E(w['would_be_falsified_by'])}</div></div>")
    return (f"<h2>What would change my mind</h2><p class='n'>A confidence score says what the "
            f"system believes. This says what it would take to change that belief, and what "
            f"would prove it wrong. Relevance is taken from the calibrated likelihood table, "
            f"not from prose.</p>{out}")


def _regret(c):
    dt = c.get("decisive_test") or {}
    m = (dt.get("hedge") or {}).get("matrix") or []
    if not m: return ""
    hyps = [x["label"][:22] for x in m[0]["cells"]]
    head = "".join(f"<th>if {E(h)}<br><span class='n'>is the cause</span></th>" for h in hyps)
    rows = ""
    for i, r in enumerate(m):
        cells = "".join(
            f"<td class='num'>{c_['value']:,.0f}<br>"
            f"<span class='n'>regret {c_['regret']:,.0f}</span></td>" for c_ in r["cells"])
        rows += (f"<tr class='{'ok' if i==0 else ''}'><td><b>{E(r['lever'])}</b><br>"
                 f"<span class='n'>{E(r['owner'])} &middot; {E(r['reversibility'])}</span></td>"
                 f"{cells}<td class='num'><b>{r['max_regret']:,.0f}</b></td></tr>")
    return (f"<h2>What it costs to be wrong</h2>"
            f"<p class='n'>{E((dt.get('hedge') or {}).get('reading',''))}</p>"
            f"<div class='tw'><table><thead><tr><th>Lever you could pull now</th>{head}"
            f"<th>Worst case</th></tr></thead>{rows}</table></div>"
            f"<p class='n'>The highlighted row is the hedge: the action whose worst outcome is "
            f"least bad across the hypotheses still standing.</p>")


def _battle(c):
    """Why two explanations cannot be separated, drawn rather than asserted."""
    v = c["verdict"]
    if v.get("abstain_type") != "collinear_causes": return ""
    top = [h for h in sorted(v["hypotheses"], key=lambda x: -x["posterior"])
           if h["posterior"] > 0.05 and h["id"] != "H_NULL"][:2]
    if len(top) < 2: return ""
    a, b = top
    return f"""<h2>Why the evidence cannot separate them</h2>
<figure><div class='tw'><svg class='dg' viewBox="0 0 840 300" role="img"
 aria-label="Two hypotheses with comparable posteriors converge on the same population and
 the same time window, leaving no control group, so neither can be established.">
<rect class='bx' x="40" y="30" width="290" height="74" rx="3"/>
<text class='tb' x="56" y="56">{E(a['label'][:34])}</text>
<text class='tx' x="56" y="76">posterior {100*a['posterior']:.0f}% &middot; rung {E(a['rung'])}</text>
<text class='tm' x="56" y="94">EVIDENCE FIRED, EFFECT NOT IDENTIFIED</text>
<rect class='bx' x="510" y="30" width="290" height="74" rx="3"/>
<text class='tb' x="526" y="56">{E(b['label'][:34])}</text>
<text class='tx' x="526" y="76">posterior {100*b['posterior']:.0f}% &middot; rung {E(b['rung'])}</text>
<text class='tm' x="526" y="94">EVIDENCE FIRED, EFFECT NOT IDENTIFIED</text>
<path class='ln' d="M185,104 L185,150 L420,150" marker-end="url(#ar)"/>
<path class='ln' d="M655,104 L655,150 L420,150" marker-end="url(#ar)"/>
<rect class='bx' x="270" y="150" width="300" height="56" rx="3"/>
<text class='tb' x="290" y="174">Same population, same window</text>
<text class='tx' x="290" y="192">the changes landed a day apart, nationally</text>
<path class='ln' d="M420,206 L420,240" marker-end="url(#ar)"/>
<rect class='hib' x="250" y="240" width="340" height="46" rx="3"/>
<text class='hit' x="272" y="262">NO CONTROL GROUP EXISTS</text>
<text class='tm' x="272" y="279">NEITHER EFFECT IS IDENTIFIED FROM THIS DATA</text>
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="currentColor"/></marker></defs>
</svg></div><figcaption>Both hypotheses fired their evidence and both sit at a comparable
posterior. Because they changed together over the same customers, no untreated population
remains, so no amount of further analysis of this data separates them. Only new data can.
</figcaption></figure>"""


def _llm_boundary(c, inv=None):
    t = c.get("telemetry", {}) or {}
    ok = t.get("llm_boundary_intact")
    invline = ""
    if inv:
        invline = (f"<p class='n'>Independently checked: the same incident was run with the "
                   f"model off and with it on, and all <b>{inv.get('fields_compared')}</b> "
                   f"computed fields were identical "
                   f"(worst relative difference {inv.get('worst_relative_difference', 0):.1e}, "
                   f"tolerance {inv.get('relative_tolerance', 0):.0e}). Only the prose changed.</p>")
    return f"""<h2>Where the model is allowed to touch the answer</h2>
<figure><div class='tw'><svg class='dg' viewBox="0 0 820 250" role="img"
 aria-label="The language model proposes hypotheses and writes prose; the deterministic core
 computes every number and adjudicates causality. No model sits on the quantitative path.">
<rect class='lane' x="30" y="20" width="760" height="66" rx="3"/>
<text class='tm' x="42" y="15">LANGUAGE MODEL</text>
<rect class='bx' x="60" y="34" width="180" height="40" rx="3"/>
<text class='tx' x="76" y="59">proposes hypotheses</text>
<rect class='bx' x="270" y="34" width="180" height="40" rx="3"/>
<text class='tx' x="286" y="59">reads documents</text>
<rect class='bx' x="480" y="34" width="180" height="40" rx="3"/>
<text class='tx' x="496" y="59">writes the narrative</text>
<path class='ln hi' d="M360,86 L360,128"/>
<text class='hit' x="372" y="112">NEVER COMPUTES A NUMBER</text>
<rect class='lane' x="30" y="140" width="760" height="86" rx="3"/>
<text class='tm' x="42" y="135">DETERMINISTIC CORE</text>
<rect class='bx' x="60" y="156" width="140" height="54" rx="3"/>
<text class='tx' x="74" y="180">SQL and the</text><text class='tx' x="74" y="196">semantic layer</text>
<rect class='bx' x="215" y="156" width="140" height="54" rx="3"/>
<text class='tx' x="229" y="180">statistics and</text><text class='tx' x="229" y="196">decomposition</text>
<rect class='bx' x="370" y="156" width="140" height="54" rx="3"/>
<text class='tx' x="384" y="180">causal</text><text class='tx' x="384" y="196">inference</text>
<rect class='bx' x="525" y="156" width="140" height="54" rx="3"/>
<text class='tx' x="539" y="180">verdict and</text><text class='tx' x="539" y="196">experiment design</text>
<rect class='bx' x="680" y="156" width="90" height="54" rx="3"/>
<text class='tx' x="694" y="180">security</text><text class='tx' x="694" y="196">and policy</text>
</svg></div><figcaption>Enforced rather than asserted: nine stages declare a backend in
routing.yaml with a written justification, six are forbidden from calling a model, the gateway
raises if they try, and every telemetry span records whether it sat on the quantitative
path.</figcaption></figure>
<p class='n'>This run: <b>{t.get('llm_calls',0)}</b> model calls, {t.get('tokens_in',0):,} in /
{t.get('tokens_out',0):,} out tokens, cost <b>${t.get('cost_usd',0):.6f}</b> per insight.
Quantitative spans that used a model: <b>{t.get('quantitative_spans_using_a_model',0)}</b>
(must be zero). Boundary intact:
<span class='pill {'ok' if ok else 'stop'}'>{ok}</span></p>{invline}"""


def _evidence(c):
    rows = ""
    for e in c["evidence"]:
        if not e["fired"]: continue
        fr = e.get("freshness", {}) or {}
        rows += (f"<tr><td>{E(e['label'])}</td>"
                 f"<td>{_pill(e['evidence_class'], 'muted')}</td>"
                 f"<td class='num'>{e['strength']:.2f}</td>"
                 f"<td class='n'>{E(e['detail'][:90])}</td>"
                 f"<td class='n'>{E(e['source'])}</td>"
                 f"<td class='n'>{E(fr.get('state','?'))}</td>"
                 f"<td class='num'>{e.get('corroboration',1)}</td>"
                 f"<td class='n'>{E(e.get('method','')[:44])}</td></tr>")
    inj = c.get("injection_defence", {}) or {}
    q = inj.get("quarantined") or []
    qn = (f"<p class='n'><b>Prompt-injection defence:</b> {inj.get('documents_scanned',0)} documents "
          f"scanned, <b>{len(q)} quarantined</b>"
          + (f" (pattern <code>{E(q[0]['pattern'])}</code> in {E(q[0]['doc_id'])})" if q else "")
          + ". Policy is quarantine, never strip-and-continue.</p>") if inj else ""
    return (f"<h2>Evidence ledger</h2><div class='tw'><table><thead><tr>"
            f"<th>Evidence</th><th>Class</th><th>Strength</th><th>Detail</th><th>Source</th>"
            f"<th>Freshness</th><th>Corrob.</th><th>Method</th></tr></thead>{rows}</table></div>{qn}")


def _tests(c):
    dt = c.get("decisive_test") or {}
    ranked = dt.get("ranked") or []
    if not ranked: return ""
    rows = ""
    for i, t in enumerate(ranked):
        cls = "ok" if i == 0 else ""
        rows += (f"<tr class='{cls}'><td>{E(t['label'])}</td>"
                 f"<td class='num'>{t['evsi']:,.0f}</td><td class='num'>{t['cost']:,.0f}</td>"
                 f"<td class='num'><b>{t['enbs_per_day']:,.0f}</b></td>"
                 f"<td class='num'>{t['days']}</td><td class='num'>{t['units_used']}</td>"
                 f"<td>{E(t['reversibility'])}</td><td>{E(t['owner_role'])}</td></tr>")
    hedge = dt.get("hedge", {}).get("chosen") or {}
    h = (f"<p class='n'><b>While the test runs:</b> {E(hedge.get('lever','none'))} "
         f"(max regret {hedge.get('max_regret',0):,.0f}, {E(hedge.get('reversibility',''))}, "
         f"owner {E(hedge.get('owner',''))}). Chosen by minimax regret across the surviving "
         f"hypotheses. EVPI ceiling {dt.get('evpi',0):,.0f}; cost of waiting "
         f"{dt.get('cost_of_waiting_per_day',0):,.0f} per day.</p>")
    return (f"<h2>The decisive test</h2><p class='n'>Ranked by expected net benefit of sampling "
            f"per day, not raw information gain, because the claim is the <i>cheapest</i> test "
            f"that settles it.</p><div class='tw'><table><thead><tr><th>Test</th><th>EVSI</th>"
            f"<th>Cost</th><th>ENBS/day</th><th>Days</th><th>Units</th><th>Reversible</th>"
            f"<th>Owner</th></tr></thead>{rows}</table></div>{h}")


def _personas(c):
    ids = list((c.get("narratives") or {}).keys())
    cid = c["incident_id"]
    tabs = "".join(
        f"<button class='ptab' data-inc='{E(cid)}' data-persona='{E(p)}'"
        f"{' aria-pressed=\'true\'' if i == 0 else ''}>{E(p.replace('_',' ').title())}</button>"
        for i, p in enumerate(ids))
    out = (f"<div class='ptabs' role='group' aria-label='persona'>{tabs}</div>")
    for pid, nar in (c.get("narratives") or {}).items():
        ent = (c.get("entitlement") or {}).get(pid, {})
        acts = (c.get("actions_by_persona") or {}).get(pid, [])
        cl = nar.get("closure", {})
        badge = (f"{cl.get('numerals_in_text',0)} numerals, {cl.get('bound',0)} bound to computed "
                 f"values, {len(cl.get('free_numerals') or [])} free")
        arows = "".join(
            f"<tr><td>{E(a['driver'])}</td><td>{E(a['action'])}</td>"
            f"<td class='num'>{a['expected_impact']:,.0f}</td><td class='num'>{a['cost']:,.0f}</td>"
            f"<td>{E(a['owner_role'])}</td><td class='num'>{100*a['confidence']:.0f}%</td>"
            f"<td>{E(a['reversibility'])}</td>"
            f"<td class='n'>{E(a['monitoring_plan']['leading_indicator'])} by day "
            f"{a['monitoring_plan']['check_after_days']}</td>"
            f"<td class='n'>{'in rights' if a['within_decision_rights'] else 'needs ' + str(a['requires_approval_from'])}</td></tr>"
            for a in acts[:3])
        cap = ("<span class='pill stop'>PROOF CAPPED BY ENTITLEMENT</span>"
               if ent.get("rung_capped_by_entitlement") else "")
        out += (f"<div class='persona' data-inc='{E(c['incident_id'])}' data-persona='{E(pid)}'>"
                f"<h3>{E(pid.replace('_',' ').title())} {cap}</h3>"
                f"<p class='n'>Regions {E(str(ent.get('regions')))} &middot; "
                f"{ent.get('evidence_admitted',0)} evidence items admitted, "
                f"<b>{ent.get('withheld_count',0)} withheld by policy</b> &middot; "
                f"max rung {E(ent.get('max_rung','?'))}. {E(ent.get('note',''))}</p>"
                f"<blockquote>{E(nar.get('text',''))}</blockquote>"
                f"<p class='n'>Rendered by <code>{E(nar.get('mode',''))}</code>. "
                f"Numeric closure check: {badge}.</p>"
                + (f"<div class='tw'><table><thead><tr><th>Driver</th><th>Lever &rarr; action</th>"
                   f"<th>Impact</th><th>Cost</th><th>Owner</th><th>Conf.</th><th>Reversible</th>"
                   f"<th>Monitoring</th><th>Rights</th></tr></thead>{arows}</table></div>"
                   if arows else "<p class='n'>No action is within this persona's scope.</p>")
                + "</div>")
    return f"<h2>Who sees what, and who may act</h2>{out}"


def _telemetry(c):
    t = c.get("telemetry", {}) or {}
    bc = t.get("by_method_class", {})
    rows = "".join(f"<tr><td>{E(k)}</td><td class='num'>{v['spans']}</td>"
                   f"<td class='num'>{v['wall_ms']:.0f}</td>"
                   f"<td class='num'>{v['cost_usd']:.6f}</td></tr>" for k, v in bc.items())
    ok = t.get("llm_boundary_intact")
    return (f"<h2>Runtime telemetry</h2>"
            f"<p class='n'><b>{t.get('spans',0)} spans</b> &middot; "
            f"{t.get('wall_ms_total',0):,.0f} ms total &middot; "
            f"{t.get('llm_calls',0)} model calls &middot; "
            f"{t.get('tokens_in',0):,} in / {t.get('tokens_out',0):,} out tokens &middot; "
            f"cost ${t.get('cost_usd',0):.6f} per insight. "
            f"<b>{t.get('quantitative_spans',0)} spans on the quantitative path, "
            f"{t.get('quantitative_spans_using_a_model',0)} of them used a model "
            f"(must be zero).</b> LLM boundary intact: "
            f"<span class='pill {'ok' if ok else 'stop'}'>{ok}</span></p>"
            f"<div class='tw'><table><thead><tr><th>Method class (the brief's vocabulary)</th>"
            f"<th>Spans</th><th>Wall ms</th><th>Cost USD</th></tr></thead>{rows}</table></div>")


CSS = """
:root{--paper:#EDF0F3;--surface:#FBFCFD;--surface-2:#E3E9EE;--ink:#131A22;--ink-2:#4A5866;
--ink-3:#77848F;--rule:#C6D0D9;--r0:#79858F;--r1:#54748A;--r2:#336480;--r3:#136065;--r4:#0A6B4F;
--flag:#95590A;--stop:#8F2C26;--ok-bg:#DFEDE7;--warn-bg:#F4EADA;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,Arial,sans-serif;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--paper:#0D1218;--surface:#141C24;
--surface-2:#1C252E;--ink:#E7EDF3;--ink-2:#A5B2BE;--ink-3:#7D8B98;--rule:#2A343E;--r0:#9BA7B3;
--r1:#82A8BF;--r2:#5AA2C0;--r3:#3FB3B0;--r4:#4AC697;--flag:#E0A54A;--stop:#E2776D;
--ok-bg:#152A22;--warn-bg:#2C2314;}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:var(--sans);font-size:15px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 90px}
h1{font-size:26px;line-height:1.25;margin:8px 0 6px;letter-spacing:-.02em}
h2{font-size:17px;margin:34px 0 10px;letter-spacing:-.01em;border-top:1px solid var(--rule);padding-top:16px}
h3{font-size:15px;margin:20px 0 6px}
.hero{border:1px solid var(--rule);border-radius:4px;padding:20px 22px;background:var(--surface)}
.hero.ok{border-left:4px solid var(--r4);background:var(--ok-bg)}
.hero.warn{border-left:4px solid var(--flag);background:var(--warn-bg)}
.sub{color:var(--ink-2);margin:6px 0 0;max-width:76ch}
.next{margin:12px 0 0;padding-top:10px;border-top:1px dashed var(--rule)}
.verdict-line{display:flex;gap:8px;align-items:center;margin-bottom:6px}
.pill{font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.06em;padding:2px 7px;
border-radius:3px;border:1px solid currentColor;text-transform:uppercase}
.pill.ok{color:var(--r4)}.pill.warn{color:var(--flag)}.pill.stop{color:var(--stop)}
.pill.muted{color:var(--ink-3)}
.rung{font-family:var(--mono);font-size:11px;font-weight:700;border:1px solid currentColor;
padding:2px 7px;border-radius:3px}
.tw{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--surface);margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-family:var(--mono);font-size:10px;text-transform:uppercase;
letter-spacing:.07em;color:var(--ink-2);background:var(--surface-2);padding:7px 10px;
border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:7px 10px;border-bottom:1px solid var(--surface-2);vertical-align:top}
tr:last-child td{border-bottom:0} tr.ok td{background:var(--ok-bg)}
td.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.k{font-weight:600;white-space:nowrap} td.n,.n{color:var(--ink-3);font-size:12.5px}
blockquote{margin:8px 0;padding:12px 16px;border-left:3px solid var(--r2);
background:var(--surface);white-space:pre-wrap}
.persona{border:1px solid var(--rule);border-radius:4px;padding:14px 16px;margin:12px 0;
background:var(--surface)}
code{font-family:var(--mono);font-size:12px;background:var(--surface-2);padding:1px 5px;border-radius:2px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:18px 0 6px}
.tabs a{font-family:var(--mono);font-size:11px;text-decoration:none;color:var(--ink-2);
border:1px solid var(--rule);padding:5px 10px;border-radius:3px;background:var(--surface)}
.tabs a:hover{color:var(--ink);border-color:var(--ink-3)}
.ladder{display:flex;gap:3px;margin:10px 0}
.ladder div{flex:1;padding:7px 9px;border-radius:3px;font-family:var(--mono);font-size:10.5px;
background:var(--surface);border:1px solid var(--rule)}
.decision{border:1px solid var(--rule);border-radius:4px;padding:18px 20px;background:var(--surface);margin:14px 0}
.decision.ok{border-left:4px solid var(--r4);background:var(--ok-bg)}
.decision.warn{border-left:4px solid var(--flag);background:var(--warn-bg)}
.dhead{display:flex;gap:8px;align-items:center;margin-bottom:10px}
.dlabel{font-family:var(--mono);font-size:10px;letter-spacing:.14em;color:var(--ink-3)}
.dtop{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap}
.dnum{font-size:34px;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.1}
.dverdict{font-size:19px;font-weight:700;letter-spacing:-.01em;padding:8px 14px;border-radius:3px;
border:2px solid currentColor;white-space:nowrap}
.dverdict.ok{color:var(--r4)} .dverdict.warn{color:var(--flag)}
.dgrid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:14px}
@media(max-width:820px){.dgrid{grid-template-columns:1fr}}
.resolve{border-left:3px solid var(--r3);padding-left:14px}
.rec{font-size:15px;margin-bottom:4px}
.recmeta{font-size:12.5px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.cmm{border:1px solid var(--rule);border-radius:3px;padding:12px 14px;margin:8px 0;background:var(--surface)}
.cmmhead{display:flex;gap:8px;align-items:center;font-weight:650;margin-bottom:8px;flex-wrap:wrap}
.cmmhead .num{margin-left:auto;font-family:var(--mono);font-variant-numeric:tabular-nums}
.cmmcols{display:grid;grid-template-columns:1fr 1fr 1.4fr;gap:14px}
@media(max-width:820px){.cmmcols{grid-template-columns:1fr}}
ul.chk{margin:4px 0 0;padding-left:16px;font-size:12.5px}
ul.chk li{margin:2px 0} ul.chk li.yes{color:var(--r4)} ul.chk li.no{color:var(--ink-3)}
ul.chk.plain li{color:var(--ink-2);list-style:square}
.ptabs{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
.ptab{font-family:var(--mono);font-size:11px;padding:5px 10px;border:1px solid var(--rule);
background:var(--surface);color:var(--ink-2);border-radius:3px;cursor:pointer}
.ptab[aria-pressed=true]{color:var(--ink);border-color:var(--r3);box-shadow:inset 0 -2px 0 var(--r3)}
.ptab:focus-visible{outline:2px solid var(--r3);outline-offset:2px}
.mem{border-left:3px solid var(--r4);padding:10px 14px;background:var(--surface);margin:8px 0}
"""


def render_case(c: dict, inv: dict | None = None) -> str:
    return (_decision_card(c) + _battle(c) + _detection(c) + _posterior(c)
            + _change_my_mind(c) + _evidence(c) + _tests(c) + _regret(c)
            + _personas(c) + _llm_boundary(c, inv) + _telemetry(c))


SCRIPT = """
<script>
(function(){
  function apply(inc, persona){
    document.querySelectorAll('.persona[data-inc="'+inc+'"]').forEach(function(el){
      el.hidden = (el.getAttribute('data-persona') !== persona);
    });
    document.querySelectorAll('.ptab[data-inc="'+inc+'"]').forEach(function(b){
      b.setAttribute('aria-pressed', String(b.getAttribute('data-persona') === persona));
    });
  }
  document.querySelectorAll('.ptab').forEach(function(b){
    b.addEventListener('click', function(){
      apply(b.getAttribute('data-inc'), b.getAttribute('data-persona'));
    });
  });
  // default to the first persona of each incident
  var seen = {};
  document.querySelectorAll('.ptab').forEach(function(b){
    var i = b.getAttribute('data-inc');
    if(!seen[i]){ seen[i] = true; apply(i, b.getAttribute('data-persona')); }
  });
})();
</script>"""


def _memory(loop: dict | None, contract: dict | None) -> str:
    """Decision memory: what the organisation learned, and what it now knows for next time."""
    if not (loop and loop.get("ran") and (loop.get("result") or {}).get("measured")):
        return ""
    m = loop["result"]["measured"]; p = loop["plan"]; led = loop.get("ledger") or {}
    edge = None
    for e in ((contract or {}).get("kpi_graph", {}) or {}).get("edges", []):
        if e.get("provenance") == "MEASURED": edge = e
    steps = [
        ("INCIDENT", f"{E(str(loop['before']['verdict']))} ({E(str(loop['before']['abstain_type']))}) "
                     f"at rung {E(str(loop['before']['max_rung']))}"),
        ("EXPERIMENT", f"pre-registered <code>{E(p['plan_hash'])}</code>, seed "
                       f"{p['randomisation_seed']}, {len(p['treated_units'])} treated vs "
                       f"{len(p['control_units'])} control, randomised before any data existed"),
        ("OUTCOME", f"<b>{100*m['effect_pct']:+.2f}%</b> (90% band {100*m['ci_low']:+.2f}% to "
                    f"{100*m['ci_high']:+.2f}%), placebo p={m['placebo_p']:.3f}, rung {E(m['rung'])}"),
        ("KNOWLEDGE", (f"edge <code>{E(edge['from'])} &rarr; {E(edge['to'])}</code> upgraded "
                       f"<b>{E(str(edge.get('provenance')))}</b>, effect {edge.get('effect')}, "
                       f"n={edge.get('n')}, measured {E(str(edge.get('measured_on')))} "
                       f"<span class='n'>({E(str(edge.get('measured_on_basis','')))})</span>")
                      if edge else "no edge upgraded"),
        ("NEXT TIME", "the Action Engine now sources this lever's expected impact from a "
                      "measured edge instead of an assumption, and says so on the card"),
    ]
    rows = "".join(f"<div class='mem'><div class='n'>{k}</div><div>{v}</div></div>"
                   for k, v in steps)
    pe = led.get("prediction_error_pct")
    err = (f"<p class='n'>Predicted impact was {led.get('predicted_impact'):.4f} and the "
           f"measured effect was {led.get('measured_effect'):.4f}, an error of "
           f"{pe:+.1f}%. That error is itself recorded, because a system that never checks "
           f"its own forecasts cannot claim to be calibrated.</p>"
           if pe is not None and led.get("predicted_impact") else "")
    return (f"<h2>Decision memory</h2><p class='n'>Every recommended action becomes a measured "
            f"natural experiment, and the result is written back into the contract. This is the "
            f"third arrow: uncertainty, then a decisive experiment, then knowledge that outlives "
            f"the incident.</p>{rows}{err}")


def _redteam(batch_path: Path) -> str:
    """Hardest cases, mined from the held-out evaluation rather than hand-picked."""
    try:
        import pandas as _pd
        df = _pd.read_parquet(batch_path)
    except Exception:
        return ""
    hard = df[(~df.identifiable) & (~df.naive_hit_dominant) & (~df.cf_named)]
    if hard.empty: return ""
    hard = hard.sort_values("naive_conf", ascending=False).head(6)
    rows = "".join(
        f"<tr><td class='num'>{int(r.seed)}</td><td class='num'>{int(r.n_causes)}</td>"
        f"<td>{E(str(r.dominant_cause).replace('H_',''))}</td>"
        f"<td class='num'>{100*float(r.dominant_share):.0f}%</td>"
        f"<td><span class='pill stop'>{E(str(r.naive_cause).replace('H_',''))} "
        f"@ {100*float(r.naive_conf):.0f}%</span></td>"
        f"<td><span class='pill ok'>ABSTAIN</span> <span class='n'>{E(str(r.cf_abstain_type))}</span></td></tr>"
        for r in hard.itertuples())
    return (f"<h2>The hardest cases it was given</h2>"
            f"<p class='n'>Not hand-picked. These are drawn from the held-out evaluation: "
            f"incidents built to be unidentifiable where the contribution-ranking baseline "
            f"committed to the wrong lever with the highest confidence. Reproduce any of them "
            f"with <code>python3 run_seeded.py &lt;seed&gt;</code>.</p>"
            f"<div class='tw'><table><thead><tr><th>Seed</th><th>Causes</th>"
            f"<th>Dominant true cause</th><th>Its share</th>"
            f"<th>Baseline said</th><th>CaseFile said</th></tr></thead>{rows}</table></div>")


def build(cases: list[dict], out_path: Path, eval_summary: dict | None = None,
          loop: dict | None = None, invariance: dict | None = None,
          contract: dict | None = None, batch_path: Path | None = None) -> Path:
    tabs = "".join(f"<a href='#{E(c['incident_id'])}'>{E(c['incident_id'])} &middot; "
                   f"{E(c['verdict']['decision'])}</a>" for c in cases)
    ladder = "".join(
        f"<div style='border-left:3px solid {RUNG_C[r]}'><b style='color:{RUNG_C[r]}'>{r}</b> {n}</div>"
        for r, n in [("R0", "Arithmetic"), ("R1", "Association"), ("R2", "Temporal + mechanism"),
                     ("R3", "Quasi-experimental"), ("R4", "Measured experiment")])
    ev = ""
    if eval_summary:
        u = eval_summary["unidentifiable"]; i = eval_summary["identifiable"]
        wl = eval_summary.get("wrong_lever_rate_unidentifiable", {})
        ev = (f"<h2>Measured against injected ground truth</h2>"
              f"<p class='n'>{eval_summary['n_incidents']} seeded incidents; "
              f"{eval_summary['n_unidentifiable']} are unidentifiable <i>by construction</i> "
              f"(causes made collinear on purpose, so no honest answer exists).</p>"
              f"<div class='tw'><table><thead><tr><th>Metric</th>"
              f"<th>Contribution-ranking baseline</th><th>CaseFile</th></tr></thead>"
              f"<tr><td>Top-1 cause accuracy, identifiable</td>"
              f"<td class='num'>{100*i['naive_top1_accuracy']:.1f}%</td>"
              f"<td class='num'>{100*(i['casefile_top1_accuracy'] or 0):.1f}%</td></tr>"
              f"<tr><td>Answer rate, identifiable</td><td class='num'>100.0%</td>"
              f"<td class='num'>{100*i['casefile_answer_rate']:.1f}%</td></tr>"
              f"<tr class='ok'><td><b>Wrong lever pulled, unidentifiable</b></td>"
              f"<td class='num'><b>{100*wl.get('naive',0):.1f}%</b></td>"
              f"<td class='num'><b>{100*wl.get('casefile_acted_and_wrong',0):.1f}%</b></td></tr>"
              f"<tr><td>Abstention rate, unidentifiable</td>"
              f"<td class='num'>{100*u['naive_abstention_rate']:.1f}%</td>"
              f"<td class='num'>{100*u['casefile_abstention_rate']:.1f}%</td></tr>"
              f"</table></div>"
              f"<p class='n'>The baseline is the standard industry pattern implemented faithfully: "
              f"decompose, take the largest contributor, report it as the cause, never abstain. "
              f"Both systems see identical evidence.</p>")
    lp = ""
    if loop and loop.get("ran") and loop.get("result", {}).get("measured"):
        m = loop["result"]["measured"]; p = loop["plan"]
        lp = (f"<h2>The loop, closed</h2>"
              f"<div class='tw'><table>"
              f"<tr><td class='k'>Pre-registered</td><td>plan <code>{E(p['plan_hash'])}</code>, "
              f"seed {p['randomisation_seed']}, {len(p['treated_units'])} treated vs "
              f"{len(p['control_units'])} control, randomised before any data existed</td></tr>"
              f"<tr><td class='k'>Falsifier</td><td class='n'>{E(p['falsification_criterion'])}</td></tr>"
              f"<tr><td class='k'>Measured</td><td><b>{100*m['effect_pct']:+.2f}%</b> "
              f"(90% band {100*m['ci_low']:+.2f}% to {100*m['ci_high']:+.2f}%), "
              f"placebo p={m['placebo_p']:.3f}, rung {m['rung']}</td></tr>"
              f"<tr><td class='k'>Before</td><td>{E(loop['before']['verdict'])} "
              f"({E(str(loop['before']['abstain_type']))}) at rung {E(loop['before']['max_rung'])}</td></tr>"
              f"<tr><td class='k'>After</td><td><b>{E(loop['after']['verdict'])}</b> on "
              f"{E(loop['after']['established_cause'])} at rung {E(loop['after']['rung'])}</td></tr>"
              f"<tr><td class='k'>Contract</td><td>edge <code>promo_depth &rarr; conversion_rate</code> "
              f"upgraded to <b>MEASURED</b>; the next incident inherits this effect size as a prior</td></tr>"
              f"</table></div>")
    body = "".join(f"<section id='{E(c['incident_id'])}'><h2 style='border-top-width:3px'>"
                   f"{E(c['incident_id'])} &middot; {E(c['label'])}</h2>"
                   f"{render_case(c, invariance)}</section>" for c in cases)
    mem = _memory(loop, contract)
    red = _redteam(batch_path) if batch_path else ""
    doc = (f"<title>CaseFile Decision Workspace</title><style>{CSS}</style>"
           f"<div class='wrap'><p class='n' style='font-family:var(--mono);font-size:11px;"
           f"letter-spacing:.1em'>CASEFILE &middot; KPI INTELLIGENCE-TO-ACTION ENGINE &middot; "
           f"VANTAGE RETAIL GROUP (SYNTHETIC)</p>"
           f"<div class='tabs'>{tabs}</div>"
           f"<h2 style='border-top:0;margin-top:12px'>Proof ladder</h2>"
           f"<p class='n'>Every claim carries the rung of the strongest evidence behind it. "
           f"The word <i>caused</i> is unavailable below R3.</p><div class='ladder'>{ladder}</div>"
           f"{ev}{lp}{mem}{red}{body}</div>{SCRIPT}")
    out_path.write_text(doc)
    return out_path

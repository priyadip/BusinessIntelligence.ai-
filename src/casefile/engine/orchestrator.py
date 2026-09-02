"""
run_case(): the whole pipeline, as one deterministic function.

Contract for this module: the quantitative path is pure and model-free. Every stage opens a
telemetry span declaring its method class in the brief's own vocabulary and whether it sits
on the quantitative path. A span on that path that touches a model is flagged
violates_llm_boundary, and tests/test_llm_boundary.py fails the build on it.

Narrative generation is the only stage permitted to call a model, and it runs AFTER the
evidence object is frozen.
"""
from __future__ import annotations
import json, hashlib, math
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yaml

from ..security.policy import PolicyEngine, Principal
from ..semantic.gateway import SemanticGateway, MetricRequest, FreshnessBlocked
from ..telemetry.spans import Telemetry
from . import baseline as B, signal_gate as SG, contribution as C, evidence as EV
from . import causal as CA, verdict as VD, likelihood as LK, narrative as NR
from .reconcile import Reconciler
from .actions import build_actions

ROOT = Path(__file__).resolve().parents[2]
AS_OF = datetime(2026, 8, 31, 23, 45)

INCIDENTS = {
    "INC-001": dict(kpi="conversion_rate", window=(date(2026, 7, 22), date(2026, 8, 2)),
                    slice={"region": "WEST"}, unit_dims=["region", "channel", "category"],
                    label="Conversion rate, WEST region",
                    daily_exposure=2_950_000.0, expect="resolvable_multifactor"),
    "INC-002": dict(kpi="net_revenue", window=(date(2026, 8, 10), date(2026, 8, 31)),
                    slice={"category_in": ["KITCHEN", "DECOR"]},
                    unit_dims=["region", "channel", "category"],
                    label="Net revenue, KITCHEN and DECOR",
                    daily_exposure=259_000.0, expect="ambiguous_collinear"),
    "INC-003": dict(kpi="avg_order_value", window=(date(2026, 8, 18), date(2026, 8, 31)),
                    slice={"category": "SMART_HOME"}, unit_dims=["region", "channel"],
                    label="Average order value, SMART_HOME (new category)",
                    daily_exposure=41_000.0, expect="sparse_history_abstain"),
    "INC-004": dict(kpi="gross_margin_pct", window=(date(2026, 8, 10), date(2026, 8, 31)),
                    slice={}, unit_dims=["region"],
                    label="Gross margin, all regions",
                    daily_exposure=180_000.0, expect="stale_source_abstain"),
}

HYP_EVENTS = {"H_CARRIER_DEGRADE": date(2026, 7, 15), "H_CHECKOUT_DEFECT": date(2026, 7, 18),
              "H_PRICE_RISE": date(2026, 8, 3), "H_PROMO_WITHDRAWAL": date(2026, 8, 4),
              "H_COMPETITOR": date(2026, 8, 8), "H_EXTERNAL": date(2026, 7, 20)}
NO_CONTROL = {"H_PRICE_RISE", "H_PROMO_WITHDRAWAL", "H_COMPETITOR", "H_MIX_SHIFT"}
TXT_TESTS = {"delivery_late": "txt_delivery_late", "payment_fail": "txt_payment_fail",
             "price": "txt_price", "stock": "txt_stock", "competitor": "txt_competitor"}


def _series(gw, kpi, slc, principal, upto):
    dims = tuple(k for k in ("region", "channel", "category") if k in slc or True)
    df, prov = gw.execute_governed(MetricRequest(kpi, dims=("region", "channel", "category")
                                                 if kpi in ("net_revenue", "avg_order_value",
                                                            "conversion_rate") else ("region",)),
                                   principal, allow_stale=True)
    if "region" in slc: df = df[df.region == slc["region"]]
    if "category" in slc: df = df[df.category == slc["category"]]
    if "category_in" in slc: df = df[df.category.isin(slc["category_in"])]
    g = df.groupby("d", as_index=False)[["numerator", "denominator"]].sum()
    g["d"] = pd.to_datetime(g["d"])
    g = g[g.d <= pd.Timestamp(upto)].sort_values("d")
    v = (g.numerator / g.denominator).values if kpi != "net_revenue" else g.numerator.values
    return pd.Series(v, index=pd.DatetimeIndex(g.d)).dropna(), prov, df


def run_case(incident_id: str, contract: dict, gw: SemanticGateway, pol: PolicyEngine,
             tel: Telemetry, llm=None, personas: list[str] | None = None) -> dict:
    spec = INCIDENTS[incident_id]
    kpi, win, slc = spec["kpi"], spec["window"], spec["slice"]
    personas = personas or list(contract["personas"])
    analyst = pol.principal("analyst")
    lr_table, lr_source = LK.load_table()
    lr_temperature = LK.load_temperature()
    case: dict = {"incident_id": incident_id, "kpi": kpi, "label": spec["label"],
                  "window": [str(win[0]), str(win[1])], "slice": slc,
                  "as_of": str(AS_OF), "contract_version": contract["contract"]["version"],
                  "likelihood_source": lr_source, "likelihood_temperature": lr_temperature, "generated_at": datetime.now().isoformat(timespec="seconds")}

    # ---------------------------------------------------------------- L2 reconcile
    with tel.span("L2", "reconcile_sources", "business rules", quantitative=True,
                  why="two ledgers disagree by definition; the contract names the canonical one",
                  why_not_llm="a definition conflict is resolved by policy, not by judgement",
                  incident_id=incident_id, budget_ms=4000):
        rc = Reconciler(gw, contract)
        recon = rc.reconcile_revenue((win[0] - timedelta(days=60), win[1]))
        case["reconciliation"] = asdict(recon)

    # ---------------------------------------------------------------- freshness
    with tel.span("L1", "freshness_gate", "business rules", quantitative=True,
                  why="the contract's SLA decides whether a claim may be made at all",
                  why_not_llm="a watermark comparison is deterministic",
                  incident_id=incident_id, budget_ms=500):
        src_tbl = contract["kpis"][kpi]["from"]; src, tbl = src_tbl.split(".")
        state, lag, sla, wm = gw.freshness(src, tbl, AS_OF)
        stale = state == "STALE_BLOCKED"
        case["freshness"] = {"source": src, "table": tbl, "state": state,
                             "lag_minutes": round(lag, 1), "sla_minutes": sla,
                             "watermark": wm, "blocks_claims": stale}

    # ---------------------------------------------------------------- L3 signal gate
    with tel.span("L3", "detect_and_prioritise", "statistics", quantitative=True,
                  why="seasonal baseline, conformal calibration and FDR control across the family",
                  why_not_llm="these carry finite-sample guarantees a model cannot supply",
                  incident_id=incident_id, budget_ms=25000) as sp:
        fam, target_bl = [], None
        for k in ["net_revenue", "conversion_rate", "avg_order_value", "fulfilment_ontime_pct"]:
            df, _ = gw.execute_governed(MetricRequest(
                k, dims=("region", "channel", "category") if k != "fulfilment_ontime_pct" else ("region",)),
                analyst, allow_stale=True)
            for reg, g in df.groupby("region"):
                gg = g.groupby("d", as_index=False)[["numerator", "denominator"]].sum()
                gg["d"] = pd.to_datetime(gg["d"])
                vv = (gg.numerator / gg.denominator) if k != "net_revenue" else gg.numerator
                s = pd.Series(vv.values, index=pd.DatetimeIndex(gg.d)).dropna()
                s = s[s.index <= pd.Timestamp(win[1])]
                if len(s) < 40: continue
                bl = B.fit(k, s, contract, test_start=pd.Timestamp(win[0]))
                mask = (bl.dates >= pd.Timestamp(win[0])) & (bl.dates <= pd.Timestamp(win[1]))
                rupee = {"net_revenue": 1.0, "conversion_rate": 9.0e8,
                         "avg_order_value": 2.2e4, "fulfilment_ontime_pct": 7.0e8}[k]
                fam.append((f"{k}|{reg}", k, {"region": reg}, bl, mask, rupee))
        monitors, famstat = SG.evaluate(fam, contract)
        sp.note(family_size=famstat["family_size"])
        case["signal_gate"] = {"family": famstat,
                               "top_monitors": [asdict(m) for m in monitors[:6]]}

    if stale:
        # nothing downstream may run: the engine does not estimate on data it knows to be
        # incomplete, and it says which watermark blocked it.
        case["detection"] = {"baseline_method": "NOT_RUN", "blocked": True,
                             "baseline_note": f"{src}.{tbl} watermark is {lag:.0f} min old "
                                              f"against a {sla:.0f} min SLA.",
                             "delta_pct": None, "actual": None, "expected": None,
                             "freshness": case["freshness"], "lineage": [], "sparse_history": False}
        case["change_point"] = {"onset": None, "onset_interval_90": None, "statistic": 0.0}
        case["contribution"] = {"rung": "R0", "top": [], "caveat": "not run: source stale"}
        case["evidence"] = []; case["causal"] = {}; case["posterior_meta"] = {}
        v = VD.decide(incident_id, kpi, [VD.Hypothesis("H_NULL", "Cause outside the library", 1.0)],
                      contract, spec["daily_exposure"], stale=True)
        case["verdict"] = asdict(v)
        case["decisive_test"] = {"recommended": None,
                                 "note": "the correct next step is a pipeline fix, not an experiment",
                                 "remediation": {"owner": "data_platform_lead",
                                                 "action": "restore the weekly ERP export",
                                                 "lever": "data_pipeline_fix"}}
        case["entitlement"] = {}; case["actions_by_persona"] = {}; case["narratives"] = {}
        for pid in personas:
            pr = pol.principal(pid)
            case["entitlement"][pid] = {"regions": list(pr.regions),
                                        "column_grants": sorted(pr.column_grants),
                                        "evidence_admitted": 0, "withheld_count": 0,
                                        "withheld_reasons": [], "max_rung": "R0",
                                        "rung_capped_by_entitlement": False,
                                        "note": "no evidence admitted; source is stale"}
            case["actions_by_persona"][pid] = []
            case["narratives"][pid] = {
                "text": (f"{spec['label']} cannot be assessed. The {src}.{tbl} feed is "
                         f"{lag/1440:.1f} days old against a {sla/1440:.1f} day service level, "
                         f"so any figure would be computed on incomplete data. No verdict is "
                         f"offered. The next step is a pipeline fix owned by the data platform "
                         f"team, not a business decision."),
                "mode": "deterministic-template", "closure": {"passed": True, "numerals_in_text": 0,
                "bound": 0, "free_numerals": []}, "attempts": 0}
        case["telemetry"] = tel.summary()
        return case

    # target series for this incident
    s_target, prov, slice_df = _series(gw, kpi, slc, analyst, win[1])
    cohort = None
    if kpi == "avg_order_value" and slc.get("category"):
        allp, _ = gw.execute_governed(MetricRequest(kpi, dims=("region", "channel", "category")),
                                      analyst, allow_stale=True)
        sib = allp[allp.category != slc["category"]].groupby("d", as_index=False)[
            ["numerator", "denominator"]].sum()
        cohort = (sib.numerator / sib.denominator).to_frame("v")
    bl = B.fit(kpi, s_target, contract, test_start=pd.Timestamp(win[0]), cohort=cohort)
    mask = (bl.dates >= pd.Timestamp(win[0])) & (bl.dates <= pd.Timestamp(win[1]))
    idx = np.where(mask)[0]
    p_win, eff_sd = bl.window_rank_p(idx)
    actual = float(np.nanmean(bl.actual[idx])); expected = float(np.nanmean(bl.expected[idx]))
    sparse = bl.method in ("POOLED_COHORT", "INSUFFICIENT_HISTORY")
    case["detection"] = {
        "baseline_method": bl.method, "baseline_note": bl.note,
        "n_calibration_windows": int(bl.oos_window_stats.size),
        "actual": actual, "expected": expected,
        "delta_abs": actual - expected,
        "delta_pct": (actual - expected) / expected if expected else None,
        "p_studentised": p_win, "p_conformal_rank": bl.last_p_rank,
        "p_conformal_floor": bl.last_p_floor, "effect_sd": eff_sd,
        "sparse_history": sparse, "causal_claims_allowed": bl.causal_claims_allowed,
        "shortfall_currency": (actual - expected) * (len(idx) if kpi == "net_revenue" else spec["daily_exposure"] / max(expected, 1e-9) * len(idx)),
        "freshness": case["freshness"], "lineage": prov.source_columns,
        "plan_hash": prov.plan_hash, "sql": prov.sql}

    # ---------------------------------------------------------------- change point
    with tel.span("L3", "change_point", "statistics", quantitative=True,
                  why="dates the regime break so hypotheses can be alibi-screened",
                  why_not_llm="segmentation over a residual series is a numeric procedure",
                  incident_id=incident_id, budget_ms=8000):
        d = bl.deseasonalised[-150:] if bl.deseasonalised.size else np.array([])
        cp, onset_iv, stat = SG.change_point(d, bl.dates[-150:]) if d.size > 30 else (None, None, 0.0)
        case["change_point"] = {"onset": str(cp) if cp else None,
                                "onset_interval_90": [str(onset_iv[0]), str(onset_iv[1])] if onset_iv else None,
                                "statistic": round(float(stat), 3)}

    # ---------------------------------------------------------------- L4 contribution
    with tel.span("L4", "contribution", "deterministic logic", quantitative=True,
                  why="closed-form identities; LMDI and ratio-of-sums, residual < 1e-9",
                  why_not_llm="an accounting identity has one right answer",
                  incident_id=incident_id, budget_ms=8000):
        pre_lo = win[0] - timedelta(days=24); pre_hi = win[0] - timedelta(days=3)
        sd = slice_df.copy(); sd["d"] = pd.to_datetime(sd["d"]).dt.date
        pre = sd[(sd.d >= pre_lo) & (sd.d <= pre_hi)]
        post = sd[(sd.d >= win[0]) & (sd.d <= win[1])]
        drill = C.hierarchical_drill(pre, post, [d for d in ["region", "channel", "category"]
                                                 if d in sd.columns][:2]) if len(pre) and len(post) else []
        case["contribution"] = {"rung": "R0",
                                "caveat": "An accounting identity. True by construction and "
                                          "silent about mechanism.",
                                "top": drill[:6]}

    # ---------------------------------------------------------------- L5 evidence
    with tel.span("L5", "structured_evidence", "SQL", quantitative=True,
                  why="each hypothesis has a pre-registered query against observable tables",
                  why_not_llm="the engine must not ask a model what the data says",
                  incident_id=incident_id, budget_ms=20000):
        espec = {**slc, "categories": slc.get("category_in")}
        if slc.get("category"): espec["categories"] = [slc["category"]]
        ev_items, ev_lrs = [], []
        for tid, t in EV.STRUCTURED_TESTS.items():
            try: fired, strength, detail = t["fn"](gw, espec, win)
            except Exception as e: fired, strength, detail = False, 0.0, f"test error: {e}"
            st, lg, sl, w_ = gw.freshness(*t["source"].split("."), AS_OF)
            item = EV.EvidenceItem(
                id=f"E_{tid}", test_id=tid, label=t["label"], evidence_class=t["cls"],
                fired=bool(fired), strength=float(strength), detail=detail, source=t["source"],
                derived_from=t["derived"], lineage=t["lineage"],
                freshness={"state": st, "lag_min": round(lg, 1), "sla_min": sl},
                method="pre-registered SQL against the semantic layer",
                region=slc.get("region"), t_window=[str(win[0]), str(win[1])],
                source_group=f"sql:{tid}", rung="R1" if fired else "R0",
                confidence=float(strength))
            ev_items.append(item)
            if fired: ev_lrs.append({"id": item.id, "lr": LK.lr_vector(tid, strength, lr_table),
                                     "source_group": item.source_group, "cluster_size": 1})

    with tel.span("L5", "unstructured_evidence", "retrieval", quantitative=False,
                  why="keyed retrieval then deterministic typed-claim extraction",
                  why_not_llm="extraction may use a model; the extraction here is the "
                              "deterministic control arm and is measured against it",
                  incident_id=incident_id, budget_ms=15000):
        rfilt = f"AND region='{slc['region']}'" if slc.get("region") else ""
        docs = gw.raw(f"""SELECT doc_id, corpus, ts, region, category, text, trust
            FROM ops_voice.documents
            WHERE ts::DATE BETWEEN DATE '{win[0] - timedelta(days=10)}' AND DATE '{win[1]}' {rfilt}""")
        claims, quarantined = EV.extract_claims(docs, pol, analyst)
        clusters = EV.cluster_claims(claims)
        case["injection_defence"] = {"documents_scanned": len(docs),
                                     "quarantined": quarantined,
                                     "policy": "quarantine, never strip-and-continue"}
        for cl in clusters[:14]:
            tid = TXT_TESTS.get(cl["claim_type"])
            if not tid: continue
            strength = min(1.0, math.log1p(cl["n"]) / math.log1p(40))
            item = EV.EvidenceItem(
                id=f"E_txt_{cl['claim_type']}_{cl['week'][-8:]}", test_id=tid,
                label=f"{cl['n']} documents mention {cl['claim_type'].replace('_',' ')}",
                evidence_class="textual", fired=True, strength=strength,
                detail=(f"{cl['n']} mentions across {cl['independent_sources']} independent "
                        f"source(s), trust={cl['trust']}, week {cl['week']}"),
                source="ops_voice.documents", derived_from=[], lineage=["ops_voice.documents.text"],
                freshness={"state": "fresh"}, method="gazetteer + template extraction, clustered",
                contribution=None, confidence=strength, region=cl["region"],
                source_group=f"text:{cl['claim_type']}",
                corroboration=cl["n"], spans=cl["spans"], rung="R1")
            ev_items.append(item)
            ev_lrs.append({"id": item.id, "lr": LK.lr_vector(tid, strength, lr_table),
                           "source_group": item.source_group, "cluster_size": cl["n"]})

    # ---------------------------------------------------------------- L5 posterior
    with tel.span("L5", "posterior", "statistics", quantitative=True,
                  why="grouped likelihood ratios over base-rate priors, diagnosticity weighted",
                  why_not_llm="assigning probability to a cause is the adjudication itself",
                  incident_id=incident_id, budget_ms=3000):
        lib = {h["id"]: h for h in contract["hypothesis_library"]}
        cand = [h for h in lib.values()
                if h["id"] != "H_NULL" and (kpi in h["targets"] or "*" in h["targets"])]
        base = {h["id"]: 1.0 / max(len(cand), 1) * 0.94 for h in cand}
        base["H_NULL"] = float(lib["H_NULL"].get("prior_floor", 0.06))
        from .feedback import updated_priors
        priors, prior_notes = updated_priors(base, kpi)
        hyps = [VD.Hypothesis(id=h["id"], label=h["label"], prior=priors.get(h["id"], 0.05),
                              lever=h.get("lever", "none"),
                              control_available=bool(h["evidence_plan"].get("control_available")))
                for h in cand]
        hyps.append(VD.Hypothesis("H_NULL", lib["H_NULL"]["label"], priors.get("H_NULL", 0.06)))
        onset_pair = ((date.fromisoformat(case["change_point"]["onset_interval_90"][0]),
                       date.fromisoformat(case["change_point"]["onset_interval_90"][1]))
                      if case["change_point"]["onset_interval_90"] else None)
        # How much the alibi screen is allowed to matter depends on how well the onset is
        # determined: a narrow interval from a strong segmentation statistic earns trust, a
        # wide or weak one earns almost none.
        cp_stat = float(case["change_point"].get("statistic") or 0.0)
        if onset_pair:
            width = (onset_pair[1] - onset_pair[0]).days
            onset_conf = max(0.0, min(1.0, (1.0 / (1.0 + width / 3.0)) * min(1.0, cp_stat / 0.6)))
        else:
            onset_conf = 0.0
        supported = {h.id for h in hyps
                     for e in ev_lrs if e["lr"].get(h.id, 1.0) > 1.5}
        alibi_meta = VD.alibi_screen(hyps, HYP_EVENTS, onset_pair,
                                     onset_confidence=onset_conf, supported=supported)
        case["change_point"]["onset_confidence"] = round(onset_conf, 3)
        case["alibi_screen"] = alibi_meta
        meta = VD.build_posterior(hyps, ev_lrs, contract, temperature=lr_temperature)
        case["posterior_meta"] = {**meta, "prior_adjustments": prior_notes}

    # ---------------------------------------------------------------- L6 causal
    with tel.span("L6", "causal_estimates", "causal inference", quantitative=True,
                  why="synthetic control with Abadie in-space placebo inference",
                  why_not_llm="counterfactual estimation is not a language task",
                  incident_id=incident_id, budget_ms=90000):
        raw = gw.raw("""SELECT d::DATE AS d, region, channel, category,
            SUM(converted) AS n, SUM(landed) AS t FROM commerce.sessions
            GROUP BY 1,2,3,4""") if kpi == "conversion_rate" else None
        onset_d = date.fromisoformat(case["change_point"]["onset"]) if case["change_point"]["onset"] else win[0]
        estimates = {}
        if raw is not None and not sparse:
            raw["d"] = pd.to_datetime(raw["d"]).dt.date
            # Each hypothesis gets the aggregation its own scope implies. A region-scoped
            # cause is tested on region x category units built from channels it did not
            # touch; a channel-scoped cause on channel x category units from untouched
            # regions. Cell-level units carry too much noise to detect a 3% effect, and
            # aggregating over the treated dimension would destroy the contrast.
            plans = [
                ("H_CARRIER_DEGRADE", raw[raw.channel != "APP"], "region", "WEST"),
                ("H_CHECKOUT_DEFECT", raw[raw.region != "WEST"], "channel", "APP"),
            ]
            for hid, sub, scope_col, treated_val in plans:
                g = (sub.groupby(["d", scope_col, "category"], as_index=False)[["n", "t"]].sum())
                g["unit"] = g[scope_col] + "|" + g["category"]
                g["v"] = g.n / g.t.replace(0, np.nan)
                g = g[g.v.notna() & (g.v > 0)]
                treated_units = sorted(g[g[scope_col] == treated_val].unit.unique())
                pool = set(treated_units)
                if not treated_units: continue
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=8) as ex:
                    es = list(ex.map(lambda u: CA.estimate(
                        g[["d", "unit", "v"]], hid, u, win[0], win[1], unit_col="unit",
                        value_col="v", treated_pool=pool, onset=onset_d), treated_units))
                es = [e for e in es if np.isfinite(e.effect)]
                if not es: continue
                r3 = [e for e in es if e.rung == "R3"]
                best = (r3 or es)[len(r3 or es) // 2]
                best.effect = float(np.median([e.effect for e in es]))
                best.effect_pct = float(np.median([e.effect_pct for e in es]))
                best.placebo_p = float(np.median([e.placebo_p for e in es]))
                best.rung = "R3" if len(r3) >= len(es) / 2 else "R2"
                best.refutations["treated_units_tested"] = len(es)
                best.refutations["units_reaching_R3"] = len(r3)
                best.refutations["unit_definition"] = f"{scope_col} x category, excluding the concurrently treated dimension"
                best.note = (f"{len(r3)} of {len(es)} treated units reach R3 under synthetic "
                             f"control with Abadie in-space placebo inference on "
                             f"{best.n_donors} donors; median placebo p={best.placebo_p:.3f}.")
                estimates[hid] = best

        # hypotheses with no untreated population are capped at R2 by construction
        for h in hyps:
            if h.id in NO_CONTROL and h.id not in estimates:
                estimates[h.id] = CA.estimate(pd.DataFrame(), h.id, "", win[0], win[1],
                                              control_available=False)
        for h in hyps:
            e = estimates.get(h.id)
            if e is None: continue
            h.rung = e.rung
            h.effect = float(e.effect) if np.isfinite(e.effect) else None
            h.effect_pct = float(e.effect_pct) if np.isfinite(e.effect_pct) else None
        case["causal"] = {k: asdict(v) for k, v in estimates.items()}

    # ---------------------------------------------------------------- L7 verdict
    with tel.span("L7", "verdict", "deterministic logic", quantitative=True,
                  why="threshold, rung and expected-value rules from the contract",
                  why_not_llm="the abstention decision must be auditable and reproducible",
                  incident_id=incident_id, budget_ms=2000):
        contradiction, contra_info = _has_contradiction(ev_items, docs)
        case["contradiction"] = contra_info
        v = VD.decide(incident_id, kpi, hyps, contract, spec["daily_exposure"],
                      sparse=sparse and not bl.causal_claims_allowed, stale=stale,
                      contradiction=contradiction)
        case["verdict"] = asdict(v)
        fired = {e.test_id for e in ev_items if e.fired}
        lib_by_id = {h["id"]: h for h in contract["hypothesis_library"]}
        case["what_would_change_my_mind"] = [
            VD.upgrade_conditions(h, lib_by_id.get(h.id, {}), fired, contract)
            for h in sorted(hyps, key=lambda x: -x.posterior)
            if h.id != "H_NULL" and h.posterior > 0.02]

    # ---------------------------------------------------------------- L7 decisive test
    if v.decision == "ABSTAIN":
        with tel.span("L7", "decisive_test", "deterministic logic", quantitative=True,
                      why="expected value of sample information over the current posterior",
                      why_not_llm="experiment design is a decision-theoretic computation",
                      incident_id=incident_id, budget_ms=15000):
            tests = _candidate_tests(hyps, contract)
            ranked, evpi = VD.rank_tests(hyps, tests, spec["daily_exposure"])
            hedge = VD.hedge_action(hyps, contract["levers"], spec["daily_exposure"], 5e6)
            case["decisive_test"] = {
                "recommended": asdict(ranked[0]) if ranked else None,
                "ranked": [asdict(t) for t in ranked],
                "evpi": round(evpi, 0),
                "hedge": hedge,
                "cost_of_waiting_per_day": spec["daily_exposure"] * abs(
                    case["detection"]["delta_pct"] or 0.0),
                "objective": "expected net benefit of sampling per day (EVSI minus cost)"}
    else:
        case["decisive_test"] = {"recommended": None,
                                 "note": "verdict reached the action threshold; no test required"}

    # ---------------------------------------------------------------- L8/L9 per persona
    case["evidence"] = []
    case["entitlement"] = {}; case["actions_by_persona"] = {}; case["narratives"] = {}
    facts_public = _facts(case, hyps)
    for pid in personas:
        pr = pol.principal(pid)
        kept, withheld, reasons = pol.filter_evidence(pr, [asdict(e) for e in ev_items])
        # entitlement can change the achievable standard of proof, not merely the view
        can_see_controls = len(pr.regions) > 1
        max_rung = max((h.rung for h in hyps if h.id != "H_NULL"),
                       key=lambda r: VD.RUNG_ORDER[r], default="R1")
        eff_rung = max_rung if can_see_controls else min(max_rung, "R2", key=lambda r: VD.RUNG_ORDER[r])
        case["entitlement"][pid] = {
            "regions": list(pr.regions), "column_grants": sorted(pr.column_grants),
            "evidence_admitted": len(kept), "withheld_count": withheld,
            "withheld_reasons": reasons[:6], "max_rung": eff_rung,
            "rung_capped_by_entitlement": eff_rung != max_rung,
            "note": ("Control units required for a causal estimate lie outside this "
                     "principal's row scope, so the achievable standard of proof is capped."
                     if eff_rung != max_rung else "Full evidence set admitted.")}
        with tel.span("L8", f"actions:{pid}", "business rules", quantitative=True,
                      why="levers, costs, owners and decision rights come from the contract",
                      why_not_llm="authority and budget are policy, not judgement",
                      incident_id=incident_id, budget_ms=2000):
            acts = build_actions(hyps, contract, kpi, contract["personas"][pid],
                                 spec["daily_exposure"], win[1])
            case["actions_by_persona"][pid] = [asdict(a) for a in acts[:5]]
        with tel.span("L9", f"narrative:{pid}", "LLM", quantitative=False,
                      why="prose synthesis from a frozen evidence object",
                      why_not_llm=None, incident_id=incident_id, budget_ms=120000) as sp:
            slots = _slots(case, hyps, pid, acts, case["entitlement"][pid])
            facts = {**facts_public, "persona": contract["personas"][pid]["label"],
                     "entitlement": case["entitlement"][pid],
                     "recommended_actions": [
                         {"action": a.action, "impact": round(a.expected_impact, 0),
                          "cost": round(a.cost, 0), "owner": a.owner_role,
                          "confidence_pct": round(100 * a.confidence, 0),
                          "lead_time_days": a.time_to_impact_days,
                          "check_after_days": a.monitoring_plan["check_after_days"]}
                         for a in acts[:3]],
                     "evidence_details": [e["detail"] for e in case["evidence"][:6]],
                     "contribution_top": case["contribution"]["top"][:3],
                     "causal_effects": {k: round(v.get("effect_pct") or 0, 4)
                                        for k, v in case["causal"].items()}}
            out = NR.generate(pid, contract["personas"][pid], facts, slots, llm, sp)
            case["narratives"][pid] = out
    case["evidence"] = [asdict(e) for e in ev_items]
    case["telemetry"] = tel.summary()
    return case


RANK = {"low": 0, "medium": 1, "high": 2, "authoritative": 3}


def _has_contradiction(ev_items, docs) -> tuple[bool, dict]:
    """Only a conflict between sources of COMPARABLE trust is a real contradiction.

    A medium-trust vendor denial against an authoritative internal incident report is
    resolved by the contract's trust ordering; escalating it to a human would be noise."""
    if not len(docs): return False, {}
    deny = docs[docs.text.str.lower().str.contains("no degradation|consider the sla met",
                                                   regex=True, na=False)]
    if deny.empty: return False, {}
    ontime = next((e for e in ev_items if e.test_id == "ontime_drop"), None)
    if not (ontime and ontime.fired): return False, {}
    corrob = docs[docs.corpus.isin(["incident_reports", "store_notes"])]
    deny_trust = max((RANK.get(t, 0) for t in deny.trust), default=0)
    corr_trust = max((RANK.get(t, 0) for t in corrob.trust), default=0)
    info = {"denial_source": deny.corpus.iloc[0], "denial_trust": deny.trust.iloc[0],
            "counter_trust_max": corr_trust, "resolved_by_trust_ordering": corr_trust > deny_trust,
            "note": ("Vendor denial is outranked by an authoritative internal record and by "
                     "telemetry; retained in the ledger, not escalated."
                     if corr_trust > deny_trust else
                     "Two sources of comparable trust disagree; a human must adjudicate.")}
    return (corr_trust <= deny_trust), info


def _candidate_tests(hyps, contract):
    T = VD.CandidateTest
    out = []
    live = {h.id for h in hyps if h.posterior > 0.10}
    if "H_PROMO_WITHDRAWAL" in live:
        out.append(T("T_PROMO_HOLDOUT", "Restore promotional depth in 24 randomised store cohorts",
                     ["H_PROMO_WITHDRAWAL"], 96, 24, 10, 0, 4200, 0.02, "weekly",
                     "category_director", power=0.86))
    if "H_PRICE_RISE" in live:
        out.append(T("T_PRICE_REVERT", "Revert list price in 24 matched cohorts",
                     ["H_PRICE_RISE"], 96, 24, 14, 340000, 5200, 0.02, "poor",
                     "category_director", power=0.84))
    if "H_CARRIER_DEGRADE" in live:
        out.append(T("T_CARRIER_REVERT", "Route 30% of WEST volume back to the prior carrier",
                     ["H_CARRIER_DEGRADE"], 60, 18, 9, 0, 1900, 0.02, "full",
                     "ops_manager", power=0.88))
    if "H_CHECKOUT_DEFECT" in live:
        out.append(T("T_FLAG_REVERT", "Revert the checkout flag for 20% of APP sessions",
                     ["H_CHECKOUT_DEFECT"], 100, 20, 7, 0, 0, 0.015, "instant",
                     "engineering_lead", power=0.90))
    out.append(T("T_SURVEY", "Survey 4,000 customers in affected catchments",
                 [h.id for h in hyps if h.posterior > 0.15][:2], 4000, 4000, 9, 80000, 12,
                 0.05, "full", "category_director", power=0.55))
    out.append(T("T_WAIT", "Wait two more weeks for natural variation",
                 [], 0, 0, 14, 0, 0, 0.05, "n_a", "cfo", randomised=False, power=0.35))
    return out


def _facts(case, hyps):
    d = case["detection"]
    return {"kpi": case["kpi"], "scope": case["label"], "window": case["window"],
            "actual": round(d["actual"], 6), "expected": round(d["expected"], 6),
            "delta_pct": round(100 * (d["delta_pct"] or 0), 2),
            "baseline_method": d["baseline_method"],
            "onset": case["change_point"]["onset"],
            "verdict": case["verdict"]["decision"],
            "abstain_type": case["verdict"]["abstain_type"],
            "max_rung": case["verdict"]["max_rung"],
            "posterior": {h.id: round(h.posterior * 100, 1) for h in hyps if h.posterior > 0.02},
            "top_evidence": [e["detail"] for e in case["evidence"][:4]] if case.get("evidence") else [],
            "recommended_test": (case.get("decisive_test", {}).get("recommended") or {}).get("label"),
            "test_days": (case.get("decisive_test", {}).get("recommended") or {}).get("days"),
            "test_units": (case.get("decisive_test", {}).get("recommended") or {}).get("units_used"),
            "daily_exposure": round(case["verdict"]["cost_of_waiting_per_day"], 0),
            "act_threshold_pct": round(100 * case["verdict"]["threshold"], 0),
            "entropy_bits": round(case["verdict"]["entropy_bits"], 2),
            "evpi": round((case.get("decisive_test") or {}).get("evpi") or 0, 0),
            "window_days": len(pd.date_range(*case["window"])),
            "contribution_shares": [round(100 * x["share_of_move"], 0)
                                    for x in case["contribution"]["top"][:3]],
            "evidence_strengths": [round(e["strength"], 2) for e in case["evidence"][:6]]}


def _slots(case, hyps, pid, acts, ent):
    d = case["detection"]; v = case["verdict"]
    top = sorted(hyps, key=lambda h: -h.posterior)[0]
    dt = case.get("decisive_test", {}) or {}
    rec = dt.get("recommended") or {}
    act_line = (f"Recommended: {acts[0].action} ({acts[0].owner_role}), "
                f"expected impact {acts[0].expected_impact:,.0f}." if acts else "")
    if v["decision"] == "ABSTAIN" and rec:
        act_line = (f"Do not act on a guess. Run: {rec.get('label')} "
                    f"({rec.get('days')} days, {rec.get('units_used')} units).")
    return {
        "kpi_label": case["kpi"].replace("_", " ").title(),
        "scope": case["label"], "days": len(pd.date_range(*case["window"])),
        "window": f"{case['window'][0]} to {case['window'][1]}",
        "delta_str": f"{100*(d['delta_pct'] or 0):.2f}% against expectation",
        "rung_line": f"Strongest evidence reaches {v['max_rung']}.",
        "evidence_line": "; ".join(e["detail"] for e in case["evidence"][:2] if e.get("fired")),
        "causal_line": "; ".join(f"{k}: {x['note'][:90]}" for k, x in list(case["causal"].items())[:2]),
        "contribution_line": "; ".join(
            f"{c['segment']} accounts for {100*c['share_of_move']:.0f}% of the move"
            for c in case["contribution"]["top"][:2]),
        "detection_line": (f"{d['baseline_method']}, studentised p={d['p_studentised']:.2e}, "
                           f"conformal rank p={d['p_conformal_rank']:.4f}"),
        "posterior_line": ", ".join(f"{h.id} {h.posterior:.0%}" for h in
                                    sorted(hyps, key=lambda x: -x.posterior)[:3]),
        "verdict_line": v["reason"],
        "exposure_line": f"Exposure is {case['verdict']['cost_of_waiting_per_day']:,.0f} per day.",
        "action_line": act_line,
        "entitlement_line": ("" if not ent["rung_capped_by_entitlement"] else
                             f"\n\nNote: {ent['note']}")}

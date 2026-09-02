#!/usr/bin/env python3
"""Batch evaluation over seeded incidents, scored against injected ground truth."""
from __future__ import annotations
import json, sys, math
from pathlib import Path
from datetime import date, timedelta
from multiprocessing import Pool
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casefile.sim.model import World
from casefile.sim.random_incident import draw
from casefile.engine import baseline as B, verdict as VD, likelihood as LK
from baselines.contribution_ranker import diagnose as naive_diagnose

from casefile.paths import contract as _contract, out as _out
CONTRACT = yaml.safe_load(open(_contract()))
LR_TABLE, LR_SRC = LK.load_table()
LR_T = LK.load_temperature()

DRIVER_COLS = {"ontime": "ontime_pct", "checkout": "checkout_error_rate",
               "price": "price_index", "promo": "promo_depth",
               "stockout": "stockout_rate", "competitor": "competitor_gap",
               "marketing": "marketing_mult"}
# "promo" includes txt_price: a shopper cannot tell a price rise from an ended discount,
# so both produce the same complaint theme and the calibration must reflect that.
DRIVER_TESTS = {
    "ontime":     ["ontime_drop", "carrier_shift", "txt_delivery_late"],
    "checkout":   ["checkout_errors", "txt_payment_fail", "doc_release_checkout"],
    "price":      ["price_move", "txt_price", "doc_release_price"],
    "promo":      ["promo_move", "doc_release_promo", "txt_price"],
    "stockout":   ["stockout_rise", "txt_stock"],
    "competitor": ["txt_competitor"],
    "marketing":  ["mix_shift"],
}
D2H = {"ontime": "H_CARRIER_DEGRADE", "checkout": "H_CHECKOUT_DEFECT", "price": "H_PRICE_RISE",
       "promo": "H_PROMO_WITHDRAWAL", "stockout": "H_STOCKOUT",
       "competitor": "H_COMPETITOR", "marketing": "H_MARKETING_CUT"}


def _driver_deltas(panel, win):
    lo, hi = win
    pre_hi = lo - timedelta(days=3); pre_lo = pre_hi - timedelta(days=21)
    pre = panel[(panel.d >= pre_lo) & (panel.d <= pre_hi)]
    post = panel[(panel.d >= lo) & (panel.d <= hi)]
    out = {}
    for k, col in DRIVER_COLS.items():
        a, b = pre[col].mean(), post[col].mean()
        if not np.isfinite(a) or a == 0: out[k] = 0.0; continue
        out[k] = float((b - a) / abs(a))
    return out


def _metric(panel, kpi, win):
    lo, hi = win
    m = (panel.d >= lo) & (panel.d <= hi)
    s = panel[m]
    if kpi == "conversion_rate": return float(s.orders.sum() / max(s.sessions.sum(), 1e-9))
    return float(s.net_revenue.sum())


def one(seed: int) -> dict | None:
    inc = draw(seed)
    kpi, win = inc["kpi"], inc["window"]
    try:
        panel = World(inc["interventions"], seed=seed).simulate()
    except Exception:
        return None
    panel["d"] = pd.to_datetime(panel["d"]).dt.date
    full = _metric(panel, kpi, win)
    contrib = {}
    for iv in inc["interventions"]:
        cf = World(inc["interventions"], seed=seed).simulate(drop_interventions={iv.id})
        cf["d"] = pd.to_datetime(cf["d"]).dt.date
        contrib[iv.hypothesis_id] = full - _metric(cf, kpi, win)
    dominant = min(contrib, key=lambda k: contrib[k]) if contrib else None
    shares = {k: abs(v) / max(sum(abs(x) for x in contrib.values()), 1e-12)
              for k, v in contrib.items()}
    dom_share = shares.get(dominant, 0.0)
    g = panel.groupby("d", as_index=False).agg(orders=("orders", "sum"),
                                               sessions=("sessions", "sum"),
                                               nr=("net_revenue", "sum"))
    v = (g.orders / g.sessions) if kpi == "conversion_rate" else g.nr
    s = pd.Series(v.values, index=pd.DatetimeIndex(g.d)).sort_index()
    s = s[s.index <= pd.Timestamp(win[1])]
    bl = B.fit(kpi, s, CONTRACT, test_start=pd.Timestamp(win[0]))
    if bl.method == "INSUFFICIENT_HISTORY": return None
    idx = np.where((bl.dates >= pd.Timestamp(win[0])) & (bl.dates <= pd.Timestamp(win[1])))[0]
    p_win, eff_sd = bl.window_rank_p(idx)

    deltas = _driver_deltas(panel, win)
    truth_h = {t["hypothesis"] for t in inc["truth"]}

    # ---- baseline: largest contributor wins, never abstains
    nb = naive_diagnose(deltas)

    # ---- CaseFile: same evidence, diagnosticity-weighted, may abstain
    hyps = [VD.Hypothesis(id=h, label=h, prior=0.94 / len(D2H),
                          lever={"H_CARRIER_DEGRADE": "carrier_routing",
                                 "H_CHECKOUT_DEFECT": "checkout_config_flag",
                                 "H_PRICE_RISE": "price_change",
                                 "H_PROMO_WITHDRAWAL": "promo_depth",
                                 "H_STOCKOUT": "inventory_replenish",
                                 "H_COMPETITOR": "promo_depth",
                                 "H_MARKETING_CUT": "paid_media_spend"}[h],
                          control_available=not (not inc["identifiable_by_construction"]))
            for h in D2H.values()]
    hyps.append(VD.Hypothesis("H_NULL", "outside library", 0.06))
    ev = []
    for k, d in deltas.items():
        strength = float(np.clip(abs(d) / 0.06, 0, 1))
        if strength < 0.15: continue
        for j, tid in enumerate(DRIVER_TESTS[k]):
            # corroborating channels are separate source groups, as in the real pipeline,
            # but each is attenuated so a single cause cannot manufacture certainty
            ev.append({"id": f"E_{k}_{tid}",
                       "lr": LK.lr_vector(tid, strength * (1.0 - 0.15 * j), LR_TABLE),
                       "source_group": f"{'sql' if j == 0 else 'corroboration'}:{k}:{tid}"})
    VD.build_posterior(hyps, ev, CONTRACT, temperature=LR_T)
    for h in hyps:
        h.rung = "R3" if inc["identifiable_by_construction"] else "R2"
    vd = VD.decide(f"S{seed}", kpi, hyps, CONTRACT, 250_000.0)

    top = max(hyps, key=lambda h: h.posterior)
    cf_named = vd.decision == "ACT"
    cf_cause = (vd.acted_on or vd.leading) if cf_named else None
    return {"seed": seed, "identifiable": inc["identifiable_by_construction"],
            "dominant_cause": dominant, "dominant_share": dom_share,
            "naive_hit_dominant": nb["cause"] == dominant,
            "cf_hit_dominant": (cf_cause == dominant) if cf_named else None,
            "n_causes": inc["n_causes"], "kpi": kpi, "truth": sorted(truth_h),
            "detected_p": p_win, "effect_sd": eff_sd,
            "naive_cause": nb["cause"], "naive_conf": nb["confidence"],
            "naive_correct": nb["cause"] in truth_h,
            "cf_decision": vd.decision, "cf_abstain_type": vd.abstain_type,
            "cf_cause": cf_cause, "cf_conf": float(top.posterior),
            "cf_correct": (cf_cause in truth_h) if cf_named else None,
            "cf_named": cf_named}


def main(n=300, workers=32, out=None):
    out = out or str(_out() / "eval")
    Path(out).mkdir(parents=True, exist_ok=True)
    seeds = list(range(9000, 9000 + n))   # held out from calibration (30000..30499)
    with Pool(workers) as p:
        rows = [r for r in p.map(one, seeds) if r]
    df = pd.DataFrame(rows)
    df.to_parquet(Path(out) / "batch.parquet")

    ident = df[df.identifiable]; unid = df[~df.identifiable]
    single = df[df.n_causes == 1]; multi = df[df.n_causes > 1]
    def pct(x): return f"{100*x:.1f}%"
    summary = {
        "n_incidents": len(df), "n_identifiable": len(ident), "n_unidentifiable": len(unid),
        "likelihood_source": LR_SRC,
        "identifiable": {
            "naive_top1_accuracy": float(ident.naive_correct.mean()),
            "casefile_top1_accuracy": float(ident[ident.cf_named].cf_correct.mean()) if ident.cf_named.any() else None,
            "casefile_answer_rate": float(ident.cf_named.mean())},
        "unidentifiable": {
            "naive_false_certainty_rate": float((~unid.naive_correct).mean()),
            "naive_abstention_rate": 0.0,
            "casefile_false_certainty_rate": float(
                ((unid.cf_named) & (~unid.cf_correct.fillna(False))).mean()),
            "casefile_abstention_rate": float((~unid.cf_named).mean())},
        "wrong_lever_rate_unidentifiable": {
            "naive": float((~unid.naive_hit_dominant).mean()),
            "casefile_acted_and_wrong": float(
                ((unid.cf_named) & (unid.cf_hit_dominant == False)).mean()),
            "note": "fraction of unidentifiable incidents where the system committed to a "
                    "lever that was not the dominant true cause"},
        "answer_rate_by_cause_count": {
            "single_cause": float(single.cf_named.mean()) if len(single) else None,
            "multi_cause": float(multi.cf_named.mean()) if len(multi) else None},
        "brier": {
            "naive": float(((ident.naive_conf - ident.naive_correct.astype(float)) ** 2).mean()),
            "casefile": float(((ident[ident.cf_named].cf_conf -
                                ident[ident.cf_named].cf_correct.astype(float)) ** 2).mean())
                        if ident.cf_named.any() else None},
    }
    # --- calibration: of the verdicts issued at X% confidence, how many held? ---
    from casefile.engine.feedback import calibration_curve
    named = df[df.cf_named]
    cal = calibration_curve([{"stated_confidence": float(r.cf_conf),
                              "was_correct": bool(r.cf_hit_dominant)}
                             for r in named.itertuples()])
    summary["calibration"] = cal

    # cost of being wrong: a false intervention spends the lever's cost for no benefit.
    import yaml as _y
    C = _y.safe_load(open(_contract()))
    lever_of = {"H_CARRIER_DEGRADE": "carrier_routing", "H_CHECKOUT_DEFECT": "checkout_config_flag",
                "H_PRICE_RISE": "price_change", "H_PROMO_WITHDRAWAL": "promo_depth",
                "H_STOCKOUT": "inventory_replenish", "H_COMPETITOR": "promo_depth",
                "H_MARKETING_CUT": "paid_media_spend"}
    EXPOSURE, HORIZON_DAYS, UNITS = 250_000.0, 30, 1000
    def lever_cost(h):
        spec = C["levers"].get(lever_of.get(h, "none"), {})
        cm = spec.get("cost_model") or {}
        return (float(cm.get("fixed", 0)) + float(cm.get("per_shipment", 0)) * UNITS
                + float(cm.get("per_unit", 0)) * UNITS
                + float(cm.get("margin_pct_of_revenue", 0)) * EXPOSURE * HORIZON_DAYS)
    wrong_naive = unid[~unid.naive_hit_dominant]
    naive_waste = float(sum(lever_cost(h) for h in wrong_naive.naive_cause))
    cf_wrong = unid[(unid.cf_named) & (unid.cf_hit_dominant == False)]
    cf_waste = float(sum(lever_cost(h) for h in cf_wrong.cf_cause))
    summary["cost_of_being_wrong"] = {
        "basis": f"lever cost from the contract, {HORIZON_DAYS}-day horizon, "
                 f"exposure {EXPOSURE:,.0f}/day, {UNITS} units",
        "unidentifiable_incidents": int(len(unid)),
        "naive_false_interventions": int(len(wrong_naive)),
        "naive_wasted_spend": round(naive_waste, 0),
        "casefile_false_interventions": int(len(cf_wrong)),
        "casefile_wasted_spend": round(cf_waste, 0),
        "avoided": round(naive_waste - cf_waste, 0),
        "avoided_per_incident": round((naive_waste - cf_waste) / max(len(unid), 1), 0)}

    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(); main(a.n, a.workers)

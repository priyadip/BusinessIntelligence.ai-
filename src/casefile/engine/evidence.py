"""
Evidence tests, and the ledger that weighs them.

Every test is a deterministic query against OBSERVABLE tables. The engine never reads the
simulator's latent drivers; if a hypothesis has no observable proxy it cannot be tested,
which is itself a finding (abstain type: missing-evidence).

Two design choices carry most of the weight here.

DIAGNOSTICITY, not consistency. An LLM asked to explain a drop collects everything
consistent with its favourite story. Heuer's method inverts that: evidence consistent with
EVERY hypothesis has no diagnostic value, no matter how compelling it reads. "Complaints
rose" fits a price rise, a carrier failure, a competitor promotion and bad weather equally,
so its likelihood-ratio vector is flat and it contributes nothing to the posterior. The
ledger shows that explicitly rather than quietly including it.

CLUSTERING BEFORE SCORING. Forty reviews about late delivery from one week are one piece
of evidence with forty mentions, not forty independent pieces. Multiplying their likelihood
ratios would drive the posterior to certainty on the strength of a single syndicated feed.
Claims are clustered by (entity, time bucket, claim type); each cluster contributes ONE
log-likelihood-ratio, taking the strongest member, and the corroboration count is displayed
beside it without entering the arithmetic.
"""
from __future__ import annotations
import json, math, re
import numpy as np, pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta


@dataclass
class EvidenceItem:
    id: str
    test_id: str
    label: str
    evidence_class: str          # statistical|temporal|causal|textual|business-rule|historical
    fired: bool
    strength: float              # 0..1, graded outcome of the test
    detail: str
    source: str
    derived_from: list[str] = field(default_factory=list)
    lineage: list[str] = field(default_factory=list)
    freshness: dict = field(default_factory=dict)
    method: str = ""
    contribution: float | None = None
    confidence: float | None = None
    region: str | None = None
    t_window: tuple | None = None
    source_group: str = ""
    corroboration: int = 1
    spans: list = field(default_factory=list)
    rung: str = "R1"
    quarantined: bool = False


# ============================================================== structured tests
def _pre_post(gw, sql: str, w, pre_days=21, gap=3):
    lo, hi = w
    pre_hi = lo - timedelta(days=gap); pre_lo = pre_hi - timedelta(days=pre_days)
    df = gw.raw(sql)
    df["d"] = pd.to_datetime(df["d"]).dt.date
    pre = df[(df.d >= pre_lo) & (df.d <= pre_hi)]
    post = df[(df.d >= lo) & (df.d <= hi)]
    return pre, post


def _rate(df, num, den):
    d = df[den].sum()
    return float(df[num].sum() / d) if d else float("nan")


def _delta(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or a == 0: return 0.0
    return (b - a) / abs(a)


STRUCTURED_TESTS = {}
def stest(tid, label, cls, source, derived, lineage):
    def deco(fn):
        STRUCTURED_TESTS[tid] = dict(fn=fn, label=label, cls=cls, source=source,
                                     derived=derived, lineage=lineage)
        return fn
    return deco


@stest("ontime_drop", "Fulfilment on-time fell in the affected cohort", "statistical",
       "ops_voice.shipment_events", [], ["ops_voice.shipment_events.delivered_ts"])
def _(gw, spec, w):
    rf = f"AND region='{spec['region']}'" if spec.get("region") else ""
    pre, post = _pre_post(gw, f"""SELECT promised_ts::DATE AS d,
        SUM(CASE WHEN delivered_ts<=promised_ts THEN 1 ELSE 0 END) n, COUNT(*) t
        FROM ops_voice.shipment_events WHERE status='DELIVERED' {rf} GROUP BY 1""", w)
    a, b = _rate(pre, "n", "t"), _rate(post, "n", "t")
    d = b - a
    return (d < -0.008, min(1.0, abs(d) / 0.05), f"on-time {a:.1%} -> {b:.1%} ({d*100:+.2f}pp)")


@stest("carrier_shift", "Shipment share moved to a different carrier", "temporal",
       "ops_voice.shipment_events", [], ["ops_voice.shipment_events.carrier_id"])
def _(gw, spec, w):
    rf = f"AND region='{spec['region']}'" if spec.get("region") else ""
    pre, post = _pre_post(gw, f"""SELECT promised_ts::DATE AS d, carrier_id, COUNT(*) t
        FROM ops_voice.shipment_events WHERE 1=1 {rf} GROUP BY 1,2""", w)
    if pre.empty or post.empty: return False, 0.0, "no shipment data"
    a = pre.groupby("carrier_id").t.sum() / pre.t.sum()
    b = post.groupby("carrier_id").t.sum() / post.t.sum()
    sh = (b - a.reindex(b.index).fillna(0)).sort_values(ascending=False)
    top, mv = sh.index[0], float(sh.iloc[0])
    return (mv > 0.08, min(1.0, mv / 0.4), f"share to {top} {mv*100:+.1f}pp")


@stest("checkout_errors", "Checkout error rate rose", "statistical",
       "commerce.checkout_events", [], ["commerce.checkout_events.errors"])
def _(gw, spec, w):
    cf = f"AND channel='{spec['channel']}'" if spec.get("channel") else ""
    pre, post = _pre_post(gw, f"""SELECT d::DATE AS d, SUM(errors) n, SUM(attempts) t
        FROM commerce.checkout_events WHERE 1=1 {cf} GROUP BY 1""", w)
    a, b = _rate(pre, "n", "t"), _rate(post, "n", "t")
    return (b > a * 1.5 and b - a > 0.002, float(np.clip(_delta(a, b) / 4.0, 0.0, 1.0)),
            f"checkout error {a:.3%} -> {b:.3%}")


@stest("price_move", "List price index moved", "business-rule",
       "commerce.price_promo_daily", [], ["commerce.price_promo_daily.list_price_index"])
def _(gw, spec, w):
    cf = (f"AND category IN ({','.join(repr(c) for c in spec['categories'])})"
          if spec.get("categories") else "")
    pre, post = _pre_post(gw, f"""SELECT d::DATE AS d, AVG(list_price_index) v
        FROM commerce.price_promo_daily WHERE 1=1 {cf} GROUP BY 1""", w)
    a, b = pre.v.mean(), post.v.mean()
    d = _delta(a, b)
    return (d > 0.015, min(1.0, d / 0.08), f"price index {a:.4f} -> {b:.4f} ({d*100:+.2f}%)")


@stest("promo_move", "Promotional depth changed", "business-rule",
       "commerce.price_promo_daily", [], ["commerce.price_promo_daily.promo_depth"])
def _(gw, spec, w):
    cf = (f"AND category IN ({','.join(repr(c) for c in spec['categories'])})"
          if spec.get("categories") else "")
    pre, post = _pre_post(gw, f"""SELECT d::DATE AS d, AVG(promo_depth) v
        FROM commerce.price_promo_daily WHERE 1=1 {cf} GROUP BY 1""", w)
    a, b = pre.v.mean(), post.v.mean()
    return (b - a < -0.008, min(1.0, abs(b - a) / 0.05), f"promo depth {a:+.3f} -> {b:+.3f}")


@stest("stockout_rise", "Out-of-stock minutes rose", "statistical",
       "ops_voice.inventory_daily", [], ["ops_voice.inventory_daily.oos_minutes"])
def _(gw, spec, w):
    rf = f"AND region='{spec['region']}'" if spec.get("region") else ""
    pre, post = _pre_post(gw, f"""SELECT d::DATE AS d, AVG(oos_minutes) v
        FROM ops_voice.inventory_daily WHERE 1=1 {rf} GROUP BY 1""", w)
    a, b = pre.v.mean(), post.v.mean()
    d = _delta(a, b)
    return (d > 0.25, min(1.0, d / 1.5), f"oos minutes/day {a:.0f} -> {b:.0f} ({d*100:+.0f}%)")


@stest("source_reconciles", "Commerce reconciles to the finance ledger", "business-rule",
       "finance_erp.fin_revenue", [], ["finance_erp.fin_revenue.net_revenue_fin"])
def _(gw, spec, w):
    com = gw.raw(f"""SELECT strftime(order_ts,'%G-W%V') wk,
        SUM(gross_amount-discount_amount-returns_amount) v FROM commerce.orders
        WHERE order_ts::DATE BETWEEN DATE '{w[0]}' AND DATE '{w[1]}' GROUP BY 1""")
    fin = gw.raw("SELECT iso_week wk, SUM(net_revenue_fin) v FROM finance_erp.fin_revenue GROUP BY 1")
    j = com.merge(fin, on="wk", suffixes=("_c", "_f"))
    # only weeks fully covered by BOTH sources may be compared; a partial week is not a gap
    wm = gw.raw("""SELECT max_event_ts FROM meta.watermarks
                   WHERE source='finance_erp' AND table_name='fin_revenue'""")
    fin_wm = str(wm.max_event_ts.iloc[0])[:10] if len(wm) else str(w[1])
    full = gw.raw(f"""SELECT strftime(order_ts,'%G-W%V') wk
        FROM commerce.orders WHERE order_ts::DATE BETWEEN DATE '{w[0]}' AND DATE '{w[1]}'
        GROUP BY 1
        HAVING COUNT(DISTINCT order_ts::DATE)=7
           AND MAX(order_ts::DATE) <= DATE '{fin_wm}'""")
    j = j[j.wk.isin(set(full.wk))]
    if j.empty: return False, 0.0, "no week is complete in both sources yet"
    if j.empty: return False, 0.0, "no overlapping weeks"
    d = abs(j.v_f.sum() - j.v_c.sum()) / max(j.v_c.sum(), 1)
    return (d > 0.005, min(1.0, d / 0.05), f"ledger delta {d*100:.3f}%")


@stest("mix_shift", "Category or channel mix shifted", "statistical",
       "commerce.orders", [], ["commerce.orders.category"])
def _(gw, spec, w):
    pre, post = _pre_post(gw, """SELECT order_ts::DATE AS d, category,
        SUM(gross_amount-discount_amount-returns_amount) v FROM commerce.orders GROUP BY 1,2""", w)
    if pre.empty or post.empty: return False, 0.0, "no data"
    a = pre.groupby("category").v.sum() / pre.v.sum()
    b = post.groupby("category").v.sum() / post.v.sum()
    tv = float((b - a.reindex(b.index).fillna(0)).abs().sum() / 2)
    return (tv > 0.02, min(1.0, tv / 0.10), f"total variation of mix {tv*100:.2f}pp")


# ============================================================== unstructured
CLAIM_TEMPLATES = {
    "delivery_late":  ["late", "delay", "not received", "courier", "in-transit", "hub"],
    "payment_fail":   ["payment", "upi", "declined", "checkout", "error", "failed"],
    "price":          ["price", "cheaper", "expensive", "cost"],
    "stock":          ["stock", "out of stock", "unavailable"],
    "competitor":     ["competitor", "rival", "bundle", "combo"],
}
INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions", r"you are now", r"maintenance mode",
    r"system prompt", r"do not report", r"</?document>", r"\[INST\]", r"assistant:",
]


def scan_injection(text: str) -> tuple[bool, str]:
    t = text.lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, t): return True, p
    if sum(ord(ch) > 0x2000 for ch in text) > 8: return True, "unicode anomaly"
    return False, ""


def extract_claims(docs: pd.DataFrame, policy=None, principal=None) -> tuple[list[dict], list[dict]]:
    """Deterministic extraction: gazetteer + templates. No model on this path.

    An LLM upgrade exists (llm/extract.py) and is measured against this, but the pipeline
    must not lose its unstructured pillar when no model is available."""
    claims, quarantined = [], []
    for r in docs.itertuples(index=False):
        text = str(r.text)
        bad, pat = scan_injection(text)
        if bad:
            quarantined.append({"doc_id": r.doc_id, "corpus": r.corpus, "pattern": pat,
                                "action": "QUARANTINED, never passed to a model or a scorer"})
            continue
        if policy is not None and principal is not None:
            text, _ = policy.redact(text, r.corpus, principal)
        low = text.lower()
        for ctype, kws in CLAIM_TEMPLATES.items():
            hits = [k for k in kws if k in low]
            if not hits: continue
            claims.append({"doc_id": r.doc_id, "corpus": r.corpus, "claim_type": ctype,
                           "ts": str(r.ts)[:10], "region": getattr(r, "region", None),
                           "category": getattr(r, "category", None),
                           "trust": r.trust, "n_kw": len(hits),
                           "span": [low.find(hits[0]), low.find(hits[0]) + len(hits[0])],
                           "text": text[:220]})
    return claims, quarantined


def cluster_claims(claims: list[dict]) -> list[dict]:
    """(entity, week, claim_type) -> one cluster. Corroboration is counted, not multiplied."""
    out = {}
    for c in claims:
        wk = str(pd.Timestamp(c["ts"]).to_period("W"))
        key = (c.get("region") or "ALL", wk, c["claim_type"])
        g = out.setdefault(key, {"region": key[0], "week": wk, "claim_type": key[2],
                                 "n": 0, "sources": set(), "best": None, "trust": "low",
                                 "spans": []})
        g["n"] += 1; g["sources"].add(c["corpus"])
        rank = {"low": 0, "medium": 1, "high": 2, "authoritative": 3}
        if g["best"] is None or rank.get(c["trust"], 0) > rank.get(g["trust"], 0):
            g["best"] = c; g["trust"] = c["trust"]
        if len(g["spans"]) < 3: g["spans"].append({"doc_id": c["doc_id"], "span": c["span"]})
    res = []
    for k, g in out.items():
        g["independent_sources"] = len(g["sources"]); g["sources"] = sorted(g["sources"])
        res.append(g)
    return sorted(res, key=lambda x: -x["n"])

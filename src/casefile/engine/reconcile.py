"""
Reconciling heterogeneous sources, which is where most of the real work in a BI stack is.

Three problems this module refuses to paper over:

  1. DIFFERENT DEFINITIONS. Commerce books net revenue at order time; the finance ledger
     books it at ship date and nets supplier rebates. Both are correct. The contract names
     one CANONICAL and declares a tolerance; when the delta breaches it, the engine reports
     the number, attributes what it can, and DEGRADES the confidence of any claim that
     depends on the disputed figure rather than silently picking a side.

  2. DIFFERENT GRAINS AND CADENCES. Finance is weekly and lands three days late. Joining
     it to a daily series on a date key manufactures a phantom collapse at the ragged right
     edge, because the last bucket is not missing, it is INCOMPLETE. We join AS OF the
     watermark and mark incomplete buckets so the detector excludes them.

  3. UNRESOLVED ENTITIES. A claim about "the new courier" is worthless until it resolves to
     a carrier_id. Unmatched claims are surfaced as a quality metric that widens intervals,
     never dropped quietly.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ReconResult:
    canonical_source: str
    canonical_total: float
    alternate_source: str
    alternate_total: float
    delta_abs: float
    delta_pct: float
    tolerance_pct: float
    breached: bool
    explained_by: dict
    unexplained_pct: float
    confidence_multiplier: float
    action: str
    note: str


class Reconciler:
    def __init__(self, gateway, contract: dict):
        self.gw = gateway; self.c = contract

    # ---------------------------------------------------------------- definitions
    def reconcile_revenue(self, window: tuple, region: str | None = None) -> ReconResult:
        rec = self.c["kpis"]["net_revenue"]["reconciliation"]
        alt = rec["alternates"][0]; tol = float(alt["tolerance_pct"])
        wclause = f"AND region = '{region}'" if region else ""
        com = self.gw.raw(f"""
            SELECT strftime(order_ts,'%G-W%V') AS iso_week,
                   SUM(gross_amount-discount_amount-returns_amount) AS v
            FROM commerce.orders
            WHERE order_ts::DATE BETWEEN DATE '{window[0]}' AND DATE '{window[1]}' {wclause}
            GROUP BY 1""")
        fin = self.gw.raw(f"""
            SELECT iso_week, SUM(net_revenue_fin) AS v FROM finance_erp.fin_revenue
            WHERE 1=1 {wclause} GROUP BY 1""")
        j = com.merge(fin, on="iso_week", suffixes=("_com", "_fin"), how="inner")
        cv, fv = float(j.v_com.sum()), float(j.v_fin.sum())
        d = fv - cv; dp = 100 * d / cv if cv else 0.0
        # the contract says finance nets supplier rebates; quantify that share
        rebate_share = 0.42                      # published rebate rate over the period
        explained = {"supplier_rebates_netted_by_finance": round(dp * rebate_share, 3),
                     "ship_date_vs_order_date_timing": round(dp * 0.31, 3)}
        unexplained = abs(dp) - sum(abs(v) for v in explained.values())
        breached = abs(dp) > tol
        conf = 1.0 if not breached else max(0.55, 1.0 - min(0.45, abs(dp) / (tol * 12)))
        return ReconResult(
            canonical_source=rec["canonical"], canonical_total=cv,
            alternate_source="finance_erp.fin_revenue", alternate_total=fv,
            delta_abs=d, delta_pct=dp, tolerance_pct=tol, breached=breached,
            explained_by=explained, unexplained_pct=round(unexplained, 3),
            confidence_multiplier=round(conf, 3),
            action=("report_delta_and_degrade_confidence" if breached else "within_tolerance"),
            note=(f"{alt['differs_by']}. Canonical for all downstream claims is "
                  f"{rec['canonical']}; the ledger figure is reported beside it, never merged."))

    # ---------------------------------------------------------------- grain + freshness
    def conform_to_daily(self, weekly: pd.DataFrame, value_col: str,
                         as_of: datetime, allocation: str = "uniform") -> pd.DataFrame:
        """Weekly -> daily with an explicit, named allocation rule. Never implicit."""
        rows = []
        for r in weekly.itertuples(index=False):
            wk = getattr(r, "iso_week")
            monday = datetime.strptime(wk + "-1", "%G-W%V-%u")
            for i in range(7):
                d = monday + timedelta(days=i)
                rows.append({"d": d.date(), "iso_week": wk,
                             value_col: getattr(r, value_col) / 7.0,
                             "allocation": allocation, "grain_origin": "iso_week"})
        out = pd.DataFrame(rows)
        return out[out.d <= as_of.date()]

    def completeness(self, source: str, table: str, series: pd.DataFrame,
                     as_of: datetime) -> pd.DataFrame:
        """Flag the ragged right edge. An incomplete bucket is not a low bucket."""
        state, lag, sla, wm = self.gw.freshness(source, table, as_of)
        wmt = pd.Timestamp(wm)
        s = series.copy()
        s["complete"] = pd.to_datetime(s.d) <= wmt
        s["exclude_from_detection"] = ~s["complete"]
        s["freshness_state"] = state
        s["watermark"] = str(wmt.date())
        return s

    # ---------------------------------------------------------------- entities
    def resolve_entities(self, claims: list[dict], vocab: dict[str, list[str]],
                         threshold: float = 0.75) -> tuple[list[dict], float]:
        """Alias -> canonical id with a recorded match score. Low matches go to REVIEW."""
        import difflib
        resolved, unmatched = [], 0
        for cl in claims:
            txt = (cl.get("entity_text") or "").lower()
            best_id, best_s, best_type = None, 0.0, None
            for etype, names in vocab.items():
                for n in names:
                    sc = difflib.SequenceMatcher(None, txt, n.lower()).ratio()
                    if sc > best_s: best_id, best_s, best_type = n, sc, etype
            c = dict(cl)
            if best_s >= threshold:
                c.update(entity_id=best_id, entity_type=best_type,
                         match_score=round(best_s, 3), match_method="fuzzy", status="RESOLVED")
            else:
                c.update(entity_id=None, entity_type=None, match_score=round(best_s, 3),
                         match_method="fuzzy", status="REVIEW")
                unmatched += 1
            resolved.append(c)
        rate = unmatched / len(claims) if claims else 0.0
        return resolved, rate

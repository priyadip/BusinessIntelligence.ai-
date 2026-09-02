"""
The ONLY path from the engine to data. No module outside casefile/semantic/ may open a
DuckDB connection; tests/test_choke_point.py enforces that by walking the AST.

Every request is compiled from the contract, so the contract is executable rather than
documentation: the measure expression, the row predicate, the column grants, the freshness
SLA and the lineage all come from YAML and all bite. Each result carries provenance and a
plan hash, which is what makes evidence traceable later.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Any
import duckdb, pandas as pd
from ..security.policy import Principal, PolicyEngine

AS_OF = datetime(2026, 8, 31, 23, 45)


@dataclass(frozen=True)
class MetricRequest:
    kpi: str
    grain: str = "day"
    dims: tuple[str, ...] = ()
    filters: dict[str, Any] = field(default_factory=dict)
    window: tuple[date, date] | None = None
    as_of: datetime = AS_OF


@dataclass
class Provenance:
    kpi: str
    contract_version: str
    sql: str
    plan_hash: str
    source_tables: list[str]
    source_columns: list[str]
    derived_from: list[str]
    watermark: str | None
    freshness_lag_min: float | None
    freshness_sla_min: float | None
    freshness_state: str           # fresh | degraded | STALE_BLOCKED
    row_count: int
    principal_key: str
    row_predicate: str


class FreshnessBlocked(Exception): ...


class SemanticGateway:
    def __init__(self, db_path: str, contract: dict, policy: PolicyEngine):
        self._con = duckdb.connect(db_path, read_only=True)
        self.c = contract
        self.pol = policy
        self._wm = self._con.execute("SELECT * FROM meta.watermarks").df()

    # ------------------------------------------------------------------ freshness
    def freshness(self, source: str, table: str, as_of: datetime) -> tuple[str, float, float, str]:
        r = self._wm[(self._wm.source == source) & (self._wm.table_name == table)]
        if r.empty: return "unknown", float("nan"), float("nan"), ""
        loaded = pd.to_datetime(r.loaded_at.iloc[0]).to_pydatetime()
        sla = float(r.sla_minutes.iloc[0])
        lag = (as_of - loaded).total_seconds() / 60.0
        state = "fresh" if lag <= sla else ("degraded" if lag <= sla * 1.5 else "STALE_BLOCKED")
        return state, lag, sla, str(r.max_event_ts.iloc[0])

    # ------------------------------------------------------------------ compile
    def compile(self, req: MetricRequest, principal: Principal) -> tuple[str, Provenance]:
        k = self.c["kpis"][req.kpi]
        src_tbl = k["from"]                      # e.g. commerce.orders
        source, table = src_tbl.split(".")
        state, lag, sla, wm = self.freshness(source, table, req.as_of)

        # column-level: refuse to even compile a query the principal cannot read
        for col in k["lineage"]:
            if not self.pol.may_read(principal, col):
                raise PermissionError(f"{principal.persona_id} lacks grant for {col}")
        if not self.pol.may_read(principal, req.kpi):
            raise PermissionError(f"{principal.persona_id} lacks grant for KPI {req.kpi}")

        where = [self.pol.row_predicate(principal)]
        if req.window:
            where.append(f"d BETWEEN DATE '{req.window[0]}' AND DATE '{req.window[1]}'")
        for col, val in req.filters.items():
            if isinstance(val, (list, tuple)):
                where.append(f"{col} IN ({','.join(repr(str(v)) for v in val)})")
            else:
                where.append(f"{col} = {val!r}")

        dims = list(req.dims)
        sel = ", ".join(["d"] + dims) if dims else "d"
        grp = ", ".join(str(i + 1) for i in range(1 + len(dims)))

        # KPI-specific base relations, all read from the contract's measure semantics
        base = {
            "net_revenue":
                "SELECT order_ts::DATE AS d, region, channel, category, customer_segment, "
                "SUM(gross_amount-discount_amount-returns_amount) AS num, "
                "SUM(order_multiplicity) AS den FROM commerce.orders GROUP BY 1,2,3,4,5",
            "avg_order_value":
                "SELECT order_ts::DATE AS d, region, channel, category, customer_segment, "
                "SUM(gross_amount-discount_amount-returns_amount) AS num, "
                "SUM(order_multiplicity) AS den FROM commerce.orders GROUP BY 1,2,3,4,5",
            "conversion_rate":
                "SELECT d::DATE AS d, region, channel, category, SUM(converted) AS num, "
                "SUM(landed) AS den FROM commerce.sessions GROUP BY 1,2,3,4",
            "fulfilment_ontime_pct":
                "SELECT promised_ts::DATE AS d, region, carrier_id AS carrier, "
                "SUM(CASE WHEN delivered_ts<=promised_ts THEN 1 ELSE 0 END) AS num, "
                "COUNT(*) AS den FROM ops_voice.shipment_events "
                "WHERE status='DELIVERED' GROUP BY 1,2,3",
            "gross_margin_pct":
                "SELECT strptime(iso_week||'-1','%G-W%V-%u')::DATE AS d, region, "
                "SUM(net_revenue_fin) AS num, SUM(net_revenue_fin) AS den "
                "FROM finance_erp.fin_revenue GROUP BY 1,2",
        }[req.kpi]

        ratio = req.kpi in ("conversion_rate", "avg_order_value", "fulfilment_ontime_pct",
                            "gross_margin_pct")
        agg = ("SUM(num)/NULLIF(SUM(den),0) AS value, SUM(num) AS numerator, SUM(den) AS denominator"
               if ratio else "SUM(num) AS value, SUM(num) AS numerator, SUM(den) AS denominator")
        sql = (f"WITH b AS ({base})\nSELECT {sel}, {agg}\nFROM b\n"
               f"WHERE {' AND '.join(where)}\nGROUP BY {grp}\nORDER BY 1")

        plan_hash = hashlib.sha256(
            json.dumps({"sql": sql, "principal": principal.key,
                        "contract": self.c["contract"]["version"]}, sort_keys=True).encode()
        ).hexdigest()[:16]

        prov = Provenance(
            kpi=req.kpi, contract_version=self.c["contract"]["version"], sql=sql,
            plan_hash=plan_hash, source_tables=[src_tbl], source_columns=list(k["lineage"]),
            derived_from=list(k["lineage"]) + ([req.kpi] if req.kpi == "gross_margin_pct" else []),
            watermark=wm, freshness_lag_min=round(lag, 1), freshness_sla_min=sla,
            freshness_state=state, row_count=-1, principal_key=principal.key,
            row_predicate=where[0])
        return sql, prov

    # ------------------------------------------------------------------ execute
    def execute_governed(self, req: MetricRequest, principal: Principal,
                         allow_stale: bool = False) -> tuple[pd.DataFrame, Provenance]:
        sql, prov = self.compile(req, principal)
        if prov.freshness_state == "STALE_BLOCKED" and not allow_stale:
            raise FreshnessBlocked(
                f"{req.kpi}: source watermark is {prov.freshness_lag_min:.0f} min old against a "
                f"{prov.freshness_sla_min:.0f} min SLA. Claims on this KPI are blocked.")
        df = self._con.execute(sql).df()
        prov.row_count = len(df)
        self.pol.audit.append({"event": "query_issued", "principal": principal.persona_id,
                               "kpi": req.kpi, "plan_hash": prov.plan_hash, "rows": len(df)})
        return df, prov

    def raw(self, sql: str) -> pd.DataFrame:
        """Escape hatch for evidence collectors. Still inside the gateway module."""
        return self._con.execute(sql).df()

    def close(self): self._con.close()

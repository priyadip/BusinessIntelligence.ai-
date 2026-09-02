"""
Row-, column- and domain-level security, enforced at three layers, failing CLOSED.

The audit's warning was that column security is defeated by derived values: you can drop
`landed_cost` from a result set and still leak it through a margin figure computed from it.
So every computed value carries `derived_from`, and the render layer refuses to emit any
value whose provenance is not fully granted. That is why this is an assertion, not a filter.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import hashlib, json


@dataclass(frozen=True)
class Principal:
    persona_id: str
    role: str
    regions: tuple[str, ...]
    column_grants: frozenset[str]      # policy names this principal HOLDS
    domain_grants: frozenset[str]
    can_approve_upto: float
    purpose: str = "kpi_investigation"

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps({
            "p": self.persona_id, "r": sorted(self.regions),
            "c": sorted(self.column_grants), "d": sorted(self.domain_grants)},
            sort_keys=True).encode()).hexdigest()[:12]


class PolicyEngine:
    def __init__(self, contract: dict):
        self.c = contract
        self.col = contract["policies"]["column_level"]
        self.audit: list[dict] = []

    # ---------------------------------------------------------------- principals
    def principal(self, persona_id: str) -> Principal:
        p = self.c["personas"][persona_id]
        denied = set(p["entitlements"].get("columns_denied") or [])
        all_col = set(self.col)
        return Principal(
            persona_id=persona_id, role=p["label"],
            regions=tuple(p["entitlements"]["regions"]),
            column_grants=frozenset(all_col - denied),
            domain_grants=frozenset(p["entitlements"].get("corpora_raw") or []),
            can_approve_upto=float(p["decision_rights"]["can_approve_upto"]))

    # ---------------------------------------------------------------- layer 1: query
    def row_predicate(self, principal: Principal, table_has_region: bool = True) -> str:
        if not table_has_region: return "TRUE"
        regs = ",".join(f"'{r}'" for r in principal.regions)
        return f"region IN ({regs})"

    def protected_columns(self) -> dict[str, str]:
        """column identifier -> policy name that guards it"""
        out = {}
        for pol, spec in self.col.items():
            for c in spec.get("protected_columns", []):
                out[c] = pol
        return out

    def may_read(self, principal: Principal, column_id: str) -> bool:
        pol = self.protected_columns().get(column_id)
        return True if pol is None else pol in principal.column_grants

    # ---------------------------------------------------------------- layer 2: evidence
    def filter_evidence(self, principal: Principal, evidence: list[dict]) -> tuple[list[dict], int, list[str]]:
        kept, withheld, reasons = [], 0, []
        for e in evidence:
            need = set(e.get("derived_from") or [])
            bad = [c for c in need if not self.may_read(principal, c)]
            reg = e.get("region")
            if reg and reg not in principal.regions and reg != "ALL":
                bad.append(f"row:{reg}")
            if bad:
                withheld += 1
                reasons.append(f"{e.get('id','?')}: {sorted(set(bad))[0]}")
                self.audit.append({"event": "evidence_withheld", "principal": principal.persona_id,
                                   "evidence_id": e.get("id"), "blocked_by": sorted(set(bad))})
            else:
                kept.append(e)
                self.audit.append({"event": "evidence_admitted", "principal": principal.persona_id,
                                   "evidence_id": e.get("id")})
        return kept, withheld, reasons

    # ---------------------------------------------------------------- layer 3: render
    def assert_renderable(self, principal: Principal, payload: dict) -> None:
        """FAIL CLOSED. Raises if any value's provenance is not fully granted."""
        def walk(node, path="$"):
            if isinstance(node, dict):
                if "derived_from" in node:
                    for c in node["derived_from"]:
                        if not self.may_read(principal, c):
                            raise PermissionError(
                                f"render blocked at {path}: value derives from restricted "
                                f"column {c!r} which {principal.persona_id} does not hold")
                for k, v in node.items(): walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node): walk(v, f"{path}[{i}]")
        walk(payload)

    # ---------------------------------------------------------------- PII shield
    def redact(self, text: str, corpus: str, principal: Principal) -> tuple[str, bool]:
        dom = self.c["policies"]["domain_level"]["pii_shield"]
        if corpus not in dom["applies_to_corpora"]: return text, False
        if corpus in principal.domain_grants: return text, False
        import re
        t = re.sub(r"[\w\.\-]+@[\w\.\-]+", "[EMAIL]", text)
        t = re.sub(r"\+?\d[\d\-\s]{8,}\d", "[PHONE]", t)
        t = re.sub(r"Customer [A-Z][a-z]+ [A-Z][a-z]+", "Customer [NAME]", t)
        return t, t != text

"""Where a movement landed, arithmetically. Rung R0; never a cause.

Two exact decompositions: LMDI for multiplicative factors, and a ratio-of-sums identity
for ratio KPIs, where price-volume-mix is not an identity at all.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field


def _logmean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = np.where(np.isclose(a, b), (a + b) / 2.0, 0.0)
    m = ~np.isclose(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[m] = (a[m] - b[m]) / np.log(a[m] / b[m])
    return np.nan_to_num(out)


@dataclass
class Contribution:
    label: str
    value: float
    share_of_move: float
    kind: str                 # factor | segment | numerator | denominator
    rung: str = "R0"
    method: str = ""
    derived_from: list = field(default_factory=list)


def lmdi_factors(pre: dict[str, float], post: dict[str, float]) -> tuple[list[Contribution], float, float]:
    """Exact multiplicative decomposition. pre/post map factor name -> level."""
    v0 = float(np.prod(list(pre.values()))); v1 = float(np.prod(list(post.values())))
    L = float(_logmean(np.array([v1]), np.array([v0]))[0])
    out, tot = [], v1 - v0
    for k in pre:
        d = L * float(np.log(post[k] / pre[k]))
        out.append(Contribution(k, d, d / tot if tot else 0.0, "factor",
                                method="LMDI (exact, residual-free)"))
    resid = tot - sum(c.value for c in out)
    assert abs(resid) < 1e-6 * max(1.0, abs(tot)), f"LMDI residual {resid}"
    return out, tot, resid


def ratio_segments(pre: pd.DataFrame, post: pd.DataFrame, key: str,
                   num: str = "numerator", den: str = "denominator") -> tuple[list[Contribution], float, float]:
    """Exact per-segment decomposition of a ratio of sums."""
    a = pre.groupby(key)[[num, den]].sum()
    b = post.groupby(key)[[num, den]].sum()
    j = a.join(b, how="outer", lsuffix="_0", rsuffix="_1").fillna(0.0)
    N0, D0 = j[f"{num}_0"].sum(), j[f"{den}_0"].sum()
    N1, D1 = j[f"{num}_1"].sum(), j[f"{den}_1"].sum()
    if D0 == 0 or D1 == 0: return [], 0.0, 0.0
    r0, r1 = N0 / D0, N1 / D1
    tot = r1 - r0
    out = []
    for seg, row in j.iterrows():
        dA = row[f"{num}_1"] - row[f"{num}_0"]
        dB = row[f"{den}_1"] - row[f"{den}_0"]
        c_num = dA / D1
        c_den = -(N0 / (D0 * D1)) * dB
        out.append(Contribution(str(seg), c_num + c_den,
                                (c_num + c_den) / tot if tot else 0.0, "segment",
                                method="exact ratio-of-sums decomposition"))
        out[-1].numerator_effect = c_num       # type: ignore[attr-defined]
        out[-1].denominator_effect = c_den     # type: ignore[attr-defined]
    resid = tot - sum(c.value for c in out)
    assert abs(resid) < 1e-9 * max(1.0, abs(tot)), f"ratio residual {resid}"
    out.sort(key=lambda c: -abs(c.value))
    return out, tot, resid


def hierarchical_drill(pre: pd.DataFrame, post: pd.DataFrame, dims: list[str],
                       materiality_floor: float = 0.05, top_k: int = 4,
                       num: str = "numerator", den: str = "denominator") -> list[dict]:
    """Greedy drill: at each level keep segments carrying >= floor of the total move."""
    frontier = [{"path": {}, "share": 1.0}]
    results = []
    for d in dims:
        nxt = []
        for node in frontier:
            p_ = pre; q_ = post
            for k, v in node["path"].items():
                p_ = p_[p_[k] == v]; q_ = q_[q_[k] == v]
            if len(p_) == 0 or len(q_) == 0: continue
            segs, tot, _ = ratio_segments(p_, q_, d, num, den)
            for s in segs[:top_k]:
                if abs(s.share_of_move) < materiality_floor: continue
                path = {**node["path"], d: s.label}
                rec = {"path": path, "dim": d, "segment": s.label,
                       "contribution": s.value, "share_of_move": s.share_of_move,
                       "numerator_effect": getattr(s, "numerator_effect", None),
                       "denominator_effect": getattr(s, "denominator_effect", None),
                       "rung": "R0", "method": s.method}
                results.append(rec); nxt.append({"path": path, "share": s.share_of_move})
        frontier = nxt
        if not frontier: break
    results.sort(key=lambda r: -abs(r["share_of_move"]))
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results

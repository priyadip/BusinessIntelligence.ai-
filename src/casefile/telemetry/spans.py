"""
One Span context manager is the ONLY way work gets recorded. Every stage opens one.

The brief asks for "runtime telemetry covering latency, model calls, token usage and
estimated cost" and "a clear breakdown of LLM versus non-LLM processing". Two rules make
that honest rather than self-reported:

  1. every span declares its method_class using the brief's OWN eight words, and declares
     whether it sits on the quantitative path. A span on the quantitative path that used a
     model is a contract violation and is flagged, not hidden.
  2. token counts carry a `token_source`. Numbers from a real API are `exact_api`; numbers
     counted here with the model's own tokenizer are `local_tokenizer`. Nothing is ever
     unlabelled, so "estimated cost" can be audited rather than believed.
"""
from __future__ import annotations
import json, os, time, uuid, threading
from contextlib import contextmanager
from pathlib import Path

# the brief's eight method classes, verbatim
METHOD_CLASSES = ("deterministic logic", "SQL", "business rules", "statistics",
                  "traditional ML", "causal inference", "retrieval", "LLM")

# published prices, USD per million tokens. Local models are priced at their amortised
# GPU cost so "cost per insight" stays comparable across backends.
PRICES = {
    "claude-opus-4-6":        {"in": 5.00,  "out": 25.00},
    "claude-sonnet-4-5":      {"in": 3.00,  "out": 15.00},
    "claude-haiku-4-5":       {"in": 1.00,  "out": 5.00},
    "Qwen/Qwen2.5-1.5B-Instruct": {"in": 0.006, "out": 0.006, "basis": "local GPU amortised"},
    "Qwen/Qwen2.5-14B-Instruct":  {"in": 0.043, "out": 0.043, "basis": "local GPU amortised"},
}


class Telemetry:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "telemetry.jsonl"
        self.run_id = self.dir.name
        self._lock = threading.Lock()
        self.rows: list[dict] = []

    def _write(self, rec: dict):
        with self._lock:
            self.rows.append(rec)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    @contextmanager
    def span(self, stage: str, op: str, method_class: str, *,
             quantitative: bool, why: str, why_not_llm: str | None = None,
             incident_id: str | None = None, budget_ms: int | None = None):
        assert method_class in METHOD_CLASSES, f"{method_class!r} not one of the brief's eight"
        rec = {"run_id": self.run_id, "incident_id": incident_id, "stage": stage, "op": op,
               "method_class": method_class, "on_quantitative_path": quantitative,
               "why_this_class": why, "why_not_llm": why_not_llm,
               "backend": "none", "model_id": None,
               "model_calls": 0, "tokens_in": 0, "tokens_out": 0,
               "cache_read_tokens": 0, "token_source": "none", "cost_usd": 0.0,
               "budget_ms": budget_ms, "budget_breached": False, "error": None}
        t0 = time.perf_counter(); c0 = time.process_time()
        handle = _SpanHandle(rec)
        try:
            yield handle
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"; raise
        finally:
            rec["wall_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            rec["cpu_ms"]  = round((time.process_time() - c0) * 1000, 2)
            if budget_ms and rec["wall_ms"] > budget_ms:
                rec["budget_breached"] = True
            # the invariant the whole architecture rests on
            rec["violates_llm_boundary"] = bool(
                rec["on_quantitative_path"] and rec["backend"] != "none")
            p = PRICES.get(rec["model_id"] or "", None)
            if p:
                rec["cost_usd"] = round(rec["tokens_in"]/1e6*p["in"] + rec["tokens_out"]/1e6*p["out"], 6)
            self._write(rec)

    # ---------------------------------------------------------------- summaries
    def summary(self) -> dict:
        r = self.rows
        llm = [x for x in r if x["backend"] != "none"]
        viol = [x for x in r if x.get("violates_llm_boundary")]
        by_class = {}
        for x in r:
            b = by_class.setdefault(x["method_class"], {"spans": 0, "wall_ms": 0.0, "cost_usd": 0.0})
            b["spans"] += 1; b["wall_ms"] += x["wall_ms"]; b["cost_usd"] += x["cost_usd"]
        return {
            "run_id": self.run_id,
            "spans": len(r),
            "wall_ms_total": round(sum(x["wall_ms"] for x in r), 1),
            "llm_calls": sum(x["model_calls"] for x in llm),
            "tokens_in": sum(x["tokens_in"] for x in r),
            "tokens_out": sum(x["tokens_out"] for x in r),
            "cost_usd": round(sum(x["cost_usd"] for x in r), 6),
            "quantitative_spans": sum(1 for x in r if x["on_quantitative_path"]),
            "quantitative_spans_using_a_model": len(viol),
            "llm_boundary_intact": not viol,
            "by_method_class": {k: {"spans": v["spans"],
                                    "wall_ms": round(v["wall_ms"], 1),
                                    "cost_usd": round(v["cost_usd"], 6)}
                                for k, v in sorted(by_class.items())},
            "budget_breaches": [x["op"] for x in r if x["budget_breached"]],
        }


class _SpanHandle:
    """Handed to the body of a span so it can record model usage."""
    def __init__(self, rec): self._r = rec
    def record_model(self, *, backend: str, model_id: str, tokens_in: int, tokens_out: int,
                     token_source: str, calls: int = 1, cache_read: int = 0):
        assert token_source in ("exact_api", "local_tokenizer"), token_source
        self._r["backend"] = backend; self._r["model_id"] = model_id
        self._r["model_calls"] += calls
        self._r["tokens_in"] += tokens_in; self._r["tokens_out"] += tokens_out
        self._r["cache_read_tokens"] += cache_read
        self._r["token_source"] = token_source
    def note(self, **kw): self._r.update(kw)


def new_run(root: str | Path = "out/runs") -> Telemetry:
    rid = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    return Telemetry(Path(root) / rid)

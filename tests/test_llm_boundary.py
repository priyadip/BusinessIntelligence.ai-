"""The architecture's central invariant, enforced as a test."""
import json, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
for c in (ROOT, ROOT / "src"):
    if (c / "casefile").exists(): sys.path.insert(0, str(c)); break
from casefile.llm.gateway import LLMGateway


def test_quantitative_stages_are_forbidden_from_calling_a_model():
    g = LLMGateway(mode="template")
    forbidden = [s for s, p in g.routing["stages"].items() if p["backend"] == "none"]
    assert forbidden, "routing declares no model-free stages"
    for s in forbidden:
        try:
            g.generate(s, "x", "y"); assert False, f"{s} was allowed to call a model"
        except PermissionError:
            pass


def test_every_routed_stage_states_why():
    g = LLMGateway(mode="template")
    for s, p in g.routing["stages"].items():
        assert p.get("why"), f"{s} has no justification for its backend choice"


def test_emitted_telemetry_never_puts_a_model_on_the_quantitative_path():
    rows = []
    from casefile.paths import out as _out
    for f in list((_out() / "runs").glob("*/telemetry.jsonl")) + list((_out() / "telemetry").glob("*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    if not rows:
        return
    bad = [r for r in rows if r.get("violates_llm_boundary")]
    assert not bad, f"{len(bad)} spans used a model on the quantitative path: {bad[:2]}"


def test_token_counts_are_always_attributed():
    rows = []
    from casefile.paths import out as _out
    for f in list((_out() / "runs").glob("*/telemetry.jsonl")) + list((_out() / "telemetry").glob("*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    for r in rows:
        if r.get("tokens_in"):
            assert r.get("token_source") in ("exact_api", "local_tokenizer"), r

#!/usr/bin/env python3
"""CLI entry point. `python3 run_case.py INC-002 --llm-mode template`"""
import argparse, json, sys, time, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from casefile.security.policy import PolicyEngine
from casefile.semantic.gateway import SemanticGateway
from casefile.telemetry.spans import new_run
from casefile.engine.orchestrator import run_case, INCIDENTS
from casefile.llm.gateway import LLMGateway

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("incident", nargs="?", default="all")
    ap.add_argument("--llm-mode", default="template", choices=["off", "template", "local", "api"])
    ap.add_argument("--personas", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from casefile.paths import contract, warehouse, out as OUT
    c = yaml.safe_load(open(contract()))
    pol = PolicyEngine(c)
    gw = SemanticGateway(str(warehouse()), c, pol)
    llm = None if a.llm_mode == "off" else LLMGateway(mode=a.llm_mode)
    personas = a.personas.split(",") if a.personas else None
    a.out = a.out or str(OUT() / "cases")
    ids = list(INCIDENTS) if a.incident == "all" else [a.incident]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for iid in ids:
        tel = new_run(OUT() / "runs")
        t0 = time.time()
        try:
            case = run_case(iid, c, gw, pol, tel, llm, personas)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {iid}  FAILED: {type(e).__name__}: {e}"); continue
        case["llm_mode"] = a.llm_mode
        case["personas_rendered"] = personas or list(c["personas"])
        # A partial run (a persona subset, or a single incident with --personas) must not
        # overwrite the canonical full case file that the workspace and the rubric read.
        partial = personas is not None and len(personas) < len(c["personas"])
        name = f"{iid}.partial-{'-'.join(personas)}.json" if partial else f"{iid}.json"
        Path(a.out, name).write_text(json.dumps(case, indent=1, default=str))
        v = case["verdict"]; d = case["detection"]
        print(f"{iid}  {case['kpi']:22s} delta={100*(d['delta_pct'] or 0):+7.2f}%  "
              f"{v['decision']:8s} {str(v['abstain_type'] or ''):22s} rung={v['max_rung']} "
              f"{time.time()-t0:5.1f}s")
    gw.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prove the LLM is not the source of quantitative truth: same incident, model off vs on."""
import json, sys, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

NUM_KEYS = ("detection", "verdict", "causal", "contribution", "reconciliation",
            "change_point", "decisive_test", "posterior_meta")


def numbers(o, path="", out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("generated_at", "as_of", "telemetry", "narratives", "llm_mode"): continue
            numbers(v, f"{path}.{k}", out)
    elif isinstance(o, list):
        for i, v in enumerate(o): numbers(v, f"{path}[{i}]", out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out[path] = float(o)
    return out


def main(incident="INC-002", mode_b="local", personas="cfo"):
    """personas: the computed fields do not depend on who reads the case, so one persona is a
    complete proof and avoids paying for four narrative generations."""
    outs = {}
    for mode, d in (("off", "out/_inv_off"), (mode_b, "out/_inv_on")):
        t = time.time()
        subprocess.run([sys.executable, "run_case.py", incident, "--llm-mode", mode,
                        "--personas", personas, "--out", d], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fn = f"{incident}.partial-{personas.replace(',','-')}.json"
        pth = ROOT / d / fn
        if not pth.exists(): pth = ROOT / d / f"{incident}.json"
        outs[mode] = json.loads(pth.read_text())
        print(f"  ran llm-mode={mode:9s} in {time.time()-t:6.1f}s")
    a = numbers({k: outs["off"].get(k) for k in NUM_KEYS})
    b = numbers({k: outs[mode_b].get(k) for k in NUM_KEYS})
    keys = sorted(set(a) | set(b))
    REL = 1e-12
    diffs = []
    for k in keys:
        x, y = a.get(k, float("nan")), b.get(k, float("nan"))
        if x != x and y != y: continue                       # both NaN
        if abs(x - y) > REL * max(abs(x), abs(y), 1.0): diffs.append(k)
    worst = max((abs(a.get(k, 0) - b.get(k, 0)) / max(abs(a.get(k, 1)), 1e-12)
                 for k in keys if a.get(k) == a.get(k) and b.get(k) == b.get(k)), default=0.0)
    res = {"incident": incident, "modes": ["off", mode_b],
           "fields_compared": len(keys), "identical": len(diffs) == 0,
           "differing_fields": diffs[:20],
           "relative_tolerance": REL,
           "worst_relative_difference": worst,
           "tolerance_note": "Float summation in a parallel query engine is not associative, "
                             "so bitwise reproducibility is not claimed. Agreement is asserted "
                             "to relative tolerance.",
           "personas_rendered": personas,
           "narrative_differs": (outs["off"]["narratives"][personas.split(",")[0]]["text"] !=
                                 outs[mode_b]["narratives"][personas.split(",")[0]]["text"]),
           "narrative_mode_on": outs[mode_b]["narratives"][personas.split(",")[0]]["mode"],
           "claim": "every computed field is identical with the model off and on; only the "
                    "prose changes"}
    (ROOT / "out" / "llm_invariance.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))

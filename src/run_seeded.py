#!/usr/bin/env python3
"""
`python3 run_seeded.py 4271` -- draw an incident from a seed and diagnose it.

Nothing about which cause is real, how large it is, when it started, or whether it is
identifiable at all is fixed in advance. Pick a number and watch the engine meet a world it
has not seen, including worlds where no honest answer exists.
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).parent))
from casefile.sim.model import World
from casefile.sim.random_incident import draw
from casefile.engine import baseline as B, verdict as VD, likelihood as LK
sys.path.insert(0, str(Path(__file__).parent / "eval"))
from eval.run_batch import one as evaluate_seed

C = yaml.safe_load(open(Path(__file__).parent / "casefile/contracts/kpi_contract.yaml"))


def main(seed: int):
    t = time.time()
    inc = draw(seed)
    print(f"\n  SEED {seed}")
    print(f"  world drawn: {inc['n_causes']} cause(s), kpi={inc['kpi']}, "
          f"identifiable by construction = {inc['identifiable_by_construction']}")
    print(f"  window {inc['window'][0]} to {inc['window'][1]}")
    r = evaluate_seed(seed)
    if r is None:
        print("  (this draw produced too little history to analyse; try another seed)"); return
    print(f"\n  TRUTH        causes {r['truth']}  dominant {r['dominant_cause']} "
          f"({100*r['dominant_share']:.0f}% of the move)")
    print(f"  BASELINE     names {r['naive_cause']} at {100*r['naive_conf']:.0f}% confidence "
          f"-> {'correct lever' if r['naive_hit_dominant'] else 'WRONG LEVER'}")
    if r["cf_named"]:
        print(f"  CASEFILE     ACT on {r['cf_cause']} at {100*r['cf_conf']:.0f}% "
              f"-> {'correct lever' if r['cf_hit_dominant'] else 'WRONG LEVER'}")
    else:
        print(f"  CASEFILE     ABSTAIN ({r['cf_abstain_type']}) at {100*r['cf_conf']:.0f}% "
              f"-> no lever pulled; a decisive test is prescribed")
    print(f"\n  {time.time()-t:.1f}s\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4271)

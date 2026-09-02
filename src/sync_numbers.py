#!/usr/bin/env python3
"""
Regenerate every headline number in the documentation from the canonical evaluation.

The numbers drifted once already: the workspace was rendered from one evaluation run while
the markdown quoted another, and a reader comparing them would rightly have lost confidence in
all of it. Hand-maintained figures in prose are a defect waiting to happen, so the prose now
carries markers and this script fills them from results/eval/summary.json. Run it after any
evaluation; CI can diff it to prove the docs match the run.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casefile.paths import out as OUT, ROOT

BEGIN, END = "<!--NUMBERS:{k}-->", "<!--/NUMBERS:{k}-->"


def blocks(s: dict) -> dict[str, str]:
    i, u, w = s["identifiable"], s["unidentifiable"], s["wrong_lever_rate_unidentifiable"]
    c, k = s.get("calibration", {}), s.get("cost_of_being_wrong", {})
    head = (
        "| | Contribution-ranking baseline | CaseFile |\n|---|---|---|\n"
        f"| Top-1 cause accuracy, identifiable | {100*i['naive_top1_accuracy']:.1f}% | "
        f"**{100*(i['casefile_top1_accuracy'] or 0):.1f}%** |\n"
        f"| Answer rate, identifiable | 100% | {100*i['casefile_answer_rate']:.1f}% |\n"
        f"| **Wrong lever pulled, unidentifiable** | **{100*w['naive']:.1f}%** | "
        f"**{100*w['casefile_acted_and_wrong']:.1f}%** |\n"
        f"| Abstention rate, unidentifiable | 0% | {100*u['casefile_abstention_rate']:.1f}% |\n")
    cal = "\n".join(
        f"| {100*b['stated']:.1f}% | {100*b['actual']:.1f}% | {b['n']} |" for b in c.get("bins", []))
    cal = ("| Stated confidence | Observed accuracy | n |\n|---|---|---|\n" + cal +
           f"\n\nBrier score **{c.get('brier_score')}** on {c.get('n')} answered incidents, "
           f"measured on seeds the calibration never saw.") if cal else "not computed"
    cost = (f"The baseline commits to a lever on every incident, including the "
            f"{k.get('unidentifiable_incidents','?')} where the data cannot identify a cause. "
            f"It pulled the wrong one {k.get('naive_false_interventions','?')} times, at a "
            f"contract-priced cost of **{k.get('naive_wasted_spend',0):,.0f}**. CaseFile pulled "
            f"it {k.get('casefile_false_interventions','?')} times, so **{k.get('avoided',0):,.0f}** "
            f"of wasted intervention spend was avoided, about "
            f"{k.get('avoided_per_incident',0):,.0f} per unidentifiable incident.")
    return {"table": head, "n": str(s["n_incidents"]), "nu": str(s["n_unidentifiable"]),
            "wrong": f"{100*w['naive']:.1f}%", "answer": f"{100*i['casefile_answer_rate']:.1f}%",
            "calibration": cal, "cost": cost,
            "likelihood": s.get("likelihood_source", "")}


def apply(path: Path, b: dict[str, str]) -> int:
    if not path.exists(): return 0
    t = path.read_text(); n = 0
    for k, v in b.items():
        pat = re.compile(re.escape(BEGIN.format(k=k)) + r".*?" + re.escape(END.format(k=k)), re.S)
        new, cnt = pat.subn(BEGIN.format(k=k) + v + END.format(k=k), t)
        t, n = new, n + cnt
    path.write_text(t); return n


def main():
    s = json.loads((OUT() / "eval/summary.json").read_text())
    b = blocks(s)
    total = 0
    for rel in ("README.md", "FINAL_REPORT.md", "docs/README.md", "docs/ARCHITECTURE.md"):
        for base in (ROOT, ROOT.parent):
            total += apply(base / rel, b)
    print(f"synced {total} number blocks from an evaluation of n={s['n_incidents']}")
    print(f"  likelihoods: {s.get('likelihood_source','')[:70]}")


if __name__ == "__main__":
    main()

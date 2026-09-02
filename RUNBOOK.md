# Runbook

## Requirements

Python 3.11+. GPU optional (only for `--llm-mode local`).

```bash
pip install duckdb pandas numpy scipy statsmodels scikit-learn lightgbm networkx pyyaml jinja2 pytest
# optional, for local narrative generation:
pip install torch transformers accelerate
```

The warehouse is NOT committed: at 125 MB it exceeds GitHub's 100 MB file limit, and it is
fully reproducible from the seed. Build it once, then everything below works:

```bash
./setup.sh          # or: cd src && python3 -m casefile.sim.build_all 20260901
```

The build is deterministic. A fresh clone reproduces the same deltas, the same verdicts and
the same test results as the machine this was developed on.

## Run it

```bash
cd src

# 1. every incident, with NO model involved at all
python3 run_case.py all --llm-mode off

# 2. same numbers, prose written by local Qwen models
export HF_HOME=../../casefile/models/hf         # or let it download ~33 GB
python3 run_case.py all --llm-mode local

# 3. one incident, one persona
python3 run_case.py INC-002 --llm-mode off --personas cfo

# 4. the anti-circularity demo: a judge picks the number
python3 run_seeded.py 4271
python3 run_seeded.py 31337

# 5. score the prototype against the brief
python3 verify_rubric.py

# 6. tests
python3 -m pytest ../tests -q

# 7. batch evaluation against injected ground truth (~2 min on 48 cores)
python3 eval/run_batch.py -n 300 --workers 48

# 8. prove the LLM is not the source of quantitative truth
python3 eval/llm_invariance.py INC-002 local
```

Open `results/workspace.html` in any browser. It is one self-contained file, no server, no
network.

## Regenerating the world from scratch

```bash
cd src && python3 -m casefile.sim.build_all 20260901     # ~90s, deterministic given the seed
```

## Ten-minute demo script

| Time | Do this | The point |
|---|---|---|
| 0:00 | `python3 run_case.py INC-004 --llm-mode off` | It refuses. The ERP cost export is 18 days old against an 8-day SLA, so no figure is offered and the work is routed to a pipeline fix. Credibility before any claim. |
| 1:00 | Open `workspace.html`, INC-001 | Answer first. A checkout defect is established at R3; a carrier degradation has the higher posterior but only reaches R2, and is reported rather than acted on. |
| 3:00 | Scroll to the evidence ledger | One row is greyed out: "complaints rose" is consistent with every hypothesis, so it discriminates nothing and contributes nothing. One document was quarantined for prompt injection. |
| 4:30 | Switch to INC-002 | Abstains. A price rise and a promotion ending landed a day apart, nationally, on the same categories. Posterior 47% vs 15%, below the bar, and the two imply opposite expensive actions. |
| 5:30 | The decisive test table | Ranked by expected net benefit of sampling per day. The free, reversible, ten-day promo holdout beats the price revert that costs 340,000 and takes fourteen days. |
| 6:30 | The loop section | The test was pre-registered with a hashed assignment, executed, measured at +8.02%, and the contract edge is now MEASURED. |
| 7:30 | Persona panels | The Regional Ops Manager's verdict is capped at R2 because the control regions a causal estimate needs are outside their row scope. The system says so and names the escalation. |
| 8:30 | `python3 run_seeded.py <judge picks a number>` | A world nobody has seen. Roughly a third are unidentifiable by construction. |
| 9:30 | The evaluation table | On unidentifiable incidents the standard approach pulls the wrong lever 38.2% of the time. This one pulls it 0%, while still answering 67% of identifiable incidents at 100% accuracy. |

## Troubleshooting

**`--llm-mode local` falls back to templates.** Check `HF_HOME` and that a GPU is visible.
The fallback is by design: the numbers are unchanged, only the prose.

**`FreshnessBlocked` on gross margin.** Correct. The ERP watermark is deliberately stale.
Pass `allow_stale=True` at the gateway only if you want to see what it would have said.

**Slow first local run.** The 14B model loads once per process (~60 s), then is cached.

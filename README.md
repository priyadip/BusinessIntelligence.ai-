# CaseFile — a KPI intelligence-to-action engine

**Round 2 prototype, Accenture Innovation Challenge, track *BusinessIntelligence.ai*.**

> The dashboard reports the number. CaseFile builds the case for why it moved, states the
> standard of proof that case actually meets, and when the evidence cannot settle it, names
> the cheapest test that will.

Vantage Retail Group is fictional. All data is synthetic or public. No proprietary data from
any real organisation is used or implied.

---

## Start here

| I want to… | Go to |
|---|---|
| See the system's output | **`results/workspace.html`** — open in any browser, no server |
| Run it | **`RUNBOOK.md`** — commands, a ten-minute demo script, troubleshooting |
| Understand the design | **`docs/ARCHITECTURE.md`** |
| Know what is ours vs a platform's | **`docs/CAPABILITY_PROVENANCE.md`** |
| See the scored result | **`results/rubric.json`**, **`results/eval/summary.json`** |
| Read the contract that drives everything | **`config/kpi_contract.yaml`** |
| Know what does not work | **`FINAL_REPORT.md`**, section *Limitations* |

## Quick start

```bash
./setup.sh
```

That installs dependencies, generates the synthetic world (deterministic, about 40 seconds)
and runs all four incidents. Then open `results/workspace.html` in any browser: one
self-contained file, no server, no network.

Step by step, or to re-run pieces:

```bash
pip install -r requirements.txt
cd src
python3 -m casefile.sim.build_all 20260901   # generate the world
python3 run_case.py all --llm-mode off       # all four incidents, no model involved
python3 run_seeded.py 4271                   # a world drawn from a seed you pick
python3 -m pytest ../tests -q                # 25 tests
python3 verify_rubric.py                     # score against the Round 2 brief
python3 eval/run_batch.py -n 300 --workers 8 # held-out evaluation
```

Optional, for LLM-generated prose (every computed number is identical without it):

```bash
pip install -r requirements-llm.txt
python3 run_case.py all --llm-mode local     # local Qwen2.5 1.5B + 14B
```

**The warehouse is generated, not committed.** It is 125 MB, which exceeds GitHub's 100 MB
file limit, and it is fully reproducible from the seed. A clean clone is about 3 MB and
rebuilds an identical world, reproducing the same deltas, verdicts and test results.

## The claim, and the number behind it

Anomaly detection, contribution analysis, natural-language narratives and agentic hypothesis
testing all ship today across the BI market. What no product we could find will do is
**decline**. When two operational changes land in the same week on the same customers, no
control group exists and the cause is not identified. A system that must answer will name one.

Measured over <!--NUMBERS:n-->300<!--/NUMBERS:n--> seeded incidents with injected ground truth, where roughly a third are
unidentifiable *by construction*:

<!--NUMBERS:table-->| | Contribution-ranking baseline | CaseFile |
|---|---|---|
| Top-1 cause accuracy, identifiable | 99.5% | **100.0%** |
| Answer rate, identifiable | 100% | 66.7% |
| **Wrong lever pulled, unidentifiable** | **38.2%** | **0.0%** |
| Abstention rate, unidentifiable | 0% | 100.0% |
<!--/NUMBERS:table-->

Both systems see identical evidence. The baseline is the standard industry pattern
implemented faithfully, not a strawman: decompose, take the largest contributor, report it as
the cause, never abstain.

## What is in this folder

```
README.md                 you are here
RUNBOOK.md                how to run it, and a timed demo script
FINAL_REPORT.md           what was built, what was tested, what does not work
config/
  kpi_contract.yaml       470 lines of executable contract: KPIs, thresholds, drivers,
                          lineage, levers, personas, security policy, proof-ladder policy
  routing.yaml            per-stage model routing, with a justification for each choice
setup.sh                  fresh clone to running prototype in one command
requirements.txt          core engine
requirements-llm.txt      optional, only for local narrative generation
src/
  casefile/               the engine (sim, semantic, security, engine, llm, telemetry, render)
  run_case.py             CLI: run one incident or all four
  run_seeded.py           CLI: draw a random incident from a seed a judge picks
  verify_rubric.py        mechanical scoring against the Round 2 brief
  sync_numbers.py         regenerates every headline figure in the docs from the canonical
                          evaluation, so prose and results cannot drift apart
  eval/                   batch evaluation, likelihood calibration, LLM-invariance proof
  baselines/              the industry-standard comparator
scripts/
  fetch_public_data.py    downloads the third-party UCI dataset on demand
data/
  likelihood_table.json   the CALIBRATED evidence model: measured on 500 incidents from seeds
                          disjoint from evaluation, plus the locked temperature. Versioned,
                          because it is a trained artifact rather than a build product.
  (generated, not committed)
  vantage.duckdb          the warehouse: 3 schemas, 11 tables, 1.2M orders. 125 MB, rebuilt
                          in ~40s by scripts below; GitHub rejects files over 100 MB.
  ground_truth.json       exact Shapley values over 64 counterfactual re-simulations
  documents.jsonl         4,006 anchored documents, ~74% unrelated to any incident
  online_retail_ii.parquet  fetched on demand by scripts/fetch_public_data.py; third-party
                          data is not redistributed here
results/
  workspace.html          the Decision Workspace
  cases/*.json            full case files for the four incidents
  eval/summary.json       the table above
  rubric.json             brief coverage
  llm_invariance.json     196 computed fields, model off vs on
  loop_INC-002.json       the closed loop, pre-registration through measured outcome
tests/                    25 tests, including the two architectural invariants
docs/                     architecture, capability provenance, the brief
```

## The one thing to look at

`results/workspace.html`, incident INC-002. Net revenue is 6% below a
seasonality-and-covariate-adjusted expectation. A price rise and a promotion ending landed a
day apart, nationally, on the same categories, so no control group exists and neither can be
established. The engine says so, ranks the experiments that would settle it by expected net
benefit per day, picks the free reversible one over the one that costs 340,000, and proposes
a hedge for the interim. Then the loop section shows that test actually being run, measured
at +8.02%, and written back into the contract.

## Licence

MIT, see `LICENSE`. Chosen as the conventional default for a portfolio and competition
submission: it lets a judge clone, run and quote the work without friction. Swap it if you
want stronger patent language (Apache-2.0) or something more restrictive.

## Data provenance

- **Vantage Retail Group is fictional.** The warehouse, the document corpus and every incident
  are generated by the simulator in `src/casefile/sim/`. No proprietary data from any real
  organisation is used or implied.
- **Synthetic personal data.** The generated support tickets contain fabricated names, phone
  numbers and addresses at the reserved `mail.example` domain. They are not real people, and
  the PII shield redacts them before anything reaches a model or a narrative.
- **Third-party data is fetched, not redistributed.** `scripts/fetch_public_data.py` downloads
  UCI Online Retail II on demand for the Tier-2 pass, under its own terms.

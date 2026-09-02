# CaseFile

**A KPI intelligence-to-action engine that states the standard of proof its answer meets,
declines when the evidence cannot decide, and prescribes the cheapest test that would.**

Round 2 prototype for the Accenture Innovation Challenge, track *BusinessIntelligence.ai*.
Company, data and incidents are synthetic. No proprietary data from any real organisation is
used or implied.

---

## The claim in one paragraph

Anomaly detection, contribution analysis, natural-language narratives and agentic hypothesis
testing all ship today across the BI market. What no product we could find will do is decline.
When two operational changes land in the same week on the same customers, no control group
exists and the cause is not identified. Every system that must answer will name one, and the
cost of that is not a slow report, it is an expensive intervention aimed at a coincidence. On
seeded incidents built to be unidentifiable, a faithful implementation of the standard
contribution-ranking approach pulls the wrong lever **<!--NUMBERS:wrong-->38.2%<!--/NUMBERS:wrong-->** of the time. CaseFile pulls it
**0%** of the time, because it abstains and returns an experiment, while still answering
**<!--NUMBERS:answer-->66.7%<!--/NUMBERS:answer-->** of identifiable incidents at **100%** top-1 accuracy.

## Quick start

```bash
pip install duckdb pandas numpy scipy statsmodels scikit-learn lightgbm networkx pyyaml jinja2
python3 -m casefile.sim.build_all          # generate the world (~90s)
python3 run_case.py all --llm-mode off     # run all incidents, no model at all
python3 run_case.py all --llm-mode local   # same numbers, prose from local Qwen models
python3 -m pytest tests -q                 # 25 tests
python3 verify_rubric.py                   # score against the brief
open out/workspace.html                    # the Decision Workspace
```

`--llm-mode off` is not a degraded mode. It is the control arm: every computed field is
identical with the model off and on, and `eval/llm_invariance.py` proves it over <!--NUMBERS:fields-->258<!--/NUMBERS:fields--> fields.

## The four demonstration incidents

| Incident | KPI | Verdict | Why |
|---|---|---|---|
| INC-001 | Conversion rate, WEST | **ACT** | Multi-factor. A checkout defect is established at R3 (-11.8%, placebo p=0.022). A carrier degradation carries the higher posterior but reaches only R2, so it is reported as a contributor and not acted on. |
| INC-002 | Net revenue, KITCHEN + DECOR | **ABSTAIN** `collinear_causes` | A price rise and a promotion ending landed a day apart, nationally, on the same categories. No control group. Returns an EVSI-ranked experiment plus a minimax-regret hedge. |
| INC-003 | AOV, SMART_HOME | **ABSTAIN** `sparse_history` | 22 days of history against a 180-day contract floor. Pooled cohort baseline, widened intervals, causal claims disabled by policy. |
| INC-004 | Gross margin | **ABSTAIN** `stale_source` | The ERP cost export is 18 days old against an 8-day SLA. The engine refuses to compute on data it knows is incomplete and routes to a pipeline fix. |

Plus **the loop**: INC-002's recommended test is pre-registered, randomised, executed against
the same generative process, measured by synthetic control at **+8.02%**, and written back so
the contract edge `promo_depth -> conversion_rate` becomes `MEASURED`.

## Architecture: what computes, and what only writes

Numbers and causal verdicts are produced by deterministic code. The language model proposes
hypotheses from a governed library, turns documents into typed claims, and writes prose. It
never computes, never assigns a probability, never adjudicates causality. This is enforced,
not asserted: `casefile/llm/routing.yaml` declares a backend per stage, the gateway raises on
any attempt to call a model from a stage marked `backend: none`, every telemetry span records
whether it sat on the quantitative path, and `tests/test_llm_boundary.py` fails the build if
any span used a model there.

```
sources (3 systems, 3 grains, 3 cadences)
  -> L1 semantic contract  ......... every SQL, policy, threshold and lever comes from YAML
  -> L2 reconciliation ............. definition conflict, ragged edge, entity crosswalk
  -> L3 signal gate ................ MSTL baseline, conformal calibration, weighted FDR,
                                     change-point -> alibi window
  -> L4 contribution ............... LMDI + exact ratio decomposition, labelled R0
  -> L5 evidence ledger ............ diagnosticity-weighted LRs, source-group collapsing
  -> L6 causal engine .............. synthetic control, Abadie placebo, proof rung
  -> L7 verdict .................... act / typed abstention / EVSI decisive test / hedge
  -> L8 action engine .............. driver -> lever -> action -> impact -> owner ->
                                     confidence -> monitoring plan
  -> L9 narrative .................. per persona, numeric closure check
  -> L10 outcome ledger ............ measure the action, upgrade the graph edge
```

## The proof ladder

Every claim carries the rung of the strongest evidence behind it, and the rung controls the
verb. `caused` is grammatically unavailable below R3.

| Rung | Meaning | Permitted language |
|---|---|---|
| R0 | Arithmetic (a decomposition identity) | "accounts for" |
| R1 | Association | "is associated with" |
| R2 | Temporal + mechanism | "is consistent with" |
| R3 | Quasi-experimental, refutations passed | "the evidence suggests X caused Y" |
| R4 | Measured randomised experiment | "caused" |

## Repository map

```
casefile/contracts/kpi_contract.yaml   the executable contract (470 lines)
casefile/sim/                          world model, scenarios, corpus, warehouse, seeded incidents
casefile/semantic/gateway.py           the ONLY path to data; AST test forbids bypass
casefile/security/policy.py            row / column / domain, fail-closed render
casefile/engine/                       baseline, signal_gate, contribution, reconcile,
                                       evidence, causal, verdict, actions, narrative,
                                       feedback, loop, orchestrator, likelihood
casefile/llm/                          routing policy + three-backend gateway
casefile/telemetry/spans.py            method-class spans, token provenance
casefile/render/workspace.py           the self-contained HTML workspace
baselines/                             the industry-standard comparator
eval/                                  batch evaluation, LLM invariance
tests/                                 25 tests including the two invariants
verify_rubric.py                       mechanical scoring against the brief
```

## Confidence calibration

<!--NUMBERS:calibration-->
| Stated confidence | Observed accuracy | n |
|---|---|---|
| 75.5% | 56.8% | 37 |
| 92.7% | 75.8% | 95 |

Brier score **0.23** on 132 answered incidents, measured on seeds the calibration never saw.
<!--/NUMBERS:calibration-->

Likelihood ratios are <!--NUMBERS:likelihood-->CALIBRATED from 500 simulated incidents on held-out seeds, Jeffreys-smoothed<!--/NUMBERS:likelihood-->.

## What being wrong costs

<!--NUMBERS:cost-->
The baseline commits to a lever on every incident, including the 102 where the data cannot identify a cause. It pulled the wrong one 39 times, at a contract-priced cost of **6,492,000**. CaseFile pulled it 0 times, so **6,492,000** of wasted intervention spend was avoided, about 63,647 per unidentifiable incident.
<!--/NUMBERS:cost-->

## Known limitations

Stated plainly, because a prototype that hides them is worse than one that does not.

1. **Likelihood ratios are a declared prior table, not yet calibrated.** The calibration path
   exists (`engine/likelihood.py` loads a measured table when present) and the simulator can
   produce it, but the numbers shipped here come from the declared table. Every case file says
   which was used.
2. **The carrier hypothesis in INC-001 cannot reach R3.** Its true effect is about -3.1% and
   with 15 donor units the placebo floor is 0.053. That is an honest power limit, and the
   engine reports it as a contributor rather than inflating the evidence.
3. **A from-history forecast and a counterfactual are different quantities.** For net revenue
   they agree to 0.04pp (-8.02% measured against -7.98% injected). For conversion they differ
   by about 4pp because the counterfactual removes causes that were also present during the
   training window. Root-cause ranking is scored, not total-effect recovery.
4. **Float summation in a parallel query engine is not bitwise reproducible.** LLM invariance
   is asserted to a relative tolerance of 1e-12; the worst observed difference is 1.5e-14.
5. **Two personas share a lever in some incidents**, so their recommended actions can coincide
   even though their entitlements and narratives differ.
6. **The UCI Online Retail II pass is detection-only.** Only 2 of 5 KPIs are computable on it.

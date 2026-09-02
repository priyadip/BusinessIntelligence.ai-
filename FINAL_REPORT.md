# Final report

## What was built

A working KPI intelligence-to-action engine, ~4,300 lines of Python across thirteen layers,
running on a synthetic three-source enterprise warehouse with injected ground truth.

**The executable contract (470 lines of YAML).** Not documentation. Every SQL statement is
compiled from it, every access decision is enforced from it, every freshness gate and alert
threshold reads from it, every hypothesis must resolve to an entry in its library, and every
recommended action must resolve to a lever with a named owner and a cost model. A test walks
the AST of every module and fails the build if anything opens a database connection outside
the semantic gateway.

**Three heterogeneous sources.** Commerce at order-line grain refreshing every 15 minutes;
an ERP ledger at SKU-week grain landing three days late with a *different revenue definition*;
ops telemetry and a 3,752-document corpus at hourly cadence with shorter history. The
reconciliation layer arbitrates the definition conflict against a declared tolerance, marks
the ragged right edge INCOMPLETE rather than reading it as a collapse, and routes unresolved
entities to a REVIEW bucket instead of dropping them.

**Detection that survives a statistician.** MSTL fitted on training data only and projected
forward, with the level estimated from non-anomalous days so a prior incident cannot depress
the forecast. Calibration by rolling-origin forecast errors at the same horizon, not in-sample
residuals. Materiality enters as Genovese-Roeder-Wasserman p-value weights with
Benjamini-Hochberg on the full family. A change-point on the deseasonalised, detrended series
gives a bootstrap onset interval that is then used as an **alibi screen**: a candidate cause
whose event lies outside the window is eliminated before any evidence is weighed.

**Exact decompositions.** LMDI for multiplicative factors and a closed-form ratio-of-sums
decomposition for ratio KPIs, because three of the five KPIs are ratios where price-volume-mix
is not an identity at all. Residuals verified at 5.6e-09 and 4.3e-19.

**Causal estimation with real inference.** Simplex-constrained synthetic control via NNLS with
Abadie in-space placebo inference on the MSPE ratio. Donor pools exclude every unit touched by
a concurrent cause. The fitting period ends at the detected onset, not the analysis window. A
pre-trend placebo offset from the onset must pass or the estimate is demoted with the failure
shown in the case file.

**The decision protocol, which is the differentiator.** A permanent none-of-the-above
hypothesis; diagnosticity weighting so evidence consistent with every hypothesis contributes
nothing; source-group collapsing so forty reviews from one week are one fact with a
corroboration count; a three-condition rule to act; a typed abstention taxonomy; and when
abstaining, candidate experiments ranked by **expected net benefit of sampling per day** with
EVPI as the ceiling and a minimax-regret hedge for the interim.

**The closed loop, executed.** The recommended experiment is pre-registered with a hashed
randomised assignment, run against the same generative process, measured by synthetic control
at +8.02%, written to an outcome ledger, and the contract edge `promo_depth -> conversion_rate`
is upgraded to MEASURED with an effect, a confidence band and an n.

**A bounded LLM layer.** Qwen2.5-1.5B for extraction and Qwen2.5-14B for narrative, running
locally on two L40S cards. Six model calls per incident, ~3,800 tokens in and ~1,100 out,
$0.000211 per insight measured with the models' own tokenizers. Nine stages declare a backend
and a written justification; six are `backend: none` and the gateway raises if they try.

## What was tested

| Test | Result |
|---|---|
| Unit and invariant suite (`pytest tests`) | **22 passed** |
| Semantic-gateway choke point (AST walk over every module) | **pass** — nothing bypasses it |
| LLM boundary (no model on the quantitative path, in emitted telemetry) | **pass** |
| LLM invariance: same incident, model off vs local, 196 computed fields | **identical** to 1e-12 relative; worst observed 1.0e-14 |
| LMDI decomposition residual | 5.6e-09 |
| Ratio decomposition residual | 4.3e-19 |
| Shapley ground truth sums to total effect | exact, all incidents |
| Detection accuracy vs injected truth (net revenue) | **-8.02% measured vs -7.98% injected** |
| Change-point onset vs injected onset (INC-001) | detected 2026-07-16, 90% interval 07-15..07-16; true 07-15 |
| Causal effect vs raw DiD (checkout) | -11.99% estimated vs -12.39% raw |
| Batch evaluation, <!--NUMBERS:n-->300<!--/NUMBERS:n--> seeded incidents | see table below |
| Four demonstration incidents end to end | ACT / collinear / sparse / stale, all as designed |
| PII shield on 475 extracted claims, ops_manager view | **0 leaks** |
| Prompt-injection quarantine | planted payload caught, quarantined not stripped |
| Rubric coverage against the Round 2 brief | see `results/rubric.json` |

### Batch evaluation, <!--NUMBERS:n-->300<!--/NUMBERS:n--> seeded incidents

<!--NUMBERS:table-->| | Contribution-ranking baseline | CaseFile |
|---|---|---|
| Top-1 cause accuracy, identifiable | 99.5% | **100.0%** |
| Answer rate, identifiable | 100% | 66.7% |
| **Wrong lever pulled, unidentifiable** | **38.2%** | **0.0%** |
| Abstention rate, unidentifiable | 0% | 100.0% |
<!--/NUMBERS:table-->

## Notable bugs found and fixed during the build

These are listed because each was a real defect that would have been fatal on stage.

1. **MSTL was fitting the anomaly.** Baseline trained on the full series, so the trend absorbed
   the drop and expected tracked actual (-0.31% instead of -9%). Fixed by training strictly
   before the window and projecting forward.
2. **Calibration compared in-sample residuals to out-of-sample forecast errors**, so every
   monitor looked extreme and every p-value pinned to the floor. Fixed with rolling-origin
   forecast errors at the same horizon.
3. **A prior incident in the training tail depressed the projected level**, making the next
   window read +12% when it was -8%. Fixed by estimating the level from non-anomalous days
   against a stiffened trend.
4. **The change-point found the train/test seam**, not the real onset, because residuals were
   in-sample before the split and out-of-sample after. Fixed by running detection on a
   homogeneous deseasonalised, detrended series.
5. **Degenerate synthetic-control donors.** A category launched mid-window had no pre-period
   data; pivot back-fill turned it into a constant series that dragged every estimate to zero.
   Fixed with an 80% coverage requirement.
6. **The synthetic-control fitting period ran past the onset**, absorbing part of the treatment
   and shrinking effects toward zero. Fixed by ending the fit at the detected onset.
7. **The pre-trend placebo sat on top of the intervention ramp** and demoted every true effect
   by construction. Fixed with a buffer measured from the onset.
8. **Operator precedence** made every collinear case report as `underpowered`.
9. **Text evidence was counted once per region-week**, so one story told sixteen times moved
   the posterior sixteen times. Fixed by collapsing to one group per claim type.
10. **`BatchEncoding` subclasses `UserDict`, not `dict`**, so an `isinstance` guard wrapped the
    encoding inside itself and every local generation failed silently to a template.
11. **`device_map="auto"` sharded a 1.5B model across two GPUs** and produced NaN logits.
12. **The world was never reproducible across processes.** Per-cell noise was seeded with
    `hash((region, channel, category))`, and Python salts string hashing per interpreter. The
    ground truth was generated in one process and the warehouse in another, so they silently
    disagreed, and KPI deltas moved between runs for reasons I had partly misattributed to
    code changes. Replaced with a CRC32-based `stable_hash`; the world is now identical across
    processes and ground truth matches the warehouse to 1e-9.
13. **The alibi screen could eliminate every evidenced hypothesis.** A change-point that
    landed two weeks early once killed both true causes and handed a 37% posterior to a data
    artifact that had fired no evidence at all, which is exactly the "best of a bad set"
    failure this system exists to prevent. The screen is now a soft, confidence-weighted
    penalty capped so that elimination requires a well-determined onset, and it disables
    itself and says so when it would otherwise eliminate everything carrying evidence.
14. **Text evidence was modelled asymmetrically.** Only the price rise generated customer
    complaints, while an ended promotion generated none, which made a genuinely confounded
    pair look separable. A shopper sees the price they pay, not its decomposition, so both
    now generate the same complaint theme and the calibration learns that the theme is
    non-diagnostic between them (LR 3.8 versus 3.1).
15. **A partial run silently overwrote a canonical case file.** Running the CLI with a persona
    subset wrote a one-persona case file over the shipped four-persona one, which would have
    degraded the workspace and the rubric without any error. Partial runs now write to a
    distinct `<incident>.partial-<personas>.json` and leave the canonical artifact alone.

## Confidence calibration

<!--NUMBERS:calibration-->| Stated confidence | Observed accuracy | n |
|---|---|---|
| 75.5% | 56.8% | 37 |
| 92.7% | 75.8% | 95 |

Brier score **0.23** on 132 answered incidents, measured on seeds the calibration never saw.<!--/NUMBERS:calibration-->

Likelihood ratios are <!--NUMBERS:likelihood-->CALIBRATED from 500 simulated incidents on held-out seeds, Jeffreys-smoothed<!--/NUMBERS:likelihood-->.

## What being wrong costs

<!--NUMBERS:cost-->The baseline commits to a lever on every incident, including the 102 where the data cannot identify a cause. It pulled the wrong one 39 times, at a contract-priced cost of **6,492,000**. CaseFile pulled it 0 times, so **6,492,000** of wasted intervention spend was avoided, about 63,647 per unidentifiable incident.<!--/NUMBERS:cost-->

## Limitations

Stated plainly. A prototype that hides these is worse than one that does not.

1. **Confidence is better calibrated but still overconfident.** Likelihood ratios are now
   measured from 500 incidents on calibration seeds disjoint from evaluation, and a single
   temperature is fitted on those same seeds and locked before any evaluation seed is touched
   (Brier 0.28 to 0.21 on the calibration set). On held-out seeds the engine still states
   about 93% where it is right about 76%. Two honest reasons: in this synthetic world the
   mapping from cause to observable evidence is nearly deterministic, so many ratios saturate
   the clip; and "correct" is scored strictly as naming the *dominant* cause, so naming a real
   but secondary contributor counts as wrong. We report the residual rather than tuning the
   metric until it disappears.
2. **The carrier hypothesis in INC-001 cannot reach R3.** Its true effect is about -3.1% and
   with 15 donor units the placebo p-value floor is 0.053. That is an honest power limit; the
   engine reports it as a contributor rather than inflating the evidence to reach a verdict.
3. **A from-history forecast and a counterfactual are different quantities.** For net revenue
   they agree to 0.04pp. For conversion they differ by about 4pp, because the counterfactual
   removes causes that were also active during the training window. Root-cause *ranking* is
   scored, not total-effect recovery.
4. **Float summation in a parallel query engine is not associative**, so bitwise
   reproducibility is not claimed. LLM invariance holds to 1e-12 relative.
5. **Two of four persona narratives fall back to the template on some incidents**, when the
   14B model cannot produce prose in which every numeral binds. The fallback is correct
   behaviour, but it means the LLM path is exercised for roughly half the personas.
6. **The UCI Online Retail II pass is detection-only.** Only 2 of 5 KPIs are computable on it.
7. **The evaluation harness runs a lighter evidence model than the full pipeline** (driver
   deltas rather than the eight SQL tests plus corpus extraction), so its absolute accuracy
   figures are not directly comparable to a full case file.
8. **Personas can share a lever**, so two roles occasionally see the same recommended action
   even though their entitlements, narratives and proof rungs differ.

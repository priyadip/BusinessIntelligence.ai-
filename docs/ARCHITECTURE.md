# Architecture

## The invariant everything rests on

> Numbers and causal verdicts are computed by deterministic code. The language model
> proposes, reads and writes. It never computes, never assigns a probability, never
> adjudicates causality.

This is enforced in four places, not asserted in one:

1. `casefile/llm/routing.yaml` declares a backend per stage and a written justification for
   each. Nine stages; six are `backend: none`.
2. `LLMGateway.generate()` raises `PermissionError` if a stage marked `none` tries to call a
   model.
3. Every telemetry span records `on_quantitative_path` and `backend`; the span writer
   computes `violates_llm_boundary` and cannot be talked out of it.
4. `tests/test_llm_boundary.py` fails the build if any emitted span violated it, and
   `eval/llm_invariance.py` re-runs an incident with the model off and on and compares every
   computed field.

## Data: three systems, three grains, three cadences

| Source | Grain | Refresh | Lag | Quality | Notes |
|---|---|---|---|---|---|
| `commerce` | order line / session | 15 min | 20 min | gold | orders, sessions, checkout telemetry, price and promo daily |
| `finance_erp` | sku x ISO week | weekly | 3 days | silver | **different revenue definition**, column-restricted, cost export deliberately stale |
| `ops_voice` | shipment event / document | hourly | 45 min | bronze | telemetry plus 3,752 anchored documents, 79% unrelated to any incident |

Heterogeneity is the point, not an accident: the reconciliation layer has real work because
the two revenue definitions genuinely differ, the ERP lands late, and the ops corpus has
shorter history than commerce.

## Layers

**L1 Semantic contract.** 470 lines of YAML that behave as code. Every SQL statement is
compiled from `kpis.*.measure`; every access decision reads `policies`; every freshness gate
reads `sources.*.freshness_sla_minutes`; every threshold reads `kpis.*.thresholds`; every
hypothesis comes from `hypothesis_library` and must resolve to a `levers` entry with an owner.
`tests/test_choke_point.py` walks the AST of every module and fails if anything opens a DuckDB
connection outside `casefile/semantic/`.

**L2 Reconciliation.** Commerce books revenue at order time; the ledger books at ship date and
nets supplier rebates. The contract names commerce canonical, declares a 0.5% tolerance, and
on breach reports the delta, attributes what it can, and multiplies down the confidence of any
dependent claim. Weekly-to-daily conformance uses a named allocation rule, and the ragged right
edge is marked INCOMPLETE rather than materialised as zeros.

**L3 Signal gate.** MSTL on a variance-stabilised scale, fitted on training data only and
projected forward, with the level estimated from non-anomalous days so a prior incident in the
training tail cannot depress the forecast. Calibration is by rolling-origin forecast errors at
the same horizon, not in-sample residuals. Materiality enters as Genovese-Roeder-Wasserman
p-value weights and Benjamini-Hochberg runs on the full family; materiality never filters
before the procedure. A binary-segmentation change-point on the deseasonalised, detrended
series yields a bootstrap onset interval.

**L4 Contribution.** LMDI for multiplicative factors, an exact ratio-of-sums decomposition for
ratio KPIs. Residuals verified at 5.6e-09 and 4.3e-19. Labelled R0 and never described as a
cause.

**L5 Evidence.** Eight structured tests against observable tables, plus deterministic typed
claim extraction from the corpus. Evidence is weighted by *diagnosticity*: a likelihood-ratio
vector that is flat across hypotheses contributes nothing. Claims are clustered so forty
reviews from one week are one fact with a corroboration count, not forty independent facts.

**L6 Causal.** Simplex-constrained synthetic control via NNLS, with Abadie in-space placebo
inference on the MSPE ratio. Donor pools exclude every unit touched by a concurrent cause. The
fitting period ends at the detected onset, not the window start. A pre-trend placebo offset
from the onset must pass or the estimate is demoted to R2 with the failure shown.

**L7 Verdict.** Act only if the leading posterior clears the threshold, its mechanism reaches
R3, and expected value stays positive across most of the posterior mass. Otherwise a typed
abstention. When abstaining, candidate experiments are ranked by expected net benefit of
sampling per day, EVPI is reported as the ceiling, and a minimax-regret hedge is proposed.

**L8 Actions.** `driver -> lever -> action -> expected impact -> owner -> confidence ->
monitoring plan`, exactly as the brief specifies. Impact comes from a MEASURED graph edge where
one exists; otherwise the action is stamped `NO_MEASURED_EFFECT` and ranked below everything
that has one.

**L9 Narrative.** Per persona, from a frozen evidence object. Every numeral in generated prose
must bind to a computed value or the draft is rejected and regenerated.

**L10 Outcome ledger.** The recommended experiment is pre-registered with a hashed assignment
vector, executed, measured by synthetic control, and written back. The graph edge
`promo_depth -> conversion_rate` is upgraded to MEASURED with an effect, a CI and an n.

## Personas differ by decision, not depth

The confidence bar is a function of what the action costs and how reversible it is. More
importantly, **entitlement can change the achievable standard of proof**: the Regional Ops
Manager is scoped to WEST, so the control regions a causal estimate needs lie outside their
row scope and their verdict is capped at R2. The case file says so explicitly and names the
escalation path rather than silently degrading.

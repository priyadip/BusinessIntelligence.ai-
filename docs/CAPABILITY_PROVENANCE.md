# Capability provenance: native, configured, custom, integrated

The Round 2 brief asks teams to distinguish native, configured, custom-built and externally
integrated capabilities, and permits platform-native solutions. This is where each layer of
CaseFile sits, and why.

**Why fully custom rather than platform-native.** Six of the thirteen layers below have a
credible native equivalent, and in a real deployment we would use them: metric semantics,
anomaly detection, contribution analysis, retrieval, narrative generation and workflow are
all commodity. We built them here for one reason: the differentiator is a *decision protocol*
that spans those layers (calibrated uncertainty to decisive experiment to measured outcome),
and a prototype that stitched six vendor APIs together would demonstrate the integration, not
the protocol. The custom layers are the ones no platform exposes: abstention with a typed
taxonomy, expected value of sample information over a hypothesis posterior, and an outcome
ledger that upgrades a causal graph edge. Those are ~1,100 lines of the ~4,000 written.

| Layer | Classification | Native equivalent we would use in production | What we built and why |
|---|---|---|---|
| L0 Simulator + ground truth | **CUSTOM** | none | Ground truth for "why" does not exist in real data. Required to measure the abstention claim at all. |
| L1 Semantic / KPI contract | **CONFIGURED** | Databricks Unity Catalog metric views; dbt MetricFlow; Snowflake semantic views | A 470-line YAML contract compiled into SQL. Same shape as a metric view, plus fields no platform models: proof-rung policy, abstention thresholds, decision rights. |
| L2 Reconciliation | **CUSTOM** | partial: dbt tests, Monte Carlo data observability | Definition conflict between commerce and the ERP ledger, ragged-edge completeness, entity crosswalk with a REVIEW bucket. Platforms detect drift; none arbitrate a definition conflict and degrade downstream confidence. |
| L3 Signal gate | **NATIVE-EQUIVALENT, rebuilt** | Databricks Lakehouse Monitoring; Snowflake ML anomaly detection; Tableau Pulse | Rebuilt to add two things the native detectors do not expose: weighted FDR across the whole monitor family, and a change-point interval used as an alibi screen. |
| L4 Contribution | **NATIVE-EQUIVALENT, rebuilt** | Fabric decomposition tree / key influencers; Snowflake Contribution Explorer (ex-Sisu) | Rebuilt only because three of five KPIs are ratios, where price-volume-mix is not an identity. LMDI and an exact ratio-of-sums decomposition, residual verified below 1e-9. |
| L5 Evidence ledger | **CUSTOM** | partial: Cortex Search, Qlik Answers for retrieval | Retrieval is commodity; diagnosticity weighting and source-group collapsing are not. |
| L6 Causal engine | **INTEGRATED (libraries)** | causaLens decisionOS | scipy NNLS synthetic control with Abadie in-space placebo inference. Library maths, custom wiring to the proof ladder. |
| L7 Verdict, abstention, EVSI | **CUSTOM** | none found | The core claim. Typed abstention, permanent none-of-the-above hypothesis, EVSI ranked on expected net benefit of sampling, minimax-regret hedge. |
| L8 Action engine | **CONFIGURED** | Palantir Ontology actions; ServiceNow | Levers, owners, costs, reversibility and decision rights all come from the contract. |
| L9 Narrative | **INTEGRATED (local models)** | AI/BI narratives; Copilot; Spotter | Qwen2.5-1.5B and 14B running locally. The custom part is the numeric closure check and rung-gated verbs. |
| L10 Feedback + outcome ledger | **CUSTOM** | none found | Writes the measured effect back and upgrades the graph edge. The third arrow of the loop. |
| L11 Security | **CONFIGURED** | Unity Catalog row/column masking; Snowflake policies | Three-layer enforcement with a fail-closed render assertion on derived values, which masking alone does not give. |
| L12 Telemetry | **CUSTOM** | platform system tables | Per-span method class in the brief's own vocabulary, plus a machine-checked LLM-boundary invariant. |

**Honest summary.** Roughly 40% of this system is a rebuild of commodity capability, done to
control the seams. Roughly 30% is configuration of things a platform would own. The remaining
30% (L7, L10, L12 and the parts of L2 and L5 that feed them) is the part we would still have
to build on any platform, and is where the differentiation lives.

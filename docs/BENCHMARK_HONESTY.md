# The benchmark measures the system, not the harness

## What was wrong

`src/eval/run_batch.py` used to contain this line:

```python
h.rung = "R3" if inc["identifiable_by_construction"] else "R2"
```

`identifiable_by_construction` is the simulator's own label for whether it built the
incident with separable causes. It is the answer. The contract requires rung R3 before any
cause may be named, so feeding that label in meant:

```
identifiable == False  ->  every rung R2  ->  R2 < R3  ->  ABSTAIN, always
```

Abstention on unidentifiable incidents was therefore **structurally guaranteed**. The
reported "0.0% wrong lever" was a property of the test harness. No data could have produced
a different number. Worse, the comparison was asymmetric: the baseline received only
observed driver movements and never saw the flag.

## What replaced it

The rung is now produced by actually running the causal estimator.

**Which units were treated is an observation, not a lookup.** For each driver the harness
measures where it moved between the pre-period and the window:

```python
denom = np.maximum(pre.abs(), max(float(pre.std(ddof=0)), 1e-9))
d = (post - pre) / denom.replace(0, np.nan)
return set(d[d.abs() >= DRIVER_MOVED_THRESHOLD].index)
```

Every column used here (`price_index`, `promo_depth`, `checkout_error_rate`, `ontime_pct`,
`stockout_rate`, `competitor_gap`, `marketing_mult`) is a measured series the evidence layer
already reads. A real deployment can see where a change landed without being told which
change caused anything.

Donors are the units where **no** driver moved. Then:

- If a driver moved everywhere, no donor pool exists, `control_available=False`, and the
  estimator caps the rung at R2 by itself. This is the outcome the flag used to assert.
- If a driver moved in some units and not others, synthetic control runs with Abadie
  in-space placebo inference and the pre-trend placebo, exactly as the four demonstration
  incidents do. R3 requires placebo p <= 0.10 and a passing pre-trend.
- If a driver never moved, there is no scope to estimate and the hypothesis stays at R1.
  This is an observation that the driver was flat, not a hint that it is innocent.

The flag survives in the code in exactly one place: scoring. It splits the results into
identifiable and unidentifiable **after** every decision has been made.

A test enforces this. `test_the_batch_evaluation_never_reads_the_ground_truth_flag_to_decide`
parses the estimator's syntax tree, strips the docstring, and fails if the flag appears in
the code.

## The result

Both modes are still runnable, so the difference can be inspected rather than described:

```bash
python3 eval/run_batch.py -n 300 --rung-mode estimated   # the default
python3 eval/run_batch.py -n 300 --rung-mode oracle      # the old behaviour, for comparison
```

| Measure | Oracle (old) | **Estimated (now)** |
|---|---|---|
| Baseline top-1 accuracy, identifiable | 99.5% | 99.5% |
| CaseFile top-1 accuracy, identifiable | 100.0% | **100.0%** |
| CaseFile answer rate, identifiable | 66.7% | **20.2%** |
| **Wrong lever pulled, unidentifiable: baseline** | 38.2% | **38.2%** |
| **Wrong lever pulled, unidentifiable: CaseFile** | 0.0% | **0.0%** |
| Abstention rate, unidentifiable | 100.0% | **99.0%** |

### The headline survived

The wrong-lever rate is still 0.0%, and it is now a measurement. Of 102 unidentifiable
incidents the engine acted on exactly one, seed 9249, and **it was right**: it named
`H_PROMO_WITHDRAWAL`, which was the dominant true cause.

That single case is worth understanding. `random_incident.draw()` gives category-scoped
causes their own two categories even when the incident is labelled unidentifiable, drawn
independently per cause. When two such causes land in different categories they are
genuinely separable, and the label is wrong. The estimator found the truth where the
ground-truth flag did not, which is the strongest possible evidence that it is not simply
reproducing the label.

Rung distribution confirms the separation is real rather than assumed:

| | R1 | R2 | R3 |
|---|---|---|---|
| Identifiable (198) | 28 | 109 | **61** |
| Unidentifiable (102) | 45 | 56 | **1** |

### The cost, which is not small

**The answer rate on identifiable incidents falls from 66.7% to 20.2%.** The oracle was
inflating it more than threefold. Estimating a counterfactual with a real placebo test is
far harder than being told one exists, and the honest number is the lower one.

That is the trade this system is built to make, now measured on its own benchmark rather
than assumed: it answers a fifth of the time, and when it answers it is right.

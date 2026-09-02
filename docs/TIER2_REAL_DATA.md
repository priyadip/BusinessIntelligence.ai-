# Tier-2: the engine against real public data

Everything else in this repository is scored against a world produced by
`src/casefile/sim/`, written by the same author as the engine. A good result there
establishes internal consistency and nothing else. This pass runs the same statistical
machinery over **UCI Online Retail II**: 1,067,371 real transactions from a UK online
retailer, December 2009 to December 2011, which nobody here generated.

```bash
python3 scripts/fetch_public_data.py        # once, needs the internet
cd src && python3 eval/tier2_real.py        # about 8 seconds on 32 workers
```

Results land in `results/tier2/`. Two of the four tests are **negative controls**: they
are designed to be able to fail, and one of them does.

## What the real data forced

| Reality | Consequence |
|---|---|
| The shop is closed on Saturdays (1 Saturday in 105) | The weekly cycle is **6 days, not 7**. `baseline.py` now reads the cycle length from the contract; the default stays 7, so the synthetic world is unchanged |
| 19,494 cancellations, 3,457 non-positive quantities, 2,750 non-positive prices | Each is dropped under a named rule and counted. A test asserts every dropped row is attributed to a rule |
| 236,121 rows have no customer id | Kept. They are guest checkouts, and dropping them would bias revenue down by about a fifth |
| 31 trading days missing (public holidays) | Linearly interpolated, count reported |
| Only 4 countries have 200+ days | Countries cannot be a donor pool. Units for synthetic control are stockcodes |
| Stockcode revenue is sparse day to day | `causal.py` drops units below 80% coverage. Only 69 of 397 stockcodes survive. Using a looser filter silently empties the donor pool |

97.59% of rows survive cleaning. The UK series is 634 trading days.

## T2-B  Do the exact decompositions still close?

**Yes.** Residual `2.84e-13` over 31 country segments, against a tolerance of `1e-9`.

This was expected, since the ratio-of-sums identity is algebraic, but it confirms the
adapter maps real data into the engine's shape correctly.

## T2-C  How often does the detector fire when nothing happened?  **FAILS, and the reason is exact**

240 random 14-day windows, no known intervention.

| Nominal alpha | Empirical fire rate |
|---|---|
| 0.01 | **0.279** |
| 0.05 | 0.388 |
| 0.10 | 0.463 |

At a nominal 1% the detector fires 28% of the time. That is a 28-fold overshoot and it
would make the system unusable on this data.

The failure is not general. It is entirely seasonal:

| Month | Fire rate at alpha=0.01 |
|---|---|
| Feb | 0.083 |
| Mar to Aug | **0.000** |
| Sep | 0.344 |
| Oct | 0.310 |
| Nov | 0.615 |
| Dec | 0.812 |
| Jan | **1.000** |

Restricted to February through August, 120 windows:

| Nominal alpha | Empirical | Verdict |
|---|---|---|
| 0.01 | **0.0083** | calibrated |
| 0.05 | 0.0333 | conservative |
| 0.10 | 0.0750 | conservative |

**Diagnosis.** `baseline.py` will only use a seasonal period when the training block holds
two full cycles. The annual cycle is 313 trading days here, so it needs 626 training days.
The series is 634 days long and the training block always ends before the tested window, so
**the annual cycle is never available**. MSTL removes the weekly cycle and the trend
smoother cannot follow the Christmas ramp, so every window from September to January is
compared against an expectation that ignores Christmas.

This is a real limitation of the method as configured, not a coding error. It says: *the
detector requires two full years of history before it can be trusted through a seasonal
peak, and it degrades silently rather than refusing.* The engine has an
`INSUFFICIENT_HISTORY` path for a series that is too short outright; it has no equivalent
for a series long enough to fit but too short for its dominant cycle.

## T2-D  Does synthetic control invent causal effects?  **PASSES, conservatively**

150 trials. Each picks a random date with no intervention of any kind, a random treated
stockcode, and 39 donors, then runs the full estimator including Abadie in-space placebo
inference and the pre-trend placebo. Every R3 verdict here is a false positive by
construction.

| | |
|---|---|
| Trials | 150 |
| Trials where the placebo floor permits R3 at all | 131 |
| Trials blocked by the floor (too few usable donors) | 19 |
| **Spurious R3 among reachable trials** | **6.1%** |
| Threshold, so the expected rate | 10% |
| Placebo p quartiles | 0.264 / 0.513 / 0.777 |

The rate is reported on the 131 reachable trials rather than all 150, because a trial whose
donor pool gives a floor above 0.10 cannot return R3 whatever the data says, and including
those would flatter the result.

**6.1% against a nominal 10%, on real data with no intervention.** The placebo p-values sit
slightly high of uniform, which is the signature of a conservative test. This is the
strongest evidence in the repository that the causal layer does not over-claim, and unlike
the synthetic benchmark it could have come out the other way.

## T2-E  Is a known shock recovered?  **Low power, and a level bias**

A multiplicative shock is injected into a real window; the baseline is fitted on the
untouched history. Windows are drawn from February to August only, because a power
measurement taken across the Christmas ramp measures baseline bias instead of power.

| Injected | Detected at alpha=0.05 | Median estimate | Bias |
|---|---|---|---|
| -2%  | 7%  | +5.5% | +7.5pp |
| -5%  | 7%  | +2.3% | +7.3pp |
| -10% | 7%  | -3.1% | +6.9pp |
| -20% | 27% | -13.9% | +6.1pp |

Two things are true at once. The **differences** are accurate: moving the injected shock
from -2% to -20% moves the estimate by -19.4 points against an expected -18. But there is a
persistent **+6 to +7 point level bias**, and power is poor: a genuine 20% revenue collapse
is detected in only 27% of windows at alpha=0.05.

For comparison, the synthetic benchmark reports detection of a -8% move at p < 1e-12. The
difference is the noise: the real UK series has a daily coefficient of variation far above
the simulated one, and the simulator's incidents are large relative to its noise in a way
this real series is not.

## What this pass changes about the claims

| Claim made elsewhere in the repository | Status after Tier-2 |
|---|---|
| Exact decompositions, residual under 1e-9 | **Confirmed on real data** |
| Synthetic control does not over-claim | **Confirmed on real data**, 6.1% against a 10% threshold |
| The engine detects material KPI movements | **Qualified.** True given two full seasonal cycles. Without them the detector fires on 28% of null windows and has 27% power against a 20% shock |
| The system is portable to another schema | **Partly demonstrated.** The adapter is 180 lines and the entire mathematical core needed no change. One engine change was required: the seasonal cycle length had to become contract-declarable |

## The honest caveat on T2-C

Real windows are not a true null. Genuine shocks occur in this data and are not labelled,
so the 27.9% aggregate is an upper bound on the false-positive rate rather than an unbiased
estimate of it. The February-to-August figure of 0.83% at a nominal 1% is the more
informative number, and it is an upper bound too, which makes the calibration result there
stronger rather than weaker.

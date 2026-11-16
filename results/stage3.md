# Stage 3 results: PBO, regime switching and alpha decay

Reproduce with `uv run python -m overfitlab all` (about 10 minutes on 8 cores).
The 13 earlier world types, their truth and all earlier results are unchanged
to the last digit. Seven world types and one filter are new.

## What is new

**Filter: probability of backtest overfitting (PBO)** by combinatorially
symmetric cross-validation, checked against Bailey, Borwein, Lopez de Prado &
Zhu (2017), Algorithm 2.3 and Section 3.1, read from the paper. 16 blocks,
12,870 in-sample/out-of-sample combinations (the paper prints 12,780; C(16, 8)
is 12,870). PBO judges the whole selection process, so as a filter it can
accept at most one strategy per world: the one with the best research-sample
Sharpe, **if PBO is below 5%**. The 5% cut is my choice by analogy with a 5%
test; the paper gives no decision rule.

**Regime switching.** A hidden two-state Markov chain. Calm spells last about
250 days and crises about 50, so about one day in six is a crisis. In a crisis
the common factor's volatility triples and asset-specific volatility rises by
half. Measured against the plain no-signal world: annual volatility 23% to
29%, average correlation 0.25 to 0.39, kurtosis 7 to 11.

**Alpha decay.** The planted signal's strength halves every 5 years. Truth is
then the Sharpe a strategy earns in the **5 years after the backtest**, because
a decayed signal is real in the sample and gone afterwards. Measured on 2,000
paths per world type, so the standard error of a true Sharpe stays at 0.010
(largest 0.013), as before.

New world types, 200 worlds each:

| World type | Planted-rule Sharpe that counts as truth | True strategies of 76 |
|---|---|---|
| No signal, regimes | none | 0 |
| Momentum / cross-sectional / mean reversion, medium, regimes | 0.68 / 0.92 / 0.64 (plain: about 1.0) | 32 / 26 / 14 |
| Momentum, obvious, decay | 0.14 after the backtest | 6 |
| Cross-sectional, obvious, decay | 0.08 after the backtest | 0 |
| Mean reversion, obvious, decay | -0.07 after the backtest | 0 |

In a separate check on 300 paths (not part of `all`), the planted rule in the
decay worlds earned a Sharpe of 1.0 to 1.2 in the research sample and 0.24 to
0.43 in the lockbox. So these are worlds where a strategy looks like Sharpe 1
in the backtest and is worth almost nothing once traded.

## Known-answer tests

| Test | Expected | Measured |
|---|---|---|
| PBO, number of combinations for 16 and 4 blocks | 12,870 and 6 | same |
| PBO, one strategy always best: logit and PBO | ln(0.75 / 0.25), 0 | same |
| PBO on interchangeable noise strategies | about 0.5 | 0.487 (mean of 60 runs) |
| PBO when in-sample winners reverse out of sample | near 1 | 0.944 |
| Regimes: share of crisis days | 1/6 | within 0.04 |
| Existing worlds reproduce exactly after the simulator change | identical | identical |

## PBO on the plain worlds

| Worlds | Share of worlds where the winner is accepted | Share of those that are false |
|---|---|---|
| No signal | 1.0% | all |
| Barely | 8.2% | about half (FDR 4.2% [2.7, 5.8]) |
| Weak | 17.3% | FDR 1.0% |
| Medium | 47.3% | FDR 0.8% |
| Obvious | 96.8% | none |

PBO behaves well as a yes/no check on the best strategy: it almost never
approves a winner picked from noise and almost always approves one picked from
an obvious signal. Its "power" in the other tables is tiny by construction,
because it can accept one strategy while 14 to 34 are truly profitable.

## Regime switching

Share of no-signal worlds with at least one false discovery:

| Filter | Plain | Regimes |
|---|---|---|
| Naive Sharpe > 1 | 4.0% [1.5, 7.0] | 10.0% [6.0, 14.5] |
| Holdout | 0.0% | 0.0% |
| Walk-forward | 0.5% [0.0, 1.5] | 5.5% [2.5, 8.5] |
| Deflated Sharpe, paper | 0.0% | 0.0% |
| Deflated Sharpe, variant | 0.0% | 1.0% [0.0, 2.5] |
| CPCV | 1.0% [0.0, 2.5] | 4.5% [2.0, 7.5] |
| Benjamini-Hochberg | 0.0% | 1.5% [0.0, 3.5] |
| Benjamini-Yekutieli | 0.0% | 0.0% |
| Romano-Wolf | 0.5% [0.0, 1.5] | 2.0% [0.5, 4.0] |
| PBO | 1.0% [0.0, 2.5] | 4.0% [2.0, 7.0] |

Medium signal, FDR / power:

| Filter | Plain | Regimes |
|---|---|---|
| Naive Sharpe > 1 | 1.6% / 16.4% | 1.9% / 10.0% |
| Holdout | 0.3% / 6.6% | 0.1% / 2.8% |
| Walk-forward | 0.7% / 11.8% | 0.6% / 6.0% |
| Deflated Sharpe, variant | 0.2% / 4.4% | 0.4% / 2.2% |
| CPCV | 1.1% / 8.1% | 1.4% / 5.1% |
| Benjamini-Hochberg | 1.1% / 18.5% | 0.8% / 10.0% |
| Benjamini-Yekutieli | 0.2% / 8.1% | 0.4% / 3.5% |
| Romano-Wolf | 0.8% / 9.5% | 0.8% / 5.2% |
| PBO | 0.8% / 1.8% | 0.3% / 1.3% |

**On noise, regimes raise false discoveries for the rules without a
calibrated test.** The naive rule goes from 4% to 10% of worlds, walk-forward
from 0.5% to 5.5%, CPCV from 1% to 4.5%. A fixed Sharpe bar means less when
returns are more erratic. The filters built on a test with a stated level
(Benjamini-Hochberg, Benjamini-Yekutieli, Romano-Wolf, the holdout) rise at
most to 2% and stay below their 5%.

**With a signal, regimes cost power and leave FDR about where it was.** Every
filter finds roughly half as many real strategies. Part of that is simply a
weaker signal relative to the noise: the planted rule's own Sharpe falls from
about 1.0 to between 0.64 and 0.92. The ranking of the filters does not change.

## Alpha decay

Obvious signal, FDR at each filter's conventional cut. "After" is the truth
that matters: Sharpe of at least 0.2 in the 5 years after the backtest.

| Filter | Plain | Decay | Decay, any positive Sharpe counts | Accepted per world |
|---|---|---|---|---|
| Naive Sharpe > 1 | 0.1% | 68.5% [65.1, 71.7] | 26.8% | 5.4 |
| Holdout | 0.0% | 18.0% [14.9, 21.0] | 3.4% | 0.8 |
| Walk-forward | 0.0% | 48.6% [44.7, 52.2] | 15.4% | 2.8 |
| Deflated Sharpe, paper | 0.0% | 0.0% | 0.0% | 0.0 |
| Deflated Sharpe, variant | 0.0% | 37.1% [33.5, 40.6] | 13.5% | 1.7 |
| CPCV | 0.1% | 50.9% [46.8, 54.3] | 19.0% | 2.8 |
| Benjamini-Hochberg | 0.3% | 51.0% [47.2, 54.4] | 19.7% | 6.3 |
| Benjamini-Yekutieli | 0.1% | 35.4% [31.9, 38.8] | 12.2% | 3.0 |
| Romano-Wolf | 0.0% | 54.5% [50.7, 57.9] | 22.3% | 3.2 |
| PBO | 0.0% | 45.3% [41.3, 49.3] | 15.0% | 0.5 |

**This is the largest effect in the project so far.** Every filter that works
on the research sample lets through mostly worthless strategies: between a
third and two thirds of what they accept earns less than Sharpe 0.2 afterwards.
Multiple-testing corrections do not help, and the reason is not a flaw in
them. They answer "was there an edge in this sample?", and the answer is
correctly yes. The question that matters is "will there be one after?", and no
test on past data can see that.

**The holdout is the only filter that helps much**, at 18% against 35% to 69%,
because its lockbox is the most recent 3 years, where the signal has already
faded. It is protection through recency, not through statistics.

**The size of the effect depends on the truth threshold.** Many strategies keep
a small positive Sharpe after decay. If any positive Sharpe counts as real, the
FDRs fall to between 12% and 27% for the research-sample filters and 3.4% for
the holdout: still far above the plain worlds, but much lower. The six
strategies that count as true at 0.2 in the momentum decay world have Sharpe
ratios of 0.21 to 0.26, and 7 of 76 labels there lie within 2 standard errors
of the threshold, so that world's power figures are fragile and are left out
of this table (they are in `exp1_grid_points.csv`).

![Plain worlds against regime switching and alpha decay](figures/exp1_grid_regimes_decay.png)

## Caveats

- The regime worlds keep the same planted drift as the plain ones, so they
  differ in two ways at once: non-stationary volatility and a weaker signal
  relative to noise. The no-signal comparison isolates the first; the
  medium-signal comparison mixes both.
- One decay speed (half-life 5 years) on one base strength (obvious). A slower
  decay would give a smaller effect.
- With decay, two of the three world types have no true strategy at all, so
  their FDR is the share of worlds in which anything was accepted.

## Files

- `exp1_grid_points.csv`, `exp1_grid_curves.csv`, `truth.csv`: now 20 world
  types and 10 filters. Pooled groups for the new worlds are named
  `all_medium_regimes` and `all_obvious_decay`.
- `figures/exp1_grid_regimes_decay.png`: the comparison above.

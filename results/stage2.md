# Stage 2 results: deflated Sharpe, CPCV, Benjamini-Hochberg, Romano-Wolf

Reproduce with `uv run python -m overfitlab all` (about 6 minutes on 8 cores).
Same worlds, seeds and truth as Stage 1; the Stage 1 numbers are unchanged to
the last digit. The four new filters see only the research sample (first 70%).

## Sources checked

| Method | Checked against | Result |
|---|---|---|
| Deflated Sharpe | Bailey & Lopez de Prado (2014), Eq. (1), (2) and the worked example, read from the paper | Matches |
| Benjamini-Hochberg | Benjamini & Hochberg (1995), procedure (1), Theorem 1 and the 15 p-value example, read from the paper | Matches |
| Romano-Wolf | Clarke, Romano & Wolf (2019, IZA DP 12845), Eq. (4)-(7) and the adjusted p-value algorithm; Romano & Wolf (2005) for the block bootstrap | Matches, with one simplification (below) |
| CPCV | **Not checked against the book** (Lopez de Prado 2018, ch. 12, not freely available). Purging on both sides and embargo after the test set follow the skfolio library's documentation, which cites the book. Split and path counts are derived and tested by enumeration. | Secondary source only |

## Where I simplified or chose

- **Deflated Sharpe: N = 76**, the number of strategies tried. The paper asks
  for the number of independent trials; counting all correlated ones raises the
  bar, so it errs on the cautious side.
- **Romano-Wolf standard errors.** For time series the 2005 paper uses a
  kernel (autocorrelation-robust) standard error and a block-based one inside
  the bootstrap. I use the plain standard error in both places. Reason:
  without an edge, daily strategy returns are not autocorrelated, so the plain
  formula is still right for the mean; the circular block bootstrap (blocks of
  20 days, 1,000 resamples) carries the volatility clustering. The null test
  below checks that this holds up.
- **CPCV gap.** The plan said to purge training days whose lookback overlaps a
  test group. With lookbacks up to 250 days that would delete most of the
  training data, so I drop 20 days on each side of every test group instead
  (about one half-life of a volatility shock at persistence 0.97).

## The CPCV accept/reject rule

10 groups, 3 test groups per split: 120 splits, 36 out-of-sample paths.

1. In each split, a strategy trades the test groups only if its Sharpe on the
   purged training days is at least 1.0; otherwise it is flat. A fixed-rule
   strategy has nothing else to fit, and without a decision made on training
   data all 36 paths would be the same series.
2. **Accept if the 5% quantile of the 36 path Sharpe ratios is above zero**,
   taken as the 2nd lowest, so at most one path may lose.

Why a quantile: the paths recombine the same days, so they are not independent
and a t-test across them would be far too confident. Why not one path: a single
holdout or walk-forward path can pass on one lucky period. The 5% is not a
calibrated error rate, so it is measured: on the no-signal worlds the rule
accepted 2 strategies in 200 worlds (76 strategies each), i.e. 1.0% of worlds
[0.0, 2.5] had a false discovery.

## Known-answer tests

| Test | Expected | Measured |
|---|---|---|
| Deflated Sharpe, paper example | SR_0 = 0.1132, DSR = 0.9004 | 0.1132, 0.9004 |
| Same, 46 trials; normal returns with 88 trials | 0.9505; 0.9505 | 0.9505; 0.9505 |
| Deflated Sharpe, best of 50 independent null strategies accepted | under 5% | 0.3% (3 of 1,000) |
| BH, paper example at q = 0.05 | 4 rejections (Bonferroni: 3) | 4 (3) |
| BH against `scipy.stats.false_discovery_control` | identical | identical |
| BH, any rejection when nothing is true, independent tests | 5% | 5.03% (4,000 runs) |
| Romano-Wolf, any false acceptance among 20 correlated null strategies with volatility clustering | 5% | 4.1% (41 of 1,000 worlds) |
| Romano-Wolf with one strategy against the t-test | same p-value | within 0.03 |
| CPCV counts for (6, 2) and (10, 3) | 15 splits / 5 paths, 120 / 36 | same |

## Filters at their conventional settings

FDR / power, pooled over the three signal types (600 worlds per column, 200
for no signal). A strategy is true if its long-run net Sharpe is at least 0.2.
Intervals are in `exp1_grid_points.csv`.

| Filter | No signal | Barely (0.3) | Weak (0.6) | Medium (1.0) | Obvious (2.0) |
|---|---|---|---|---|---|
| Naive Sharpe > 1 | 4.0% | 7.8% / 2.6% | 2.2% / 6.2% | 1.6% / 16.4% | 0.1% / 49.1% |
| Holdout | 0.0% | 0.4% / 0.3% | 0.5% / 1.3% | 0.3% / 6.6% | 0.0% / 35.8% |
| Walk-forward | 0.5% | 1.8% / 1.0% | 1.1% / 3.3% | 0.7% / 11.8% | 0.0% / 45.6% |
| Deflated Sharpe > 0.95 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| CPCV | 1.0% | 3.1% / 1.1% | 1.3% / 3.0% | 1.1% / 8.1% | 0.1% / 38.2% |
| Benjamini-Hochberg 5% | 0.0% | 2.2% / 2.0% | 1.1% / 6.0% | 1.1% / 18.5% | 0.3% / 56.1% |
| Romano-Wolf 5% | 0.5% | 3.0% / 1.2% | 1.0% / 3.0% | 0.8% / 9.5% | 0.0% / 41.5% |

![FDR against power](figures/exp1_grid_fdr_power.png)

## What the numbers say

**Benjamini-Hochberg gives the best trade-off on this grid.** With a medium
signal it finds more real strategies than the naive rule (18.5% against 16.4%)
at a lower FDR (1.1% against 1.6%), and on pure noise it accepted nothing in
200 worlds. The reason is that its bar moves: when many strategies look
significant it relaxes, when few do it tightens. A fixed Sharpe cut cannot do
that. Its FDR stays below the 5% it promises in every column.

**Romano-Wolf is stricter and pays for it.** It guards against even one false
acceptance, so it finds about half of what BH finds with a weak or medium
signal (3.0% and 9.5% against 6.0% and 18.5%).

**CPCV lands close to walk-forward and Romano-Wolf**, without touching the
lockbox. Lower FDR than the naive rule from a medium signal up, at roughly half
the power.

**The holdout still has the lowest FDR among the filters that accept anything,
and the lowest power.**

**Nothing rescues a barely-detectable signal.** No filter finds more than 2.6%
of the real strategies there.

**The truth threshold and the tested hypothesis are different things.** BH and
Romano-Wolf test "mean return above zero", while a strategy counts as true
here only from Sharpe 0.2. A strategy with a real Sharpe of 0.1 is a correct
rejection for them and a false discovery in this table. With the threshold at
0, their FDR is 0.1% or less everywhere except Romano-Wolf's 0.4% in the barely
worlds.

## The deflated Sharpe ratio accepts nothing

Not one strategy passed in 2,600 worlds, at any cut down to 0.5, even where
the planted Sharpe is 2. This is the method doing what its formula says, not a
bug: the paper's own example is reproduced to four decimals.

The cause is the benchmark SR_0 = sqrt(V) x 2.43 for 76 trials, where V is the
variance of the trials' Sharpe ratios. The formula assumes that all trials have
the same true Sharpe, so that V is pure estimation noise. On this grid the
trials differ for real: where a signal exists, the strategies that follow it
have a high Sharpe and their mirror images a very negative one. V measures that
real spread, and the benchmark rises with the signal. From `truth.csv`:

| World type | SD of the 76 true Sharpes | 2.43 x SD | Best true Sharpe |
|---|---|---|---|
| No signal | 0.25 | 0.60 | 0.00 |
| Momentum, barely | 0.42 | 1.03 | 0.40 |
| Momentum, medium | 0.86 | 2.08 | 1.10 |
| Momentum, obvious | 1.60 | 3.88 | 2.27 |

The bar set by the real spread alone is already above the best strategy in
every row, before any estimation noise is added. So the stronger the signal,
the further out of reach the benchmark. Whether to keep this literal version,
or to add a variant whose V is the estimation variance under "no edge", is an
open decision.

## Files

- `exp1_grid_points.csv`, `exp1_grid_curves.csv`: now with seven filters.
- `truth.csv`: unchanged from Stage 1.

# Stage 4: the LLM agent against the filters

## Setup

- **Agent:** Claude Haiku 5.5 (`claude-haiku-5-5`), 20 rounds per world, compact
  history (each round a fresh prompt with a table of all earlier rounds), up to
  two repairs per round. Prompts and code: `src/overfitlab/agent.py`.
- **Worlds:** no signal, and the three medium-strength signals (time-series
  momentum, cross-sectional momentum, mean reversion), 30 worlds each, the
  same world seeds as the grid search (worlds 0 to 29). 120 runs, 2,400 rounds.
- **Cost:** $0.8314 for 2,510 calls, all stored in `cache/llm/`. Everything
  below is re-scored from those files with no model call.
- **Truth:** every usable agent strategy was run on the same 10,000 fresh
  simulated years per world type as the grid (100 paths of 100 years). Largest
  standard error of a true Sharpe: 0.015. The truth step took 5,531 s with
  4 workers.
- **Filters:** the ten filters of Stages 1 to 3 at their conventional cut,
  applied to each run's usable trials (on average 19.8 per world) exactly as to
  the grid's 76 strategies.
- **Intervals:** 95% percentile bootstrap over worlds (30 for the agent, 200 for
  the grid).

## Usable rounds and look-ahead

| World type | Usable rounds (share, 95% interval) | Calls per round | Look-ahead rejections |
|---|---|---|---|
| No signal | 98.7% [97.5, 99.7] | 1.06 | 0 |
| Momentum | 99.7% [99.2, 100.0] | 1.04 | 0 |
| Cross-sectional | 98.8% [97.3, 99.8] | 1.05 | 1 |
| Mean reversion | 99.5% [98.7, 100.0] | 1.03 | 1 |
| All | 99.2% [98.7, 99.6] | 1.05 | 2 of 2,510 calls (0.08%) |

20 of 2,400 rounds stayed unusable after two repairs. The look-ahead test
rejected 2 attempts in all, and both were repaired. 110 calls were repairs.
On average a run held 18.3 distinct strategies out of 20 rounds.

## What the agent proposes

- **Share of true strategies.** Of the agent's usable strategies, 85% are true
  (true Sharpe at least 0.2) on momentum worlds, 84% on cross-sectional and 68%
  on mean reversion. For the grid the shares are 42% (32 of 76), 38% and 24%.
  On no-signal worlds no strategy of either generator is true: the agent's best
  true Sharpe is -0.01, the grid's -0.003.
- **Before seeing any data.** In round 1, with nothing but a market summary,
  117 of 120 replies describe a cross-sectional or relative idea, 79 a
  momentum or trend idea and 41 a reversal idea (keyword count over the
  hypothesis text; one reply can match several). The planted signals are
  textbook ideas and the model proposes them unprompted, so its high share of
  true strategies reflects prior knowledge, not only what it learned from the
  backtests.
- **Best-in-backtest strategy.** The strategy with the highest research
  Sharpe in each run had a mean research Sharpe of 0.41 on no-signal worlds
  (none above 1) and 1.18 to 1.23 on medium worlds, where it was true in 93%
  to 100% of runs.

## FDR, pFDR and power, agent against grid

Cells: percent, with the 95% interval. `-` means undefined (no acceptance for
pFDR, no true strategy for power). Truth threshold 0.2.

### No signal (agent 30 worlds, grid 200)

Power is undefined because no true strategy exists.

| Filter | FDR agent | FDR grid | pFDR agent | pFDR grid |
|---|---|---|---|---|
| Naive Sharpe | 0.0 [0.0, 0.0] | 4.0 [1.5, 7.0] | - | 100 |
| Holdout | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | - |
| Walk-forward | 0.0 [0.0, 0.0] | 0.5 [0.0, 1.5] | - | 100 |
| Deflated Sharpe (paper) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | - |
| Deflated Sharpe (variant) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | - |
| CPCV | 3.3 [0.0, 10.0] | 1.0 [0.0, 2.5] | 100 | 100 |
| Benjamini-Hochberg | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | - |
| Benjamini-Yekutieli | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | - |
| Romano-Wolf | 3.3 [0.0, 10.0] | 0.5 [0.0, 1.5] | 100 | 100 |
| PBO (best only) | 0.0 [0.0, 0.0] | 1.0 [0.0, 2.5] | - | 100 |

On no-signal worlds every acceptance is false, so pFDR is 100% whenever it is
defined. The agent's 3.3% for CPCV and Romano-Wolf is one world out of 30 each.

### Medium signal, three types pooled (agent 90 worlds, grid 600)

| Filter | FDR agent | FDR grid | pFDR agent | pFDR grid | Power agent | Power grid |
|---|---|---|---|---|---|---|
| Naive Sharpe | 1.1 [0.0, 3.3] | 1.6 [0.9, 2.4] | 1.6 [0.0, 4.8] | 2.2 [1.3, 3.4] | 37.4 [30.4, 44.6] | 16.4 [14.8, 18.0] |
| Holdout | 0.0 [0.0, 0.0] | 0.3 [0.0, 0.7] | 0.0 [0.0, 0.0] | 0.6 [0.0, 1.5] | 17.9 [12.7, 23.6] | 6.6 [5.8, 7.6] |
| Walk-forward | 0.0 [0.0, 0.0] | 0.7 [0.2, 1.2] | 0.0 [0.0, 0.0] | 1.1 [0.4, 2.0] | 29.1 [22.6, 36.1] | 11.8 [10.5, 13.1] |
| Deflated Sharpe (paper) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | 0.2 [0.0, 0.7] | 0.0 [0.0, 0.0] |
| Deflated Sharpe (variant) | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.7] | 0.0 [0.0, 0.0] | 0.7 [0.0, 1.9] | 26.0 [19.6, 33.0] | 4.4 [3.6, 5.2] |
| CPCV | 0.0 [0.0, 0.0] | 1.1 [0.5, 1.9] | 0.0 [0.0, 0.0] | 2.1 [0.9, 3.5] | 22.9 [17.2, 29.2] | 8.1 [7.2, 9.1] |
| Benjamini-Hochberg | 1.4 [0.1, 4.0] | 1.1 [0.6, 1.7] | 2.0 [0.1, 5.5] | 2.1 [1.1, 3.2] | 54.1 [45.3, 62.6] | 18.5 [16.5, 20.7] |
| Benjamini-Yekutieli | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.4] | 0.0 [0.0, 0.0] | 0.6 [0.1, 1.4] | 38.5 [30.4, 46.7] | 8.1 [6.7, 9.6] |
| Romano-Wolf | 1.1 [0.0, 3.3] | 0.8 [0.4, 1.4] | 1.6 [0.0, 4.7] | 1.5 [0.7, 2.5] | 42.9 [35.4, 50.8] | 9.5 [8.4, 10.7] |
| PBO (best only) | 0.0 [0.0, 0.0] | 0.8 [0.2, 1.7] | 0.0 [0.0, 0.0] | 1.8 [0.4, 3.5] | 1.2 [0.5, 2.2] | 1.8 [1.6, 1.9] |

### Medium signal, by type: FDR / power for the agent, with the grid alongside

| Filter | Momentum, agent | Momentum, grid | Cross-sectional, agent | Cross-sectional, grid | Mean reversion, agent | Mean reversion, grid |
|---|---|---|---|---|---|---|
| Naive Sharpe | 0.0 / 33.0 [21.2, 45.0] | 0.1 / 27.8 | 3.3 [0, 10] / 38.5 [25.7, 51.8] | 0.9 / 11.8 | 0.0 / 40.7 [29.0, 51.8] | 3.8 / 9.6 |
| Holdout | 0.0 / 16.6 [8.2, 25.4] | 0.0 / 12.4 | 0.0 / 15.6 [8.4, 23.5] | 0.6 / 3.7 | 0.0 / 21.7 [11.2, 33.3] | 0.2 / 3.8 |
| Walk-forward | 0.0 / 30.1 [18.9, 41.3] | 0.1 / 20.8 | 0.0 / 28.7 [17.6, 41.3] | 0.8 / 7.3 | 0.0 / 28.5 [17.4, 40.5] | 1.1 / 7.2 |
| Deflated Sharpe (paper) | 0.0 / 0.5 [0.0, 1.6] | 0.0 / 0.0 | 0.0 / 0.2 [0.0, 0.5] | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| Deflated Sharpe (variant) | 0.0 / 24.3 [13.8, 35.3] | 0.0 / 8.2 | 0.0 / 26.7 [15.3, 39.2] | 0.2 / 2.7 | 0.0 / 27.1 [15.6, 39.4] | 0.5 / 2.2 |
| CPCV | 0.0 / 20.5 [11.8, 29.8] | 0.0 / 13.7 | 0.0 / 21.5 [10.9, 33.5] | 0.9 / 5.1 | 0.0 / 26.8 [16.6, 37.7] | 2.5 / 5.5 |
| Benjamini-Hochberg | 0.7 [0, 1.7] / 49.2 [34.0, 63.1] | 0.2 / 34.8 | 3.6 [0, 10.5] / 57.5 [42.5, 71.8] | 0.6 / 13.5 | 0.0 / 55.5 [40.4, 69.5] | 2.4 / 7.2 |
| Benjamini-Yekutieli | 0.0 / 35.5 [22.0, 49.1] | 0.1 / 16.8 | 0.0 / 40.3 [25.6, 55.3] | 0.3 / 5.0 | 0.0 / 39.8 [25.7, 53.0] | 0.2 / 2.6 |
| Romano-Wolf | 0.0 / 36.5 [23.3, 50.0] | 0.1 / 15.5 | 3.3 [0, 10] / 44.9 [31.4, 58.7] | 0.2 / 5.6 | 0.0 / 47.5 [35.4, 59.2] | 2.2 / 7.3 |
| PBO (best only) | 0.0 / 1.5 [0.0, 3.9] | 0.0 / 2.1 | 0.0 / 0.4 [0.0, 0.9] | 0.0 / 1.3 | 0.0 / 1.8 [0.5, 3.3] | 2.5 / 1.8 |

An agent FDR of 0.0 has the interval [0.0, 0.0]; the grid's intervals are in
[exp1_agent_tables.md](exp1_agent_tables.md) and the CSV files.

## How to read the comparison

- **Power has a different denominator for each generator.** It is the share
  of the generator's own true strategies that a filter accepts. The agent's
  true strategies are the ones it proposed (about 16 per medium world, most of
  them strong), the grid's are the true ones among its 76 (18 to 32 per world
  type, many of them weak). So "power agent" and "power grid" do not measure
  the same thing, and a higher agent power does not mean the filters work
  better on agent output in general.
- **Fewer strategies per world.** The agent tries about 20 strategies per
  world, the grid 76. The corrections (Benjamini-Hochberg, Romano-Wolf, the
  deflated Sharpe) charge less for 20 trials than for 76.
- **Small number of worlds.** 30 worlds per type give wide intervals: an FDR
  of 3.3% is one world, and the upper ends near 10% come from that.
- **Sensitivity to the truth threshold** (pooled medium worlds, agent): at 0
  and 0.2 all FDRs are at most 1.4%; at 0.5 they rise to as much as 11%
  (Benjamini-Hochberg 11.0%, Romano-Wolf 9.5%, naive 8.8%, holdout 4.9%; the
  published deflated Sharpe and PBO stay below 1%), because many of the
  agent's strategies have true Sharpes between 0.2 and 0.5. Full table:
  `exp1_agent_points.csv`.

## Files

- `exp1_agent_runs.csv`: per run: usable rounds, calls, look-ahead
  rejections, distinct strategies, model seconds, cost.
- `exp1_agent_truth.csv`: every usable agent strategy with its research
  Sharpe, true Sharpe and standard error.
- `exp1_agent_points.csv`, `exp1_agent_tables.md`: FDR, pFDR, power with
  intervals, at truth thresholds 0, 0.2 and 0.5.
- `figures/exp1_agent_vs_grid.png`.

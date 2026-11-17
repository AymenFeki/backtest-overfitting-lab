# Zero-cost checks

Written by `scripts/zero_cost.py`.

## Stage 4: agent and grid on no-signal worlds (research sample)

| | agent | grid |
|---|---|---|
| Strategies, worlds | 592, 30 | 15200, 200 |
| Sharpe before costs | 0.13 [0.06, 0.20] | -0.00 [-0.00, 0.00] |
| Sharpe after costs | -0.09 [-0.16, -0.02] | -0.22 [-0.22, -0.22] |
| Cost drag | 0.22 [0.19, 0.25] | 0.22 [0.22, 0.22] |
| Turnover per year, mean | 35.73 [30.53, 40.89] | 37.84 [37.73, 37.94] |
| Turnover per year, median strategy | 19.3 | 19.8 |
| Best strategy per world, before costs | 0.54 | 0.78 |
| Best strategy per world, after costs | 0.41 | 0.64 |
| Worlds with a strategy above 1, before costs | 3.3% | 11.0% |
| Worlds with a strategy above 1, after costs | 0.0% | 4.0% |

Share of worlds where the filter accepts something (all false), %:

| Filter | agent, after costs | agent, zero cost | grid, after costs | grid, zero cost |
|---|---|---|---|---|
| naive | 0.0 | 3.3 | 4.0 | 11.0 |
| holdout | 0.0 | 0.0 | 0.0 | 0.5 |
| walk_forward | 0.0 | 0.0 | 0.5 | 3.0 |
| dsr | 0.0 | 0.0 | 0.0 | 0.0 |
| dsr_null | 0.0 | 0.0 | 0.0 | 0.5 |
| cpcv | 3.3 | 3.3 | 1.0 | 5.5 |
| bh | 0.0 | 3.3 | 0.0 | 2.5 |
| by | 0.0 | 0.0 | 0.0 | 0.0 |
| romano_wolf | 3.3 | 6.7 | 0.5 | 3.0 |
| pbo | 0.0 | 0.0 | 1.0 | 0.0 |

## Experiment 2: the lockbox opened once (hill-climber, no-signal worlds)

- Claims after 40 rounds: 3 of 200 (1.5%).
- Probability of 3 or fewer at a true rate of 5%: 0.009 (exact binomial).
- Mean true Sharpe of the candidates after costs: -0.09.
- Expected claim rate for these candidates, lockbox standard error 0.58: 3.7%; probability of 3 or fewer: 0.057.
- Candidates' lockbox Sharpe after costs: mean -0.12, spread across worlds 0.58; at zero cost: mean -0.03.
- Claims at zero cost: 5 of 200.

## Experiment 3: told and not told, no-signal worlds (research sample)

| | not_told | told |
|---|---|---|
| Strategies, worlds | 299, 15 | 300, 15 |
| Sharpe before costs | 0.11 [0.04, 0.18] | 0.07 [-0.02, 0.16] |
| Sharpe after costs | -0.25 [-0.36, -0.13] | -0.30 [-0.44, -0.15] |
| Cost drag | 0.36 [0.30, 0.42] | 0.37 [0.31, 0.44] |
| Turnover per year, mean | 59.84 [49.28, 70.19] | 59.90 [50.40, 69.80] |
| Turnover per year, median strategy | 43.2 | 39.3 |
| Best strategy per world, before costs | 0.58 | 0.58 |
| Best strategy per world, after costs | 0.36 | 0.38 |
| Worlds with a strategy above 1, before costs | 0.0% | 0.0% |
| Worlds with a strategy above 1, after costs | 0.0% | 0.0% |

Share of worlds where the filter accepts something (all false), %:

| Filter | not_told, after costs | not_told, zero cost | told, after costs | told, zero cost |
|---|---|---|---|---|
| naive | 0.0 | 0.0 | 0.0 | 0.0 |
| holdout | 0.0 | 0.0 | 0.0 | 0.0 |
| walk_forward | 0.0 | 0.0 | 0.0 | 0.0 |
| dsr | 0.0 | 0.0 | 0.0 | 0.0 |
| dsr_null | 0.0 | 0.0 | 0.0 | 0.0 |
| cpcv | 0.0 | 0.0 | 0.0 | 0.0 |
| bh | 0.0 | 0.0 | 0.0 | 0.0 |
| by | 0.0 | 0.0 | 0.0 | 0.0 |
| romano_wolf | 0.0 | 0.0 | 0.0 | 0.0 |
| pbo | 0.0 | 0.0 | 0.0 | 0.0 |

Strategies passing any of the six checks shown to the agents (arm, returns: count): not_told gross: 0, not_told net: 0, told gross: 0, told net: 0.

# Stage 5: holdout reuse, Thresholdout and the Goodhart test

Results only; the interpretation follows after review.

## Experiment 2: a lockbox consulted after every round

### Setup

- **Analyst:** proposes one strategy per round for 40 rounds and after each
  round asks how it did in the lockbox (the last 3 years, 756 days). The
  candidate is the proposal with the highest answer so far.
- **Claim:** declared rule: the answer the analyst was told about the
  candidate is above 0.95, the bar of a one-sided 5% test on a 3-year lockbox
  (1.645 x standard error 0.58). Second rule, added after the first results:
  the candidate's *actual* lockbox Sharpe is above 0.95, whatever the analyst
  was told. Under "once" the two are the same.
- **Ways of answering:**
  - `once`: no lockbox answers during the search (the analyst sees the
    research Sharpe); the lockbox is read one time, for the final candidate.
    The correct use and the baseline.
  - `reuse`: the lockbox Sharpe after every round.
  - `thresholdout` (Dwork et al. 2015): answers with the research Sharpe
    unless it differs from the lockbox Sharpe by more than a noisy threshold;
    then answers with the lockbox Sharpe plus Laplace noise and spends one unit
    of budget. Declared settings: threshold 1 standard error (0.58), noise
    rate a twenty-fifth of that, budget 10. With the budget spent it gives no
    answer.
- **Analysts:** a hill-climber without a model (walks over the 76 grid
  strategies, tries a relative of the best so far, with a 20% chance of a jump
  to any untried strategy), on 200 no-signal and 200 medium-momentum worlds;
  and the Haiku agent on 15 no-signal worlds per arm.
- **Truth:** on no-signal worlds every claim is false. On momentum worlds a
  claim is true when the candidate's true Sharpe is at least 0.2 (from
  `truth.csv`).
- **Seeds:** search stream `hill_climb`, Thresholdout noise stream
  `thresholdout`, both derived from the master seed 20261116. Every
  mechanism sees the same search random numbers in a world, so differences
  come from the answers.
- **Intervals:** 95% percentile bootstrap over worlds.

### Variants added after the first results

These were not in the declared design; they were added on request after the
hill-climber preview and are reported as sensitivity runs. The agent arms kept
their declared settings.

- **Threshold sweep** (declared): 0.5, 2 and 4 standard errors.
- **Budget sweep** (declared): 5 and 40.
- **`thresholdout_noise2`:** noise rate 2 standard errors (1.15 Sharpe) at the
  declared threshold.
- **`thresholdout_paper`:** the ratios of the paper's experiment, threshold 4
  and noise rate 1 standard error. In the paper the answers were correlations
  over 10,000 samples (standard error about 0.01), the threshold 0.04 and the
  noise 0.01 (read from a summary of the paper, not checked against the
  original).
- **Long lockbox (`*_long`):** 12 years (3,024 days) instead of 3, from the
  same worlds simulated for longer (the first 10 years are identical, checked
  by a test). Standard error 0.29, so the bar for a claim is 0.47 and the
  Thresholdout threshold scales with it.

### Hill-climber, no-signal worlds (200)

Share of worlds with a (false) claim, percent with 95% interval. "Told" is the
declared rule, "actual" the candidate's real lockbox Sharpe. Reported and
actual are mean Sharpe ratios of the candidate.

| Mechanism | Round 10, told | Round 40, told | Round 40, actual | Reported | Actual lockbox | True Sharpe |
|---|---|---|---|---|---|---|
| once | 1.5 [0.0, 3.5] | 1.5 [0.0, 3.5] | 1.5 [0.0, 3.5] | -0.12 | -0.12 | -0.09 |
| reuse | 18.5 [13.5, 24.0] | 41.5 [35.0, 48.5] | 41.5 [35.0, 48.5] | 0.91 | 0.91 | -0.13 |
| thresholdout (declared) | 20.0 [14.5, 25.5] | 32.0 [25.5, 39.0] | 31.0 [24.5, 37.5] | 0.82 | 0.77 | -0.14 |
| threshold 0.5 SE | 16.5 [11.5, 21.5] | 26.0 [20.0, 32.5] | 26.5 [20.5, 33.0] | 0.76 | 0.75 | -0.14 |
| threshold 2 SE | 13.5 [9.0, 18.5] | 33.0 [26.5, 39.5] | 28.5 [22.0, 35.0] | 0.81 | 0.52 | -0.14 |
| threshold 4 SE | 3.5 [1.0, 6.0] | 8.5 [5.0, 12.5] | 5.5 [2.5, 9.0] | 0.62 | -0.00 | -0.10 |
| budget 5 | 14.0 [9.5, 19.0] | 17.5 [12.5, 23.0] | 17.5 [12.5, 23.0] | 0.66 | 0.60 | -0.15 |
| budget 40 | 18.0 [12.5, 24.0] | 43.0 [36.0, 50.0] | 39.0 [32.5, 45.5] | 0.90 | 0.85 | -0.12 |
| noise 2 SE | 62.0 [55.5, 68.5] | 90.0 [85.5, 94.0] | 7.0 [3.5, 10.5] | 2.45 | 0.09 | -0.17 |
| paper ratios | 21.5 [15.5, 27.0] | 54.5 [47.5, 61.0] | 12.5 [8.0, 17.5] | 1.15 | 0.27 | -0.13 |
| once, 12-year lockbox | 2.0 [0.5, 4.0] | 2.0 [0.5, 4.0] | 2.0 [0.5, 4.0] | -0.10 | -0.10 | -0.09 |
| reuse, 12-year lockbox | 15.5 [10.5, 20.5] | 33.5 [27.0, 40.0] | 33.5 [27.0, 40.0] | 0.41 | 0.41 | -0.07 |
| thresholdout, 12-year lockbox | 13.5 [9.0, 18.5] | 24.0 [18.0, 30.0] | 18.0 [13.0, 23.5] | 0.36 | 0.30 | -0.08 |

In round 1 (one strategy, one look) once and reuse are at 3.5% [1.5, 6.0] and
the 12-year mechanisms at 2.5% [0.5, 5.0] on the actual-lockbox rule. On the
told rule the Thresholdout variants range from 0.5% (threshold 4) to 11.5%
(noise 2 SE), because the noise moves the answer.

**Why "once" is at 1.5% and not 5%.** 3 claims in 200 worlds; if the true
rate were 5%, 3 or fewer would happen with probability 0.009 (exact
binomial). Most of the gap is costs, the rest is within chance:

- The bar of 0.95 is the 5% point for a strategy whose true Sharpe is 0. The
  candidates here lose money after costs: their mean true Sharpe is -0.09.
  With a lockbox standard error of 0.58, the expected claim rate for these
  candidates is 3.7%, and 3 or fewer claims in 200 then has probability
  0.057.
- With the same candidates at zero cost (same positions, no cost), 5 of 200
  would have been claimed (2.5%). The candidates' mean lockbox Sharpe is
  -0.12 after costs and -0.03 before. Their measured spread across worlds,
  0.58, matches the standard error used for the bar.

### Hill-climber, medium-momentum worlds (200), round 40

| Mechanism | True claims, told | True claims, actual | False claims, told | False claims, actual | Reported | Actual lockbox | True Sharpe |
|---|---|---|---|---|---|---|---|
| once | 48.0 [40.5, 55.0] | 48.0 [40.5, 55.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.85 | 0.85 | 0.88 |
| reuse | 87.0 [82.5, 91.5] | 87.0 [82.5, 91.5] | 3.0 [1.0, 5.5] | 3.0 [1.0, 5.5] | 1.52 | 1.52 | 0.78 |
| thresholdout (declared) | 85.5 [80.5, 90.0] | 72.0 [65.5, 78.0] | 1.0 [0.0, 2.5] | 1.0 [0.0, 2.5] | 1.43 | 1.30 | 0.79 |
| threshold 0.5 SE | 77.0 [71.0, 82.5] | 76.5 [70.5, 82.5] | 2.0 [0.5, 4.0] | 2.0 [0.5, 4.0] | 1.35 | 1.34 | 0.72 |
| threshold 2 SE | 90.5 [86.0, 94.5] | 60.5 [53.5, 67.5] | 2.5 [0.5, 5.0] | 2.5 [0.5, 5.0] | 1.49 | 1.18 | 0.81 |
| threshold 4 SE | 87.0 [82.0, 91.5] | 50.5 [43.5, 57.5] | 1.5 [0.0, 3.5] | 1.5 [0.0, 3.5] | 1.37 | 0.93 | 0.86 |
| budget 5 | 75.5 [69.5, 81.0] | 66.0 [59.0, 72.0] | 1.5 [0.0, 3.5] | 1.5 [0.0, 3.5] | 1.28 | 1.18 | 0.71 |
| budget 40 | 91.0 [87.0, 95.0] | 75.5 [69.5, 81.5] | 2.5 [0.5, 5.0] | 2.5 [0.5, 5.0] | 1.54 | 1.41 | 0.79 |
| noise 2 SE | 75.5 [69.5, 81.5] | 36.5 [29.5, 43.5] | 23.0 [17.0, 29.0] | 0.5 [0.0, 1.5] | 2.97 | 0.61 | 0.43 |
| paper ratios | 89.0 [84.5, 93.0] | 49.0 [42.0, 56.0] | 6.5 [3.5, 10.0] | 0.5 [0.0, 1.5] | 1.82 | 0.94 | 0.72 |
| once, 12-year lockbox | 86.5 [81.5, 91.0] | 86.5 [81.5, 91.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.89 | 0.89 | 0.88 |
| reuse, 12-year lockbox | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.20 | 1.20 | 0.95 |
| thresholdout, 12-year lockbox | 98.0 [96.0, 99.5] | 98.0 [95.5, 99.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.14 | 1.07 | 0.92 |

The best grid strategy on these worlds has a true Sharpe of 1.10.

### Where the answers came from

Per run (40 rounds), mean over 200 worlds. "From the lockbox": the answer
differs from the research Sharpe, i.e. Thresholdout flagged the gap and
answered with the noisy lockbox Sharpe. "Budget spent": share of runs that
used all their lockbox answers, and the round in which the last one was given
(median, min to max).

| Mechanism | World | Answered | From the lockbox | Share from the lockbox | Budget spent | Round (median, range) |
|---|---|---|---|---|---|---|
| thresholdout (declared) | no signal | 25.1 | 9.5 | 44.9% [42.2, 47.7] | 83% | 21 (10 to 40) |
| thresholdout (declared) | momentum | 25.4 | 9.5 | 45.7% [42.7, 48.9] | 86% | 22 (10 to 40) |
| threshold 0.5 SE | no signal | 15.3 | 10.0 | 71.1% [68.7, 73.6] | 100% | 14 (10 to 40) |
| threshold 2 SE | no signal | 38.2 | 4.4 | 13.0% [11.1, 14.9] | 16% | 29 (13 to 40) |
| threshold 4 SE | no signal | 40.0 | 0.5 | 1.1% [0.8, 1.5] | 0% | - |
| budget 5 | no signal | 14.9 | 4.9 | 47.8% [44.2, 51.3] | 94% | 11 (5 to 37) |
| budget 40 | no signal | 40.0 | 17.1 | 42.6% [40.0, 45.2] | 0% | - |
| noise 2 SE | no signal | 25.3 | 9.7 | 42.8% [40.8, 44.9] | 89% | 23 (12 to 40) |
| paper ratios | no signal | 37.5 | 7.2 | 20.4% [19.1, 21.9] | 32% | 33 (19 to 40) |
| thresholdout, 12-year lockbox | no signal | 19.4 | 10.0 | 58.4% [55.8, 61.2] | 99% | 18 (10 to 39) |

`once` answers from the lockbox 0% of the time and `reuse` 100%. Momentum
worlds give the same picture for every variant; all rows are in
`exp2_climber_sources.csv`.

### The agent, no-signal worlds (15 per arm, 40 rounds)

| Arm | Round 10 | Round 20 | Round 40, told | Round 40, actual | Reported | Actual lockbox |
|---|---|---|---|---|---|---|
| reuse | 6.7 [0.0, 20.0] | 13.3 [0.0, 33.3] | 33.3 [13.3, 60.0] | 33.3 [13.3, 60.0] | 0.82 | 0.82 |
| thresholdout | 6.7 [0.0, 20.0] | 6.7 [0.0, 20.0] | 13.3 [0.0, 33.3] | 13.3 [0.0, 33.3] | 0.72 | 0.66 |

Percent of worlds with a false claim; 6.7% is one world of 15.

**Paired comparison.** Both arms ran on the same 15 worlds, so they are
compared world by world with an exact McNemar test (`exp2_agent_paired.csv`).
After 40 rounds, reuse made a false claim in 5 worlds and Thresholdout in 2.
The discordant worlds are 3 (a claim under reuse only) and 0 (under
Thresholdout only). If the arms did not differ, each discordant world would
go either way with probability 1/2, so the two-sided p-value is
2 x 0.5^3 = 0.25. The same holds on the actual-lockbox rule. **The difference
between the arms is not significant**; with 15 worlds, at least 6 discordant
worlds, all in the same direction, would be needed for p below 0.05.

- **Usable rounds:** 587 of 600 (reuse), 598 of 600 (Thresholdout).
- **Thresholdout answers:** 26.4 of 39.9 usable rounds per run were answered;
  8.8 came from the lockbox, a share of 42.8% [31.8, 54.3]. The budget ran out
  in 9 of 15 runs, in rounds 13 to 23 (median 16); after that the agent was
  told "not available".

## Experiment 3: told against not told

### Setup

- No-signal worlds 0 to 14, 20 rounds, Haiku, compact history.
- Both arms get the same goal text and, for every past strategy, a "checks
  passed" column with letters A to F for six checks (holdout, walk-forward,
  deflated Sharpe, CPCV, Benjamini-Hochberg, Romano-Wolf), evaluated over all
  strategies tried so far. Only the "told" arm gets a description of each
  check. Reference: the Stage 4 agent on the same worlds, which saw only its
  Sharpe ratios ("Sharpe only").
- Final scoring: all ten filters on each run's usable strategies.

### Results

- **What the agents saw:** in all 30 runs (600 prompts) no strategy ever
  passed any of the six checks. The column showed "none" in every row of
  every prompt, in both arms.
- **Accepted strategies (all false):** zero for every filter in both arms.
  The Sharpe-only reference had one world of 15 (6.7% [0, 20]) passing CPCV
  and one passing Romano-Wolf. The paired differences told minus not told are
  zero for every filter.
- **How the strategies differed**, mean per run with 95% interval over the 15
  worlds; differences are paired by world:

| Measure | Sharpe only | Not told | Told | Told minus not told | Not told minus Sharpe only |
|---|---|---|---|---|---|
| Usable rounds | 19.8 | 19.9 | 20.0 | | |
| Distinct strategies | 18.7 [17.4, 19.5] | 18.9 [18.2, 19.3] | 19.5 [19.0, 19.9] | 0.67 [0.00, 1.33] | 0.20 [-0.73, 1.47] |
| Mean absolute correlation between strategies | 0.457 [0.398, 0.514] | 0.378 [0.312, 0.444] | 0.319 [0.266, 0.381] | -0.059 [-0.133, 0.013] | -0.079 [-0.141, -0.020] |
| Spread of research Sharpe ratios | 0.352 [0.305, 0.396] | 0.433 [0.388, 0.481] | 0.443 [0.377, 0.514] | 0.011 [-0.060, 0.077] | 0.081 [0.025, 0.141] |
| Best research Sharpe | 0.405 [0.300, 0.520] | 0.356 [0.235, 0.477] | 0.375 [0.231, 0.522] | 0.020 [-0.125, 0.172] | -0.050 [-0.170, 0.086] |

- **Wording of hypotheses** (keyword count): words like "robust",
  "out-of-sample", "stable" or "regime" appear in 16.3% of the told arm's
  hypotheses and 10.0% of the not-told arm's; words about multiple testing or
  overfitting in 2.7% of both.

### A documented null, and why

Experiment 3 cannot tell whether knowing the checks helps an agent game
them, and it was not re-run. The reason is that the feedback carried no
information: no strategy passed a check, so every prompt in both arms showed
"none" in every row, and the two arms differed only in the description text.
An agent cannot steer towards a check it never sees passing.

Costs are not the cause. Run again on the same strategies at zero cost (same
positions, no transaction costs), no strategy in either arm passes any of the
six checks or any of the ten filters either. These agents traded more than
the Stage 4 agent (one-way turnover about 60 a year against 36), and their
research Sharpe is -0.25 (not told) and -0.30 (told) after costs and 0.11 and
0.07 before. Twenty noise strategies with Sharpe ratios that small do not
reach the checks' bars with or without costs. The Stage 4 agent on the same
worlds got one strategy past CPCV and one past Romano-Wolf in 15 worlds, so
passes on noise are rare to begin with.

A test of gaming needs checks that can be passed by chance often enough to
learn from, for example more rounds, lower bars in the feedback than in the
final scoring, or worlds with a weak signal. None of this was run.

## Cost

Stage 5 spent $0.7939 (ledger: $0.8315 before, $1.6254 after), for 1,900 new
calls. The per-run cost columns in the CSV files add up to $0.7975, because
the two Experiment 2 arms share their identical round-1 request, which was
made once and is counted in both.

## Files

- `exp2_climber.csv`: hill-climber, per world type, mechanism and round,
  with both claim rules, at truth thresholds 0, 0.2 and 0.5.
- `exp2_climber_sources.csv`, `exp2_agent_sources.csv`: where the answers came
  from, and when the budget ran out.
- `exp2_agent.csv`, `exp2_agent_rounds.csv`: the agent, per round, summary and
  raw. `exp2_agent_paired.csv`: the McNemar test.

The zero-cost checks ("once" candidates and Experiment 3 strategies) come from
a script outside the package, not from `python -m overfitlab`.
- `exp3_points.csv`: accepted strategies per filter and arm, and paired
  differences. `exp3_runs.csv`: per run descriptors.
- `figures/exp2_holdout_reuse.png`, `figures/exp3_goodhart.png`.

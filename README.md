# Backtest Overfitting Lab

**Work in progress: Stage 3 of 6 is done and Stage 4 (the LLM agent) has
started.** This README is a placeholder; the full write-up comes with the last
stage.

**Question:** when many trading strategies are backtested on the same data,
which statistical filters stop the false discoveries, and how many real
strategies do they throw away in the process?

**Approach:** simulated markets where the truth is known (no signal, or a
planted signal of known strength), a strategy generator that tries many ideas,
and filters that decide which ideas to believe. False discovery rate and power
are measured over hundreds of simulated worlds.

## How to read the numbers

- **True strategy.** A strategy counts as real in a world type if its long-run
  Sharpe ratio after costs is at least 0.2 a year, measured on 10,000 fresh
  simulated years (standard error about 0.01). Thresholds 0 and 0.5 are
  reported as a sensitivity check.
- **FDR.** In each world, V is the number of accepted strategies that are not
  real and R the number accepted. The false discovery proportion is
  V / max(R, 1), so a world where nothing is accepted counts as zero. FDR is
  its average over worlds.
- **pFDR** (positive FDR, Storey 2003) is the same average over only the worlds
  where something was accepted: "when this filter says yes, how often is it
  wrong?". FDR = pFDR x share of worlds with an acceptance, so a filter that
  rarely accepts anything can have a low FDR and a high pFDR. Both are in the
  tables.
- **Power** is the share of the real strategies that a filter accepts.
- **The filters test a different hypothesis than the truth label.** Every
  filter with a test asks "is the mean return above zero?". The truth label
  asks "is the Sharpe at least 0.2?". A strategy with a true Sharpe of 0.1 is
  a correct rejection of "mean = 0" and still counts as a false discovery
  here. This is deliberate (an edge that small is not worth trading), but it
  means a filter can exceed its nominal 5% without being broken.
- **The noise worlds are an easier null than "mean = 0".** With no signal a
  strategy earns nothing before costs and pays costs on every trade, so its
  true Sharpe is negative: -0.22 on average over the 76 grid strategies, from
  -0.95 to 0.00. A test of "mean = 0" on them is conservative, so the
  false-positive rates in the no-signal rows are lower than the same filters
  would show on strategies with a true mean of exactly zero.

## Results so far

Brute-force grid search of 76 strategies against ten filters: naive Sharpe
threshold, single holdout, walk-forward, deflated Sharpe ratio (as published
and a variant), combinatorial purged cross-validation, Benjamini-Hochberg,
Benjamini-Yekutieli, Romano-Wolf and the probability of backtest overfitting.
All tables: [results/exp1_grid_tables.md](results/exp1_grid_tables.md). Notes:
[Stage 1](results/stage1.md), [Stage 2](results/stage2.md),
[Stage 3](results/stage3.md).

**Headline so far: no filter protects against a signal that fades.** When the
planted signal halves every 5 years, a strategy can look like Sharpe 1 in the
backtest and be worth almost nothing afterwards. At the 0.2 threshold,
35% to 69% of what the research-sample filters accept is worthless (FDR), and
in the worlds where a filter accepts anything at all, 86% to 89% is (pFDR).
Multiple-testing corrections do not help, and that is not a flaw in them: they
answer "was there an edge in this sample?", and the answer is correctly yes.
The holdout has the lowest FDR (18%), but through recency, not statistics: its
lockbox is the most recent 3 years, where the signal has already faded, so it
says yes in fewer worlds (23%). When it does say yes, 79% of what it accepts
is still worthless. Details in the [Stage 3 note](results/stage3.md).

![Plain worlds against regime switching and alpha decay](results/figures/exp1_grid_regimes_decay.png)

**The deflated Sharpe ratio, applied as published, accepts nothing.** In 4,600
simulated worlds it passed no strategy, even where the planted Sharpe is 2.
Its benchmark is built from the variance of the trials' Sharpe ratios and
assumes that this variance is pure estimation noise. On a grid where
strategies differ for real (those that follow a signal against their mirror
images), the variance measures real spread, and the benchmark rises above the
best strategy. A variant that uses the estimation variance under "no edge" and
an effective number of independent trials accepts nothing on noise and finds
32% of the real strategies when the signal is obvious. Details in the
[Stage 2 note](results/stage2.md).

![FDR against power](results/figures/exp1_grid_fdr_power.png)

## The LLM agent (Stage 4, in progress)

A language model plays the researcher: each round it writes a hypothesis and
a `positions(prices)` function, the function is checked, run in a sandbox with
the look-ahead test, backtested by the same engine as the grid, and the model
is told the result on the research sample.

- **What the agent remembers.** By default every round is a fresh prompt with
  a table of all its past rounds (hypothesis, status, research-sample Sharpe)
  and the error of the last round if it failed. It sees every result it ever
  got, so it stays adaptive. The first version kept one growing conversation
  with all earlier code instead; a small local model then copied its own
  mistakes round after round and the prompt reached 14,000 tokens in 20
  rounds (1,800 with the table). The old mode is still there:
  `--history transcript`. The price of the table: the agent no longer sees its
  earlier code, only what each idea was and how it scored.
- **Repairs.** If a round's code cannot be used (error, look-ahead, a repeat),
  the model gets the reason and up to two more attempts. Only the final
  version is a trial. Failed code was never backtested, so it told the model
  nothing about the data and does not count as a trial for the corrections.
- **Reproducibility.** Every model reply is stored under a hash of the full
  request (`cache/llm/`), and experiments are re-scored from those files
  without calling a model.

## Setup and usage

```bash
uv sync
uv run pytest -q
uv run python -m overfitlab all          # every experiment, about 12 minutes
uv run python -m overfitlab all --quick  # small check run, writes to results/quick
```

All randomness derives from one master seed (`20261116` in
`src/overfitlab/config.py`).

## Limitations

- **Simulated markets.** They have fat tails, volatility clustering, correlated
  assets, and optionally regime switching and alpha decay, but are far simpler
  than real markets.
- **Regime switching and signal strength are only partly separated.** Scaling
  the signal back up matches the planted rule's Sharpe, not how hard the world
  is for the grid; the comparison is clean for one of three signal types.
- **Daily data only.**
- **Two methods rest on secondary sources.** The CPCV chapter (Lopez de Prado
  2018) and the paper behind the effective number of trials (Galwey 2009) are
  not freely available. CPCV follows a library's documentation and my own
  derivation of the split and path counts; the effective-trials formula comes
  from an R package's documentation. Neither was checked against the original.
- **The CPCV accept rule is my own.** The book gives the paths, not a decision
  rule; the one used here is stated and tested in the Stage 2 note.
- **Alpha decay is tested in one setting.** One half-life (5 years) on one
  signal strength (obvious). A slower decay or a weaker signal would give
  different numbers; the direction of the result is what carries over.
- **Not finished.** The LLM agent, Thresholdout and the real-data check are
  still to come.

# Backtest Overfitting Lab

**Work in progress: Stage 2 of 6 is done.** This README is a placeholder; the
full write-up comes with the last stage.

**Question:** when many trading strategies are backtested on the same data,
which statistical filters stop the false discoveries, and how many real
strategies do they throw away in the process?

**Approach:** simulated markets where the truth is known (no signal, or a
planted signal of known strength), a strategy generator that tries many ideas,
and filters that decide which ideas to believe. False discovery rate and power
are measured over hundreds of simulated worlds.

## Results so far

Brute-force grid search of 76 strategies against nine filters: naive Sharpe
threshold, single holdout, walk-forward, deflated Sharpe ratio (as published
and a variant), combinatorial purged cross-validation, Benjamini-Hochberg,
Benjamini-Yekutieli and Romano-Wolf. Notes with tables:
[Stage 1](results/stage1.md), [Stage 2](results/stage2.md).

![FDR against power](results/figures/exp1_grid_fdr_power.png)

**Finding so far: the deflated Sharpe ratio, applied as published, accepts
nothing.** In 2,600 simulated worlds it passed no strategy, even where the
planted Sharpe is 2. Its benchmark is built from the variance of the trials'
Sharpe ratios and assumes that this variance is pure estimation noise. On a
grid where strategies differ for real (those that follow a signal against
their mirror images), the variance measures real spread, and the benchmark
rises above the best strategy. A variant that uses the estimation variance
under "no edge" and an effective number of independent trials accepts nothing
on noise and finds 32% of the real strategies when the signal is obvious.
Details in the [Stage 2 note](results/stage2.md).

## Setup and usage

```bash
uv sync
uv run pytest -q
uv run python -m overfitlab all          # every experiment, about 7 minutes
uv run python -m overfitlab all --quick  # small check run, writes to results/quick
```

All randomness derives from one master seed (`20261116` in
`src/overfitlab/config.py`).

## Limitations

- **Simulated markets.** They have fat tails, volatility clustering and
  correlated assets, but are far simpler than real markets.
- **Daily data only.**
- **Two methods rest on secondary sources.** The CPCV chapter (Lopez de Prado
  2018) and the paper behind the effective number of trials (Galwey 2009) are
  not freely available. CPCV follows a library's documentation and my own
  derivation of the split and path counts; the effective-trials formula comes
  from an R package's documentation. Neither was checked against the original.
- **The CPCV accept rule is my own.** The book gives the paths, not a decision
  rule; the one used here is stated and tested in the Stage 2 note.
- **Not finished.** PBO, regime switching, the LLM agent and Thresholdout are
  still to come.

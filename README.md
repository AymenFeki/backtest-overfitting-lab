# Backtest Overfitting Lab

**Work in progress: Stage 1 of 6 is done.** This README is a placeholder; the
full write-up comes with the last stage.

**Question:** when many trading strategies are backtested on the same data,
which statistical filters stop the false discoveries, and how many real
strategies do they throw away in the process?

**Approach:** simulated markets where the truth is known (no signal, or a
planted signal of known strength), a strategy generator that tries many ideas,
and filters that decide which ideas to believe. False discovery rate and power
are measured over hundreds of simulated worlds.

## Results so far

Stage 1: brute-force grid search against a naive Sharpe threshold, a single
holdout and walk-forward. Details and table in
[results/stage1.md](results/stage1.md).

![FDR against power](results/figures/exp1_grid_fdr_power.png)

## Setup and usage

```bash
uv sync
uv run pytest -q
uv run python -m overfitlab all          # every experiment, about 5 minutes
uv run python -m overfitlab all --quick  # small check run, writes to results/quick
```

All randomness derives from one master seed (`20261116` in
`src/overfitlab/config.py`).

## Limitations

- **Simulated markets.** They have fat tails, volatility clustering and
  correlated assets, but are far simpler than real markets.
- **Daily data only.**
- **Not finished.** Deflated Sharpe, CPCV, multiple-testing control, PBO, the
  LLM agent and Thresholdout are still to come.

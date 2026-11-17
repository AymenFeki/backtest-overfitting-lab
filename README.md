# Backtest Overfitting Lab

**Stages 1–5 complete; Stage 6 (real-data check) in progress.**

## Problem

When many trading strategies are backtested on the same data, which
statistical filters stop the false discoveries, and how many real strategies
do they throw away in the process?

## Approach

- **Simulated worlds with known truth.** Ten correlated assets with fat tails
  and volatility clustering, either pure noise or with a planted signal
  (momentum, cross-sectional momentum, mean reversion) of known strength,
  optionally with regime switching or a fading signal. Each strategy's true
  Sharpe is measured on 10,000 fresh simulated years.
- **Grid generator.** A brute-force search over 76 textbook strategies, run
  on 200 worlds of each type.
- **LLM agent.** Claude Haiku 5.5 plays the researcher: each round it writes
  a strategy as code, sees its backtest result and tries again.
- **Filters.** Ten ways of deciding which strategies to believe: naive Sharpe
  threshold, single holdout, walk-forward, deflated Sharpe ratio (as published
  and a variant), combinatorial purged cross-validation (CPCV),
  Benjamini-Hochberg, Benjamini-Yekutieli, Romano-Wolf and the probability of
  backtest overfitting.
- **Reusable holdout.** An analyst who consults the lockbox after every round,
  with and without Thresholdout (Dwork et al. 2015), and a test of whether an
  agent told the filters' rules learns to game them.

## Results

Medium-strength signal, the three signal types pooled, each filter at its
conventional cut; percent (grid 600 worlds, agent 90). Intervals and the
remaining filters are in the [Stage 4 note](results/stage4.md).

| Filter | FDR, grid | FDR, agent | Power, grid | Power, agent |
|---|---|---|---|---|
| Naive Sharpe | 1.6 | 1.1 | 16.4 | 37.4 |
| Holdout | 0.3 | 0.0 | 6.6 | 17.9 |
| Deflated Sharpe (as published) | 0.0 | 0.0 | 0.0 | 0.2 |
| Benjamini-Hochberg | 1.1 | 1.4 | 18.5 | 54.1 |
| Romano-Wolf | 0.8 | 1.1 | 9.5 | 42.9 |
| CPCV | 1.1 | 0.0 | 8.1 | 22.9 |

Power is measured against each generator's own real strategies, so the two
power columns are not comparable one to one.

![FDR against power for each filter, by signal strength](results/figures/exp1_grid_fdr_power.png)

- With a grid of 76 strategies and a medium-strength signal, every filter
  keeps the false discovery rate at 1.6% or below; what separates them is
  power: Benjamini-Hochberg finds 18.5% of the real strategies, Romano-Wolf
  9.5%, a single holdout 6.6%, and the deflated Sharpe ratio as published
  none ([Stage 2 note](results/stage2.md)).
- No filter protects against a signal that fades: with alpha decay, 35% to
  69% of what the research-sample filters accept is worthless
  ([Stage 3 note](results/stage3.md)).
- A Claude Haiku agent writing its own strategies proposes far more real ones
  than the grid (68% to 85% against 24% to 42% on signal worlds), and the
  same filters accept more of them with a false discovery rate of 1.4% or
  below; but it proposes the planted textbook signals before seeing any data,
  so this measures prior knowledge as much as research
  ([Stage 4 note](results/stage4.md)).
- Reporting the lockbox result after every round raises false claims on noise
  from 1.5% (lockbox opened once) to 41.5% for a simple hill-climber.
  Thresholdout at its declared settings only brings this to 32%, and the
  lockbox stays honest (7% of the chosen strategies really clear the bar)
  only when the answers are so noisy that 90% of the claims made from them
  are false ([Stage 5 note](results/stage5.md)).
- Whether knowing the filters helps an agent game them could not be tested:
  on noise no agent strategy ever passed a check, so the feedback carried no
  information ([Stage 5 note](results/stage5.md)).

## How to read the numbers

A strategy is **real** if its long-run Sharpe after costs is at least 0.2.
**FDR** is the share of accepted strategies that are not real, averaged over
worlds (a world with no acceptance counts as 0); **pFDR** averages only over
worlds where something was accepted; **power** is the share of real strategies
accepted. More in the [Stage 1 note](results/stage1.md#how-to-read-the-numbers).

## How to run

```bash
uv sync
uv run pytest -q
uv run python -m overfitlab all          # every experiment, from stored model replies
uv run python -m overfitlab all --quick  # small check run, writes to results/quick
uv run python -m overfitlab agent        # only the agent experiment, from stored replies
uv run python -m overfitlab exp2         # holdout reuse: hill-climber and agent
uv run python -m overfitlab exp3         # told against not told
uv run python scripts/zero_cost.py --jobs 6  # the same strategies without costs
```

`all` makes no model call. The one command that does is
`uv run --env-file .env python -m overfitlab generate`, which needs
`ANTHROPIC_API_KEY` in a git-ignored `.env` file, skips every reply that is
already stored and stops before the spending cap.

All randomness derives from one master seed (`20261116` in
`src/overfitlab/config.py`).

## Limitations

- **Simulated markets.** Fat tails, volatility clustering, regimes and decay,
  but far simpler than real markets; daily data only.
- **Thresholdout runs outside its guarantee.** Its theorem needs bounded
  statistics and a far larger holdout than a Sharpe ratio on 756 days.
- **Experiment 3 gave the agent nothing to game.** No strategy ever passed a
  check, so the design does not test what happens when checks can be passed.
- **Small agent samples, one model, one prompt.** 15 to 30 worlds per arm or
  type, all with Claude Haiku 5.5.
- **Two methods rest on secondary sources** (CPCV and the effective number of
  trials), and the CPCV accept rule is my own.

Full list: [results/limitations.md](results/limitations.md).

## Details

- [Stage 1](results/stage1.md): simulated worlds, the backtest engine, the
  grid, the three everyday filters, and how to read the numbers.
- [Stage 2](results/stage2.md): deflated Sharpe ratio, CPCV,
  Benjamini-Hochberg and Romano-Wolf, and why the published deflated Sharpe
  accepts nothing.
- [Stage 3](results/stage3.md): probability of backtest overfitting, regime
  switching, alpha decay, and pFDR next to FDR.
- [Stage 4](results/stage4.md): the LLM agent against the grid, the choice of
  model, and the cost drag on noise.
- [Stage 5](results/stage5.md): holdout reuse with and without Thresholdout,
  and the told-against-not-told test.

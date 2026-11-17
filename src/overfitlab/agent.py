"""A language model as strategy researcher.

Each round the model proposes a hypothesis and a `positions(prices)` function.
The function is checked, run in the sandbox and backtested by the same engine
as every other strategy; the model is then told the result on the research
sample (the first 70% of days) and tries again. It never sees prices, the
lockbox, or which world it is in.

What the model remembers (`history`):

* "compact" (default): every round starts a fresh conversation that holds a
  table of all past rounds (hypothesis, status, research Sharpe) and, if the
  last round failed, its error. The model sees every result it ever got, so it
  stays adaptive, and the prompt stays short enough for a small local model.
* "transcript": one conversation that only grows, including all earlier code.

If the code of a round cannot be used (error, look-ahead, a repeat), the model
is shown the reason and may send a corrected version, up to two times. Only
the final version is the round's trial. Code that failed was never backtested,
so the model learned nothing about the data from it and it does not count as
a trial for the multiple-testing corrections.

Experiments 2 and 3 give the model more feedback than the research Sharpe.
They pass their own system prompt and a `Column`: one more column in the
table of past rounds (compact history only).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .backtest import net_returns, sharpe
from .llm import Model, Reply, complete
from .sandbox import run_strategy

SYSTEM = """\
You are a quantitative researcher. You are looking for a daily trading \
strategy that will be profitable on future data from the same market, after \
transaction costs.

The market has 10 assets. You never see the prices yourself. Each round you \
write one strategy as a Python function, it is backtested on about 7 years of \
daily prices, and you are told the result. Use what you learn: refine what \
works, drop what does not, and try different kinds of ideas and lookbacks.

Reply in exactly this format and nothing else:

Hypothesis: <one sentence on why this could work>
```python
def positions(prices):
    ...
    return weights
```

Rules for the function:
- `prices` is a pandas DataFrame: one row per trading day (oldest first), one \
column per asset.
- Return a DataFrame of the same shape, index and columns: the weight to hold \
in each asset on each day. Positive is long, negative is short.
- The weight on a day may use prices up to and including that day only. Never \
use later rows: no shift(-1), no statistics over the whole table.
- The backtest trades the weights on the following day, scales them down if \
their absolute values sum to more than 1, and charges 5 basis points on every \
change in holdings. Do not normalise the weights yourself.
- Only `np` (numpy) and `pd` (pandas) exist, already imported. No other \
imports, no files, no random numbers, no loops over days.
- Every round must be a new strategy. Never send the same code twice.

A complete valid reply, to show the form only (it holds every asset equally \
and uses no signal):

Hypothesis: Holding all assets equally earns the average market return.
```python
def positions(prices):
    weights = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    return weights / prices.shape[1]
```
"""
REPAIRS = 2  # extra attempts per round when the code cannot be used
_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_HYPOTHESIS = re.compile(r"Hypothesis:\s*(.+)")


@dataclass
class Trial:
    round: int
    hypothesis: str
    code: str
    status: str  # "ok", or why the final version could not be used
    returns: pd.Series | None  # net daily returns over the whole world
    replies: list[Reply]  # one per attempt: the proposal and any repairs
    failures: list[str]  # why each failed attempt of this round was rejected


@dataclass(frozen=True)
class Column:
    """One more column in the agent's table of past rounds."""

    header: str
    cells: Callable[[list[Trial]], list[str]]  # the text for each round so far


def market_summary(prices: pd.DataFrame) -> str:
    """What the agent is told about the data before its first round."""
    returns = prices.iloc[: cfg.N_RESEARCH].pct_change().dropna()
    vol = returns.std() * np.sqrt(cfg.PERIODS_PER_YEAR)
    corr = returns.corr().to_numpy()
    average = corr[np.triu_indices_from(corr, k=1)].mean()
    return (
        f"Market summary: {len(returns)} trading days, {prices.shape[1]} assets, "
        f"annual volatility per asset {vol.min():.0%} to {vol.max():.0%}, "
        f"average correlation between assets {average:.2f}."
    )


def research_feedback(returns: pd.Series) -> str:
    """Backtest result on the research sample, as shown to the agent."""
    r = returns.iloc[: cfg.N_RESEARCH]
    year = cfg.PERIODS_PER_YEAR
    return (
        f"Sharpe ratio after costs {float(sharpe(r)):.2f} (annual return "
        f"{r.mean() * year:.1%}, annual volatility {r.std() * np.sqrt(year):.1%})"
    )


def parse(text: str) -> tuple[str, str]:
    """Hypothesis and code from a reply; the code is empty if there is none."""
    hypothesis = _HYPOTHESIS.search(text)
    code = _CODE.search(text)
    return (
        hypothesis.group(1).strip() if hypothesis else "",
        code.group(1).strip() if code else "",
    )


def _research_sharpe(trial: Trial) -> float:
    return float(sharpe(trial.returns.iloc[: cfg.N_RESEARCH]))


def history_table(trials: list[Trial], column: Column | None = None) -> str:
    """All past rounds as the agent sees them in compact mode."""
    if not trials:
        return "No rounds yet."
    extra = f"{column.header} | " if column else ""
    cells = column.cells(trials) if column else [""] * len(trials)
    lines = [
        "Your rounds so far (Sharpe ratio after costs on the backtest):",
        f"round | status | Sharpe | {extra}hypothesis",
    ]
    for t, cell in zip(trials, cells, strict=True):
        score = "-" if t.returns is None else f"{_research_sharpe(t):.2f}"
        status = "ok" if t.status == "ok" else "failed"
        cell = f"{cell} | " if column else ""
        lines.append(f"{t.round} | {status} | {score} | {cell}{t.hypothesis[:150]}")
    if trials[-1].returns is None:
        lines.append(f"Round {trials[-1].round} failed with: {trials[-1].status}.")
    return "\n".join(lines)


def run_agent(
    prices: pd.DataFrame,
    model: Model,
    seed: int,
    rounds: int = cfg.AGENT_ROUNDS,
    live: bool = False,
    history: str = cfg.AGENT_HISTORY,
    system: str = SYSTEM,
    column: Column | None = None,
) -> list[Trial]:
    """One agent, one world: `rounds` proposals, each with its backtest."""
    if column and history != "compact":
        raise ValueError("an extra column needs the compact history")
    trials: list[Trial] = []
    messages: list[dict] = []
    summary = market_summary(prices)
    note = summary  # transcript mode: what happened since the last message
    for number in range(1, rounds + 1):
        ask = f"Round {number} of {rounds}: propose a new strategy."
        if history == "compact":
            messages = []  # a fresh conversation every round
            ask = f"{summary}\n\n{history_table(trials, column)}\n\n{ask}"
        else:
            ask = f"{note} {ask}"
        replies: list[Reply] = []
        failures: list[str] = []
        for attempt in range(1 + REPAIRS):
            messages.append({"role": "user", "content": ask})
            call_seed = seed * 1000 + number * 10 + attempt
            replies.append(complete(model, system, messages, call_seed, live))
            messages.append({"role": "assistant", "content": replies[-1].text})
            hypothesis, code = parse(replies[-1].text)
            returns, status = None, "no Python code block in the reply"
            if code:
                weights, status = run_strategy(code, prices)
            same = [t.round for t in trials if t.status == "ok" and t.code == code]
            if status == "ok" and same:
                status = f"identical to round {same[0]}"
            if status == "ok":
                returns = net_returns(prices, weights)
                break
            failures.append(status)
            ask = (
                f"That code cannot be used: {status}. "
                "Send a corrected reply in the same format."
            )
        trial = Trial(number, hypothesis, code, status, returns, replies, failures)
        trials.append(trial)
        if returns is None:
            note = f"Round {number} could not be used: {status}."
        else:
            note = f"Round {number} result: {research_feedback(returns)}."
        done = [t for t in trials if t.returns is not None]
        if done:
            best = max(done, key=_research_sharpe)
            note += (
                f" Best so far: round {best.round}, "
                f"Sharpe ratio {_research_sharpe(best):.2f}."
            )
    return trials

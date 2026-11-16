"""A language model as strategy researcher.

Each round the model proposes a hypothesis and a `positions(prices)` function.
The function is checked, run in the sandbox and backtested by the same engine
as every other strategy; the model is then told the result on the research
sample (the first 70% of days) and tries again. It never sees prices, the
lockbox, or which world it is in. Its conversation only grows, so every round
it sees everything it tried before: that is what makes it adaptive.
"""

import re
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
- Return a DataFrame of the same shape: the weight to hold in each asset on \
each day. Positive is long, negative is short.
- The weight on a day may use prices up to and including that day only. Never \
use later rows: no shift(-1), no statistics over the whole table.
- The backtest trades the weights on the following day, scales them down if \
their absolute values sum to more than 1, and charges 5 basis points on every \
change in holdings.
- Only `np` (numpy) and `pd` (pandas) exist, already imported. No other \
imports, no files, no random numbers.
- Use vectorised pandas and numpy operations; no loops over days.
"""
_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_HYPOTHESIS = re.compile(r"Hypothesis:\s*(.+)")


@dataclass
class Trial:
    round: int
    hypothesis: str
    code: str
    status: str  # "ok", or why the proposal could not be used
    returns: pd.Series | None  # net daily returns over the whole world
    reply: Reply


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


def run_agent(
    prices: pd.DataFrame,
    model: Model,
    seed: int,
    rounds: int = cfg.AGENT_ROUNDS,
    live: bool = False,
) -> list[Trial]:
    """One agent, one world: `rounds` proposals, each with its backtest."""
    trials: list[Trial] = []
    messages: list[dict] = []
    note = market_summary(prices)
    for number in range(1, rounds + 1):
        ask = f"{note} Round {number} of {rounds}: propose a strategy."
        messages.append({"role": "user", "content": ask})
        reply = complete(model, SYSTEM, messages, seed * 100 + number, live)
        messages.append({"role": "assistant", "content": reply.text})
        hypothesis, code = parse(reply.text)
        returns = None
        if not code:
            status = "no Python code block in the reply"
        else:
            weights, status = run_strategy(code, prices)
        if status == "ok":
            returns = net_returns(prices, weights)
            same = [t.round for t in trials if t.status == "ok" and t.code == code]
            if same:
                returns, status = None, f"identical to round {same[0]}"
        trials.append(Trial(number, hypothesis, code, status, returns, reply))
        if returns is None:
            note = f"Round {number} could not be used: {status}."
        else:
            note = f"Round {number} result: {research_feedback(returns)}."
    return trials

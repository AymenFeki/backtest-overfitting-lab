"""Strategy families for the brute-force grid search.

Each rule is tried in both directions ("follow" the signal or "fade" it), the
way a researcher without a prior would. 38 rules x 2 directions = 76 strategies.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from . import config as cfg
from .worlds import ASSET_VOL, planted_signal


@dataclass(frozen=True)
class Strategy:
    name: str
    positions: Callable[[pd.DataFrame], pd.DataFrame]


def trend(prices: pd.DataFrame, lookback: int, direction: int = 1) -> pd.DataFrame:
    """Long assets that rose over the lookback, short those that fell."""
    return direction * np.sign(prices.pct_change(lookback)) / prices.shape[1]


def cross_section(
    prices: pd.DataFrame, lookback: int, direction: int = 1
) -> pd.DataFrame:
    """Long recent winners, short recent losers, by rank. Always market-neutral."""
    ranks = prices.pct_change(lookback).rank(axis=1)
    centred = ranks.sub(ranks.mean(axis=1), axis=0)
    return direction * centred.div(centred.abs().sum(axis=1), axis=0)


def ma_cross(
    prices: pd.DataFrame, fast: int, slow: int, direction: int = 1
) -> pd.DataFrame:
    """Long when the fast moving average is above the slow one, else short."""
    gap = prices.rolling(fast).mean() - prices.rolling(slow).mean()
    return direction * np.sign(gap) / prices.shape[1]


def breakout(prices: pd.DataFrame, lookback: int, direction: int = 1) -> pd.DataFrame:
    """Go long at a new high, short at a new low, and hold until the opposite."""
    at_high = prices >= prices.rolling(lookback).max()
    at_low = prices <= prices.rolling(lookback).min()
    signal = (at_high.astype(float) - at_low.astype(float)).replace(0.0, np.nan)
    return direction * signal.ffill() / prices.shape[1]


_RULES = (
    [(trend, {"lookback": n}) for n in (2, 3, 5, 10, 15, 20, 40, 60, 90, 120, 180, 250)]
    + [(cross_section, {"lookback": n}) for n in (5, 10, 20, 40, 60, 90, 120, 180, 250)]
    + [
        (ma_cross, {"fast": fast, "slow": slow})
        for slow in (50, 100, 200)
        for fast in (5, 10, 20, 50)
        if fast < slow
    ]
    + [(breakout, {"lookback": n}) for n in (10, 20, 40, 60, 120, 250)]
)


def grid() -> list[Strategy]:
    """All grid-search strategies, in a fixed order."""
    out = []
    for rule, params in _RULES:
        label = "_".join(str(v) for v in params.values())
        for direction, tag in ((1, "follow"), (-1, "fade")):
            name = f"{rule.__name__}_{label}_{tag}"
            out.append(Strategy(name, partial(rule, **params, direction=direction)))
    return out


def grid_neighbours() -> list[list[int]]:
    """For each grid strategy, the positions of its close relatives in `grid()`.

    Relatives are the same rule traded the other way round, and the rule of
    the same family with the next shorter or next longer setting, in the same
    direction. This is the "small change" a researcher makes when tuning.
    """
    out = []
    for index in range(2 * len(_RULES)):
        rule, direction = divmod(index, 2)
        near = [2 * rule + (1 - direction)]
        for other in (rule - 1, rule + 1):
            if 0 <= other < len(_RULES) and _RULES[other][0] is _RULES[rule][0]:
                near.append(2 * other + direction)
        out.append(near)
    return out


def oracle(spec: cfg.WorldSpec) -> Strategy:
    """The strategy that trades the planted rule itself.

    A yardstick for signal strength, not an upper bound: a grid strategy with
    lower turnover can end up with a higher Sharpe after costs.
    """
    lookback = cfg.SIGNAL_LOOKBACK[spec.signal]

    def positions(prices: pd.DataFrame) -> pd.DataFrame:
        trailing = prices.pct_change().rolling(lookback).sum()
        z = (trailing / (ASSET_VOL * np.sqrt(lookback))).to_numpy()
        weights = planted_signal(spec.signal, z) / (2 * prices.shape[1])
        return pd.DataFrame(weights, index=prices.index, columns=prices.columns)

    return Strategy("oracle", positions)

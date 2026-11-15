"""The one backtest engine every strategy goes through.

A strategy only says what it wants to hold: `positions(prices) -> weights`.
The engine does the rest the same way for everyone:

* weights decided at the close of day t earn the return of day t+1 (one-day lag),
* total exposure is capped at 100% of capital,
* every change in holdings pays a transaction cost.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from . import config as cfg

Positions = Callable[[pd.DataFrame], pd.DataFrame]


def net_returns(
    prices: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = cfg.COST_BPS
) -> pd.Series:
    """Daily strategy returns after costs.

    Weights are target weights, rebalanced daily; drift between rebalances is
    ignored, which is the usual simplification for daily data.
    """
    asset_returns = prices.pct_change().fillna(0.0)
    weights = weights.fillna(0.0)
    leverage = weights.abs().sum(axis=1).clip(lower=1.0)
    held = weights.div(leverage, axis=0).shift(1).fillna(0.0)
    turnover = held.diff().abs().sum(axis=1)
    return (held * asset_returns).sum(axis=1) - turnover * cost_bps / 1e4


def run_all(
    prices: pd.DataFrame,
    strategies: Sequence,
    cost_bps: float = cfg.COST_BPS,
) -> pd.DataFrame:
    """Net returns of every strategy on one price table (days x strategies)."""
    return pd.DataFrame(
        {s.name: net_returns(prices, s.positions(prices), cost_bps) for s in strategies}
    )


def sharpe(returns: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    """Annualised Sharpe ratio, column by column. Zero if a column never moves."""
    r = np.asarray(returns, dtype=float)
    mean, sd = r.mean(axis=0), r.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(cfg.PERIODS_PER_YEAR) * np.where(sd > 0, mean / sd, 0.0)

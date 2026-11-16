"""Stage 1 filters: the three things most people do.

Every filter takes the same input, a table of net daily returns (days x
strategies) for one world, and gives each strategy a score. A strategy is
accepted when its score exceeds the filter's cut (see registry.py). Sweeping
the cut traces the filter's FDR-versus-power curve.

None of these three corrects for how many strategies were tried. That is the
point of the baseline: the later filters are measured against them.
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .backtest import sharpe


def confidence(returns: np.ndarray, traded: np.ndarray | None = None) -> np.ndarray:
    """1 - p of the one-sided t-test "mean daily return <= 0", per column.

    The t statistic of the mean is the daily Sharpe ratio times sqrt(n), so
    this is also a test of "Sharpe <= 0". It stays valid under volatility
    clustering: without an edge, returns are still unpredictable from the past
    (a martingale difference sequence), which is all the central limit theorem
    behind the test needs. `traded` marks the days that count (default: all).
    """
    if traded is None:
        traded = np.ones(returns.shape, dtype=bool)
    n = traded.sum(axis=0)
    n_safe = np.maximum(n, 2)
    mean = np.where(traded, returns, 0.0).sum(axis=0) / n_safe
    squares = np.where(traded, (returns - mean) ** 2, 0.0).sum(axis=0)
    standard_error = np.sqrt(squares / (n_safe - 1) / n_safe)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(standard_error > 0, mean / standard_error, 0.0)
    return np.where(n >= 2, stats.t.cdf(t, df=n_safe - 1), 0.0)


def naive_sharpe(returns: pd.DataFrame) -> np.ndarray:
    """Score = annualised Sharpe on the research sample. Nothing else."""
    return sharpe(returns.iloc[: cfg.N_RESEARCH])


def holdout(returns: pd.DataFrame, screen: float = cfg.SCREEN_SHARPE) -> np.ndarray:
    """Pass the in-sample screen, then be significantly positive in the lockbox.

    Score = confidence of the lockbox t-test, or 0 if the screen was failed.
    The test is correct for one strategy chosen in advance. It does not know
    that many strategies were screened, so some pass both steps by luck.
    """
    r = returns.to_numpy()
    passed = sharpe(r[: cfg.N_RESEARCH]) >= screen
    return np.where(passed, confidence(r[cfg.N_RESEARCH :]), 0.0)


def walk_forward(
    returns: pd.DataFrame,
    block: int = cfg.PERIODS_PER_YEAR,
    n_initial_blocks: int = 3,
    screen: float = cfg.SCREEN_SHARPE,
) -> np.ndarray:
    """Trade each year only if the strategy passed the screen on all prior data.

    After 3 initial years, each following year is "out of sample" for the
    decision taken just before it. Score = confidence of the t-test on the
    days the strategy was actually traded, or 0 if it was never traded.
    Compared with one holdout, this spreads the test over more years, but
    early decisions rest on little data.
    """
    r = returns.to_numpy()
    traded = np.zeros(r.shape, dtype=bool)
    for start in range(n_initial_blocks * block, len(r), block):
        traded[start : start + block] = sharpe(r[:start]) >= screen
    return confidence(r, traded)

"""Probability of backtest overfitting, estimated by CSCV.

Bailey, Borwein, Lopez de Prado & Zhu (2017), Algorithm 2.3 (combinatorially
symmetric cross-validation):

1. Cut the days into S blocks of equal length.
2. For every way of choosing S/2 blocks as "in sample" (the other half is
   "out of sample"): find the strategy with the best in-sample Sharpe and
   look up its rank out of sample.
3. Turn that rank into a logit: omega = rank / (N + 1) and
   lambda = ln(omega / (1 - omega)). lambda > 0 means the in-sample winner
   was in the better half out of sample.
4. PBO = share of combinations with lambda <= 0 (their Section 3.1).

If picking the in-sample winner is no better than picking at random, PBO is
about 0.5. PBO judges the selection process, not a single strategy.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from . import config as cfg
from .backtest import sharpe

N_BLOCKS = 16  # the paper's example; C(16, 8) = 12,870 combinations


def cscv_logits(returns: np.ndarray, n_blocks: int = N_BLOCKS) -> np.ndarray:
    """One logit per in-sample/out-of-sample combination.

    A Sharpe ratio on any set of blocks only needs each block's sum and sum of
    squares, so all combinations are computed with two matrix products.
    """
    n_strategies = returns.shape[1]
    size = len(returns) // n_blocks
    blocks = returns[: size * n_blocks].reshape(n_blocks, size, n_strategies)
    sums, squares = blocks.sum(axis=1), (blocks**2).sum(axis=1)
    halves = list(combinations(range(n_blocks), n_blocks // 2))
    in_sample = np.zeros((len(halves), n_blocks))
    for i, chosen in enumerate(halves):
        in_sample[i, chosen] = 1.0
    n_days = size * n_blocks // 2

    def half_sharpe(weights: np.ndarray) -> np.ndarray:
        mean = weights @ sums / n_days
        variance = (weights @ squares - n_days * mean**2) / (n_days - 1)
        return mean / np.sqrt(variance)

    inside, outside = half_sharpe(in_sample), half_sharpe(1 - in_sample)
    winner = outside[np.arange(len(halves)), inside.argmax(axis=1)]
    rank = (outside <= winner[:, None]).sum(axis=1)  # 1 = worst, N = best
    omega = rank / (n_strategies + 1)
    return np.log(omega / (1 - omega))


def pbo(returns: np.ndarray, n_blocks: int = N_BLOCKS) -> float:
    """Share of combinations where the in-sample winner ends in the worse half."""
    return float((cscv_logits(returns, n_blocks) <= 0).mean())


def pbo_filter(returns: pd.DataFrame) -> np.ndarray:
    """Score = 1 - PBO for the strategy with the best research-sample Sharpe.

    All other strategies score 0: PBO says whether picking the winner can be
    trusted, so at most one strategy per world can be accepted.
    """
    r = returns.to_numpy()[: cfg.N_RESEARCH]
    scores = np.zeros(r.shape[1])
    scores[sharpe(r).argmax()] = 1 - pbo(r)
    return scores

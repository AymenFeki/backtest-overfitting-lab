"""Multiple-testing control across strategies.

Both methods start from one test per strategy of "mean daily return <= 0" and
then correct for the fact that many strategies were tested at once.

* Benjamini-Hochberg (1995) controls the false discovery rate: the expected
  share of accepted strategies that are false. Benjamini-Yekutieli (2001) is
  its stricter version that needs no assumption on how the tests depend.
* Romano-Wolf (2005, 2016) controls the familywise error rate: the probability
  of even one false acceptance. It uses a bootstrap, so it learns how the
  strategies move together instead of assuming the worst case.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as cfg
from .filters import confidence


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH adjusted p-values: the smallest level q at which each test is rejected.

    The paper's procedure (1): sort the m p-values, find the largest i with
    P(i) <= (i/m) q, reject the i smallest. Equivalently, scale each sorted
    p-value by m/i and take the running minimum from the largest downwards.
    Proven for independent tests; it also holds under positive dependence.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    scaled = p_values[order] * m / np.arange(1, m + 1)
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    return adjusted


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """BY adjusted p-values: BH run at level q / (1 + 1/2 + ... + 1/m).

    Benjamini & Yekutieli (2001), Theorem 1.3: with this smaller level the FDR
    is controlled under any dependence between the tests. BH itself is only
    proven for independent or positively dependent tests (their Theorem 1.2).
    """
    harmonic = (1 / np.arange(1, len(p_values) + 1)).sum()
    return np.minimum(benjamini_hochberg(p_values) * harmonic, 1.0)


def _block_sums(x: np.ndarray, block: int) -> np.ndarray:
    """Sum of `block` consecutive rows starting at every row, wrapping around."""
    wrapped = np.vstack([x, x[: block - 1]])
    cumulative = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(wrapped, axis=0)])
    return cumulative[block:] - cumulative[:-block]


@lru_cache
def _block_counts(n_days: int, block: int, n_resamples: int) -> np.ndarray:
    """How often each day starts a block, per resample: (resamples, days).

    A resample is built from blocks of consecutive days with random starting
    days (circular block bootstrap). Keeping days together in blocks preserves
    volatility clustering; using the same days for all strategies preserves
    how the strategies move together.
    """
    rng = np.random.default_rng(cfg.seed("romano_wolf"))
    n_blocks = -(-n_days // block)  # ceiling division
    starts = rng.integers(0, n_days, size=(n_resamples, n_blocks))
    counts = np.zeros((n_resamples, n_days))
    np.add.at(counts, (np.arange(n_resamples)[:, None], starts), 1)
    return counts


def romano_wolf(
    returns: np.ndarray,
    block: int = cfg.RW_BLOCK,
    n_resamples: int = cfg.RW_RESAMPLES,
) -> np.ndarray:
    """Romano-Wolf step-down adjusted p-values for "mean return <= 0", per column.

    Follows Clarke, Romano & Wolf (2019), Eqs. (4)-(6) and the adjusted
    p-value algorithm after Eq. (7):
    1. t = mean / standard error for each strategy.
    2. In each resample, t* = (resampled mean - original mean) / resampled
       standard error. Subtracting the original mean makes t* behave as if
       the strategy had no edge.
    3. Go from the most to the least significant strategy. Compare its t with
       the resampled maximum of t* over itself and all less significant ones:
       p = (#{max* >= t} + 1) / (resamples + 1). Then make the p-values
       non-decreasing along that order.

    A resampled mean is a weighted sum of block sums, so all resamples are two
    matrix products instead of 1,000 reshuffles of the data.
    """
    n, m = returns.shape
    mean = returns.mean(axis=0)
    standard_error = returns.std(axis=0, ddof=1) / np.sqrt(n)
    counts = _block_counts(n, block, n_resamples)
    size = counts[0].sum() * block  # days in one resample
    mean_star = counts @ _block_sums(returns, block) / size
    square_star = counts @ _block_sums(returns**2, block) / size
    error_star = np.sqrt((square_star - mean_star**2) / (size - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / standard_error
        t_star = (mean_star - mean) / error_star
    # A strategy that never trades has no t statistic: it can never be accepted.
    t = np.where(np.isfinite(t), t, -np.inf)
    t_star = np.where(np.isfinite(t_star), t_star, -np.inf)

    order = np.argsort(-t)
    max_star = np.maximum.accumulate(t_star[:, order][:, ::-1], axis=1)[:, ::-1]
    initial = ((max_star >= t[order]).sum(axis=0) + 1) / (n_resamples + 1)
    adjusted = np.empty(m)
    adjusted[order] = np.maximum.accumulate(initial)
    return adjusted


def bh_filter(returns: pd.DataFrame) -> np.ndarray:
    """Score = 1 - BH adjusted p-value on the research sample."""
    r = returns.to_numpy()[: cfg.N_RESEARCH]
    return 1 - benjamini_hochberg(1 - confidence(r))


def by_filter(returns: pd.DataFrame) -> np.ndarray:
    """Score = 1 - BY adjusted p-value on the research sample."""
    r = returns.to_numpy()[: cfg.N_RESEARCH]
    return 1 - benjamini_yekutieli(1 - confidence(r))


def romano_wolf_filter(returns: pd.DataFrame) -> np.ndarray:
    """Score = 1 - Romano-Wolf adjusted p-value on the research sample."""
    return 1 - romano_wolf(returns.to_numpy()[: cfg.N_RESEARCH])

"""Combinatorial purged cross-validation (Lopez de Prado, 2018, ch. 12).

Cut the research sample into N consecutive groups. Every way of choosing k of
them as test groups is one split, so there are C(N, k) splits. Each group is a
test group in C(N-1, k-1) splits, so the out-of-sample pieces can be stitched
into C(N-1, k-1) full-length backtest paths: path p takes each group from the
p-th split in which that group was tested. With N=10 and k=3 that gives 120
splits and 36 paths instead of the single path of a holdout or walk-forward.

Purging and embargo: training days right next to a test group are dropped,
because volatility clustering and positions carried across the boundary make
them informative about the test group.

Accept/reject rule used here
----------------------------
A fixed-rule strategy has nothing to fit, so by itself every path would be the
same series. The decision made on training data is the same one the other
filters use: in each split the strategy trades the test groups only if its
Sharpe on the purged training days passes the screen; otherwise it is flat.

    Accept if the 5% quantile of the 36 path Sharpe ratios is above zero.

The 5% quantile is taken as the 2nd lowest of the 36, so at most one path may
lose. A quantile is used instead of a t-test because the paths recombine the
same days: they are not independent draws, and a standard error across them
would be far too small. The 5% is therefore not a calibrated error rate; the
actual false-positive rate is measured on null worlds.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from . import config as cfg

N_GROUPS, N_TEST, GAP = 10, 3, 20
MIN_TRAIN = cfg.PERIODS_PER_YEAR  # a split with less training data trades nothing


def splits(n_groups: int, n_test: int) -> list[tuple[int, ...]]:
    """All choices of `n_test` test groups out of `n_groups`."""
    return list(combinations(range(n_groups), n_test))


def paths(n_groups: int, n_test: int) -> np.ndarray:
    """Which split supplies each group on each path: shape (paths, groups)."""
    all_splits = splits(n_groups, n_test)
    per_group = [
        [i for i, tested in enumerate(all_splits) if group in tested]
        for group in range(n_groups)
    ]
    return np.array(per_group).T


def train_masks(n_days: int, n_groups: int, n_test: int, gap: int) -> np.ndarray:
    """Training days of every split: (splits, days) booleans.

    A day is a training day if it is outside the test groups and at least
    `gap` days away from them on both sides (purge before, embargo after).
    """
    edges = np.linspace(0, n_days, n_groups + 1).astype(int)
    masks = np.ones((len(splits(n_groups, n_test)), n_days), dtype=bool)
    for i, tested in enumerate(splits(n_groups, n_test)):
        for group in tested:
            masks[i, max(edges[group] - gap, 0) : edges[group + 1] + gap] = False
    return masks


def cpcv(
    returns: pd.DataFrame,
    n_groups: int = N_GROUPS,
    n_test: int = N_TEST,
    gap: int = GAP,
    screen: float = cfg.SCREEN_SHARPE,
) -> np.ndarray:
    """Score = 5% quantile of each strategy's Sharpe across the CPCV paths."""
    r = returns.to_numpy()[: cfg.N_RESEARCH]
    n_days = len(r)
    annual = np.sqrt(cfg.PERIODS_PER_YEAR)

    # Sharpe of every strategy on every split's training days, via sums.
    train = train_masks(n_days, n_groups, n_test, gap).astype(float)
    n_train = train.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = train @ r / n_train
        variance = (train @ r**2 - n_train * mean**2) / (n_train - 1)
        traded = annual * mean / np.sqrt(variance) >= screen  # (splits, strategies)
    traded &= n_train >= MIN_TRAIN

    # Path returns = each group's returns where traded, zero where flat.
    edges = np.linspace(0, n_days, n_groups + 1).astype(int)[:-1]
    group_sum = np.add.reduceat(r, edges, axis=0)
    group_square = np.add.reduceat(r**2, edges, axis=0)
    on = traded[paths(n_groups, n_test)]  # (paths, groups, strategies)
    path_mean = (on * group_sum).sum(axis=1) / n_days
    path_square = (on * group_square).sum(axis=1)
    path_variance = (path_square - n_days * path_mean**2) / (n_days - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        path_sharpe = annual * np.where(
            path_variance > 0, path_mean / np.sqrt(path_variance), 0.0
        )
    return np.sort(path_sharpe, axis=0)[int(0.05 * len(path_sharpe))]

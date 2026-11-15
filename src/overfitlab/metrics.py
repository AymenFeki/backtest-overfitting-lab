"""False discovery rate and power, with uncertainty across simulated worlds."""

import numpy as np

from . import config as cfg


def fdp_and_power(
    accepted: np.ndarray, is_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-world false discovery proportion and power.

    accepted: (worlds, strategies) booleans, is_true: (strategies,) booleans.
    FDP = false accepted / accepted, taken as 0 when nothing is accepted
    (the Benjamini-Hochberg convention). Its average over worlds is the FDR.
    Power = share of the truly profitable strategies that were accepted; it is
    NaN when the world has none.
    """
    n_accepted = accepted.sum(axis=1)
    fdp = (accepted & ~is_true).sum(axis=1) / np.maximum(n_accepted, 1)
    if is_true.any():
        power = (accepted & is_true).sum(axis=1) / is_true.sum()
    else:
        power = np.full(len(accepted), np.nan)
    return fdp, power


def mean_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean over worlds with a percentile-bootstrap confidence interval.

    Worlds are independent, so we resample whole worlds with replacement and
    look at how much the mean moves. Returns (mean, low, high); NaNs are dropped.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(cfg.seed("bootstrap"))
    draws = rng.integers(0, len(values), size=(cfg.N_BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(values.mean()), float(low), float(high)

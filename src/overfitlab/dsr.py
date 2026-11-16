"""Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

Idea: try N strategies on noise and the best one still shows a positive Sharpe.
So do not ask "is this Sharpe above zero?" but "is it above what the best of N
worthless strategies would show?". That higher bar is SR_0, and the deflated
Sharpe ratio is the probability that the true Sharpe exceeds it, allowing for
skewed and fat-tailed returns.

All Sharpe ratios here are per period (daily, not annualised) and kurtosis is
raw (3 for a normal), as in the paper.
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .backtest import sharpe


def expected_max_sharpe(sharpe_variance: float, n_trials: float) -> float:
    """SR_0: expected best Sharpe among `n_trials` strategies with no edge.

    Paper Eq. (1) with mean zero:
        SR_0 = sqrt(V) * ((1 - g) * Z^-1[1 - 1/N] + g * Z^-1[1 - 1/(N e)])
    V is the variance of the trials' Sharpe ratios, g the Euler-Mascheroni
    constant, Z^-1 the inverse normal CDF. The formula comes from extreme value
    theory: the maximum of N normal draws.
    """
    g = np.euler_gamma
    z = (1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(
        1 - 1 / (n_trials * np.e)
    )
    return float(np.sqrt(sharpe_variance) * z)


def probabilistic_sharpe(
    observed: np.ndarray | float,
    benchmark: float,
    n_obs: int,
    skew: np.ndarray | float,
    kurtosis: np.ndarray | float,
) -> np.ndarray:
    """Probability that the true Sharpe exceeds `benchmark`. Paper Eq. (2):

        Z[(SR - SR_0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1)/4 * SR^2)]

    The square root is the standard error of a Sharpe estimate, times
    sqrt(T - 1). Negative skew and fat tails make it larger, so the same Sharpe
    counts for less. It can never be negative: any distribution has
    kurtosis >= skew^2 + 1, which keeps the quadratic in SR above zero.
    """
    spread = np.sqrt(1 - skew * observed + (kurtosis - 1) / 4 * observed**2)
    return stats.norm.cdf((observed - benchmark) * np.sqrt(n_obs - 1) / spread)


def deflated_sharpe(returns: pd.DataFrame) -> np.ndarray:
    """DSR of every strategy on the research sample.

    N is the number of strategies tried and V the variance of their Sharpe
    ratios. The paper asks for the number of *independent* trials; counting
    every correlated trial overstates SR_0, which errs on the cautious side.
    """
    r = returns.to_numpy()[: cfg.N_RESEARCH]
    daily = sharpe(r) / np.sqrt(cfg.PERIODS_PER_YEAR)
    benchmark = expected_max_sharpe(daily.var(ddof=1), r.shape[1])
    return probabilistic_sharpe(
        daily, benchmark, len(r), stats.skew(r), stats.kurtosis(r, fisher=False)
    )


def effective_trials(returns: np.ndarray) -> float:
    """Effective number of independent trials, from the correlation matrix.

    Galwey (2009): M_eff = (sum of sqrt(eigenvalue))^2 / (sum of eigenvalues),
    negative eigenvalues set to zero. 76 unrelated strategies give 76, 76
    copies of one strategy give 1, and a strategy with its mirror image gives
    1, because eigenvalues ignore the sign of a correlation.

    Formula taken from the documentation of the R package poolr (`meff`); the
    paper itself (Genetic Epidemiology 33, 559-568) is behind a paywall.
    """
    traded = returns[:, returns.std(axis=0) > 0]  # a flat strategy is no trial
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(traded, rowvar=False))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return float(np.sqrt(eigenvalues).sum() ** 2 / eigenvalues.sum())


def deflated_sharpe_null(returns: pd.DataFrame) -> np.ndarray:
    """Variant of the DSR for a set of trials with different true Sharpe ratios.

    Not the paper's formula. Two changes to the benchmark SR_0:

    * V is the variance a Sharpe estimate has when the true Sharpe is zero,
      instead of the variance across trials. A Sharpe estimate has variance
      (1 - skew * SR + (kurt - 1)/4 * SR^2) / (T - 1); at SR = 0 the skewness
      and kurtosis terms vanish and V = 1 / (T - 1) exactly.
    * N is the effective number of independent trials (see effective_trials).

    The test itself is unchanged, including its skewness/kurtosis correction.
    """
    r = returns.to_numpy()[: cfg.N_RESEARCH]
    daily = sharpe(r) / np.sqrt(cfg.PERIODS_PER_YEAR)
    benchmark = expected_max_sharpe(1 / (len(r) - 1), effective_trials(r))
    return probabilistic_sharpe(
        daily, benchmark, len(r), stats.skew(r), stats.kurtosis(r, fisher=False)
    )

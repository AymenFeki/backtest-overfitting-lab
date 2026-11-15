"""Automated look-ahead-bias test.

Idea: weights for days up to t may only depend on prices up to t. So if we
change the future, by cutting it off or by replacing it with different prices,
the weights up to t must stay exactly the same. A strategy that peeks (for
example `shift(-1)`, or standardising with the full-sample mean) fails.
"""

import numpy as np
import pandas as pd

from .backtest import Positions


def passes_lookahead_test(
    positions: Positions,
    prices: pd.DataFrame,
    cut_fractions: tuple[float, ...] = (0.5, 0.75, 0.9),
    seed: int = 0,
) -> bool:
    """True if weights up to each cut point ignore everything after it."""
    rng = np.random.default_rng(seed)
    reference = positions(prices).to_numpy()
    for fraction in cut_fractions:
        cut = int(len(prices) * fraction)
        truncated = prices.iloc[:cut]
        altered = prices.copy()
        altered.iloc[cut:] *= rng.uniform(0.5, 1.5, size=altered.iloc[cut:].shape)
        for variant in (truncated, altered):
            weights = positions(variant).to_numpy()[:cut]
            if not np.allclose(weights, reference[:cut], equal_nan=True):
                return False
    return True

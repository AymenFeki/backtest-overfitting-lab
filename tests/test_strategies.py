import numpy as np
import pandas as pd
import pytest

from overfitlab import config as cfg
from overfitlab.lookahead import passes_lookahead_test
from overfitlab.strategies import grid, oracle, trend
from overfitlab.worlds import world

PRICES = world(cfg.SPECS[0], 0, n_days=700)


def test_grid_has_76_unique_strategies():
    names = [s.name for s in grid()]
    assert len(names) == len(set(names)) == 76


@pytest.mark.parametrize("strategy", grid(), ids=lambda s: s.name)
def test_grid_strategy_is_causal_and_within_leverage(strategy):
    assert passes_lookahead_test(strategy.positions, PRICES)
    weights = strategy.positions(PRICES)
    assert weights.shape == PRICES.shape
    assert weights.abs().sum(axis=1).max() <= 1 + 1e-9
    assert weights.abs().sum().sum() > 0  # it does trade


def test_fade_is_the_mirror_of_follow():
    follow, fade = trend(PRICES, 20, 1), trend(PRICES, 20, -1)
    pd.testing.assert_frame_equal(follow, -fade)


def test_oracle_is_causal():
    spec = next(s for s in cfg.SPECS if s.name == "xsmom_medium")
    assert passes_lookahead_test(oracle(spec).positions, PRICES)


def test_lookahead_test_catches_peeking():
    def uses_tomorrow(prices: pd.DataFrame) -> pd.DataFrame:
        return np.sign(prices.pct_change().shift(-1)) / prices.shape[1]

    def uses_full_sample_mean(prices: pd.DataFrame) -> pd.DataFrame:
        return np.sign(prices - prices.mean()) / prices.shape[1]

    assert not passes_lookahead_test(uses_tomorrow, PRICES)
    assert not passes_lookahead_test(uses_full_sample_mean, PRICES)

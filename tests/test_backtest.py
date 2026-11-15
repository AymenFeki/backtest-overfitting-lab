import numpy as np
import pandas as pd
import pytest

from overfitlab.backtest import net_returns, sharpe


def test_net_returns_matches_hand_calculation():
    # Asset returns: day1 A +10%, day2 A -10%, day3 B +10%.
    prices = pd.DataFrame({"A": [100, 110, 99, 99], "B": [100, 100, 100, 110.0]})
    weights = pd.DataFrame({"A": [1.0, 0.5, 0.0, 0.0], "B": [0.0, 0.5, -1.0, 0.0]})
    # Held on day t = weights of day t-1: (0,0), (1,0), (.5,.5), (0,-1).
    # Gross: 0, +0.10, -0.05, -0.10. Turnover: 0, 1, 1, 2. Cost 10 bps each.
    expected = [0.0, 0.10 - 0.001, -0.05 - 0.001, -0.10 - 0.002]
    assert net_returns(prices, weights, cost_bps=10).tolist() == pytest.approx(expected)


def test_weights_are_lagged_one_day():
    # Weights that "know" today's move must not profit from it.
    prices = pd.DataFrame({"A": [100, 110, 121.0]})
    peeking = pd.DataFrame({"A": [0.0, 1.0, 0.0]})  # long only on day 1 itself
    result = net_returns(prices, peeking, cost_bps=0)
    assert result.tolist() == pytest.approx([0.0, 0.0, 0.10])  # earns day 2 instead


def test_leverage_is_capped_at_one():
    prices = pd.DataFrame({"A": [100, 110.0], "B": [100, 110.0]})
    weights = pd.DataFrame({"A": [2.0, 2.0], "B": [2.0, 2.0]})
    assert net_returns(prices, weights, cost_bps=0).iloc[1] == pytest.approx(0.10)


def test_sharpe_known_value_and_flat_series():
    # mean 0.005, sample sd 0.01291 -> daily 0.3873 -> annualised 6.148
    assert float(sharpe(np.array([0.01, -0.01, 0.02, 0.0]))) == pytest.approx(
        6.148, abs=1e-3
    )
    assert float(sharpe(np.zeros(10))) == 0.0

import numpy as np
import pytest

from overfitlab import config as cfg
from overfitlab.sandbox import check_code, run_on_paths, run_strategy
from overfitlab.worlds import world

GOOD = """
def positions(prices):
    return np.sign(prices.pct_change(20)) / prices.shape[1]
"""


@pytest.fixture(scope="module")
def prices():
    return world(cfg.SPECS[0], 0, n_days=600)


def test_a_plain_momentum_rule_runs_and_matches_a_direct_call(prices):
    weights, status = run_strategy(GOOD, prices)
    assert status == "ok"
    expected = (np.sign(prices.pct_change(20)) / prices.shape[1]).fillna(0.0)
    assert np.allclose(weights, expected)


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef positions(prices):\n    return prices",
        "def positions(prices):\n    open('x', 'w')\n    return prices",
        "def positions(prices):\n    prices.to_csv('x.csv')\n    return prices",
        "def positions(prices):\n    return pd.read_csv('x.csv')",
        "def positions(prices):\n    return prices.__class__",
        "def positions(prices):\n    return eval('prices')",
        "def signal(prices):\n    return prices",
        "def positions(prices):\n    return prices +",
    ],
)
def test_code_outside_the_allowed_set_is_rejected_without_running(code, prices):
    assert check_code(code) is not None
    weights, status = run_strategy(code, prices)
    assert weights is None and status.startswith("not allowed")


def test_look_ahead_is_caught(prices):
    peeking = "def positions(prices):\n    return np.sign(prices.shift(-1) - prices)"
    assert run_strategy(peeking, prices)[1].startswith("look-ahead")
    full_sample = "def positions(prices):\n    return prices - prices.mean()"
    assert run_strategy(full_sample, prices)[1].startswith("look-ahead")


def test_errors_wrong_shape_flat_and_slow_code_are_reported(prices):
    crash = "def positions(prices):\n    return prices.no_such_method()"
    assert run_strategy(crash, prices)[1].startswith("error: AttributeError")
    shape = "def positions(prices):\n    return prices.iloc[:, :3]"
    assert "shape" in run_strategy(shape, prices)[1]
    flat = "def positions(prices):\n    return prices * 0"
    assert "never trades" in run_strategy(flat, prices)[1]
    slow = "def positions(prices):\n    while True:\n        pass"
    assert run_strategy(slow, prices, timeout=2)[1].startswith("too slow")


def test_an_array_of_the_right_shape_is_accepted_and_nothing_else_is_fixed(prices):
    array = (
        "def positions(prices):\n    return np.sign(prices.pct_change(20)).to_numpy()"
    )
    weights, status = run_strategy(array, prices)
    assert status == "ok"
    assert weights.index.equals(prices.index) and list(weights.columns) == list(prices)
    assert np.allclose(weights, np.sign(prices.pct_change(20)).fillna(0.0))
    one_column = "def positions(prices):\n    return prices.pct_change(20).mean(axis=1)"
    assert "shape" in run_strategy(one_column, prices)[1]


def test_strategies_run_on_many_paths_and_a_failing_one_gives_none(prices):
    paths = np.stack([prices.to_numpy(), prices.to_numpy()[::-1]])
    crash = "def positions(prices):\n    return prices.no_such_method()"
    good, bad = run_on_paths([GOOD, crash], paths)
    assert bad is None and good.shape == paths.shape
    direct, _ = run_strategy(GOOD, prices)
    assert np.allclose(good[0], direct)

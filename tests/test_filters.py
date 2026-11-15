import numpy as np
import pandas as pd
import pytest
from scipy import stats

from overfitlab import config as cfg
from overfitlab.backtest import net_returns, sharpe
from overfitlab.filters import confidence, holdout, naive_sharpe, walk_forward
from overfitlab.strategies import trend
from overfitlab.worlds import draw_shocks, simulate, to_prices

NOISE = np.tile([0.005, -0.005], cfg.N_DAYS // 2)  # deterministic, sd 0.005


def test_confidence_matches_scipy_t_test():
    rng = np.random.default_rng(1)
    x = rng.normal(0.02, 1.0, size=(300, 4))
    expected = 1 - stats.ttest_1samp(x, 0.0, alternative="greater").pvalue
    assert confidence(x) == pytest.approx(expected)

    traded = np.zeros(x.shape, dtype=bool)
    traded[100:250] = True
    subset = 1 - stats.ttest_1samp(x[100:250], 0.0, alternative="greater").pvalue
    assert confidence(x, traded) == pytest.approx(subset)
    assert confidence(x, np.zeros(x.shape, dtype=bool)).tolist() == [0, 0, 0, 0]


def test_naive_sharpe_ignores_the_lockbox():
    returns = pd.DataFrame({"a": NOISE + 0.001, "b": NOISE})
    changed = returns.copy()
    changed.iloc[cfg.N_RESEARCH :] = -0.5
    expected = sharpe(returns.iloc[: cfg.N_RESEARCH])
    assert naive_sharpe(changed) == pytest.approx(expected)
    assert naive_sharpe(returns)[0] > 1 > naive_sharpe(returns)[1]


def test_holdout_needs_both_steps():
    good = NOISE + 0.001  # Sharpe about 3 everywhere
    fails_screen = NOISE.copy()
    fails_screen[cfg.N_RESEARCH :] += 0.002  # only good in the lockbox
    fails_lockbox = NOISE + 0.001
    fails_lockbox[cfg.N_RESEARCH :] -= 0.002  # good in sample, loses after
    scores = holdout(pd.DataFrame({"a": good, "b": fails_screen, "c": fails_lockbox}))
    assert scores[0] > 0.95
    assert scores[1] == 0.0
    assert scores[2] < 0.05


def test_walk_forward_follows_the_strategy_through_time():
    always_good = NOISE + 0.001
    turns_bad = NOISE + 0.001
    turns_bad[5 * 252 :] -= 0.002  # five good years, then it loses
    # turns_bad is traded in years 4-5 (wins) and 6-8 (loses) until its
    # running Sharpe drops below the screen, so its traded days lose overall.
    scores = walk_forward(pd.DataFrame({"a": always_good, "b": turns_bad, "c": NOISE}))
    assert scores[0] > 0.95
    assert scores[1] < 0.5
    assert scores[2] == 0.0  # never passed the screen, never traded


def test_lockbox_test_has_nominal_size_for_one_prespecified_strategy():
    """Known answer: on null data, a 5% test must reject about 5% of the time.

    One strategy fixed in advance, no costs (so its true mean is exactly zero),
    1,000 null worlds. Tolerance is 3 binomial standard errors: 0.05 +- 0.021.
    """
    rejections, n_worlds = 0, 1000
    for chunk in range(4):
        seeds = [cfg.seed("calibrate", 5000 + chunk * 250 + i) for i in range(250)]
        for returns in simulate(cfg.SPECS[0], draw_shocks(seeds, cfg.N_DAYS)):
            prices = to_prices(returns)
            lockbox = net_returns(prices, trend(prices, 60), cost_bps=0).to_numpy()
            rejections += confidence(lockbox[cfg.N_RESEARCH :, None])[0] >= 0.95
    assert abs(rejections / n_worlds - 0.05) < 3 * np.sqrt(0.05 * 0.95 / n_worlds)

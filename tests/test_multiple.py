import numpy as np
import pytest
from scipy import stats

from overfitlab import config as cfg
from overfitlab.filters import confidence
from overfitlab.multiple import (
    _block_sums,
    benjamini_hochberg,
    benjamini_yekutieli,
    romano_wolf,
)
from overfitlab.worlds import draw_shocks, simulate

# The 15 ordered p-values of the example in Benjamini & Hochberg (1995), p. 295.
PAPER_P = np.array(
    [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344]
    + [0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
)


def test_bh_reproduces_the_paper_example():
    # "we reject the four hypotheses having p-values ... less than 0.013";
    # Bonferroni at 0.05/15 rejects only three.
    assert (benjamini_hochberg(PAPER_P) <= 0.05).sum() == 4
    assert (PAPER_P <= 0.05 / 15).sum() == 3


def test_bh_matches_scipy_in_any_order():
    shuffled = np.random.default_rng(0).permutation(PAPER_P)
    expected = stats.false_discovery_control(shuffled, method="bh")
    assert benjamini_hochberg(shuffled) == pytest.approx(expected)


def test_by_is_bh_scaled_by_the_harmonic_sum():
    shuffled = np.random.default_rng(5).permutation(PAPER_P)
    expected = stats.false_discovery_control(shuffled, method="by")
    assert benjamini_yekutieli(shuffled) == pytest.approx(expected)
    # 1 + 1/2 + ... + 1/15 = 3.318, so the paper example keeps 3 of BH's 4.
    assert (benjamini_yekutieli(PAPER_P) <= 0.05).sum() == 3
    assert (benjamini_yekutieli(PAPER_P) >= benjamini_hochberg(PAPER_P)).all()


def test_bh_has_nominal_rate_when_nothing_is_true():
    """Known answer: with independent tests and no real effect, BH at 5%
    rejects something in 5% of repetitions. 4,000 repetitions, 3 SE = 0.010."""
    rng = np.random.default_rng(1)
    any_rejected = [
        (benjamini_hochberg(rng.uniform(size=20)) <= 0.05).any() for _ in range(4000)
    ]
    assert abs(np.mean(any_rejected) - 0.05) < 3 * np.sqrt(0.05 * 0.95 / 4000)


def test_block_sums_wrap_around():
    x = np.arange(10.0).reshape(5, 2)
    expected = np.array(
        [x[[i, (i + 1) % 5, (i + 2) % 5]].sum(axis=0) for i in range(5)]
    )
    assert _block_sums(x, 3) == pytest.approx(expected)


def test_romano_wolf_single_strategy_agrees_with_t_test():
    x = np.random.default_rng(2).normal(0.08, 1.0, size=(600, 1))
    assert romano_wolf(x)[0] == pytest.approx(1 - confidence(x)[0], abs=0.03)


def test_romano_wolf_finds_the_real_one_and_keeps_order():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1000, 12))
    x[:, 4] += 0.2  # t statistic about 6
    adjusted = romano_wolf(x)
    assert adjusted[4] < 0.01
    assert np.delete(adjusted, 4).min() > 0.05
    t = x.mean(axis=0) / x.std(axis=0, ddof=1)
    assert (np.diff(adjusted[np.argsort(-t)]) >= 0).all()  # monotone step-down


def test_romano_wolf_ignores_a_strategy_that_never_trades():
    x = np.random.default_rng(4).standard_normal((500, 3))
    x[:, 1] = 0.0
    assert romano_wolf(x)[1] == 1.0


def test_romano_wolf_controls_familywise_error_on_correlated_null_strategies():
    """Known answer: when no strategy has an edge, the chance of accepting even
    one must be about 5%, however correlated the strategies are.

    20 null "strategies" per world: the 10 correlated assets of a no-signal
    world (fat tails, volatility clustering) and their mirror images.
    1,000 worlds; tolerance 3 binomial standard errors = 0.021.
    """
    any_rejected, n_worlds = 0, 1000
    for chunk in range(4):
        seeds = [cfg.seed("calibrate", 9000 + chunk * 250 + i) for i in range(250)]
        for assets in simulate(cfg.SPECS[0], draw_shocks(seeds, cfg.N_RESEARCH)):
            adjusted = romano_wolf(np.hstack([assets, -assets]))
            any_rejected += adjusted.min() <= 0.05
    assert abs(any_rejected / n_worlds - 0.05) < 3 * np.sqrt(0.05 * 0.95 / n_worlds)

import numpy as np
import pandas as pd
import pytest

from overfitlab.dsr import (
    deflated_sharpe,
    deflated_sharpe_null,
    effective_trials,
    expected_max_sharpe,
    probabilistic_sharpe,
)

# Worked example in Bailey & Lopez de Prado (2014), p. 9-10: annualised Sharpe
# 2.5 on 5 years of daily data (250 days a year), N=100 trials with variance
# of annualised Sharpe 1/2, skewness -3, kurtosis 10.
DAILY_SHARPE = 2.5 / np.sqrt(250)
VARIANCE = 0.5 / 250


def test_expected_max_sharpe_matches_paper():
    assert expected_max_sharpe(VARIANCE, 100) == pytest.approx(0.1132, abs=5e-5)


def test_deflated_sharpe_matches_paper():
    benchmark = expected_max_sharpe(VARIANCE, 100)
    dsr = probabilistic_sharpe(DAILY_SHARPE, benchmark, 1250, -3, 10)
    assert dsr == pytest.approx(0.9004, abs=2e-4)  # "0.9004 < 0.95"


def test_paper_follow_up_numbers():
    # "after running only N=46 independent trials ... DSR would have been 0.9505"
    few_trials = expected_max_sharpe(VARIANCE, 46)
    assert probabilistic_sharpe(DAILY_SHARPE, few_trials, 1250, -3, 10) == (
        pytest.approx(0.9505, abs=2e-4)
    )
    # "Normal returns (skew 0, kurtosis 3), DSR = 0.9505 after N=88"
    normal = expected_max_sharpe(VARIANCE, 88)
    assert probabilistic_sharpe(DAILY_SHARPE, normal, 1250, 0, 3) == (
        pytest.approx(0.9505, abs=2e-4)
    )


def test_best_of_many_null_strategies_is_rarely_accepted():
    """Known answer: on noise, the best of 50 independent strategies should
    pass DSR > 0.95 in fewer than 5% of repetitions (a plain test of the best
    one would pass in about 92%: 1 - 0.95**50)."""
    rng = np.random.default_rng(7)
    accepted = 0
    for _ in range(1000):
        returns = pd.DataFrame(rng.standard_normal((500, 50)))
        accepted += deflated_sharpe(returns).max() > 0.95
    assert accepted / 1000 < 0.05


def test_effective_trials_known_cases():
    rng = np.random.default_rng(11)
    independent = rng.standard_normal((2000, 30))
    assert effective_trials(independent) == pytest.approx(30, rel=0.05)
    # A strategy and its mirror image are one trial, not two.
    mirrored = np.hstack([independent, -independent])
    assert effective_trials(mirrored) == pytest.approx(30, rel=0.05)
    # Thirty copies of the same strategy are one trial.
    copies = np.tile(independent[:, :1], (1, 30))
    assert effective_trials(copies) == pytest.approx(1, abs=1e-6)
    # A strategy that never trades is not a trial.
    with_flat = np.hstack([independent, np.zeros((2000, 1))])
    assert effective_trials(with_flat) == pytest.approx(effective_trials(independent))


def test_variant_rarely_accepts_noise_even_with_mirrored_pairs():
    """Known answer: on noise the best strategy should pass in under 5% of
    repetitions. 25 independent strategies plus their 25 mirror images."""
    rng = np.random.default_rng(12)
    accepted = 0
    for _ in range(1000):
        half = rng.standard_normal((500, 25))
        returns = pd.DataFrame(np.hstack([half, -half]))
        accepted += deflated_sharpe_null(returns).max() > 0.95
    assert accepted / 1000 < 0.05


def test_variant_accepts_a_strong_strategy_that_the_paper_version_rejects():
    # Real spread between trials: one strong strategy, its mirror, and noise.
    rng = np.random.default_rng(13)
    noise = rng.standard_normal((1500, 20)) * 0.01
    strong = 0.0015 + rng.standard_normal(1500) * 0.01  # annual Sharpe about 2.4
    returns = pd.DataFrame(np.column_stack([strong, -strong, noise]))
    assert deflated_sharpe_null(returns)[0] > 0.95
    assert deflated_sharpe(returns)[0] < deflated_sharpe_null(returns)[0]

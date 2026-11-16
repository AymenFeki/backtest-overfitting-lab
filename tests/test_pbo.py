from math import comb

import numpy as np
import pandas as pd
import pytest

from overfitlab.backtest import sharpe
from overfitlab.pbo import cscv_logits, pbo, pbo_filter


def test_number_of_combinations():
    returns = np.random.default_rng(0).standard_normal((320, 5))
    # The paper prints 12,780 for S = 16; C(16, 8) is 12,870.
    assert len(cscv_logits(returns, 16)) == comb(16, 8) == 12870
    assert len(cscv_logits(returns, 4)) == 6


def test_logit_is_the_out_of_sample_rank_of_the_in_sample_winner():
    # Strategy 0 always wins, in and out of sample: rank 3 of 3 everywhere.
    noise = np.tile([0.01, -0.01], 200)
    returns = np.column_stack([noise + 0.004, noise, noise - 0.004])
    logits = cscv_logits(returns, 4)
    assert logits == pytest.approx(np.log(0.75 / 0.25))  # omega = 3 / (3 + 1)
    assert pbo(returns, 4) == 0.0


def test_pbo_is_about_one_half_when_selection_is_worthless():
    """Known answer: with interchangeable noise strategies the in-sample winner
    is equally likely to land in either half out of sample."""
    rng = np.random.default_rng(1)
    values = [pbo(rng.standard_normal((800, 20))) for _ in range(60)]
    assert np.mean(values) == pytest.approx(0.5, abs=0.06)


def test_pbo_is_high_when_the_winner_reverses():
    # Each strategy is good in one half of the blocks and bad in the other,
    # so whoever wins in sample loses out of sample.
    rng = np.random.default_rng(2)
    blocks = rng.permuted(np.tile([1.0, -1.0], (30, 8)), axis=1).T  # (16, 30)
    returns = np.repeat(blocks, 50, axis=0) * 0.002
    returns = returns + rng.standard_normal(returns.shape) * 0.01
    assert pbo(returns) > 0.9


def test_filter_scores_only_the_best_strategy():
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.standard_normal((1764, 12)) * 0.01)
    scores = pbo_filter(returns)
    best = sharpe(returns).argmax()
    assert scores[best] == pytest.approx(1 - pbo(returns.to_numpy()))
    assert np.delete(scores, best).tolist() == [0.0] * 11

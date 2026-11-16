from math import comb

import numpy as np
import pandas as pd
import pytest

from overfitlab import config as cfg
from overfitlab.cpcv import GAP, N_GROUPS, N_TEST, cpcv, paths, splits, train_masks
from overfitlab.plots import STYLE
from overfitlab.registry import FILTERS

NOISE = np.tile([0.005, -0.005], cfg.N_RESEARCH // 2)  # deterministic, sd 0.005


@pytest.mark.parametrize(("n_groups", "n_test"), [(6, 2), (10, 3)])
def test_split_and_path_counts(n_groups, n_test):
    # 6 groups, 2 test -> 15 splits and 5 paths; 10 and 3 -> 120 and 36.
    n_splits, n_paths = comb(n_groups, n_test), comb(n_groups - 1, n_test - 1)
    assert n_paths == n_test * n_splits // n_groups  # the book's k/N * C(N,k)
    assert len(splits(n_groups, n_test)) == n_splits
    table = paths(n_groups, n_test)
    assert table.shape == (n_paths, n_groups)
    # Every (split, test group) pair is used on exactly one path.
    for group in range(n_groups):
        tested_in = [i for i, s in enumerate(splits(n_groups, n_test)) if group in s]
        assert sorted(table[:, group]) == tested_in


def test_counts_for_the_setup_used():
    assert (len(splits(N_GROUPS, N_TEST)), len(paths(N_GROUPS, N_TEST))) == (120, 36)


def test_training_days_keep_their_distance_from_test_groups():
    n_days = cfg.N_RESEARCH
    masks = train_masks(n_days, N_GROUPS, N_TEST, GAP)
    edges = np.linspace(0, n_days, N_GROUPS + 1).astype(int)
    days = np.arange(n_days)
    for mask, tested in zip(masks, splits(N_GROUPS, N_TEST), strict=True):
        for group in tested:
            start, end = edges[group], edges[group + 1]
            near_or_inside = (days >= start - GAP) & (days < end + GAP)
            assert not mask[near_or_inside].any()
    # Without a gap, training days are exactly the days outside the test groups.
    no_gap = train_masks(n_days, N_GROUPS, N_TEST, 0)
    assert no_gap.sum(axis=1).tolist() == pytest.approx([n_days * 7 / 10] * 120, abs=3)


def test_cpcv_rule_on_constructed_strategies():
    always_good = NOISE + 0.001
    turns_bad = NOISE + 0.001
    turns_bad[cfg.N_RESEARCH // 2 :] -= 0.002  # good first half, loses after
    scores = cpcv(pd.DataFrame({"a": always_good, "b": turns_bad, "c": NOISE}))
    assert scores[0] > 0  # accepted: every path wins
    assert scores[1] < 0  # more than one path loses
    assert scores[2] == 0  # never passes the screen, never traded: not accepted


def test_split_with_too_little_training_data_trades_nothing():
    # A gap wider than the sample leaves no training days in any split.
    always_good = pd.DataFrame({"a": NOISE + 0.001})
    assert cpcv(always_good, gap=10_000)[0] == 0
    assert cpcv(always_good)[0] > 0


def test_every_filter_has_a_unique_name_and_a_plot_style():
    names = [f.name for f in FILTERS]
    assert len(names) == len(set(names)) == 9
    assert set(names) == set(STYLE)

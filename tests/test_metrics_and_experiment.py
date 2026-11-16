import numpy as np

from overfitlab import config as cfg
from overfitlab import experiments
from overfitlab.metrics import fdp_and_power, mean_ci
from overfitlab.registry import FILTERS


def test_fdp_and_power_hand_example():
    accepted = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]], dtype=bool)
    is_true = np.array([True, False, True, False])
    fdp, power = fdp_and_power(accepted, is_true)
    assert fdp.tolist() == [0.5, 0.0, 0.0]  # nothing accepted counts as FDP 0
    assert power.tolist() == [0.5, 0.0, 1.0]


def test_null_world_every_acceptance_is_false_and_power_undefined():
    accepted = np.array([[1, 0], [0, 0]], dtype=bool)
    fdp, power = fdp_and_power(accepted, np.array([False, False]))
    assert fdp.tolist() == [1.0, 0.0]
    assert np.isnan(power).all()


def test_mean_ci_is_deterministic_and_brackets_the_mean():
    values = np.array([0.0, 1.0, 0.0, 1.0, 1.0, np.nan])
    mean, low, high = mean_ci(values)
    assert mean == 0.6  # the NaN is dropped
    assert low < mean < high
    assert mean_ci(values) == (mean, low, high)
    assert mean_ci(np.full(5, 0.3)) == (0.3, 0.3, 0.3)


def test_experiment_runs_end_to_end_on_a_tiny_setup(monkeypatch):
    specs = [cfg.SPECS[0], next(s for s in cfg.SPECS if s.name == "tsmom_obvious")]
    monkeypatch.setattr(cfg, "SPECS", specs)
    monkeypatch.setattr(cfg, "TRUTH_YEARS", 3)

    # Truth from 9 simulated years is far too noisy to label strategies; this
    # test only checks that the pieces fit together.
    truth = experiments.true_sharpes(n_paths=3)
    assert len(truth) == 2 * 77  # 76 grid strategies + the oracle, per world type

    scores = experiments.grid_scores(n_worlds=4)
    assert scores.shape == (2, 4, len(FILTERS), 76)

    table = experiments.summarise(scores, truth, threshold=0.2, sweep=False)
    assert set(table.worlds) == {"no_signal", "tsmom_obvious", "all_obvious"}
    assert table.fdr.between(0, 1).all()

    # With an unreachable threshold no strategy is true: power is undefined
    # and every accepted strategy is a false discovery.
    nothing_true = experiments.summarise(scores, truth, threshold=99.0, sweep=False)
    assert nothing_true.power.isna().all()
    assert (nothing_true.false_per_world == nothing_true.accepted_per_world).all()

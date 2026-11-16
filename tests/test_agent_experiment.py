import numpy as np
import pandas as pd

from overfitlab import config as cfg
from overfitlab.agent_experiment import agent_seed, summarise_agent
from overfitlab.registry import FILTERS


def _run(spec_name: str, naive_scores: list[float]) -> dict:
    scores = np.full((len(FILTERS), len(naive_scores)), -np.inf)
    scores[0] = naive_scores  # filter 0 is the naive rule: accept above 1.0
    spec = [s.name for s in cfg.SPECS].index(spec_name)
    rounds = list(range(1, len(naive_scores) + 1))
    return {"spec": spec, "world": 0, "rounds": rounds, "scores": scores}


def test_agent_summary_judges_each_world_by_its_own_strategies():
    runs = [_run("no_signal", [2.0, 0.0]), _run("tsmom_medium", [2.0, 2.0, 0.0])]
    truth = pd.DataFrame(
        {
            "world_type": ["no_signal"] * 2 + ["tsmom_medium"] * 3,
            "world": 0,
            "round": [1, 2, 1, 2, 3],
            "true_sharpe": [-0.1, -0.3, 0.5, 0.1, 0.4],
        }
    )
    table = summarise_agent(runs, truth, threshold=0.2)
    table = table.set_index(["worlds", "filter"])

    noise = table.loc[("no_signal", "naive")]  # one acceptance, and it is false
    assert noise.fdr == 1.0 and noise.pfdr == 1.0 and np.isnan(noise.power)
    # Accepted rounds 1 and 2: one real (0.5), one not (0.1). Real ones: 1 and 3.
    signal = table.loc[("tsmom_medium", "naive")]
    assert signal.fdr == 0.5 and signal.power == 0.5 and signal.true_per_world == 2
    assert table.loc[("all_medium", "naive")].fdr == 0.5
    quiet = table.loc[("tsmom_medium", "holdout")]  # accepts nothing
    assert quiet.fdr == 0.0 and np.isnan(quiet.pfdr) and quiet.power == 0.0


def test_agent_seed_matches_the_pilot_runs():
    # The pilot ran world 0 of two world types with seed = spec index x 1000,
    # so its stored replies are reused as the first runs of the experiment.
    assert agent_seed(3, 0) == 3000 and agent_seed(3, 7) == 3007

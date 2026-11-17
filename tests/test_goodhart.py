import numpy as np
import pandas as pd

from overfitlab import agent, goodhart
from overfitlab import config as cfg
from overfitlab.goodhart import SYSTEMS, checks_passed, describe, summarise_goodhart
from overfitlab.llm import Model, Reply
from overfitlab.registry import FILTERS


def _scripted(model, system, messages, seed, live=False):
    """A stand-in for the model: a different trend rule every round."""
    lookback = 3 + seed // 10 % 100 * 4
    return Reply(
        f"Hypothesis: trend over {lookback} days.\n```python\n"
        f"def positions(prices):\n    return np.sign(prices.pct_change({lookback}))\n```",
        100,
        50,
        0.0,
    )


def test_both_arms_get_the_same_task_and_only_one_the_descriptions():
    told, not_told = SYSTEMS["told"], SYSTEMS["not_told"]
    assert told.startswith(not_told) and not_told.startswith(agent.SYSTEM)
    assert "passes all six checks" in not_told
    assert "deflated" not in not_told and "holdout" not in not_told
    for word in ("holdout", "walk-forward", "deflated", "Hochberg", "Romano-Wolf"):
        assert word in told
    assert [f.name for f in goodhart.CHECKS] == list(cfg.GOODHART_CHECKS)


def test_agent_is_shown_which_checks_each_round_passes(monkeypatch):
    seen = []

    def fake(model, system, messages, seed, live=False):
        seen.append((system, messages[-1]["content"]))
        return _scripted(model, system, messages, seed)

    monkeypatch.setattr(agent, "complete", fake)
    monkeypatch.setattr(cfg, "AGENT_ROUNDS", 3)
    model = Model("ollama", "fake:1b")
    run = goodhart._run((model, "told", 0, False))
    assert run["arm"] == "told" and run["rounds"] == [1, 2, 3]
    assert run["scores"].shape == (len(FILTERS), 3)
    assert seen[0][0] == SYSTEMS["told"]
    # Round 2 sees one strategy: no family to correct for yet, so no check
    # is passed. Round 3 sees two rows with the same column.
    assert "round | status | Sharpe | checks passed | hypothesis" in seen[1][1]
    assert "1 | ok | " in seen[1][1] and " | none | trend over" in seen[1][1]
    assert seen[2][1].count(" | none | ") + seen[2][1].count(" | A") >= 2


def test_checks_passed_marks_failed_rounds_and_agrees_with_the_filters():
    rng = np.random.default_rng(0)
    days = cfg.N_DAYS

    def trial(number: int, drift: float, ok: bool = True) -> agent.Trial:
        returns = pd.Series(rng.normal(drift, 0.01, days)) if ok else None
        return agent.Trial(
            number, "h", "code", "ok" if ok else "error", returns, [], []
        )

    # Round 1 earns 0.2% a day (a Sharpe above 3: passes everything), round 2
    # is noise, round 3 failed to run.
    cells = checks_passed([trial(1, 0.002), trial(2, 0.0), trial(3, 0.0, ok=False)])
    assert cells == ["A B C D E F", "none", "-"]
    assert checks_passed([trial(1, 0.002)]) == ["none"]  # one strategy alone


def _run(arm: str, world: int, naive: list[float], holdout: list[float]) -> dict:
    scores = np.full((len(FILTERS), len(naive)), -np.inf)
    scores[0], scores[1] = naive, holdout  # filters 0 and 1: naive, holdout
    return {
        "arm": arm,
        "world": world,
        "rounds": list(range(1, len(naive) + 1)),
        "scores": scores,
        "research_sharpe": np.array(naive),
        "attempts": len(naive),
        "distinct": len(naive),
        "mean_abs_correlation": 0.1,
        "cost": 0.0,
    }


def test_summary_counts_false_discoveries_and_compares_the_arms_world_by_world():
    runs = [
        _run("told", 0, [2.0, 2.0, 0.0], [0.99, 0.0, 0.0]),
        _run("told", 1, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _run("not_told", 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _run("not_told", 1, [2.0, 0.0, 0.0], [0.99, 0.0, 0.0]),
    ]
    table = summarise_goodhart(runs).set_index(["arm", "filter"])
    # Naive accepts a Sharpe above 1.0: told 2 and 1, not told 0 and 1.
    assert table.loc[("told", "naive")].accepted_per_world == 1.5
    assert table.loc[("told", "naive")].worlds_accepting == 1.0
    assert table.loc[("not_told", "naive")].worlds_accepting == 0.5
    gap = table.loc[("told_minus_not_told", "naive")]
    assert gap.accepted_per_world == 1.0 and gap.worlds_accepting == 0.5
    assert gap.n_worlds == 2
    # Holdout: one world each, but not the same one: the gap is zero.
    assert table.loc[("told_minus_not_told", "holdout")].accepted_per_world == 0.0
    # Nothing passed all six checks (the other four have no score here).
    assert table.loc[("told", "all_checks")].accepted_per_world == 0.0
    per_run = describe(runs)
    assert list(per_run.usable) == [3, 3, 3, 3]
    assert per_run.best_research_sharpe.max() == 2.0

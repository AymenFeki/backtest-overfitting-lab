import numpy as np
import pandas as pd
import pytest

from overfitlab import agent, holdout_reuse
from overfitlab import config as cfg
from overfitlab.holdout_reuse import (
    MECHANISMS,
    ONCE,
    REUSE,
    THRESHOLDOUT,
    Mechanism,
    agent_runs,
    answer_sources,
    candidates,
    hill_climb,
    summarise_reuse,
)
from overfitlab.llm import Model, Reply
from overfitlab.strategies import grid, grid_neighbours
from overfitlab.worlds import world

LINE = [[j for j in (i - 1, i + 1) if 0 <= j < 30] for i in range(30)]


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_hill_climber_reaches_the_top_of_a_single_hill():
    # 30 strategies in a row, each next to its two neighbours; the answer
    # falls with the distance from strategy 20. Without jumps the climber
    # can only walk, and it must arrive whatever the starting point.
    for seed in range(10):
        proposals, answers = hill_climb(
            lambda i: -abs(i - 20), LINE, rounds=60, rng=_rng(seed), jump=0.0
        )
        assert proposals[int(np.argmax(answers))] == 20
        assert len(set(proposals)) == len(proposals) == 30  # each one asked once


def test_hill_climber_only_moves_to_a_higher_answer():
    noise = _rng(1).normal(size=30)
    proposals, answers = hill_climb(lambda i: noise[i], LINE, 20, _rng(2), jump=0.0)
    best = np.maximum.accumulate(answers)
    for k in range(1, len(proposals)):
        # Every proposal is a neighbour of the best strategy at that time,
        # or a fresh start once both of its neighbours have been tried.
        holder = proposals[int(np.argmax(answers[:k]))]
        tried = set(proposals[:k])
        assert proposals[k] in LINE[holder] or set(LINE[holder]) <= tried
        assert best[k] >= best[k - 1]


def test_hill_climber_is_reproducible_and_ignores_missing_answers():
    def run(seed: int) -> tuple:
        return hill_climb(lambda i: float(i % 7), LINE, 15, _rng(seed))

    assert run(5) == run(5) and run(5) != run(6)
    # "No answer" for everything except strategy 3: it stays the best.
    proposals, answers = hill_climb(
        lambda i: 1.0 if i == 3 else None, LINE, rounds=30, rng=_rng(0)
    )
    assert answers.count(None) == 29 and len(proposals) == 30


def test_grid_neighbours_are_the_mirror_and_the_adjacent_settings():
    names = [s.name for s in grid()]
    near = grid_neighbours()
    assert len(near) == len(names) == 76
    of = {names[i]: {names[j] for j in near[i]} for i in range(len(names))}
    assert of["trend_20_follow"] == {
        "trend_20_fade",
        "trend_15_follow",
        "trend_40_follow",
    }
    # The first and last setting of a family have one neighbour less, and a
    # neighbour never belongs to another family.
    assert of["trend_2_fade"] == {"trend_2_follow", "trend_3_fade"}
    assert of["trend_250_follow"] == {"trend_250_fade", "trend_180_follow"}
    assert all(i in near[j] for i in range(76) for j in near[i])  # symmetric


def _noise_table(mechanisms: list[Mechanism], n_worlds: int) -> pd.DataFrame:
    """The hill-climber on pure-noise Sharpe ratios, without any backtest:
    each strategy's research and lockbox Sharpe are independent draws with
    the standard errors of 7 and 3 years of data."""
    rows = []
    near = grid_neighbours()
    research_se = np.sqrt(cfg.PERIODS_PER_YEAR / cfg.N_RESEARCH)
    for index in range(n_worlds):
        rng = _rng(index)
        research = rng.normal(0.0, research_se, len(near))
        lockbox = rng.normal(0.0, cfg.LOCKBOX_SHARPE_SE, len(near))
        for mechanism in mechanisms:
            ask = mechanism.ask(_rng(10_000 + index))
            proposals, answers = hill_climb(
                lambda i, ask=ask, research=research, lockbox=lockbox: ask(
                    research[i], lockbox[i]
                ),
                near,
                cfg.REUSE_ROUNDS,
                _rng(index),
            )
            for number, (i, a) in enumerate(zip(proposals, answers, strict=True), 1):
                rows.append(
                    {
                        "world_type": "noise",
                        "world": index,
                        "mechanism": mechanism.name,
                        "round": number,
                        "answer": np.nan if a is None else a,
                        "research": research[i],
                        "lockbox": lockbox[i],
                        "true_sharpe": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_a_holdout_used_once_keeps_its_level_and_a_reused_one_loses_it():
    n = 1000
    table = summarise_reuse(_noise_table([ONCE, REUSE], n))
    last = table[table["round"] == cfg.REUSE_ROUNDS].set_index("mechanism")
    # Used once: the claim is one test at the 5% level (the bar is rounded
    # to 0.95, which is the 5.0% point), whatever the search did before.
    level = 0.05
    assert abs(last.false_claims["once"] - level) < 3 * np.sqrt(level * 0.95 / n)
    # Reused 40 times: the best of 40 lockbox values clears the bar far more
    # often, and what the analyst is told overstates the truth (zero).
    assert last.false_claims["reuse"] > 0.4
    assert last.reported["reuse"] > 0.9 and abs(last.reported["once"]) < 0.06
    first = table[table["round"] == 1].set_index("mechanism")
    assert abs(first.false_claims["reuse"] - level) < 3 * np.sqrt(level * 0.95 / n)
    assert (last.true_claims == 0).all()


def test_thresholdout_stops_answering_when_its_budget_is_used_up():
    tight = Mechanism("thresholdout", "thresholdout", threshold=0.5, budget=3)
    table = _noise_table([tight], 50)
    per_world = candidates(table, threshold=0.2).groupby("world").answered.max()
    assert per_world.max() < cfg.REUSE_ROUNDS  # every world ran out of budget
    # After the last answer the candidate can no longer change.
    for _, run in table.groupby("world"):
        asked = run.answer.notna().to_numpy()
        assert not asked[np.argmin(asked) :].any()  # no answer after the first gap


def test_answer_sources_count_lockbox_answers_and_find_the_spent_budget():
    named = {m.name: m for m in MECHANISMS}
    mechanisms = [ONCE, REUSE, named["thresholdout_b5"]]
    table = _noise_table(mechanisms, 30)
    out = answer_sources(table).set_index("mechanism")
    assert out.share_from_lockbox["once"] == 0.0
    assert out.share_from_lockbox["reuse"] == 1.0
    assert 0.0 < out.share_from_lockbox["thresholdout_b5"] < 1.0
    assert np.isnan(out.budget_round_median["reuse"])
    # Check one run by hand: the budget is spent at the 5th lockbox answer,
    # and nothing is answered after it.
    for _, run in table[table.mechanism == "thresholdout_b5"].groupby("world"):
        from_lockbox = run.answer.notna() & (run.answer != run.research)
        if from_lockbox.sum() == 5:
            spent = run["round"][from_lockbox].iloc[4]
            assert run.answer[run["round"] > spent].isna().all()
            break
    else:
        raise AssertionError("no run spent its budget")
    assert out.runs_budget_spent["thresholdout_b5"] > 0


def test_a_longer_lockbox_extends_the_same_world():
    # The search must see the same research Sharpe ratios with either
    # lockbox, so the longer world has to start with the shorter one.
    long = next(m for m in MECHANISMS if m.name == "thresholdout_long")
    assert np.isclose(long.se, THRESHOLDOUT.se / 2)
    assert long.claim == 0.47 and THRESHOLDOUT.claim == cfg.CLAIM_SHARPE
    for spec in [cfg.SPECS[0], next(s for s in cfg.SPECS if s.name == "tsmom_medium")]:
        short = world(spec, 3)
        longer = world(spec, 3, cfg.N_RESEARCH + cfg.LONG_LOCKBOX_DAYS)
        assert len(longer) == cfg.N_RESEARCH + cfg.LONG_LOCKBOX_DAYS
        pd.testing.assert_frame_equal(longer.iloc[: len(short)], short)


def test_claims_on_the_actual_lockbox_sharpe():
    table = pd.DataFrame(
        {
            "world_type": "w",
            "world": 0,
            "mechanism": "thresholdout",
            "round": [1, 2],
            "answer": [1.2, 0.3],  # told 1.2 about a strategy whose lockbox is 0.5
            "research": [0.0, 0.3],
            "lockbox": [0.5, 2.0],
            "true_sharpe": [0.0, 0.0],
        }
    )
    out = candidates(table, threshold=0.2)
    assert list(out.false_claim) == [True, True]
    assert list(out.lockbox_false_claim) == [False, False]


def test_candidates_follow_the_highest_answer_and_judge_it_by_the_truth():
    def rows(mechanism: str, answers: list, lockbox: list, true: list) -> list:
        return [
            {
                "world_type": "w",
                "world": 0,
                "mechanism": mechanism,
                "round": k + 1,
                "answer": a,
                "research": 0.0,
                "lockbox": box,
                "true_sharpe": t,
            }
            for k, (a, box, t) in enumerate(zip(answers, lockbox, true, strict=True))
        ]

    table = pd.DataFrame(
        rows("reuse", [np.nan, 1.2, 0.5, 2.0], [0, 1.2, 0.5, 2.0], [0, 0.0, 0.9, 0.9])
        + rows("once", [1.5, 0.3, 2.0, np.nan], [0.1, 3.0, 1.0, 0], [0.9, 0, 0.9, 0])
    )
    out = candidates(table, threshold=0.2).set_index(["mechanism", "round"])
    reuse, once = out.loc["reuse"], out.loc["once"]
    # reuse: no candidate yet, then round 2 (false) until round 4 (real) wins.
    assert list(reuse.false_claim) == [False, True, True, False]
    assert list(reuse.true_claim) == [False, False, False, True]
    assert np.isnan(reuse.reported[1]) and list(reuse.reported[1:]) == [1.2, 1.2, 2.0]
    assert list(reuse.answered) == [0, 1, 2, 3]
    # once: the search follows the answers (research Sharpe), the claim is
    # decided by the candidate's lockbox Sharpe: 0.1 fails, 1.0 passes.
    assert list(once.reported) == [0.1, 0.1, 1.0, 1.0]
    assert list(once.true_claim) == [False, False, True, True]


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


def test_agent_sees_the_mechanisms_answers_as_validation_sharpe(monkeypatch):
    seen = []

    def fake(model, system, messages, seed, live=False):
        seen.append((system, messages[-1]["content"]))
        return _scripted(model, system, messages, seed)

    monkeypatch.setattr(agent, "complete", fake)
    monkeypatch.setattr(cfg, "REUSE_ROUNDS", 4)
    table = agent_runs(Model("ollama", "fake:1b"), n_seeds=1)
    assert list(table.mechanism) == ["reuse"] * 4 + ["thresholdout"] * 4
    assert "validation Sharpe ratio is above 0.95" in seen[0][0]
    reuse = table[table.mechanism == "reuse"]
    # Plain reuse: the answer is the lockbox Sharpe, shown with two decimals.
    assert np.allclose(reuse.answer, reuse.lockbox)
    last = seen[3][1]
    assert "round | status | Sharpe | validation Sharpe | hypothesis" in last
    for row in reuse.itertuples():
        if row.round < 4:
            cells = f"{row.round} | ok | {row.research:.2f} | {row.answer:.2f} | "
            assert cells in last
    # Thresholdout: every answer is the research Sharpe or a noisy lockbox one.
    guarded = table[table.mechanism == "thresholdout"]
    same = np.isclose(guarded.answer, guarded.research)
    near = np.abs(guarded.answer - guarded.lockbox) < 0.5
    assert (same | near).all()
    # Both arms start from the same request, so they share the first reply.
    assert seen[0] == seen[4]
    summary = summarise_reuse(table)
    assert set(summary.mechanism) == {"reuse", "thresholdout"}
    assert summary.n_worlds.eq(1).all() and summary.true_claims.eq(0).all()
    assert holdout_reuse.AGENT_MECHANISMS == [REUSE, THRESHOLDOUT]


def test_paired_test_counts_discordant_worlds_and_gives_the_exact_p_value():
    # Reuse claims in worlds 0 to 3, Thresholdout in world 0 only: three
    # discordant worlds, all one way, p = 2 x 0.5^3 = 0.25.
    rows = [
        {
            "world_type": "w",
            "world": world_index,
            "mechanism": name,
            "round": 1,
            "answer": answer,
            "research": 0.0,
            "lockbox": answer,
            "true_sharpe": 0.0,
        }
        for world_index in range(5)
        for name, answer in [
            ("reuse", 1.5 if world_index < 4 else 0.0),
            ("thresholdout", 1.5 if world_index == 0 else 0.0),
        ]
    ]
    out = holdout_reuse.paired_test(pd.DataFrame(rows), "reuse", "thresholdout", 1)
    told = out.set_index("rule").loc["false_claim"]
    assert (told.reuse_claims, told.thresholdout_claims) == (4, 1)
    assert (told.only_reuse, told.only_thresholdout) == (3, 0)
    assert told.p_value == pytest.approx(0.25)

import pytest

from overfitlab import agent, llm
from overfitlab import config as cfg
from overfitlab.agent import history_table, parse, run_agent
from overfitlab.llm import CacheMiss, Model, Reply, complete
from overfitlab.worlds import world

MOMENTUM = (
    "Hypothesis: trends persist.\n```python\ndef positions(prices):\n"
    "    return np.sign(prices.pct_change(60))\n```"
)
REVERSAL = (
    "Hypothesis: moves reverse.\n```python\ndef positions(prices):\n"
    "    return -np.sign(prices.pct_change(5))\n```"
)
PEEKING = (
    "Hypothesis: tomorrow is known.\n```python\ndef positions(prices):\n"
    "    return np.sign(prices.shift(-1) - prices)\n```"
)


def test_parse_reads_hypothesis_and_code():
    assert parse(MOMENTUM) == (
        "trends persist.",
        "def positions(prices):\n    return np.sign(prices.pct_change(60))",
    )
    assert parse("I think momentum works.") == ("", "")


def test_replies_are_stored_and_never_fetched_twice(tmp_path, monkeypatch):
    calls = []

    def fake(model, system, messages, seed):
        calls.append(seed)
        return Reply("hello", 10, 2, 0.1)

    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "_ollama", fake)
    model = Model("ollama", "fake:1b")
    question = [{"role": "user", "content": "hi"}]
    with pytest.raises(CacheMiss):  # nothing stored, live calls off
        complete(model, "system", question, seed=1)
    assert complete(model, "system", question, seed=1, live=True).text == "hello"
    assert complete(model, "system", question, seed=1).text == "hello"
    llm._stored.cache_clear()  # a new process reads the file
    assert complete(model, "system", question, seed=1).text == "hello"
    assert calls == [1]
    with pytest.raises(CacheMiss):  # a different seed is a different request
        complete(model, "system", question, seed=2)


def test_api_call_is_refused_before_the_budget_is_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cfg, "API_BUDGET_USD", 0.0)
    model = Model("anthropic", cfg.API_MODEL)
    with pytest.raises(llm.BudgetExceeded):
        complete(model, "system", [{"role": "user", "content": "hi"}], 1, live=True)


def test_agent_loop_feeds_results_back_and_allows_two_repairs(monkeypatch):
    # Round 1 works. Round 2 peeks, then is repaired. Round 3 fails three
    # times (no code, a repeat of round 1, look-ahead) and stays unusable.
    script = [MOMENTUM, PEEKING, REVERSAL, "no code here", MOMENTUM, PEEKING]
    seen = []

    def fake(model, system, messages, seed, live=False):
        seen.append(messages[-1]["content"])
        return Reply(script[len(seen) - 1], 0, 0, 0.0)

    monkeypatch.setattr(agent, "complete", fake)
    prices = world(cfg.SPECS[0], 0)
    model = Model("ollama", "fake:1b")
    trials = run_agent(prices, model, seed=0, rounds=3, history="transcript")
    assert [t.status.split(":")[0] for t in trials] == ["ok", "ok", "look-ahead"]
    assert [len(t.replies) for t in trials] == [1, 2, 3]
    assert trials[1].hypothesis == "moves reverse." and trials[2].returns is None
    assert len(trials[0].returns) == cfg.N_DAYS
    assert seen[0].startswith("Market summary: 1763 trading days, 10 assets")
    assert "Round 1 result: Sharpe ratio after costs" in seen[1]
    assert seen[2].startswith("That code cannot be used: look-ahead")
    assert "Round 2 result" in seen[3] and "Best so far: round" in seen[3]
    assert seen[4].startswith("That code cannot be used: no Python code block")
    assert seen[5].startswith("That code cannot be used: identical to round 1")


def test_compact_history_shows_every_past_round_in_a_fresh_conversation(monkeypatch):
    script = [MOMENTUM, PEEKING, PEEKING, PEEKING, REVERSAL]
    seen = []

    def fake(model, system, messages, seed, live=False):
        seen.append(list(messages))
        return Reply(script[len(seen) - 1], 0, 0, 0.0)

    monkeypatch.setattr(agent, "complete", fake)
    prices = world(cfg.SPECS[0], 0)
    trials = run_agent(prices, Model("ollama", "fake:1b"), seed=0, rounds=3)
    assert [t.status.split(":")[0] for t in trials] == ["ok", "look-ahead", "ok"]
    assert [len(f) for f in (t.failures for t in trials)] == [0, 3, 0]
    # Round 1: nothing to show. Round 3 starts over with one message that
    # lists both earlier rounds and the error of the failed one.
    assert len(seen[0]) == 1 and "No rounds yet." in seen[0][0]["content"]
    assert len(seen[4]) == 1
    last = seen[4][0]["content"]
    assert last.startswith("Market summary: 1763 trading days")
    assert "1 | ok | " in last and "| trends persist." in last
    assert "2 | failed | - | tomorrow is known." in last
    assert "Round 2 failed with: look-ahead" in last
    assert last.endswith("Round 3 of 3: propose a new strategy.")
    assert history_table(trials).count("\n") == 4  # header, columns, 3 rounds

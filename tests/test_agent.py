import pytest

from overfitlab import agent, llm
from overfitlab import config as cfg
from overfitlab.agent import parse, run_agent
from overfitlab.llm import CacheMiss, Model, Reply, complete
from overfitlab.worlds import world

MOMENTUM = (
    "Hypothesis: trends persist.\n```python\ndef positions(prices):\n"
    "    return np.sign(prices.pct_change(60))\n```"
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


def test_agent_loop_feeds_results_back_and_flags_bad_proposals(monkeypatch):
    script = [MOMENTUM, PEEKING, "no code here", MOMENTUM]
    seen = []

    def fake(model, system, messages, seed, live=False):
        seen.append(messages[-1]["content"])
        return Reply(script[len(seen) - 1], 0, 0, 0.0)

    monkeypatch.setattr(agent, "complete", fake)
    prices = world(cfg.SPECS[0], 0)
    trials = run_agent(prices, Model("ollama", "fake:1b"), seed=0, rounds=4)
    assert [t.status.split(":")[0] for t in trials] == [
        "ok",
        "look-ahead",
        "no Python code block in the reply",
        "identical to round 1",
    ]
    assert len(trials[0].returns) == cfg.N_DAYS and trials[1].returns is None
    assert seen[0].startswith("Market summary: 1763 trading days, 10 assets")
    assert "Round 1 result: Sharpe ratio after costs" in seen[1]
    assert "Round 2 could not be used: look-ahead" in seen[2]

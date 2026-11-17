"""Zero-cost checks: what the results would be without transaction costs.

Every strategy is run a second time with the same positions and the cost set
to 0, through the same engine (one-day lag, exposure capped at 100%). Three
questions from the Stage 4 and 5 notes:

1. Stage 4, no-signal worlds: research Sharpe before and after costs, turnover,
   and which filters accept something, for the agent's and the grid's
   strategies.
2. Experiment 2: why the lockbox "opened once" claims less than 5% on noise.
3. Experiment 3: whether costs are why no strategy ever passed a check.

No model call: the agent strategies are replayed from the stored replies.
Writes results/zero_cost.md. Takes about 10 minutes with 6 workers.

    uv run python scripts/zero_cost.py --jobs 6
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from overfitlab import config as cfg
from overfitlab import goodhart
from overfitlab.agent import Column, run_agent
from overfitlab.agent_experiment import agent_seed
from overfitlab.backtest import net_returns, sharpe
from overfitlab.experiments import GRID, _parallel
from overfitlab.holdout_reuse import ONCE, candidates, climb
from overfitlab.llm import Model
from overfitlab.metrics import mean_ci
from overfitlab.registry import FILTERS
from overfitlab.sandbox import run_strategy
from overfitlab.worlds import world

MODEL = Model(cfg.AGENT_PROVIDER, cfg.API_MODEL)
NO_SIGNAL = cfg.SPECS[0]
R = cfg.N_RESEARCH
OUT = Path("results/zero_cost.md")


def turnover(weights: pd.DataFrame) -> float:
    """One-way turnover per year on the research sample, as the engine counts it."""
    weights = weights.fillna(0.0)
    leverage = weights.abs().sum(axis=1).clip(lower=1.0)
    held = weights.div(leverage, axis=0).shift(1).fillna(0.0)
    return float(held.diff().abs().sum(axis=1).iloc[:R].mean() * cfg.PERIODS_PER_YEAR)


def score_world(prices: pd.DataFrame, positions: dict, label: dict) -> tuple:
    """Per strategy: Sharpe before and after costs and turnover. Per filter:
    number of strategies accepted, after costs and at zero cost."""
    net = pd.DataFrame({k: net_returns(prices, w) for k, w in positions.items()})
    gross = pd.DataFrame({k: net_returns(prices, w, 0.0) for k, w in positions.items()})
    rows = pd.DataFrame(
        {
            "net": sharpe(net.iloc[:R]),
            "gross": sharpe(gross.iloc[:R]),
            "turnover": [turnover(w) for w in positions.values()],
        }
    ).assign(**label)
    accepted = [
        label
        | {"returns": kind}
        | {f.name: int((f.score(t) > f.cut).sum()) for f in FILTERS}
        for kind, t in (("net", net), ("gross", gross))
    ]
    return rows, accepted


def agent_positions(trials: list, prices: pd.DataFrame) -> dict:
    """Weights of each usable agent strategy, run again in the sandbox."""
    out = {}
    for t in trials:
        if t.returns is None:
            continue
        weights, status = run_strategy(t.code, prices)
        if status != "ok":
            raise RuntimeError(f"round {t.round} no longer runs: {status}")
        out[t.round] = weights
    return out


def stage4_agent(index: int) -> tuple:
    prices = world(NO_SIGNAL, index)
    trials = run_agent(prices, MODEL, agent_seed(0, index))
    return score_world(
        prices, agent_positions(trials, prices), {"who": "agent", "world": index}
    )


def stage4_grid(index: int) -> tuple:
    prices = world(NO_SIGNAL, index)
    positions = {s.name: s.positions(prices) for s in GRID}
    return score_world(prices, positions, {"who": "grid", "world": index})


def exp3_run(task: tuple[str, int]) -> tuple:
    arm, index = task
    prices = world(NO_SIGNAL, index)
    trials = run_agent(
        prices,
        MODEL,
        seed=index,
        rounds=cfg.AGENT_ROUNDS,
        system=goodhart.SYSTEMS[arm],
        column=Column("checks passed", goodhart.checks_passed),
    )
    return score_world(
        prices, agent_positions(trials, prices), {"who": arm, "world": index}
    )


def once_lockbox(task: tuple[int, str]) -> dict:
    """Lockbox Sharpe of the "once" candidate, after costs and at zero cost."""
    index, name = task
    prices = world(NO_SIGNAL, index)
    weights = next(s for s in GRID if s.name == name).positions(prices)
    return {
        "world": index,
        "net": float(sharpe(net_returns(prices, weights).iloc[R:])),
        "gross": float(sharpe(net_returns(prices, weights, 0.0).iloc[R:])),
    }


def ci(values) -> str:
    mean, low, high = mean_ci(np.asarray(values, dtype=float))
    return f"{mean:.2f} [{low:.2f}, {high:.2f}]"


def describe(rows: pd.DataFrame, accepted: pd.DataFrame, groups: list[str]) -> str:
    """Markdown: Sharpe ratios and turnover (mean over a world's strategies,
    95% interval over worlds), then the share of worlds where each filter
    accepts something."""
    summary = {}
    for g in groups:
        mine = rows[rows.who == g]
        by_world = mine.groupby("world")
        best_net, best_gross = by_world.net.max(), by_world.gross.max()
        summary[g] = {
            "Strategies, worlds": f"{len(mine)}, {by_world.ngroups}",
            "Sharpe before costs": ci(by_world.gross.mean()),
            "Sharpe after costs": ci(by_world.net.mean()),
            "Cost drag": ci(by_world.gross.mean() - by_world.net.mean()),
            "Turnover per year, mean": ci(by_world.turnover.mean()),
            "Turnover per year, median strategy": f"{mine.turnover.median():.1f}",
            "Best strategy per world, before costs": f"{best_gross.mean():.2f}",
            "Best strategy per world, after costs": f"{best_net.mean():.2f}",
            "Worlds with a strategy above 1, before costs": f"{100 * (best_gross > 1).mean():.1f}%",
            "Worlds with a strategy above 1, after costs": f"{100 * (best_net > 1).mean():.1f}%",
        }
    table = pd.DataFrame(summary)
    lines = ["| | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    lines += [
        f"| {name} | " + " | ".join(table.loc[name]) + " |" for name in table.index
    ]

    lines += [
        "",
        "Share of worlds where the filter accepts something (all false), %:",
        "",
    ]
    header = [f"{g}, {kind}" for g in groups for kind in ("after costs", "zero cost")]
    lines += [
        "| Filter | " + " | ".join(header) + " |",
        "|---" * (len(header) + 1) + "|",
    ]
    for f in FILTERS:
        cells = []
        for g in groups:
            for kind in ("net", "gross"):
                part = accepted[(accepted.who == g) & (accepted.returns == kind)]
                cells.append(f"{100 * (part[f.name] > 0).mean():.1f}")
        lines.append(f"| {f.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect(results: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.concat([r for r, _ in results], ignore_index=True)
    accepted = pd.DataFrame([a for _, pair in results for a in pair])
    return rows, accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    jobs = parser.parse_args().jobs
    report = ["# Zero-cost checks", "", "Written by `scripts/zero_cost.py`.", ""]

    # 1. Stage 4, no-signal worlds.
    agent = list(_parallel(stage4_agent, list(range(cfg.AGENT_SEEDS)), jobs))
    grid = list(_parallel(stage4_grid, list(range(cfg.N_WORLDS)), jobs))
    rows, accepted = collect(agent + grid)
    report += [
        "## Stage 4: agent and grid on no-signal worlds (research sample)",
        "",
        describe(rows, accepted, ["agent", "grid"]),
        "",
    ]

    # 2. Experiment 2: the "once" candidates of the hill-climber.
    table = climb(pd.read_csv("results/truth.csv"), cfg.N_WORLDS, jobs, [ONCE])
    table = table[table.world_type == NO_SIGNAL.name]
    last = candidates(table, cfg.TRUE_SHARPE_MIN)
    last = last[last["round"] == cfg.REUSE_ROUNDS].set_index("world")
    chosen = {
        i: run.loc[run.answer.idxmax(), "strategy"] for i, run in table.groupby("world")
    }
    lockbox = pd.DataFrame(_parallel(once_lockbox, list(chosen.items()), jobs))
    lockbox = lockbox.set_index("world").loc[last.index]
    if not np.allclose(lockbox.net, last.lockbox):
        raise RuntimeError("the candidates' lockbox Sharpe ratios do not match")
    n, claims, bar = len(last), int(last.false_claim.sum()), cfg.CLAIM_SHARPE
    expected = float(
        (1 - stats.norm.cdf((bar - last.true_sharpe) / cfg.LOCKBOX_SHARPE_SE)).mean()
    )
    report += [
        "## Experiment 2: the lockbox opened once (hill-climber, no-signal worlds)",
        "",
        (
            f"- Claims after {cfg.REUSE_ROUNDS} rounds: {claims} of {n} "
            f"({100 * claims / n:.1f}%)."
        ),
        (
            f"- Probability of {claims} or fewer at a true rate of 5%: "
            f"{stats.binom.cdf(claims, n, 0.05):.3f} (exact binomial)."
        ),
        (
            f"- Mean true Sharpe of the candidates after costs: "
            f"{last.true_sharpe.mean():.2f}."
        ),
        (
            f"- Expected claim rate for these candidates, lockbox standard error "
            f"{cfg.LOCKBOX_SHARPE_SE:.2f}: {100 * expected:.1f}%; probability of "
            f"{claims} or fewer: {stats.binom.cdf(claims, n, expected):.3f}."
        ),
        (
            f"- Candidates' lockbox Sharpe after costs: mean {lockbox.net.mean():.2f}, "
            f"spread across worlds {lockbox.net.std():.2f}; at zero cost: mean "
            f"{lockbox.gross.mean():.2f}."
        ),
        f"- Claims at zero cost: {int((lockbox.gross > bar).sum())} of {n}.",
        "",
    ]

    # 3. Experiment 3.
    tasks = [(arm, i) for arm in goodhart.ARMS for i in range(cfg.GOODHART_SEEDS)]
    rows, accepted = collect(list(_parallel(exp3_run, tasks, jobs)))
    checks = [c.name for c in goodhart.CHECKS]
    passes = accepted.groupby(["who", "returns"])[checks].sum().sum(axis=1)
    report += [
        "## Experiment 3: told and not told, no-signal worlds (research sample)",
        "",
        describe(rows, accepted, list(goodhart.ARMS)),
        "",
        "Strategies passing any of the six checks shown to the agents "
        "(arm, returns: count): "
        + ", ".join(f"{who} {kind}: {n}" for (who, kind), n in passes.items())
        + ".",
        "",
    ]
    OUT.write_text("\n".join(report))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()

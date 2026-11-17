"""Experiment 2: what happens to a holdout that is looked at every round.

An analyst proposes one strategy per round and, after each, asks how it did
in the lockbox. Whatever answers best so far is the candidate, and a discovery
is claimed when the candidate's answer clears the bar of the 5% test
(`config.CLAIM_SHARPE`). Three ways of answering are compared:

* "once": no answers during the search (the analyst sees only the research
  Sharpe); the lockbox is opened one time, for the final candidate. This is
  the correct use of a holdout and the baseline: 5% false claims on noise.
* "reuse": the lockbox Sharpe of every proposal, as it is.
* "thresholdout": the lockbox is asked through Thresholdout.

Two analysts:

* a hill-climber without any model, on hundreds of worlds. It walks over the
  76 grid strategies, whose true Sharpe is already known from Experiment 1.
* the language-model agent, on a few no-signal worlds.

In a no-signal world every claim is false, without any simulation: returns
there cannot be predicted from the past, so every strategy earns zero before
costs on average and less than zero after.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import config as cfg
from .agent import SYSTEM, Column, Trial, run_agent
from .backtest import run_all, sharpe
from .experiments import GRID, _parallel
from .llm import Model, spent_usd
from .metrics import mean_ci
from .strategies import grid_neighbours
from .thresholdout import Thresholdout
from .worlds import world

Ask = Callable[[float, float], float | None]  # (research, lockbox) -> answer


@dataclass(frozen=True)
class Mechanism:
    """One way of answering the analyst's questions about the lockbox.

    Threshold and noise rate are in standard errors of the lockbox Sharpe;
    a noise rate of None means `THRESHOLDOUT_SIGMA_SHARE` of the threshold.
    """

    name: str
    kind: str  # "once", "reuse" or "thresholdout"
    threshold: float = cfg.THRESHOLDOUT_T
    budget: int = cfg.THRESHOLDOUT_BUDGET
    sigma: float | None = None
    lockbox_days: int = cfg.N_LOCKBOX

    @property
    def se(self) -> float:
        """Standard error of an annualised Sharpe near zero on the lockbox."""
        return float(np.sqrt(cfg.PERIODS_PER_YEAR / self.lockbox_days))

    @property
    def claim(self) -> float:
        """The bar for a claim: the one-sided 5% test on the lockbox."""
        return round(1.645 * self.se, 2)

    def ask(self, rng: np.random.Generator) -> Ask:
        """A fresh answering function for one world."""
        if self.kind == "once":
            return lambda research, lockbox: research
        if self.kind == "reuse":
            return lambda research, lockbox: lockbox
        t = self.threshold * self.se
        sigma = t * cfg.THRESHOLDOUT_SIGMA_SHARE
        if self.sigma is not None:
            sigma = self.sigma * self.se
        return Thresholdout(t, sigma, self.budget, rng).answer


ONCE = Mechanism("once", "once")
REUSE = Mechanism("reuse", "reuse")
THRESHOLDOUT = Mechanism("thresholdout", "thresholdout")
_LONG = cfg.LONG_LOCKBOX_DAYS
MECHANISMS = (
    [ONCE, REUSE, THRESHOLDOUT]
    + [
        Mechanism(f"thresholdout_t{t:g}", "thresholdout", threshold=t)
        for t in cfg.THRESHOLDOUT_T_SWEEP
    ]
    + [
        Mechanism(f"thresholdout_b{b}", "thresholdout", budget=b)
        for b in cfg.THRESHOLDOUT_BUDGET_SWEEP
    ]
    + [
        Mechanism(
            "thresholdout_noise2", "thresholdout", sigma=cfg.THRESHOLDOUT_NOISE_SE
        ),
        Mechanism(
            "thresholdout_paper",
            "thresholdout",
            threshold=cfg.THRESHOLDOUT_PAPER[0],
            sigma=cfg.THRESHOLDOUT_PAPER[1],
        ),
        Mechanism("once_long", "once", lockbox_days=_LONG),
        Mechanism("reuse_long", "reuse", lockbox_days=_LONG),
        Mechanism("thresholdout_long", "thresholdout", lockbox_days=_LONG),
    ]
)
AGENT_MECHANISMS = [REUSE, THRESHOLDOUT]
# The agent's noise streams are numbered after the first 8 mechanisms. The
# number is fixed: it set the answers the agent saw when its replies were
# generated, and another number would change the prompts.
_AGENT_NOISE_OFFSET = 8


# --- the hill-climber --------------------------------------------------------
def hill_climb(
    ask: Callable[[int], float | None],
    neighbours: list[list[int]],
    rounds: int,
    rng: np.random.Generator,
    jump: float = cfg.CLIMB_JUMP,
) -> tuple[list[int], list[float | None]]:
    """Greedy search: try a relative of the best strategy so far, keep it if
    its answer is higher.

    `ask(i)` is the answer for strategy i, or None for "no answer".
    `neighbours[i]` lists the relatives of strategy i. No strategy is asked
    twice. With probability `jump`, or when every relative has been tried,
    the next proposal is any untried strategy. Returns the proposals and
    their answers, in order; the best so far after k rounds is the proposal
    with the highest answer among the first k.
    """
    untried = set(range(len(neighbours)))
    proposals: list[int] = []
    answers: list[float | None] = []
    best, best_answer = None, -np.inf
    for _ in range(min(rounds, len(neighbours))):
        near = [] if best is None else [i for i in neighbours[best] if i in untried]
        pool = near if near and rng.random() >= jump else sorted(untried)
        choice = int(rng.choice(pool))
        untried.remove(choice)
        answer = ask(choice)
        proposals.append(choice)
        answers.append(answer)
        if answer is not None and answer > best_answer:
            best, best_answer = choice, answer
    return proposals, answers


def _ask_strategy(
    answer: Ask, research: np.ndarray, lockbox: np.ndarray, i: int
) -> float | None:
    return answer(research[i], lockbox[i])


def _climb_world(
    spec_index: int, index: int, mechanisms: list[Mechanism], rounds: int
) -> list[dict]:
    """The hill-climber in one world under each mechanism: one row per round."""
    spec = cfg.SPECS[spec_index]
    sharpes = {}  # lockbox length: (research Sharpe, lockbox Sharpe) per strategy
    for days in sorted({m.lockbox_days for m in mechanisms}):
        returns = run_all(world(spec, index, cfg.N_RESEARCH + days), GRID)
        sharpes[days] = (
            sharpe(returns.iloc[: cfg.N_RESEARCH]),
            sharpe(returns.iloc[cfg.N_RESEARCH :]),
        )
    rows = []
    for m_index, mechanism in enumerate(mechanisms):
        research, lockbox = sharpes[mechanism.lockbox_days]
        # The search uses the same random numbers under every mechanism, so
        # differences come from the answers alone.
        search = np.random.default_rng(cfg.seed("hill_climb", spec_index, index))
        noise = cfg.seed("thresholdout", spec_index, index, m_index)
        answer = mechanism.ask(np.random.default_rng(noise))
        proposals, answers = hill_climb(
            partial(_ask_strategy, answer, research, lockbox),
            grid_neighbours(),
            rounds,
            search,
        )
        for number, (i, a) in enumerate(zip(proposals, answers, strict=True), 1):
            rows.append(
                {
                    "world_type": spec.name,
                    "world": index,
                    "mechanism": mechanism.name,
                    "round": number,
                    "strategy": GRID[i].name,
                    "answer": np.nan if a is None else a,
                    "research": research[i],
                    "lockbox": lockbox[i],
                }
            )
    return rows


def _climb_batch(task: tuple[int, range, list[Mechanism], int]) -> pd.DataFrame:
    spec_index, worlds, mechanisms, rounds = task
    rows = []
    for index in worlds:
        rows += _climb_world(spec_index, index, mechanisms, rounds)
    return pd.DataFrame(rows)


def climb(
    truth: pd.DataFrame,
    n_worlds: int = cfg.N_WORLDS,
    jobs: int = 1,
    mechanisms: list[Mechanism] = MECHANISMS,
    rounds: int = cfg.REUSE_ROUNDS,
) -> pd.DataFrame:
    """The hill-climber under every mechanism; one row per world and round.

    `truth` is the table of true Sharpe ratios from Experiment 1.
    """
    names = [spec.name for spec in cfg.SPECS]
    tasks = [
        (names.index(name), range(start, min(start + 10, n_worlds)), mechanisms, rounds)
        for name in cfg.REUSE_WORLDS
        for start in range(0, n_worlds, 10)
    ]
    table = pd.concat(_parallel(_climb_batch, tasks, jobs), ignore_index=True)
    true = truth.set_index(["world", "strategy"]).true_sharpe
    keys = pd.MultiIndex.from_frame(table[["world_type", "strategy"]])
    table["true_sharpe"] = true.reindex(keys).to_numpy()
    if table.true_sharpe.isna().any():
        raise ValueError("the truth table does not cover every grid strategy")
    return table


# --- the agent ---------------------------------------------------------------
SYSTEM_REUSE = (
    SYSTEM
    + f"""
Validation: each strategy is also measured on about 3 further years of data \
from the same market, which the backtest does not use. After each round you \
are told this validation Sharpe ratio as well; in some rounds it may not be \
available. A strategy counts as a discovery when its validation Sharpe ratio \
is above {cfg.CLAIM_SHARPE:.2f}. Aim for a strategy with the highest \
validation Sharpe ratio you can reach.
"""
)


class LockboxAnswers:
    """The validation column of the agent's table, answered by a mechanism.

    Each usable round is asked about once, when it first appears, and the
    answer is kept: asking again would use the budget and draw new noise.
    """

    def __init__(self, ask: Ask) -> None:
        self.ask = ask
        self.answers: dict[int, float | None] = {}

    def cells(self, trials: list[Trial]) -> list[str]:
        out = []
        for trial in trials:
            if trial.returns is None:
                out.append("-")
                continue
            if trial.round not in self.answers:
                research = float(sharpe(trial.returns.iloc[: cfg.N_RESEARCH]))
                lockbox = float(sharpe(trial.returns.iloc[cfg.N_RESEARCH :]))
                self.answers[trial.round] = self.ask(research, lockbox)
            answer = self.answers[trial.round]
            out.append("not available" if answer is None else f"{answer:.2f}")
        return out


def _agent_run(task: tuple[Model, int, int, bool]) -> pd.DataFrame:
    """One agent in one no-signal world under one mechanism: a row per round."""
    model, m_index, index, live = task
    mechanism = AGENT_MECHANISMS[m_index]
    # Its own noise stream, apart from the hill-climber's.
    noise = cfg.seed("thresholdout", 0, index, _AGENT_NOISE_OFFSET + m_index)
    lockbox = LockboxAnswers(mechanism.ask(np.random.default_rng(noise)))
    trials = run_agent(
        world(cfg.SPECS[0], index),
        model,
        seed=index,
        rounds=cfg.REUSE_ROUNDS,
        live=live,
        system=SYSTEM_REUSE,
        column=Column("validation Sharpe", lockbox.cells),
    )
    lockbox.cells(trials)  # the last round has not been asked about yet
    rows = []
    for trial in trials:
        usable = trial.returns is not None
        answer = lockbox.answers.get(trial.round)
        rows.append(
            {
                "world_type": cfg.SPECS[0].name,
                "world": index,
                "mechanism": mechanism.name,
                "round": trial.round,
                "answer": np.nan if answer is None else answer,
                "research": sharpe(trial.returns.iloc[: cfg.N_RESEARCH])
                if usable
                else np.nan,
                "lockbox": sharpe(trial.returns.iloc[cfg.N_RESEARCH :])
                if usable
                else np.nan,
                "true_sharpe": np.nan,  # not measured: below zero by design
                "attempts": len(trial.replies),
                "cost_usd": sum(r.cost_usd for r in trial.replies),
            }
        )
    return pd.DataFrame(rows)


def _agent_tasks(model: Model, n_seeds: int, live: bool) -> list:
    return [
        (model, m_index, index, live)
        for index in range(n_seeds)
        for m_index in range(len(AGENT_MECHANISMS))
    ]


def generate_agents(model: Model, n_seeds: int = cfg.REUSE_AGENT_SEEDS) -> None:
    """Make the model calls of Experiment 2. Stored replies are not repeated."""
    for task in _agent_tasks(model, n_seeds, live=True):
        table = _agent_run(task)
        print(
            f"{table.mechanism[0]} world {task[2]}: {table.answer.notna().sum()} of "
            f"{len(table)} rounds answered, spent ${spent_usd(model):.4f}",
            flush=True,
        )


def agent_runs(
    model: Model, n_seeds: int = cfg.REUSE_AGENT_SEEDS, jobs: int = 1
) -> pd.DataFrame:
    """The agent's runs replayed from stored replies; one row per round."""
    tasks = _agent_tasks(model, n_seeds, live=False)
    return pd.concat(_parallel(_agent_run, tasks, jobs), ignore_index=True)


# --- summary -----------------------------------------------------------------
def candidates(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """After each round of each run: the candidate and whether it is claimed.

    The candidate is the proposal with the highest answer so far. Under
    "reuse" and "thresholdout" the claim rests on that answer; under "once"
    the answers are research Sharpe ratios and the claim rests on the
    candidate's lockbox Sharpe, read one time. A candidate is real when its
    true Sharpe is at least `threshold`; in a no-signal world none is.

    Second rule, `lockbox_*_claim`: the candidate's actual lockbox Sharpe
    clears the bar, whatever the analyst was told. Under "once" the two rules
    are the same.
    """
    by_name = {m.name: m for m in MECHANISMS}
    rows = []
    for (world_type, index, name), run in table.groupby(
        ["world_type", "world", "mechanism"], sort=False
    ):
        mechanism = by_name[name]
        best, answered = None, 0
        for row in run.sort_values("round").itertuples():
            if not np.isnan(row.answer):
                answered += 1
                if best is None or row.answer > best.answer:
                    best = row
            reported = lockbox = true_sharpe = np.nan
            if best is not None:
                lockbox, true_sharpe = best.lockbox, best.true_sharpe
                reported = lockbox if mechanism.kind == "once" else best.answer
            claimed = reported > mechanism.claim  # False while there is none
            lockbox_claimed = lockbox > mechanism.claim
            real = best is not None and true_sharpe >= threshold
            rows.append(
                {
                    "world_type": world_type,
                    "world": index,
                    "mechanism": name,
                    "round": row.round,
                    "reported": reported,
                    "lockbox": lockbox,
                    "true_sharpe": true_sharpe,
                    "false_claim": claimed and not real,
                    "true_claim": claimed and real,
                    "lockbox_false_claim": lockbox_claimed and not real,
                    "lockbox_true_claim": lockbox_claimed and real,
                    "answered": answered,
                }
            )
    return pd.DataFrame(rows)


def summarise_reuse(
    table: pd.DataFrame, threshold: float = cfg.TRUE_SHARPE_MIN
) -> pd.DataFrame:
    """Per world type, mechanism and round: share of worlds with a false and
    with a true claim (95% intervals over worlds), what the analyst was told
    about the candidate, and what the candidate is really worth."""
    rows = []
    groups = candidates(table, threshold).groupby(
        ["world_type", "mechanism", "round"], sort=False
    )
    for (world_type, name, number), part in groups:
        row = {
            "world_type": world_type,
            "mechanism": name,
            "round": number,
            "n_worlds": len(part),
        }
        for prefix, column in [
            ("false", "false_claim"),
            ("true", "true_claim"),
            ("lockbox_false", "lockbox_false_claim"),
            ("lockbox_true", "lockbox_true_claim"),
        ]:
            share, low, high = mean_ci(part[column].to_numpy())
            row |= {
                f"{prefix}_claims": share,
                f"{prefix}_low": low,
                f"{prefix}_high": high,
            }
        rows.append(
            row
            | {
                "reported": part.reported.mean(),
                "lockbox": part.lockbox.mean(),
                "true_sharpe": part.true_sharpe.mean(),
                "answered": part.answered.mean(),
            }
        )
    return pd.DataFrame(rows)


def paired_test(
    table: pd.DataFrame, first: str, second: str, number: int = cfg.REUSE_ROUNDS
) -> pd.DataFrame:
    """Exact McNemar test of two mechanisms run on the same worlds.

    Per world, does each mechanism have a false claim after `number` rounds?
    Only the discordant worlds (a claim under one and not the other) carry
    information; under "no difference" each is equally likely to go either
    way, so their split is Binomial(discordant, 1/2). Two-sided p-value. One
    row per claim rule.
    """
    claims = candidates(table, cfg.TRUE_SHARPE_MIN)
    claims = claims[claims["round"] == number]
    rows = []
    for rule in ["false_claim", "lockbox_false_claim"]:
        wide = claims.pivot(index="world", columns="mechanism", values=rule)
        a, b = wide[first].astype(bool), wide[second].astype(bool)
        only_first, only_second = int((a & ~b).sum()), int((~a & b).sum())
        p = 1.0
        if only_first + only_second:
            p = binomtest(only_first, only_first + only_second).pvalue
        rows.append(
            {
                "rule": rule,
                "round": number,
                "n_worlds": len(wide),
                f"{first}_claims": int(a.sum()),
                f"{second}_claims": int(b.sum()),
                f"only_{first}": only_first,
                f"only_{second}": only_second,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def answer_sources(table: pd.DataFrame) -> pd.DataFrame:
    """Per world type and mechanism: where the answers came from.

    An answer came from the lockbox when it differs from the research Sharpe:
    "reuse" always answers with the lockbox, "once" never, and Thresholdout
    answers with the research Sharpe itself unless the gap is flagged, and
    then with the noisy lockbox Sharpe (an exact tie has probability zero).
    `budget_round` is the round of the last answer Thresholdout could give
    from the lockbox, in the runs where the budget ran out; NaN otherwise.
    """
    by_name = {m.name: m for m in MECHANISMS}
    runs = []
    for (world_type, index, name), run in table.groupby(
        ["world_type", "world", "mechanism"], sort=False
    ):
        run = run.sort_values("round")
        answered = run.answer.notna()
        from_lockbox = answered & (run.answer != run.research)
        budget_round = np.nan
        mechanism = by_name[name]
        if mechanism.kind == "thresholdout" and from_lockbox.sum() >= mechanism.budget:
            budget_round = run["round"][from_lockbox].iloc[mechanism.budget - 1]
        runs.append(
            {
                "world_type": world_type,
                "mechanism": name,
                "usable": run.research.notna().sum(),
                "answered": answered.sum(),
                "from_lockbox": from_lockbox.sum(),
                "share_from_lockbox": from_lockbox.sum() / max(answered.sum(), 1),
                "budget_round": budget_round,
            }
        )
    runs = pd.DataFrame(runs)
    rows = []
    for (world_type, name), part in runs.groupby(
        ["world_type", "mechanism"], sort=False
    ):
        share, low, high = mean_ci(part.share_from_lockbox.to_numpy().astype(float))
        spent = part.budget_round.dropna()
        rows.append(
            {
                "world_type": world_type,
                "mechanism": name,
                "n_worlds": len(part),
                "usable_per_run": part.usable.mean(),
                "answered_per_run": part.answered.mean(),
                "from_lockbox_per_run": part.from_lockbox.mean(),
                "share_from_lockbox": share,
                "share_low": low,
                "share_high": high,
                "runs_budget_spent": len(spent) / len(part),
                "budget_round_median": spent.median() if len(spent) else np.nan,
                "budget_round_min": spent.min() if len(spent) else np.nan,
                "budget_round_max": spent.max() if len(spent) else np.nan,
            }
        )
    return pd.DataFrame(rows)

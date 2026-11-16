"""Experiment 1 with the language-model agent as strategy generator.

Three steps, as for the grid search:

1. runs: one agent per world, 20 rounds each. `generate` makes the model
   calls (one at a time, so the spending cap is checked before every call);
   `score_runs` replays the stored replies, so it needs no model.
2. truth: the long-run Sharpe of every strategy the agent wrote, measured on
   the same fresh paths as the grid's truth.
3. summary: FDR, pFDR and power per filter. Unlike the grid, every world has
   its own set of strategies, so "true" is decided strategy by strategy.
"""

import numpy as np
import pandas as pd

from . import config as cfg
from .agent import run_agent
from .backtest import net_returns, sharpe
from .experiments import _parallel
from .llm import Model, spent_usd
from .metrics import fdp_and_power, mean_ci
from .registry import FILTERS
from .sandbox import run_on_paths
from .worlds import draw_shocks, simulate, to_prices, world

_BATCH = 10  # truth paths simulated by one worker call


def _worlds(n_seeds: int) -> list[tuple[int, int]]:
    names = [spec.name for spec in cfg.SPECS]
    return [(names.index(w), i) for w in cfg.AGENT_WORLDS for i in range(n_seeds)]


def agent_seed(spec_index: int, world_index: int) -> int:
    return spec_index * 1000 + world_index


# --- 1. runs -----------------------------------------------------------------
def generate(model: Model, n_seeds: int = cfg.AGENT_SEEDS) -> None:
    """Make the model calls for every world. Stored replies are not repeated."""
    for count, (spec_index, index) in enumerate(_worlds(n_seeds), start=1):
        prices = world(cfg.SPECS[spec_index], index)
        trials = run_agent(prices, model, agent_seed(spec_index, index), live=True)
        usable = sum(t.status == "ok" for t in trials)
        print(
            f"run {count}: {cfg.SPECS[spec_index].name} world {index}, "
            f"{usable} of {len(trials)} usable, spent ${spent_usd(model):.4f}",
            flush=True,
        )


def _score_run(task: tuple[int, int, Model]) -> dict:
    """One world: replay the agent from stored replies and score its strategies."""
    spec_index, index, model = task
    prices = world(cfg.SPECS[spec_index], index)
    trials = run_agent(prices, model, agent_seed(spec_index, index))
    usable = [t for t in trials if t.status == "ok"]
    returns = pd.DataFrame({t.round: t.returns for t in usable})
    scores = np.full((len(FILTERS), len(usable)), -np.inf)
    distinct = 0
    if len(usable) >= 2:  # the corrections need something to correct for
        scores = np.array([f.score(returns) for f in FILTERS])
        copies = np.triu(returns.corr().abs().to_numpy(), k=1) > 0.99
        distinct = int((~copies.any(axis=0)).sum())
    return {
        "spec": spec_index,
        "world": index,
        "rounds": [t.round for t in usable],
        "codes": [t.code for t in usable],
        "scores": scores,
        "research_sharpe": sharpe(returns.iloc[: cfg.N_RESEARCH]),
        "attempts": sum(len(t.replies) for t in trials),
        "lookahead": sum(
            f.startswith("look-ahead") for t in trials for f in t.failures
        ),
        "distinct": distinct,
        "cost": sum(r.cost_usd for t in trials for r in t.replies),
        "seconds": sum(r.seconds for t in trials for r in t.replies),
    }


def score_runs(
    model: Model, n_seeds: int = cfg.AGENT_SEEDS, jobs: int = 1
) -> list[dict]:
    tasks = [(spec_index, index, model) for spec_index, index in _worlds(n_seeds)]
    return list(_parallel(_score_run, tasks, jobs))


# --- 2. truth ----------------------------------------------------------------
def _truth_batch(task: tuple[int, range, list[str]]) -> np.ndarray:
    """Sharpe of each strategy on each fresh path, shape (paths, strategies)."""
    spec_index, paths, codes = task
    spec = cfg.SPECS[spec_index]
    seeds = [cfg.seed("truth", spec_index, p) for p in paths]
    days = cfg.TRUTH_YEARS * cfg.PERIODS_PER_YEAR
    tables = [to_prices(r) for r in simulate(spec, draw_shocks(seeds, days))]
    prices = np.stack([t.to_numpy() for t in tables])
    out = np.full((len(tables), len(codes)), np.nan)
    for column, weights in enumerate(run_on_paths(codes, prices)):
        if weights is None:
            continue  # failed on long paths: its truth stays unknown
        for row, table in enumerate(tables):
            held = pd.DataFrame(weights[row], columns=table.columns)
            out[row, column] = sharpe(net_returns(table, held))
    return out


def agent_truth(
    runs: list[dict], n_paths: int = cfg.TRUTH_PATHS, jobs: int = 1
) -> pd.DataFrame:
    """True Sharpe and its standard error for every usable agent strategy."""
    specs = sorted({run["spec"] for run in runs})
    by_spec = {s: [run for run in runs if run["spec"] == s] for s in specs}
    tasks = [
        (
            spec_index,
            range(start, min(start + _BATCH, n_paths)),
            [code for run in by_spec[spec_index] for code in run["codes"]],
        )
        for spec_index in specs
        for start in range(0, n_paths, _BATCH)
    ]
    results = list(_parallel(_truth_batch, tasks, jobs))
    frames = []
    for spec_index, mine in by_spec.items():
        per_path = np.concatenate(
            [r for task, r in zip(tasks, results, strict=True) if task[0] == spec_index]
        )
        frames.append(
            pd.DataFrame(
                {
                    "world_type": cfg.SPECS[spec_index].name,
                    "world": [run["world"] for run in mine for _ in run["rounds"]],
                    "round": [r for run in mine for r in run["rounds"]],
                    "research_sharpe": np.concatenate(
                        [run["research_sharpe"] for run in mine]
                    ),
                    "true_sharpe": per_path.mean(axis=0),
                    "se": per_path.std(axis=0, ddof=1) / np.sqrt(len(per_path)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- 3. summary --------------------------------------------------------------
def summarise_agent(
    runs: list[dict], truth: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """FDR, pFDR and power per group of worlds and filter, at the usual cuts."""
    lookup = truth.set_index(["world_type", "world", "round"]).true_sharpe
    groups = {name: [name] for name in cfg.AGENT_WORLDS}
    groups["all_medium"] = [n for n in cfg.AGENT_WORLDS if n != "no_signal"]
    rows = []
    for group, names in groups.items():
        mine = [run for run in runs if cfg.SPECS[run["spec"]].name in names]
        if not mine:
            continue
        is_true = [
            np.array(
                [
                    lookup[(cfg.SPECS[run["spec"]].name, run["world"], r)] >= threshold
                    for r in run["rounds"]
                ],
                dtype=bool,
            )
            for run in mine
        ]
        for f_index, flt in enumerate(FILTERS):
            fdp, power, n_accepted = [], [], []
            for run, true in zip(mine, is_true, strict=True):
                accepted = run["scores"][f_index] > flt.cut
                world_fdp, world_power = fdp_and_power(accepted[None, :], true)
                fdp.append(world_fdp[0])
                power.append(world_power[0])
                n_accepted.append(accepted.sum())
            fdp, n_accepted = np.array(fdp), np.array(n_accepted)
            fdr, fdr_low, fdr_high = mean_ci(fdp)
            pfdr, pfdr_low, pfdr_high = mean_ci(fdp[n_accepted > 0])
            pw, pw_low, pw_high = mean_ci(np.array(power))
            rows.append(
                {
                    "worlds": group,
                    "filter": flt.name,
                    "true_threshold": threshold,
                    "n_worlds": len(mine),
                    "fdr": fdr,
                    "fdr_low": fdr_low,
                    "fdr_high": fdr_high,
                    "pfdr": pfdr,
                    "pfdr_low": pfdr_low,
                    "pfdr_high": pfdr_high,
                    "worlds_accepting": (n_accepted > 0).mean(),
                    "power": pw,
                    "power_low": pw_low,
                    "power_high": pw_high,
                    "accepted_per_world": n_accepted.mean(),
                    "strategies_per_world": np.mean([len(t) for t in is_true]),
                    "true_per_world": np.mean([t.sum() for t in is_true]),
                }
            )
    return pd.DataFrame(rows)

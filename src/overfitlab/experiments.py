"""Experiment 1 with the grid-search generator.

Steps:
1. truth: the long-run Sharpe of every grid strategy in every world type,
2. scores: every filter's score for every strategy in every simulated world,
3. summary: FDR and power per world type, filter and cut.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as cfg
from .backtest import net_returns, run_all, sharpe
from .metrics import fdp_and_power, mean_ci
from .registry import FILTERS
from .strategies import grid, oracle
from .worlds import draw_shocks, simulate, to_prices, world

GRID = grid()
_BATCH = 10  # worlds or truth paths handled by one worker call


def _parallel(function: Callable, tasks: list, jobs: int) -> Iterable:
    if jobs == 1:
        return map(function, tasks)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(function, tasks))


def _batches(n_specs: int, n_items: int) -> list[tuple[int, range]]:
    return [
        (spec_index, range(start, min(start + _BATCH, n_items)))
        for spec_index in range(n_specs)
        for start in range(0, n_items, _BATCH)
    ]


# --- 1. truth ----------------------------------------------------------------
def _truth_batch(task: tuple[int, range]) -> np.ndarray:
    """Sharpe of every grid strategy, plus the oracle, on fresh 100-year paths."""
    spec_index, paths = task
    spec = cfg.SPECS[spec_index]
    seeds = [cfg.seed("truth", spec_index, p) for p in paths]
    n_days = cfg.TRUTH_YEARS * cfg.PERIODS_PER_YEAR
    out = []
    for returns in simulate(spec, draw_shocks(seeds, n_days)):
        prices = to_prices(returns)
        sharpes = sharpe(run_all(prices, GRID))
        best = np.nan
        if spec.kappa:
            best = sharpe(net_returns(prices, oracle(spec).positions(prices)))
        out.append(np.append(sharpes, best))
    return np.array(out)


def true_sharpes(n_paths: int = cfg.TRUTH_PATHS, jobs: int = 1) -> pd.DataFrame:
    """Long-run net Sharpe of each strategy in each world type, with its error.

    Each path gives one Sharpe estimate; paths are independent, so the standard
    error of their mean is their standard deviation / sqrt(number of paths).
    """
    tasks = _batches(len(cfg.SPECS), n_paths)
    results = list(_parallel(_truth_batch, tasks, jobs))
    names = [s.name for s in GRID] + ["oracle"]
    frames = []
    for spec_index, spec in enumerate(cfg.SPECS):
        per_path = np.concatenate(
            [r for (i, _), r in zip(tasks, results, strict=True) if i == spec_index]
        )
        frames.append(
            pd.DataFrame(
                {
                    "world": spec.name,
                    "strategy": names,
                    "true_sharpe": per_path.mean(axis=0),
                    "se": per_path.std(axis=0, ddof=1) / np.sqrt(len(per_path)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- 2. scores ---------------------------------------------------------------
def _score_batch(task: tuple[int, range]) -> np.ndarray:
    """Filter scores, shape (worlds in batch, filters, strategies)."""
    spec_index, worlds = task
    spec = cfg.SPECS[spec_index]
    out = []
    for index in worlds:
        returns = run_all(world(spec, index), GRID)
        out.append([f.score(returns) for f in FILTERS])
    return np.array(out)


def grid_scores(n_worlds: int = cfg.N_WORLDS, jobs: int = 1) -> np.ndarray:
    """Scores of all filters, shape (world types, worlds, filters, strategies)."""
    tasks = _batches(len(cfg.SPECS), n_worlds)
    results = list(_parallel(_score_batch, tasks, jobs))
    return np.concatenate(results).reshape(
        len(cfg.SPECS), n_worlds, len(FILTERS), len(GRID)
    )


# --- 3. summary --------------------------------------------------------------
def _groups() -> dict[str, list[int]]:
    """World types one by one, and pooled by signal strength."""
    groups = {spec.name: [i] for i, spec in enumerate(cfg.SPECS)}
    for strength in cfg.STRENGTH_TARGETS:
        groups[f"all_{strength}"] = [
            i for i, spec in enumerate(cfg.SPECS) if spec.strength == strength
        ]
    return {name: indices for name, indices in groups.items() if indices}


def summarise(
    scores: np.ndarray, truth: pd.DataFrame, threshold: float, sweep: bool
) -> pd.DataFrame:
    """FDR and power per group of worlds, filter and cut, with 95% intervals.

    A strategy is "true" in a world type when its long-run net Sharpe is at
    least `threshold`. With `sweep=False` only each filter's conventional cut
    is reported.
    """
    names = [s.name for s in GRID]
    wide = truth.pivot(index="world", columns="strategy", values="true_sharpe")
    is_true = wide.loc[[s.name for s in cfg.SPECS], names].to_numpy() >= threshold
    rows = []
    for group, spec_indices in _groups().items():
        for f_index, flt in enumerate(FILTERS):
            for cut in flt.sweep if sweep else [flt.cut]:
                fdp, power, n_accepted = [], [], []
                for s in spec_indices:
                    accepted = scores[s, :, f_index, :] > cut
                    world_fdp, world_power = fdp_and_power(accepted, is_true[s])
                    fdp.append(world_fdp)
                    power.append(world_power)
                    n_accepted.append(accepted.sum(axis=1))
                fdp, power = np.concatenate(fdp), np.concatenate(power)
                n_accepted = np.concatenate(n_accepted)
                fdr, fdr_low, fdr_high = mean_ci(fdp)
                pw, pw_low, pw_high = mean_ci(power)
                rows.append(
                    {
                        "worlds": group,
                        "filter": flt.name,
                        "cut": cut,
                        "true_threshold": threshold,
                        "n_worlds": len(fdp),
                        "fdr": fdr,
                        "fdr_low": fdr_low,
                        "fdr_high": fdr_high,
                        "power": pw,
                        "power_low": pw_low,
                        "power_high": pw_high,
                        "accepted_per_world": n_accepted.mean(),
                        "false_per_world": (fdp * n_accepted).mean(),
                    }
                )
    return pd.DataFrame(rows)

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
from .registry import FILTERS, Filter
from .strategies import grid, oracle
from .worlds import draw_shocks, simulate, to_prices, world

GRID = grid()
_BATCH = 10  # worlds or truth paths handled by one worker call
_AFTER = cfg.DEPLOYMENT_YEARS


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
    """Sharpe of every grid strategy, plus the oracle, on fresh paths.

    Normally a path is 100 years. With alpha decay the truth that matters is
    what a strategy earns after the backtest, so a path is the 10 backtest
    years plus the 5 years after, and only those 5 years are scored.
    """
    spec_index, paths = task
    spec = cfg.SPECS[spec_index]
    seeds = [cfg.seed("truth", spec_index, p) for p in paths]
    start, years = 0, cfg.TRUTH_YEARS
    if spec.half_life_years:
        start, years = cfg.N_DAYS, cfg.N_DAYS // cfg.PERIODS_PER_YEAR + _AFTER
    out = []
    for returns in simulate(spec, draw_shocks(seeds, years * cfg.PERIODS_PER_YEAR)):
        prices = to_prices(returns)
        sharpes = sharpe(run_all(prices, GRID).iloc[start:])
        best = np.nan
        if spec.kappa:
            trades = net_returns(prices, oracle(spec).positions(prices))
            best = sharpe(trades.iloc[start:])
        out.append(np.append(sharpes, best))
    return np.array(out)


def true_sharpes(n_paths: int = cfg.TRUTH_PATHS, jobs: int = 1) -> pd.DataFrame:
    """Long-run net Sharpe of each strategy in each world type, with its error.

    Each path gives one Sharpe estimate; paths are independent, so the standard
    error of their mean is their standard deviation / sqrt(number of paths).
    """
    # Decay worlds score 5 years per path instead of 100, so they get 20 times
    # as many paths: the same 10,000 scored years per world type.
    more = cfg.TRUTH_YEARS // _AFTER
    tasks = [
        (spec_index, range(start, min(start + _BATCH, total)))
        for spec_index, spec in enumerate(cfg.SPECS)
        for total in [n_paths * (more if spec.half_life_years else 1)]
        for start in range(0, total, _BATCH)
    ]
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
def _score_batch(task: tuple[int, range, list[Filter]]) -> np.ndarray:
    """Filter scores, shape (worlds in batch, filters, strategies)."""
    spec_index, worlds, filters = task
    spec = cfg.SPECS[spec_index]
    out = []
    for index in worlds:
        returns = run_all(world(spec, index), GRID)
        out.append([f.score(returns) for f in filters])
    return np.array(out)


def grid_scores(
    n_worlds: int = cfg.N_WORLDS, jobs: int = 1, filters: list[Filter] = FILTERS
) -> np.ndarray:
    """Scores of the filters, shape (world types, worlds, filters, strategies)."""
    tasks = [(*batch, filters) for batch in _batches(len(cfg.SPECS), n_worlds)]
    results = list(_parallel(_score_batch, tasks, jobs))
    return np.concatenate(results).reshape(
        len(cfg.SPECS), n_worlds, len(filters), len(GRID)
    )


# --- 3. summary --------------------------------------------------------------
def _groups() -> dict[str, list[int]]:
    """World types one by one, and pooled by signal strength and variant.

    Pooled names: all_medium, all_medium_regimes, all_obvious_decay, ...
    """
    groups = {spec.name: [i] for i, spec in enumerate(cfg.SPECS)}
    for i, spec in enumerate(cfg.SPECS):
        if spec.kappa:
            suffix = "" if spec.variant == "base" else f"_{spec.variant}"
            groups.setdefault(f"all_{spec.strength}{suffix}", []).append(i)
    return groups


def summarise(
    scores: np.ndarray,
    truth: pd.DataFrame,
    threshold: float,
    sweep: bool,
    filters: list[Filter] = FILTERS,
) -> pd.DataFrame:
    """FDR, pFDR and power per group of worlds, filter and cut, with 95% intervals.

    A strategy is "true" in a world type when its long-run net Sharpe is at
    least `threshold`. With `sweep=False` only each filter's conventional cut
    is reported.

    pFDR (positive FDR, Storey 2003) is the mean false discovery proportion
    over the worlds where the filter accepted something. FDR counts the other
    worlds as zero, so FDR = pFDR x share of worlds with an acceptance. The
    pFDR interval resamples only those worlds; it is NaN when there are none.
    """
    names = [s.name for s in GRID]
    wide = truth.pivot(index="world", columns="strategy", values="true_sharpe")
    is_true = wide.loc[[s.name for s in cfg.SPECS], names].to_numpy() >= threshold
    rows = []
    for group, spec_indices in _groups().items():
        for f_index, flt in enumerate(filters):
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
                pfdr, pfdr_low, pfdr_high = mean_ci(fdp[n_accepted > 0])
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
                        "pfdr": pfdr,
                        "pfdr_low": pfdr_low,
                        "pfdr_high": pfdr_high,
                        "worlds_accepting": (n_accepted > 0).mean(),
                        "power": pw,
                        "power_low": pw_low,
                        "power_high": pw_high,
                        "accepted_per_world": n_accepted.mean(),
                        "false_per_world": (fdp * n_accepted).mean(),
                    }
                )
    return pd.DataFrame(rows)

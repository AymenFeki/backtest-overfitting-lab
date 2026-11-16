"""Simulated markets where the truth is known.

Daily return of asset i:

    r[i, t] = mu[i, t] + beta[i] * f[t] + e[i, t]

* `f` is a common factor, `e` is asset-specific noise. Sharing one factor is
  the simplest way to get correlated assets.
* Both have Student-t shocks (fat tails) and GARCH(1,1) variance (volatility
  clustering: calm and wild periods come in runs).
* `mu` is the planted signal. In the null world it is zero for every asset and
  every day, so no strategy has an edge and costs make all of them lose.

Two optional complications (Stage 3):

* Regime switching: a hidden two-state Markov chain (calm/crisis). In a crisis
  the factor's volatility triples and asset-specific volatility rises by half,
  so everything moves more and moves together.
* Alpha decay: the planted signal's strength halves every few years.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import config as cfg

BETAS = np.linspace(*cfg.BETA_RANGE, cfg.N_ASSETS)
# Long-run daily variance of [factor, asset 1 noise, ..., asset N noise].
_TARGET_VAR = (
    np.r_[cfg.FACTOR_VOL**2, np.full(cfg.N_ASSETS, cfg.IDIO_VOL**2)]
    / cfg.PERIODS_PER_YEAR
)
ASSET_VOL = np.sqrt(BETAS**2 * _TARGET_VAR[0] + _TARGET_VAR[1:])  # daily
_CRISIS_SCALE = np.r_[cfg.CRISIS_FACTOR_VOL, np.full(cfg.N_ASSETS, cfg.CRISIS_IDIO_VOL)]


def planted_signal(signal: str, z: np.ndarray) -> np.ndarray:
    """Direction and size of the planted edge, given standardised trailing returns.

    `z` has assets on the last axis. Clipping at +-2 keeps one extreme day from
    dominating the expected return.
    """
    if signal == "tsmom":  # each asset keeps going its own way
        return np.clip(z, -2, 2)
    if signal == "xsmom":  # recent winners beat recent losers
        return np.clip(z - z.mean(axis=-1, keepdims=True), -2, 2)
    if signal == "meanrev":  # short-term moves partly reverse
        return -np.clip(z, -2, 2)
    raise ValueError(f"unknown signal {signal!r}")


def draw_shocks(seeds: Sequence[np.random.SeedSequence], n_days: int) -> np.ndarray:
    """All randomness of each path, shape (paths, days, 2 + assets).

    Columns: unit-variance Student-t shocks for the factor and each asset,
    then one uniform draw per day that drives the regime switch. One seed per
    path, so each path can be regenerated on its own.
    """
    df = cfg.STUDENT_T_DF
    scale = np.sqrt(df / (df - 2))  # a t variable has variance df / (df - 2)
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shocks = rng.standard_t(df, (n_days, 1 + cfg.N_ASSETS)) / scale
        out.append(np.column_stack([shocks, rng.uniform(size=n_days)]))
    return np.stack(out)


def simulate(spec: cfg.WorldSpec, shocks: np.ndarray) -> np.ndarray:
    """Daily simple returns, shape (paths, days, assets), for the given shocks.

    The loop over days is needed because today's variance and today's planted
    drift both depend on yesterday's outcome. Everything inside the loop is
    vectorised over paths and assets. A crisis scales the shock itself; the
    GARCH recursion keeps running on the unscaled shock, so the two sources of
    changing volatility multiply instead of feeding each other.
    """
    n_paths, n_days, _ = shocks.shape
    omega = (1 - cfg.GARCH_ALPHA - cfg.GARCH_BETA) * _TARGET_VAR
    variance = np.tile(_TARGET_VAR, (n_paths, 1))
    lookback = cfg.SIGNAL_LOOKBACK.get(spec.signal, 0)
    cumulative = np.zeros((n_paths, n_days + 1, cfg.N_ASSETS))
    returns = np.empty((n_paths, n_days, cfg.N_ASSETS))
    crisis = np.zeros(n_paths, dtype=bool)
    half_life = spec.half_life_years * cfg.PERIODS_PER_YEAR

    for t in range(n_days):
        eps = np.sqrt(variance) * shocks[:, t, :-1]
        shock = eps
        if spec.regimes:
            draw = shocks[:, t, -1]
            crisis = np.where(crisis, draw > cfg.REGIME_EXIT, draw < cfg.REGIME_ENTER)
            shock = eps * np.where(crisis[:, None], _CRISIS_SCALE, 1.0)
        r = BETAS * shock[:, :1] + shock[:, 1:]
        if spec.kappa and t >= lookback:
            trailing = cumulative[:, t] - cumulative[:, t - lookback]
            z = trailing / (ASSET_VOL * np.sqrt(lookback))
            kappa = spec.kappa * (0.5 ** (t / half_life) if half_life else 1.0)
            r += kappa * ASSET_VOL * planted_signal(spec.signal, z)
        r = np.clip(r, -0.5, 0.5)  # keeps prices positive in a freak tail draw
        returns[:, t] = r
        cumulative[:, t + 1] = cumulative[:, t] + r
        variance = omega + cfg.GARCH_ALPHA * eps**2 + cfg.GARCH_BETA * variance
    return returns


def to_prices(returns: np.ndarray) -> pd.DataFrame:
    """Price table (days x assets) from one path of returns, starting near 100."""
    columns = [f"A{i}" for i in range(returns.shape[1])]
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), columns=columns)


def world(spec: cfg.WorldSpec, index: int, n_days: int = cfg.N_DAYS) -> pd.DataFrame:
    """Prices of world number `index` of this type. Same arguments, same world."""
    seed = cfg.seed("world", cfg.SPECS.index(spec), index)
    return to_prices(simulate(spec, draw_shocks([seed], n_days))[0])

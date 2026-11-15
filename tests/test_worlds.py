import numpy as np
import pandas as pd

from overfitlab import config as cfg
from overfitlab.backtest import net_returns, sharpe
from overfitlab.strategies import oracle
from overfitlab.worlds import draw_shocks, simulate, to_prices, world

NULL = cfg.SPECS[0]


def _shocks(n_paths: int, n_days: int) -> np.ndarray:
    return draw_shocks(
        [cfg.seed("calibrate", 1000 + p) for p in range(n_paths)], n_days
    )


def test_world_is_reproducible_and_seeds_differ():
    pd.testing.assert_frame_equal(world(NULL, 3), world(NULL, 3))
    assert not world(NULL, 3).equals(world(NULL, 4))
    assert world(NULL, 0).shape == (cfg.N_DAYS, cfg.N_ASSETS)


def test_spec_names_are_unique():
    names = [s.name for s in cfg.SPECS]
    assert len(names) == len(set(names)) == 13


def test_null_world_has_no_drift_but_fat_tails_and_vol_clustering():
    returns = simulate(NULL, _shocks(20, 5000))  # 100,000 days per asset
    flat = returns.reshape(-1, cfg.N_ASSETS)
    t_stats = flat.mean(axis=0) / (flat.std(axis=0, ddof=1) / np.sqrt(len(flat)))
    assert np.abs(t_stats).max() < 3.5  # no asset has a detectable mean

    kurtosis = pd.DataFrame(flat).kurt().mean() + 3  # a normal has 3
    assert kurtosis > 4.5

    # Volatility clustering: big moves follow big moves.
    acf = np.mean([pd.Series(path[:, 0] ** 2).autocorr(1) for path in returns])
    assert acf > 0.05

    correlation = np.corrcoef(flat.T)
    mean_correlation = (correlation.sum() - cfg.N_ASSETS) / (cfg.N_ASSETS * 9)
    assert 0.15 < mean_correlation < 0.35


def test_stronger_planted_signal_gives_higher_oracle_sharpe():
    shocks = _shocks(10, 50 * cfg.PERIODS_PER_YEAR)
    for signal in cfg.KAPPA:
        measured = []
        for strength in ("barely", "medium", "obvious"):
            spec = next(s for s in cfg.SPECS if s.name == f"{signal}_{strength}")
            trader = oracle(spec)
            measured.append(
                np.mean(
                    [
                        sharpe(net_returns(p, trader.positions(p)))
                        for p in map(to_prices, simulate(spec, shocks))
                    ]
                )
            )
        assert measured[0] < measured[1] < measured[2]
        assert measured[2] > 1.5  # "obvious" is calibrated to about 2

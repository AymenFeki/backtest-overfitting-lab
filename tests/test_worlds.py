import numpy as np
import pandas as pd
import pytest

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
    assert len(names) == len(set(names)) == 20


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


def test_existing_worlds_never_change():
    # Regression guard: results are only reproducible if world k stays world k.
    assert world(NULL, 0).iloc[-1, 0] == pytest.approx(60.32286454758064)
    assert world(cfg.SPECS[7], 5).iloc[100, 3] == pytest.approx(106.48235616165904)


def test_regime_switching_raises_volatility_and_correlation():
    regimes = next(s for s in cfg.SPECS if s.name == "no_signal_regimes")
    shocks = _shocks(20, 5000)
    plain, switching = simulate(NULL, shocks), simulate(regimes, shocks)

    def volatility_and_correlation(returns):
        flat = returns.reshape(-1, cfg.N_ASSETS)
        correlation = np.corrcoef(flat.T)
        return flat.std(), (correlation.sum() - cfg.N_ASSETS) / (cfg.N_ASSETS * 9)

    plain_vol, plain_corr = volatility_and_correlation(plain)
    crisis_vol, crisis_corr = volatility_and_correlation(switching)
    assert crisis_vol > 1.15 * plain_vol
    assert crisis_corr > plain_corr + 0.08

    # The chain spends about one day in six in a crisis: enter / (enter + exit).
    crisis, share = np.zeros(20, dtype=bool), []
    for t in range(5000):
        draw = shocks[:, t, -1]
        crisis = np.where(crisis, draw > cfg.REGIME_EXIT, draw < cfg.REGIME_ENTER)
        share.append(crisis.mean())
    assert np.mean(share) == pytest.approx(1 / 6, abs=0.04)


def test_alpha_decay_makes_the_signal_fade():
    spec = next(s for s in cfg.SPECS if s.name == "tsmom_obvious_decay")
    trader = oracle(spec)
    early, late = [], []
    for path in simulate(spec, _shocks(30, 15 * cfg.PERIODS_PER_YEAR)):
        prices = to_prices(path)
        returns = net_returns(prices, trader.positions(prices)).to_numpy()
        early.append(sharpe(returns[: cfg.N_RESEARCH]))
        late.append(sharpe(returns[cfg.N_DAYS :]))
    assert np.mean(early) > 0.7  # looks like a good strategy in the backtest
    assert np.mean(late) < 0.4  # and is worth little in the 5 years after

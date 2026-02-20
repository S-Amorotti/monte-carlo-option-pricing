import numpy as np

from mc_option_pricing.config import MarketParams, SimulationParams
from mc_option_pricing.models.gbm import GBMModel, GBMParams
from mc_option_pricing.rng import make_rng


def test_gbm_log_returns_mean_var():
    market = MarketParams(spot=100.0, rate=0.03, dividend=0.01)
    params = GBMParams(sigma=0.2)
    model = GBMModel(params)
    sim = SimulationParams(n_paths=50000, n_steps=1, maturity=1.0, seed=123)

    rng = make_rng(sim.seed)
    terminal = model.simulate_paths(market, sim, rng, terminal_only=True)
    log_returns = np.log(terminal / market.spot)

    mean = float(np.mean(log_returns))
    var = float(np.var(log_returns))

    theo_mean = (market.rate - market.dividend - 0.5 * params.sigma**2) * sim.maturity
    theo_var = params.sigma**2 * sim.maturity

    assert np.isclose(mean, theo_mean, atol=5e-3)
    assert np.isclose(var, theo_var, atol=5e-3)

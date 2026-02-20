import numpy as np

from mc_option_pricing.config import MarketParams, SimulationParams
from mc_option_pricing.models.heston import HestonModel, HestonParams
from mc_option_pricing.rng import make_rng


def test_heston_full_truncation_neg_freq():
    market = MarketParams(spot=100.0, rate=0.03, dividend=0.0)
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04)
    model = HestonModel(params)
    sim = SimulationParams(n_paths=10000, n_steps=50, maturity=1.0, seed=42)

    rng = make_rng(sim.seed)
    s, v, neg_freq = model.simulate_paths(market, sim, rng)
    assert np.all(np.isfinite(v))
    assert neg_freq < 0.2

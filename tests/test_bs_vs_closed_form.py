import numpy as np

from mc_option_pricing.config import MarketParams, SimulationParams
from mc_option_pricing.models.gbm import GBMModel, GBMParams
from mc_option_pricing.payoffs.vanilla import VanillaPayoff
from mc_option_pricing.pricers.european_mc import EuropeanMCPricer
from mc_option_pricing.utils.black_scholes import bs_price


def test_mc_converges_to_bs():
    market = MarketParams(spot=100.0, rate=0.05, dividend=0.0)
    params = GBMParams(sigma=0.2)
    model = GBMModel(params)
    payoff = VanillaPayoff(strike=100.0, option_type="call")
    sim = SimulationParams(n_paths=200000, n_steps=1, maturity=1.0, seed=7)

    pricer = EuropeanMCPricer()
    res = pricer.price(model, market, payoff, sim)

    analytic = bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, "call")
    assert np.isclose(res.price, analytic, rtol=2e-2)

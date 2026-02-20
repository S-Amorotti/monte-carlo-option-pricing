import numpy as np

from mc_option_pricing.config import MarketParams, SimulationParams
from mc_option_pricing.models.gbm import GBMModel, GBMParams
from mc_option_pricing.payoffs.vanilla import VanillaPayoff
from mc_option_pricing.pricers.european_mc import EuropeanMCPricer


def test_antithetic_reduces_variance():
    market = MarketParams(spot=100.0, rate=0.01, dividend=0.0)
    params = GBMParams(sigma=0.2)
    model = GBMModel(params)
    payoff = VanillaPayoff(strike=100.0, option_type="call")

    base = SimulationParams(n_paths=20000, n_steps=1, maturity=1.0, seed=42)
    anti = SimulationParams(
        n_paths=20000, n_steps=1, maturity=1.0, seed=42, antithetic=True
    )

    pricer = EuropeanMCPricer()
    res_base = pricer.price(model, market, payoff, base)
    res_anti = pricer.price(model, market, payoff, anti)

    assert res_anti.stderr <= res_base.stderr


def test_control_variate_reduces_variance():
    market = MarketParams(spot=100.0, rate=0.03, dividend=0.0)
    params = GBMParams(sigma=0.2)
    model = GBMModel(params)
    payoff = VanillaPayoff(strike=100.0, option_type="call")

    base = SimulationParams(n_paths=20000, n_steps=1, maturity=1.0, seed=1)
    cv = SimulationParams(
        n_paths=20000, n_steps=1, maturity=1.0, seed=1, use_control_variate=True
    )

    pricer = EuropeanMCPricer()
    res_base = pricer.price(model, market, payoff, base)
    res_cv = pricer.price(model, market, payoff, cv)

    assert res_cv.stderr <= res_base.stderr

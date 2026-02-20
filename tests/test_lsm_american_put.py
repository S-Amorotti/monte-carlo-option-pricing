import numpy as np

from mc_option_pricing.config import MarketParams, SimulationParams
from mc_option_pricing.models.gbm import GBMModel, GBMParams
from mc_option_pricing.payoffs.vanilla import VanillaPayoff
from mc_option_pricing.pricers.lsm_american import LSMAmericanPricer, LSMConfig


def american_put_binomial(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    sigma: float,
    maturity: float,
    steps: int,
) -> float:
    dt = maturity / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-rate * dt)
    p = (np.exp((rate - dividend) * dt) - d) / (u - d)

    prices = spot * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps + 1))
    values = np.maximum(strike - prices, 0.0)

    for _ in range(steps):
        prices = prices[:-1] / u
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        exercise = np.maximum(strike - prices, 0.0)
        values = np.maximum(values, exercise)
    return float(values[0])


def test_lsm_american_put_vs_binomial():
    market = MarketParams(spot=100.0, rate=0.05, dividend=0.0)
    params = GBMParams(sigma=0.2)
    model = GBMModel(params)
    payoff = VanillaPayoff(strike=100.0, option_type="put")

    sim = SimulationParams(n_paths=30000, n_steps=50, maturity=1.0, seed=123)
    pricer = LSMAmericanPricer(LSMConfig(train_fraction=0.5))
    res = pricer.price(model, market, payoff, sim)

    benchmark = american_put_binomial(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, steps=200)
    assert np.isclose(res.price, benchmark, rtol=0.1)

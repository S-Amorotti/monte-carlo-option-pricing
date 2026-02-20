from mc_option_pricing.config import MarketParams, SimulationParams
from mc_option_pricing.models.gbm import GBMModel, GBMParams
from mc_option_pricing.payoffs.vanilla import VanillaPayoff
from mc_option_pricing.pricers.european_mc import EuropeanMCPricer


def main() -> None:
    market = MarketParams(spot=100.0, rate=0.03, dividend=0.0)
    model = GBMModel(GBMParams(sigma=0.2))
    payoff = VanillaPayoff(strike=100.0, option_type="call")

    pricer = EuropeanMCPricer()
    base = SimulationParams(n_paths=20000, n_steps=1, maturity=1.0, seed=1)
    anti = SimulationParams(n_paths=20000, n_steps=1, maturity=1.0, seed=1, antithetic=True)
    cv = SimulationParams(n_paths=20000, n_steps=1, maturity=1.0, seed=1, use_control_variate=True)

    res_base = pricer.price(model, market, payoff, base)
    res_anti = pricer.price(model, market, payoff, anti)
    res_cv = pricer.price(model, market, payoff, cv)

    print("Base stderr:", res_base.stderr)
    print("Antithetic stderr:", res_anti.stderr)
    print("Control variate stderr:", res_cv.stderr)


if __name__ == "__main__":
    main()

from mc_option_pricing.config import MarketParams, SimulationParams
from mc_option_pricing.models.gbm import GBMModel, GBMParams
from mc_option_pricing.payoffs.vanilla import VanillaPayoff
from mc_option_pricing.pricers.lsm_american import LSMAmericanPricer


def main() -> None:
    market = MarketParams(spot=100.0, rate=0.05, dividend=0.0)
    model = GBMModel(GBMParams(sigma=0.2))
    payoff = VanillaPayoff(strike=100.0, option_type="put")
    sim = SimulationParams(n_paths=30000, n_steps=50, maturity=1.0, seed=123)

    pricer = LSMAmericanPricer()
    res = pricer.price(model, market, payoff, sim)

    print("American put (LSM) price:", res.price)
    print("In-sample price:", res.extra["in_sample_price"])


if __name__ == "__main__":
    main()

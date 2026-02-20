import numpy as np

from mc_option_pricing.calibration.calibration_engine import CalibrationConfig, HestonCalibrator
from mc_option_pricing.config import MarketParams, SimulationParams
from mc_option_pricing.models.heston import HestonModel, HestonParams
from mc_option_pricing.payoffs.vanilla import VanillaPayoff
from mc_option_pricing.pricers.european_mc import EuropeanMCPricer


def generate_synthetic_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    market = MarketParams(spot=100.0, rate=0.02, dividend=0.0)
    params = HestonParams(kappa=1.5, theta=0.04, xi=0.5, rho=-0.6, v0=0.04)
    model = HestonModel(params)

    strikes = np.array([80, 90, 100, 110, 120], dtype=float)
    maturities = np.array([0.5, 1.0, 2.0], dtype=float)
    sim = SimulationParams(n_paths=40000, n_steps=50, maturity=1.0, seed=123)

    pricer = EuropeanMCPricer()
    prices = np.zeros((len(maturities), len(strikes)))
    for i, t in enumerate(maturities):
        sim_t = SimulationParams(
            n_paths=sim.n_paths,
            n_steps=sim.n_steps,
            maturity=float(t),
            seed=sim.seed,
        )
        for j, k in enumerate(strikes):
            payoff = VanillaPayoff(strike=float(k), option_type="call")
            res = pricer.price(model, market, payoff, sim_t)
            prices[i, j] = res.price

    return strikes, maturities, prices


def main() -> None:
    strikes, maturities, prices = generate_synthetic_surface()
    market = MarketParams(spot=100.0, rate=0.02, dividend=0.0)
    sim = SimulationParams(n_paths=20000, n_steps=50, maturity=1.0, seed=7)

    calibrator = HestonCalibrator(
        market=market,
        strikes=strikes,
        maturities=maturities,
        market_prices=prices,
        sim=sim,
        config=CalibrationConfig(objective="iv_sse", n_starts=3),
    )
    result = calibrator.calibrate()
    print("Calibration success:", result.success)
    print("Objective:", result.objective)
    print("Params:", result.params)
    print("Feller condition:", result.extra["feller"])


if __name__ == "__main__":
    main()

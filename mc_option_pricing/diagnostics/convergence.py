from __future__ import annotations

import numpy as np

from ..config import MarketParams, SimulationParams
from ..models.gbm import GBMModel
from ..payoffs.base import Payoff
from ..pricers.european_mc import EuropeanMCPricer


def price_convergence(
    model: GBMModel,
    market: MarketParams,
    payoff: Payoff,
    sim: SimulationParams,
    grid: list[int],
) -> np.ndarray:
    pricer = EuropeanMCPricer()
    results = []
    for n in grid:
        sim_n = SimulationParams(
            n_paths=n,
            n_steps=sim.n_steps,
            maturity=sim.maturity,
            seed=sim.seed,
            dtype=sim.dtype,
            antithetic=sim.antithetic,
            use_control_variate=sim.use_control_variate,
            use_qmc=sim.use_qmc,
        )
        res = pricer.price(model, market, payoff, sim_n)
        results.append([n, res.price, res.stderr])
    return np.array(results)

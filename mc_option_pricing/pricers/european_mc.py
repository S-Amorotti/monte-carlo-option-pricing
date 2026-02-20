from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from ..config import MarketParams, SimulationParams
from ..models.gbm import GBMModel
from ..models.heston import HestonModel
from ..payoffs.base import Payoff
from ..rng import make_rng
from ..utils.black_scholes import bs_price
from ..variance_reduction.antithetic import antithetic_pairs
from ..variance_reduction.control_variate import control_variate_adjust
from .base import PricingResult


@dataclass(frozen=True)
class EuropeanMCConfig:
    confidence: float = 0.95


class EuropeanMCPricer:
    def __init__(self, config: EuropeanMCConfig | None = None) -> None:
        self.config = config or EuropeanMCConfig()

    def price(
        self,
        model: GBMModel | HestonModel,
        market: MarketParams,
        payoff: Payoff,
        sim: SimulationParams,
    ) -> PricingResult:
        start = time.perf_counter()
        rng = make_rng(sim.seed)
        market.validate()
        sim.validate()

        n_paths = sim.n_paths
        if sim.antithetic:
            if n_paths % 2 != 0:
                raise ValueError("n_paths must be even for antithetic variates")
            if not isinstance(model, GBMModel):
                raise ValueError("antithetic variates are implemented for GBMModel only")
            if sim.use_qmc:
                raise ValueError("antithetic variates are not supported with QMC")

        if sim.use_qmc:
            if not isinstance(model, GBMModel):
                raise ValueError("QMC is currently supported for GBMModel only")
            terminal = self._simulate_qmc(model, market, sim)
        else:
            if isinstance(model, GBMModel):
                if sim.antithetic:
                    terminal = self._simulate_antithetic_gbm(model, market, sim, rng)
                else:
                    terminal = model.simulate_paths(market, sim, rng, terminal_only=True)
            elif isinstance(model, HestonModel):
                if sim.antithetic:
                    raise ValueError("antithetic variates are not supported for HestonModel")
                spot_paths, _neg_freq = model.simulate_spot_paths(market, sim, rng)
                terminal = spot_paths[:, -1]
            else:
                raise TypeError("Unsupported model type")

        payoffs = payoff(terminal)
        method_flags = ["mc"]

        if sim.antithetic:
            payoffs = antithetic_pairs(payoffs)
            method_flags.append("antithetic")

        if sim.use_control_variate:
            payoffs = self._apply_control_variate(
                model, market, sim, payoff, payoffs, terminal
            )
            method_flags.append("control_variate")

        discount = np.exp(-market.rate * sim.maturity)
        price = discount * float(np.mean(payoffs))
        stderr = discount * float(np.std(payoffs, ddof=1) / np.sqrt(len(payoffs)))
        ci_low, ci_high = self._confidence_interval(price, stderr)

        runtime = time.perf_counter() - start
        return PricingResult(
            price=price,
            stderr=stderr,
            ci_low=ci_low,
            ci_high=ci_high,
            n_paths=len(payoffs),
            runtime_sec=runtime,
            method="+".join(method_flags),
        )

    def _confidence_interval(self, price: float, stderr: float) -> tuple[float, float]:
        alpha = 1.0 - self.config.confidence
        z = float(norm.ppf(1.0 - alpha / 2.0))
        return price - z * stderr, price + z * stderr

    def _apply_control_variate(
        self,
        model: GBMModel | HestonModel,
        market: MarketParams,
        sim: SimulationParams,
        payoff: Payoff,
        payoffs: npt.NDArray[np.floating[Any]],
        terminal: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        if not isinstance(model, GBMModel):
            return payoffs

        if not hasattr(payoff, "strike") or not hasattr(payoff, "option_type"):
            return payoffs

        strike = float(payoff.strike)
        option_type = str(payoff.option_type)

        analytic = bs_price(
            spot=market.spot,
            strike=strike,
            maturity=sim.maturity,
            rate=market.rate,
            dividend=market.dividend,
            sigma=model.params.sigma,
            option_type=option_type,
        )
        if option_type == "call":
            control = cast(npt.NDArray[np.floating[Any]], np.maximum(terminal - strike, 0.0))
        else:
            control = cast(npt.NDArray[np.floating[Any]], np.maximum(strike - terminal, 0.0))
        return cast(
            npt.NDArray[np.floating[Any]],
            control_variate_adjust(payoffs, control, analytic),
        )

    def _simulate_qmc(
        self,
        model: GBMModel,
        market: MarketParams,
        sim: SimulationParams,
    ) -> npt.NDArray[np.floating[Any]]:
        from ..variance_reduction.qmc import sobol_normals

        n_paths = sim.n_paths
        n_steps = 1
        dt = sim.maturity / n_steps
        z = sobol_normals(n_paths, n_steps, seed=sim.seed)
        drift = (market.rate - market.dividend - 0.5 * model.params.sigma**2) * dt
        vol = model.params.sigma * np.sqrt(dt)
        log_return = drift + vol * z[:, 0]
        return cast(npt.NDArray[np.floating[Any]], market.spot * np.exp(log_return))

    def _simulate_antithetic_gbm(
        self,
        model: GBMModel,
        market: MarketParams,
        sim: SimulationParams,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.floating[Any]]:
        half = sim.n_paths // 2
        dt = sim.maturity
        z = rng.standard_normal(size=half)
        drift = (market.rate - market.dividend - 0.5 * model.params.sigma**2) * dt
        vol = model.params.sigma * np.sqrt(dt)
        log_return = drift + vol * z
        log_return_anti = drift - vol * z
        terminal = market.spot * np.exp(np.concatenate([log_return, log_return_anti]))
        return cast(npt.NDArray[np.floating[Any]], terminal)

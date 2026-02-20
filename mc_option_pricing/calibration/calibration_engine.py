from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from ..config import MarketParams, SimulationParams
from ..models.heston import HestonModel, HestonParams
from ..payoffs.vanilla import VanillaPayoff
from ..pricers.european_mc import EuropeanMCPricer
from .base import CalibrationResult
from .implied_vol import implied_vol_surface


@dataclass(frozen=True)
class CalibrationConfig:
    option_type: str = "call"
    n_starts: int = 5
    objective: str = "iv_sse"
    train_fraction: float = 0.7


class HestonCalibrator:
    def __init__(
        self,
        market: MarketParams,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        sim: SimulationParams,
        config: CalibrationConfig | None = None,
    ) -> None:
        self.market = market
        self.strikes = strikes
        self.maturities = maturities
        self.market_prices = market_prices
        self.sim = sim
        self.config = config or CalibrationConfig()
        self._train_idx, self._val_idx = self._split_maturities()

    def _split_maturities(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.maturities)
        cut = max(1, int(self.config.train_fraction * n))
        train_idx = np.arange(cut)
        val_idx = np.arange(cut, n)
        return train_idx, val_idx

    def calibrate(self) -> CalibrationResult:
        best = None
        rng = np.random.default_rng(123)

        for _ in range(self.config.n_starts):
            x0 = np.array(
                [
                    rng.uniform(0.5, 5.0),
                    rng.uniform(0.01, 0.2),
                    rng.uniform(0.1, 1.0),
                    rng.uniform(-0.8, 0.8),
                    rng.uniform(0.01, 0.2),
                ]
            )

            res = minimize(
                lambda x: self._objective(x),
                x0=x0,
                bounds=[
                    (1e-3, 10.0),
                    (1e-3, 1.0),
                    (1e-3, 5.0),
                    (-0.999, 0.999),
                    (1e-4, 1.0),
                ],
                method="L-BFGS-B",
            )
            if best is None or res.fun < best.fun:
                best = res

        assert best is not None
        params = self._vector_to_params(best.x)
        eval_error = self._evaluate(params)
        return CalibrationResult(
            params=params.__dict__,
            objective=float(best.fun),
            success=bool(best.success),
            message=str(best.message),
            extra={"feller": params.feller_condition(), "val_error": eval_error},
        )

    def _objective(self, x: np.ndarray) -> float:
        params = self._vector_to_params(x)
        model = HestonModel(params)
        prices = self._model_prices(model)
        prices_train = prices[self._train_idx]
        market_train = self.market_prices[self._train_idx]
        if self.config.objective == "price_sse":
            diff = prices_train - market_train
            return float(np.mean(diff**2))

        market_iv = implied_vol_surface(
            market_train,
            self.market.spot,
            self.strikes,
            self.maturities[self._train_idx],
            self.market.rate,
            self.market.dividend,
            self.config.option_type,
        )
        model_iv = implied_vol_surface(
            prices_train,
            self.market.spot,
            self.strikes,
            self.maturities[self._train_idx],
            self.market.rate,
            self.market.dividend,
            self.config.option_type,
        )
        diff = model_iv - market_iv
        return float(np.mean(diff**2))

    def _evaluate(self, params: HestonParams) -> float:
        model = HestonModel(params)
        prices = self._model_prices(model)
        if self._val_idx.size == 0:
            return float("nan")
        prices_val = prices[self._val_idx]
        market_val = self.market_prices[self._val_idx]
        if self.config.objective == "price_sse":
            diff = prices_val - market_val
            return float(np.mean(diff**2))
        market_iv = implied_vol_surface(
            market_val,
            self.market.spot,
            self.strikes,
            self.maturities[self._val_idx],
            self.market.rate,
            self.market.dividend,
            self.config.option_type,
        )
        model_iv = implied_vol_surface(
            prices_val,
            self.market.spot,
            self.strikes,
            self.maturities[self._val_idx],
            self.market.rate,
            self.market.dividend,
            self.config.option_type,
        )
        diff = model_iv - market_iv
        return float(np.mean(diff**2))

    def _model_prices(self, model: HestonModel) -> np.ndarray:
        pricer = EuropeanMCPricer()
        prices = np.zeros_like(self.market_prices)
        for i, t in enumerate(self.maturities):
            sim = SimulationParams(
                n_paths=self.sim.n_paths,
                n_steps=self.sim.n_steps,
                maturity=float(t),
                seed=self.sim.seed,
                dtype=self.sim.dtype,
            )
            for j, k in enumerate(self.strikes):
                payoff = VanillaPayoff(strike=float(k), option_type=self.config.option_type)
                res = pricer.price(model, self.market, payoff, sim)
                prices[i, j] = res.price
        return prices

    def _vector_to_params(self, x: np.ndarray) -> HestonParams:
        return HestonParams(
            kappa=float(x[0]),
            theta=float(x[1]),
            xi=float(x[2]),
            rho=float(x[3]),
            v0=float(x[4]),
        )

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import MarketParams, SimulationParams
from ..models.gbm import GBMModel
from ..payoffs.vanilla import VanillaPayoff
from ..rng import make_rng
from .base import PricingResult


@dataclass(frozen=True)
class LSMConfig:
    basis: str = "poly"
    ridge: float = 1e-8
    train_fraction: float = 0.5


class LSMAmericanPricer:
    def __init__(self, config: LSMConfig | None = None) -> None:
        self.config = config or LSMConfig()

    def price(
        self,
        model: GBMModel,
        market: MarketParams,
        payoff: VanillaPayoff,
        sim: SimulationParams,
    ) -> PricingResult:
        if payoff.option_type not in {"call", "put"}:
            raise ValueError("LSM supports call/put only")

        start = time.perf_counter()
        rng = make_rng(sim.seed)
        market.validate()
        sim.validate()

        paths = model.simulate_paths(market, sim, rng, terminal_only=False)
        n_paths, n_steps_plus = paths.shape
        n_steps = n_steps_plus - 1
        dt = sim.maturity / n_steps
        discount = np.exp(-market.rate * dt)

        n_train = int(self.config.train_fraction * n_paths)
        if n_train <= 0 or n_train >= n_paths:
            raise ValueError("train_fraction must yield non-empty train and eval sets")

        train_paths = paths[:n_train]
        eval_paths = paths[n_train:]

        train_cashflows, coefs = self._fit_policy(train_paths, payoff, discount)
        eval_cashflows = self._apply_policy(eval_paths, payoff, discount, coefs)

        price_in = float(np.mean(train_cashflows)) * np.exp(-market.rate * sim.maturity)
        price_out = float(np.mean(eval_cashflows)) * np.exp(-market.rate * sim.maturity)
        stderr = float(np.std(eval_cashflows, ddof=1) / np.sqrt(len(eval_cashflows))) * np.exp(
            -market.rate * sim.maturity
        )

        runtime = time.perf_counter() - start
        return PricingResult(
            price=price_out,
            stderr=stderr,
            ci_low=price_out - 1.96 * stderr,
            ci_high=price_out + 1.96 * stderr,
            n_paths=len(eval_cashflows),
            runtime_sec=runtime,
            method="lsm",
            extra={"in_sample_price": price_in},
        )

    def _fit_policy(
        self, paths: np.ndarray, payoff: VanillaPayoff, discount: float
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        n_paths, n_steps_plus = paths.shape
        n_steps = n_steps_plus - 1
        cashflows = payoff(paths)
        coefs: list[np.ndarray] = [np.zeros(1) for _ in range(n_steps_plus)]

        for t in range(n_steps - 1, 0, -1):
            s_t = paths[:, t]
            itm = self._in_the_money(s_t, payoff)
            if not np.any(itm):
                cashflows *= discount
                continue

            x = self._basis(s_t[itm])
            y = cashflows[itm] * discount
            coef = self._ridge_regression(x, y)
            coefs[t] = coef
            continuation = x @ coef
            exercise = payoff_value(s_t[itm], payoff)
            exercise_now = exercise > continuation

            cashflows *= discount
            idx_itm = np.where(itm)[0]
            cashflows[idx_itm[exercise_now]] = exercise[exercise_now]

        return cashflows, coefs

    def _apply_policy(
        self,
        paths: np.ndarray,
        payoff: VanillaPayoff,
        discount: float,
        coefs: list[np.ndarray],
    ) -> np.ndarray:
        n_paths, n_steps_plus = paths.shape
        n_steps = n_steps_plus - 1
        cashflows = payoff(paths)

        for t in range(n_steps - 1, 0, -1):
            coef = coefs[t]
            if coef.size == 1:
                cashflows *= discount
                continue

            s_t = paths[:, t]
            itm = self._in_the_money(s_t, payoff)
            if not np.any(itm):
                cashflows *= discount
                continue

            x = self._basis(s_t[itm])
            continuation = x @ coef
            exercise = payoff_value(s_t[itm], payoff)
            exercise_now = exercise > continuation

            cashflows *= discount
            idx_itm = np.where(itm)[0]
            cashflows[idx_itm[exercise_now]] = exercise[exercise_now]

        return cashflows

    def _basis(self, s: np.ndarray) -> np.ndarray:
        if self.config.basis == "poly":
            return np.column_stack([np.ones_like(s), s, s**2])
        raise ValueError("unknown basis")

    def _ridge_regression(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        reg = self.config.ridge
        xtx = x.T @ x + reg * np.eye(x.shape[1])
        xty = x.T @ y
        return np.linalg.solve(xtx, xty)

    def _in_the_money(self, s: np.ndarray, payoff: VanillaPayoff) -> np.ndarray:
        if payoff.option_type == "call":
            return s > payoff.strike
        return s < payoff.strike


def payoff_value(s: np.ndarray, payoff: VanillaPayoff) -> np.ndarray:
    if payoff.option_type == "call":
        return np.maximum(s - payoff.strike, 0.0)
    return np.maximum(payoff.strike - s, 0.0)

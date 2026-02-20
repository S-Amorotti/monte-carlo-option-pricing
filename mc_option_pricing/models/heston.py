from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import MarketParams, SimulationParams
from .base import ModelParams


@dataclass(frozen=True)
class HestonParams(ModelParams):
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def validate(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        if self.xi <= 0.0:
            raise ValueError("xi must be positive")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError("rho must be in [-1, 1]")
        if self.v0 <= 0.0:
            raise ValueError("v0 must be positive")

    def feller_condition(self) -> float:
        return 2.0 * self.kappa * self.theta - self.xi**2


class HestonModel:
    def __init__(self, params: HestonParams) -> None:
        params.validate()
        self.params = params

    def simulate_paths(
        self,
        market: MarketParams,
        sim: SimulationParams,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        market.validate()
        sim.validate()

        n_paths = sim.n_paths
        n_steps = sim.n_steps
        dt = sim.maturity / n_steps
        dtype = np.float64 if sim.dtype == "float64" else np.float32

        z1 = rng.standard_normal(size=(n_paths, n_steps), dtype=dtype)
        z2 = rng.standard_normal(size=(n_paths, n_steps), dtype=dtype)
        z2 = self.params.rho * z1 + np.sqrt(1.0 - self.params.rho**2) * z2

        s = np.empty((n_paths, n_steps + 1), dtype=dtype)
        v = np.empty((n_paths, n_steps + 1), dtype=dtype)
        s[:, 0] = dtype(market.spot)
        v[:, 0] = dtype(self.params.v0)

        neg_count = 0
        for t in range(n_steps):
            v_pos = np.maximum(v[:, t], 0.0)
            neg_count += int(np.sum(v[:, t] < 0.0))

            dv = (
                self.params.kappa * (self.params.theta - v_pos) * dt
                + self.params.xi * np.sqrt(v_pos) * np.sqrt(dt) * z2[:, t]
            )
            v[:, t + 1] = v[:, t] + dv

            s[:, t + 1] = s[:, t] * np.exp(
                (market.rate - market.dividend - 0.5 * v_pos) * dt
                + np.sqrt(v_pos) * np.sqrt(dt) * z1[:, t]
            )

        neg_freq = neg_count / (n_paths * n_steps)
        return s, v, neg_freq

    def simulate_spot_paths(
        self,
        market: MarketParams,
        sim: SimulationParams,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        s, _v, neg_freq = self.simulate_paths(market, sim, rng)
        return s, neg_freq

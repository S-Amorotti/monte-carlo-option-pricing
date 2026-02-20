from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import MarketParams, SimulationParams
from .base import ModelParams


@dataclass(frozen=True)
class MertonParams(ModelParams):
    sigma: float
    jump_intensity: float
    jump_mean: float
    jump_std: float

    def validate(self) -> None:
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if self.jump_intensity < 0.0:
            raise ValueError("jump_intensity must be non-negative")
        if self.jump_std < 0.0:
            raise ValueError("jump_std must be non-negative")


class MertonModel:
    """Scaffold for Merton jump-diffusion.

    This implementation is intentionally minimal and is not wired into pricers yet.
    """

    def __init__(self, params: MertonParams) -> None:
        params.validate()
        self.params = params

    def simulate_paths(
        self,
        market: MarketParams,
        sim: SimulationParams,
        rng: np.random.Generator,
    ) -> np.ndarray:
        market.validate()
        sim.validate()

        n_paths = sim.n_paths
        n_steps = sim.n_steps
        dt = sim.maturity / n_steps
        dtype = np.float64 if sim.dtype == "float64" else np.float32

        z = rng.standard_normal(size=(n_paths, n_steps), dtype=dtype)
        n_jumps = rng.poisson(self.params.jump_intensity * dt, size=(n_paths, n_steps))
        jump_sizes = rng.normal(
            loc=self.params.jump_mean,
            scale=self.params.jump_std,
            size=(n_paths, n_steps),
        )

        drift_adj = self.params.jump_intensity * (
            np.exp(self.params.jump_mean + 0.5 * self.params.jump_std**2) - 1.0
        )
        drift = (market.rate - market.dividend - 0.5 * self.params.sigma**2 - drift_adj) * dt
        diffusion = self.params.sigma * np.sqrt(dt) * z
        jumps = n_jumps * jump_sizes

        increments = drift + diffusion + jumps
        log_paths = np.cumsum(increments, axis=1)
        s0 = dtype(market.spot)
        paths = s0 * np.exp(log_paths)
        paths = np.concatenate([np.full((n_paths, 1), s0, dtype=dtype), paths], axis=1)
        return paths

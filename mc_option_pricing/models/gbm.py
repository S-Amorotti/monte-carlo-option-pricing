from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing import Any, cast

from ..config import MarketParams, SimulationParams
from .base import ModelParams


@dataclass(frozen=True)
class GBMParams(ModelParams):
    sigma: float

    def validate(self) -> None:
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")


class GBMModel:
    def __init__(self, params: GBMParams) -> None:
        params.validate()
        self.params = params

    def simulate_paths(
        self,
        market: MarketParams,
        sim: SimulationParams,
        rng: np.random.Generator,
        terminal_only: bool = False,
    ) -> npt.NDArray[np.floating[Any]]:
        market.validate()
        sim.validate()

        n_paths = sim.n_paths
        n_steps = 1 if terminal_only else sim.n_steps
        dt = sim.maturity / n_steps
        dtype = np.float64 if sim.dtype == "float64" else np.float32

        z = rng.standard_normal(size=(n_paths, n_steps), dtype=dtype)
        drift = (market.rate - market.dividend - 0.5 * self.params.sigma**2) * dt
        vol = self.params.sigma * np.sqrt(dt)

        increments = drift + vol * z
        log_paths = np.cumsum(increments, axis=1)
        s0 = dtype(market.spot)
        paths = s0 * np.exp(log_paths)
        if terminal_only:
            return cast(npt.NDArray[np.floating[Any]], paths[:, -1])

        paths = np.concatenate([np.full((n_paths, 1), s0, dtype=dtype), paths], axis=1)
        return cast(npt.NDArray[np.floating[Any]], paths)

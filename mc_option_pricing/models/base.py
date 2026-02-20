from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config import MarketParams, SimulationParams


class PathModel(Protocol):
    def simulate_paths(
        self,
        market: MarketParams,
        sim: SimulationParams,
        rng: np.random.Generator,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ModelParams:
    def validate(self) -> None:
        return None

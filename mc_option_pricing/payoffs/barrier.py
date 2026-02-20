from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BarrierUpAndOutCall:
    strike: float
    barrier: float

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.barrier <= self.strike:
            raise ValueError("barrier must be greater than strike")

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim != 2:
            raise ValueError("paths must be 2D for barrier payoff")
        breached = np.any(paths[:, 1:] >= self.barrier, axis=1)
        terminal = paths[:, -1]
        payoff = np.maximum(terminal - self.strike, 0.0)
        payoff[breached] = 0.0
        return payoff

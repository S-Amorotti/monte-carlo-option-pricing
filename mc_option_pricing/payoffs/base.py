from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Payoff(Protocol):
    def __call__(self, paths: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class PayoffBase:
    strike: float

    def validate(self) -> None:
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import PayoffBase


@dataclass(frozen=True)
class VanillaPayoff(PayoffBase):
    option_type: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        super().validate()
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        s = paths[:, -1] if paths.ndim == 2 else paths
        if self.option_type == "call":
            return np.maximum(s - self.strike, 0.0)
        return np.maximum(self.strike - s, 0.0)


@dataclass(frozen=True)
class DigitalPayoff(PayoffBase):
    option_type: str
    payout: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        super().validate()
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.payout <= 0.0:
            raise ValueError("payout must be positive")

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        s = paths[:, -1] if paths.ndim == 2 else paths
        if self.option_type == "call":
            return self.payout * (s > self.strike)
        return self.payout * (s < self.strike)

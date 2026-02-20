from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .base import PayoffBase


@dataclass(frozen=True)
class AsianArithmeticPayoff(PayoffBase):
    option_type: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        super().validate()
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

    def __call__(
        self, paths: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        if paths.ndim != 2:
            raise ValueError("paths must be 2D for Asian payoff")
        avg = np.mean(paths[:, 1:], axis=1)
        if self.option_type == "call":
            return cast(npt.NDArray[np.floating[Any]], np.maximum(avg - self.strike, 0.0))
        return cast(npt.NDArray[np.floating[Any]], np.maximum(self.strike - avg, 0.0))

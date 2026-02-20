from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketParams:
    spot: float
    rate: float
    dividend: float = 0.0

    def validate(self) -> None:
        if self.spot <= 0.0:
            raise ValueError("spot must be positive")
        if not np.isfinite(self.rate):
            raise ValueError("rate must be finite")
        if not np.isfinite(self.dividend):
            raise ValueError("dividend must be finite")


@dataclass(frozen=True)
class SimulationParams:
    n_paths: int
    n_steps: int
    maturity: float
    seed: int | None = 12345
    dtype: str = "float64"
    antithetic: bool = False
    use_control_variate: bool = False
    use_qmc: bool = False

    def validate(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.use_qmc and self.antithetic:
            raise ValueError("antithetic is not compatible with QMC in this implementation")

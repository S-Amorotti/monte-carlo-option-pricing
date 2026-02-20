from __future__ import annotations

import numpy as np


def clamp(x: float, low: float, high: float) -> float:
    return float(max(low, min(high, x)))


def ensure_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")

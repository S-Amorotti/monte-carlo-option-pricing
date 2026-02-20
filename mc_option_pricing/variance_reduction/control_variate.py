from __future__ import annotations

import numpy as np


def control_variate_adjust(
    target: np.ndarray, control: np.ndarray, control_expectation: float
) -> np.ndarray:
    if target.shape != control.shape:
        raise ValueError("target and control must have the same shape")
    cov = np.cov(target, control, ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    return target - beta * (control - control_expectation)

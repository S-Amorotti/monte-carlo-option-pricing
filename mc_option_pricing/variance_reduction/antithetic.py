from __future__ import annotations

import numpy as np


def antithetic_pairs(samples: np.ndarray) -> np.ndarray:
    if samples.size % 2 != 0:
        raise ValueError("samples length must be even for antithetic")
    half = samples.size // 2
    return 0.5 * (samples[:half] + samples[half:])

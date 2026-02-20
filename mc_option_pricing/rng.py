from __future__ import annotations

from typing import Optional

import numpy as np


def make_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a deterministic NumPy RNG.

    Uses PCG64 for reproducibility across platforms.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.PCG64(seed))

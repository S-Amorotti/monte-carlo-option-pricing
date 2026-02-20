from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt


def antithetic_pairs(samples: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    if samples.size % 2 != 0:
        raise ValueError("samples length must be even for antithetic")
    half = samples.size // 2
    return cast(npt.NDArray[np.floating[Any]], 0.5 * (samples[:half] + samples[half:]))

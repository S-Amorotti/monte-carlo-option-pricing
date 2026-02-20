from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from typing import Any, cast


def sobol_normals(
    n_paths: int, dim: int, seed: int | None = None
) -> npt.NDArray[np.float64]:
    sampler = Sobol(d=dim, scramble=True, seed=seed)
    u = sampler.random(n=n_paths)
    return cast(npt.NDArray[np.float64], norm.ppf(u))

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol


def sobol_normals(n_paths: int, dim: int, seed: int | None = None) -> np.ndarray:
    sampler = Sobol(d=dim, scramble=True, seed=seed)
    u = sampler.random(n=n_paths)
    return norm.ppf(u)

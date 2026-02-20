from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PricingResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    runtime_sec: float
    method: str
    extra: Optional[dict] = None

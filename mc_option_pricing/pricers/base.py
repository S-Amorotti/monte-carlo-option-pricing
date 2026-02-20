from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PricingResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    runtime_sec: float
    method: str
    extra: dict[str, Any] | None = None

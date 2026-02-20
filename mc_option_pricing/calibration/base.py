from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CalibrationResult:
    params: dict[str, Any]
    objective: float
    success: bool
    message: str
    extra: dict[str, Any]


class Calibrator(Protocol):
    def calibrate(self) -> CalibrationResult:
        ...

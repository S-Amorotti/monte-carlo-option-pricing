from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CalibrationResult:
    params: dict
    objective: float
    success: bool
    message: str
    extra: dict


class Calibrator(Protocol):
    def calibrate(self) -> CalibrationResult:
        ...

"""Problem adapter contracts shared by MATLAB and validation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .archive import EvaluationArchive
from .config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    X: np.ndarray
    F: np.ndarray
    C: np.ndarray


@dataclass(frozen=True, slots=True)
class Stage1Output:
    archive: EvaluationArchive
    lower: np.ndarray
    upper: np.ndarray


class ProblemAdapter(Protocol):
    def run_stage1(self, config: ExperimentConfig) -> Stage1Output:
        """Run the configured base EA and return every true evaluation."""

    def evaluate(self, X: np.ndarray, config: ExperimentConfig) -> EvaluationBatch:
        """Repair and truly evaluate one batch without hiding FE increments."""

    def reference_set(self, count: int, config: ExperimentConfig) -> np.ndarray:
        """Return a dense PF reference set in objective space."""

    def bounds(self, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
        """Return decision bounds without consuming function evaluations."""

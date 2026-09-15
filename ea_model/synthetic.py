"""Deterministic validation backend; formal experiments should use PlatEMO."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .archive import EvaluationArchive
from .config import ExperimentConfig
from .interfaces import EvaluationBatch, Stage1Output


def dtlz2(X: np.ndarray, n_obj: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < n_obj:
        raise ValueError("DTLZ2 requires D >= M")
    g = np.sum((X[:, n_obj - 1 :] - 0.5) ** 2, axis=1)
    F = np.tile((1.0 + g)[:, None], (1, n_obj))
    for objective in range(n_obj):
        cosine_count = n_obj - objective - 1
        if cosine_count:
            F[:, objective] *= np.prod(
                np.cos(X[:, :cosine_count] * np.pi / 2.0), axis=1
            )
        if objective:
            F[:, objective] *= np.sin(X[:, cosine_count] * np.pi / 2.0)
    return F


class DeterministicDTLZ2Adapter:
    """Validation-only quasi-random Stage 1 with the same adapter contract."""

    def run_stage1(self, config: ExperimentConfig) -> Stage1Output:
        dimension = config.D or config.M + 9
        sampler = qmc.LatinHypercube(d=dimension, seed=config.seed)
        X = sampler.random(config.ea_budget)
        F = dtlz2(X, config.M)
        archive = EvaluationArchive(
            X,
            F,
            np.arange(1, len(X) + 1),
            np.full(len(X), "EA"),
            np.zeros((len(X), 0)),
        )
        return Stage1Output(archive, np.zeros(dimension), np.ones(dimension))

    def evaluate(self, X: np.ndarray, config: ExperimentConfig) -> EvaluationBatch:
        repaired = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        return EvaluationBatch(repaired, dtlz2(repaired, config.M), np.zeros((len(repaired), 0)))

    def reference_set(self, count: int, config: ExperimentConfig) -> np.ndarray:
        rng = np.random.default_rng(config.seed + 104729)
        positive = rng.exponential(size=(count, config.M))
        return positive / np.linalg.norm(positive, axis=1, keepdims=True)

    def bounds(self, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
        dimension = config.D or config.M + 9
        return np.zeros(dimension), np.ones(dimension)

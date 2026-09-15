"""Unbounded archive and Pareto filtering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _as_2d(values: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1 and array.size == 0:
        array = array.reshape(0, 0)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def nondominated_mask(objectives: np.ndarray, constraints: np.ndarray | None = None) -> np.ndarray:
    """Return a minimization, feasibility-first non-dominated mask."""
    F = _as_2d(objectives, "objectives")
    n = len(F)
    if n == 0:
        return np.zeros(0, dtype=bool)

    if constraints is None or np.asarray(constraints).size == 0:
        cv = np.zeros(n)
    else:
        C = _as_2d(constraints, "constraints")
        if len(C) != n:
            raise ValueError("objectives and constraints must have equal row counts")
        cv = np.maximum(C, 0.0).sum(axis=1)

    feasible = cv <= 1e-12
    keep = np.zeros(n, dtype=bool)
    candidates = np.flatnonzero(feasible)
    if len(candidates) == 0:
        minimum = np.min(cv)
        return np.isclose(cv, minimum, rtol=0.0, atol=1e-12)

    Fc = F[candidates]
    local_keep = np.ones(len(candidates), dtype=bool)
    for i, point in enumerate(Fc):
        dominates_i = np.all(Fc <= point, axis=1) & np.any(Fc < point, axis=1)
        if np.any(dominates_i):
            local_keep[i] = False
    keep[candidates[local_keep]] = True
    return keep


@dataclass(slots=True)
class EvaluationArchive:
    """Every true evaluation, including dominated and duplicate records."""

    X: np.ndarray
    F: np.ndarray
    FE: np.ndarray
    source: np.ndarray
    C: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.X = _as_2d(self.X, "X")
        self.F = _as_2d(self.F, "F")
        self.FE = np.asarray(self.FE, dtype=np.int64).reshape(-1)
        self.source = np.asarray(self.source, dtype="U16").reshape(-1)
        if self.C is None:
            self.C = np.zeros((len(self.X), 0), dtype=float)
        else:
            self.C = _as_2d(self.C, "C")
        self.validate()

    @classmethod
    def empty(cls, n_var: int, n_obj: int, n_con: int = 0) -> "EvaluationArchive":
        return cls(
            np.empty((0, n_var)),
            np.empty((0, n_obj)),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype="U16"),
            np.empty((0, n_con)),
        )

    def validate(self) -> None:
        n = len(self.X)
        if len(self.F) != n or len(self.FE) != n or len(self.source) != n or len(self.C) != n:
            raise ValueError("X, F, C, FE, and source must have equal row counts")
        if n and (np.any(self.FE <= 0) or np.any(np.diff(self.FE) <= 0)):
            raise ValueError("FE indices must be positive and strictly increasing")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def last_fe(self) -> int:
        return int(self.FE[-1]) if len(self) else 0

    def append(
        self,
        X: np.ndarray,
        F: np.ndarray,
        source: str,
        C: np.ndarray | None = None,
    ) -> np.ndarray:
        X_new = _as_2d(X, "X")
        F_new = _as_2d(F, "F")
        if X_new.shape[1] != self.X.shape[1] or F_new.shape[1] != self.F.shape[1]:
            raise ValueError("appended X/F dimensions do not match the archive")
        if len(X_new) != len(F_new):
            raise ValueError("appended X and F must have equal row counts")
        if C is None:
            C_new = np.zeros((len(X_new), self.C.shape[1]))
        else:
            C_new = _as_2d(C, "C")
            if C_new.shape != (len(X_new), self.C.shape[1]):
                raise ValueError("appended constraint dimensions do not match the archive")
        indices = np.arange(self.last_fe + 1, self.last_fe + len(X_new) + 1, dtype=np.int64)
        self.X = np.vstack((self.X, X_new))
        self.F = np.vstack((self.F, F_new))
        self.C = np.vstack((self.C, C_new))
        self.FE = np.concatenate((self.FE, indices))
        self.source = np.concatenate((self.source, np.full(len(X_new), source, dtype="U16")))
        self.validate()
        return indices

    def subset(self, indices: np.ndarray) -> "EvaluationArchive":
        return EvaluationArchive(
            self.X[indices], self.F[indices], self.FE[indices], self.source[indices], self.C[indices]
        )

    def without_duplicates(self) -> "EvaluationArchive":
        if not len(self):
            return self
        key = np.hstack((self.X, self.F, self.C))
        _, first = np.unique(key, axis=0, return_index=True)
        return self.subset(np.sort(first))

    def nondominated(self, remove_duplicates: bool = True) -> "EvaluationArchive":
        archive = self.without_duplicates() if remove_duplicates else self
        return archive.subset(np.flatnonzero(nondominated_mask(archive.F, archive.C)))


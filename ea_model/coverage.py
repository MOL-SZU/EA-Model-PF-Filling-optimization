"""Distance-based PF coverage and hole localization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
    return values


@dataclass(frozen=True, slots=True)
class HoleScores:
    reference_set: np.ndarray
    distances: np.ndarray
    nearest_archive_indices: np.ndarray

    @property
    def igd_infinity(self) -> float:
        return float(np.max(self.distances))

    @property
    def largest_index(self) -> int:
        return int(np.argmax(self.distances))


def calculate_hole_distances(reference_set: np.ndarray, archive_objectives: np.ndarray) -> HoleScores:
    Z = _validate_points(reference_set, "reference_set")
    F = _validate_points(archive_objectives, "archive_objectives")
    if Z.shape[1] != F.shape[1]:
        raise ValueError("reference and objective dimensions must match")
    pairwise = cdist(Z, F, metric="euclidean")
    nearest = np.argmin(pairwise, axis=1)
    return HoleScores(Z, pairwise[np.arange(len(Z)), nearest], nearest)


def _automatic_separation(reference_set: np.ndarray) -> float:
    if len(reference_set) < 2:
        return 0.0
    distances = cdist(reference_set, reference_set)
    np.fill_diagonal(distances, np.inf)
    spacing = np.median(np.min(distances, axis=1))
    return float(2.0 * spacing)


def select_multiple_holes(
    scores: HoleScores,
    count: int,
    method: str = "separated_topB",
    min_separation: float | None = None,
) -> np.ndarray:
    """Select project-specific top-B hole targets by descending h(z)."""
    if count < 1:
        raise ValueError("count must be positive")
    if method not in {"topB", "separated_topB"}:
        raise ValueError("unknown hole selection method")
    order = np.argsort(-scores.distances, kind="stable")
    count = min(count, len(order))
    if method == "topB":
        return order[:count]

    separation = _automatic_separation(scores.reference_set) if min_separation is None else min_separation
    selected: list[int] = []
    for index in order:
        if not selected or np.all(
            np.linalg.norm(scores.reference_set[selected] - scores.reference_set[index], axis=1)
            >= separation
        ):
            selected.append(int(index))
            if len(selected) == count:
                break
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(int(i) for i in order if int(i) not in selected_set)
    return np.asarray(selected[:count], dtype=int)


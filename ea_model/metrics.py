"""IGD-infinity, IGD, and hypervolume metrics for minimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .archive import nondominated_mask
from .coverage import calculate_hole_distances


@dataclass(frozen=True, slots=True)
class MetricValues:
    igd_infinity: float
    igd: float
    hv: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def igd_infinity(reference_set: np.ndarray, approximation: np.ndarray) -> float:
    return calculate_hole_distances(reference_set, approximation).igd_infinity


def igd(reference_set: np.ndarray, approximation: np.ndarray) -> float:
    return float(np.mean(calculate_hole_distances(reference_set, approximation).distances))


def default_hv_reference(reference_set: np.ndarray) -> np.ndarray:
    Z = np.asarray(reference_set, dtype=float)
    span = np.ptp(Z, axis=0)
    margin = np.where(span > 0, 0.1 * span, 1.0)
    return np.max(Z, axis=0) + margin


def hypervolume(approximation: np.ndarray, reference_point: np.ndarray) -> float:
    F = np.asarray(approximation, dtype=float)
    ref = np.asarray(reference_point, dtype=float).reshape(-1)
    if F.ndim != 2 or F.shape[1] != len(ref) or not np.all(np.isfinite(F)):
        raise ValueError("invalid approximation or hypervolume reference point")
    F = F[np.all(F < ref, axis=1)]
    if not len(F):
        return 0.0
    F = F[nondominated_mask(F)]
    try:
        from pymoo.indicators.hv import HV

        return float(HV(ref_point=ref)(F))
    except ImportError:
        if F.shape[1] != 2:
            raise RuntimeError("pymoo is required for hypervolume with more than two objectives")
        order = np.argsort(F[:, 0])
        area = 0.0
        previous_y = ref[1]
        for x, y in F[order]:
            if y < previous_y:
                area += (ref[0] - x) * (previous_y - y)
                previous_y = y
        return float(area)


def compute_metrics(
    reference_set: np.ndarray,
    approximation: np.ndarray,
    hv_reference_point: np.ndarray | None = None,
) -> MetricValues:
    ref = default_hv_reference(reference_set) if hv_reference_point is None else hv_reference_point
    return MetricValues(
        igd_infinity(reference_set, approximation),
        igd(reference_set, approximation),
        hypervolume(approximation, ref),
    )

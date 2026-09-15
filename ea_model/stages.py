"""Independently callable steps of the EA+Model workflow."""

from __future__ import annotations

import numpy as np

from .archive import EvaluationArchive
from .config import ExperimentConfig
from .coverage import HoleScores, calculate_hole_distances, select_multiple_holes
from .interfaces import EvaluationBatch, ProblemAdapter, Stage1Output
from .metrics import MetricValues, compute_metrics
from .psm import PSMDataset, TrainedPSM, build_psm_dataset, repair_candidates, train_psm


def run_stage1_ea(adapter: ProblemAdapter, config: ExperimentConfig) -> Stage1Output:
    return adapter.run_stage1(config)


def run_extract_archive(archive: EvaluationArchive) -> EvaluationArchive:
    return archive.nondominated(remove_duplicates=True)


def run_generate_reference_set(adapter: ProblemAdapter, config: ExperimentConfig) -> np.ndarray:
    Z = np.asarray(adapter.reference_set(config.reference_point_num, config), dtype=float)
    if Z.ndim != 2 or Z.shape[1] != config.M or not np.all(np.isfinite(Z)):
        raise ValueError("the PF reference set is invalid or has the wrong objective dimension")
    return Z


def run_hole_detection(
    reference_set: np.ndarray,
    nd_objectives: np.ndarray,
    count: int,
    method: str,
    min_separation: float | None = None,
) -> tuple[HoleScores, np.ndarray]:
    scores = calculate_hole_distances(reference_set, nd_objectives)
    indices = select_multiple_holes(scores, count, method, min_separation)
    return scores, indices


def run_build_psm_dataset(
    archive: EvaluationArchive, reference_set: np.ndarray
) -> PSMDataset:
    nd = run_extract_archive(archive)
    return build_psm_dataset(nd.X, nd.F, reference_set)


def run_train_psm(
    dataset: PSMDataset,
    lower: np.ndarray,
    upper: np.ndarray,
    config: ExperimentConfig,
) -> TrainedPSM:
    return train_psm(dataset, lower, upper, config.model, config.seed)


def run_generate_candidates(
    model: TrainedPSM,
    targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return repair_candidates(model.predict(targets), lower, upper)


def run_true_evaluation(
    adapter: ProblemAdapter, candidates: np.ndarray, config: ExperimentConfig
) -> EvaluationBatch:
    return adapter.evaluate(candidates, config)


def run_archive_update(archive: EvaluationArchive, batch: EvaluationBatch) -> np.ndarray:
    return archive.append(batch.X, batch.F, source="model", C=batch.C)


def run_metrics(
    reference_set: np.ndarray,
    archive: EvaluationArchive,
    hv_reference_point: np.ndarray | None = None,
) -> MetricValues:
    nd = run_extract_archive(archive)
    return compute_metrics(reference_set, nd.F, hv_reference_point)


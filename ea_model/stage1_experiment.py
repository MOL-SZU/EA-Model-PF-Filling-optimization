"""Standalone Stage 1 execution and checkpointing without model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch

from .archive import EvaluationArchive
from .checkpoints import CheckpointManager
from .config import ExperimentConfig
from .interfaces import ProblemAdapter
from .metrics import MetricValues
from .stages import (
    run_extract_archive,
    run_generate_reference_set,
    run_metrics,
    run_stage1_ea,
)


@dataclass(slots=True)
class Stage1ExperimentResult:
    archive: EvaluationArchive
    nd_archive: EvaluationArchive
    reference_set: np.ndarray
    metrics: MetricValues
    result_directory: Path


def run_stage1_experiment(
    config: ExperimentConfig, adapter: ProblemAdapter
) -> Stage1ExperimentResult:
    """Run and save only the EA portion of an experiment."""
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    result_directory = Path(config.results_dir)
    checkpoints = CheckpointManager(result_directory)
    config.to_json(result_directory / "config_resolved.json")

    stage1 = run_stage1_ea(adapter, config)
    archive = stage1.archive
    if len(archive) != config.ea_budget or archive.last_fe != config.ea_budget:
        raise RuntimeError(
            f"Stage 1 recorded {len(archive)} evaluations with last FE "
            f"{archive.last_fe}; expected exactly {config.ea_budget}"
        )

    nd_archive = run_extract_archive(archive)
    if len(nd_archive) < 2:
        raise RuntimeError("Stage 1 produced fewer than two non-dominated solutions")
    reference_set = run_generate_reference_set(adapter, config)
    hv_reference = (
        None
        if config.hv_reference_point is None
        else np.asarray(config.hv_reference_point, dtype=float)
    )
    metrics = run_metrics(reference_set, archive, hv_reference)

    checkpoints.save_archive("stage1_archive.mat", archive)
    checkpoints.save_archive("stage1_nd_archive.mat", nd_archive)
    checkpoints.save_reference_set(reference_set)
    checkpoints.save_bounds(stage1.lower, stage1.upper)
    checkpoints.save_json(
        "stage1_summary.json",
        {
            "algorithm": config.algorithm,
            "problem": config.problem,
            "seed": config.seed,
            "stage1_FE": archive.last_fe,
            "archive_size": len(archive),
            "nd_archive_size": len(nd_archive),
            "metrics": metrics.to_dict(),
        },
    )
    return Stage1ExperimentResult(
        archive, nd_archive, reference_set, metrics, result_directory
    )

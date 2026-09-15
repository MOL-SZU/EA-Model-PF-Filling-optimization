"""End-to-end two-stage EA+Model pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .archive import EvaluationArchive, nondominated_mask
from .checkpoints import CheckpointManager
from .config import ExperimentConfig
from .interfaces import ProblemAdapter
from .metrics import MetricValues
from .stages import (
    run_archive_update,
    run_build_psm_dataset,
    run_extract_archive,
    run_generate_candidates,
    run_generate_reference_set,
    run_hole_detection,
    run_metrics,
    run_stage1_ea,
    run_train_psm,
    run_true_evaluation,
)


LOGGER = logging.getLogger("ea_model")


@dataclass(slots=True)
class FillingTrace:
    iteration: int
    fe_start: int
    fe_end: int
    targets: np.ndarray
    old_h: np.ndarray
    candidates: np.ndarray
    objectives: np.ndarray
    target_distances: np.ndarray
    duplicate_candidates: np.ndarray
    nondominated: np.ndarray
    new_h: np.ndarray
    old_igd_infinity: float
    new_igd_infinity: float


@dataclass(slots=True)
class ExperimentResult:
    archive: EvaluationArchive
    nd_archive: EvaluationArchive
    reference_set: np.ndarray
    stage1_metrics: MetricValues
    stage2_metrics: MetricValues
    trajectory: list[dict[str, float]]
    traces: list[FillingTrace]
    result_directory: Path


class EAModelPipeline:
    def __init__(self, config: ExperimentConfig, adapter: ProblemAdapter):
        config.validate()
        self.config = config
        self.adapter = adapter
        self.checkpoints = CheckpointManager(config.results_dir)

    def _configure_reproducibility(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        level = logging.DEBUG if self.config.validation_mode else logging.INFO
        if not logging.getLogger().handlers:
            logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
        LOGGER.setLevel(level)

    def run(self) -> ExperimentResult:
        self._configure_reproducibility()
        config = self.config
        result_dir = Path(config.results_dir)
        config.to_json(result_dir / "config_resolved.json")

        LOGGER.info(
            "Stage 1: %s + %s, N=%d, budget=%d", config.algorithm, config.problem, config.N, config.ea_budget
        )
        stage1 = run_stage1_ea(self.adapter, config)
        archive = stage1.archive
        if len(archive) != config.ea_budget or archive.last_fe != config.ea_budget:
            raise RuntimeError(
                f"Stage 1 recorded {len(archive)} evaluations, expected exactly {config.ea_budget}. "
                "The selected EA must respect the population-aligned PlatEMO budget."
            )
        if stage1.lower.shape != (archive.X.shape[1],) or stage1.upper.shape != (archive.X.shape[1],):
            raise ValueError("decision bounds returned by the adapter have the wrong dimension")
        self.checkpoints.save_archive("stage1_archive.mat", archive)
        self.checkpoints.save_bounds(stage1.lower, stage1.upper)

        nd_archive = run_extract_archive(archive)
        if len(nd_archive) < 2:
            raise RuntimeError("Stage 1 produced fewer than two non-dominated training samples")
        self.checkpoints.save_archive("stage1_nd_archive.mat", nd_archive)
        reference_set = run_generate_reference_set(self.adapter, config)
        self.checkpoints.save_reference_set(reference_set)
        hv_reference = (
            None if config.hv_reference_point is None else np.asarray(config.hv_reference_point, dtype=float)
        )
        stage1_metrics = run_metrics(reference_set, archive, hv_reference)
        trajectory = [{"FE": float(archive.last_fe), **stage1_metrics.to_dict()}]

        return self._run_stage2(
            archive,
            stage1.lower,
            stage1.upper,
            reference_set,
            stage1_metrics,
            trajectory,
        )

    def run_stage2_from_checkpoints(self) -> ExperimentResult:
        """Resume at model training from a completed, immutable Stage 1 archive."""
        self._configure_reproducibility()
        config = self.config
        result_dir = Path(config.results_dir)
        required = (
            result_dir / "config_resolved.json",
            result_dir / "stage1_archive.mat",
            result_dir / "stage1_nd_archive.mat",
            result_dir / "reference_set.mat",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Stage 2 requires completed Stage 1 checkpoints; missing: "
                + ", ".join(missing)
            )
        if (result_dir / "final_archive.mat").exists():
            raise RuntimeError(
                f"Stage 2 is already complete in {result_dir}; refusing to overwrite it"
            )
        partial = [
            path.name
            for path in (
                result_dir / "psm_model.pt",
                result_dir / "stage2_progress_archive.mat",
            )
            if path.exists()
        ]
        partial.extend(path.name for path in result_dir.glob("filling_iteration_*.mat"))
        if partial:
            raise RuntimeError(
                "Partial Stage 2 artifacts already exist and will not be overwritten: "
                + ", ".join(sorted(partial))
            )

        stored = json.loads((result_dir / "config_resolved.json").read_text(encoding="utf-8"))
        identity_fields = ("algorithm", "problem", "N", "M", "D", "maxFE", "EA_ratio", "seed")
        mismatches = [
            field
            for field in identity_fields
            if stored.get(field) != config.to_dict().get(field)
        ]
        if mismatches:
            raise RuntimeError(
                "Stage 2 configuration does not match its Stage 1 checkpoints: "
                + ", ".join(mismatches)
            )

        archive = self.checkpoints.load_archive(result_dir / "stage1_archive.mat")
        if len(archive) != config.ea_budget or archive.last_fe != config.ea_budget:
            raise RuntimeError(
                f"Stage 1 checkpoint has {len(archive)} rows and last FE {archive.last_fe}; "
                f"expected exactly {config.ea_budget}"
            )
        if archive.X.shape[1] != (config.D or archive.X.shape[1]) or archive.F.shape[1] != config.M:
            raise RuntimeError("Stage 1 checkpoint dimensions do not match the configuration")

        stored_nd = self.checkpoints.load_archive(result_dir / "stage1_nd_archive.mat")
        recomputed_nd = run_extract_archive(archive)
        if stored_nd.X.shape != recomputed_nd.X.shape or not np.allclose(
            stored_nd.X, recomputed_nd.X, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("stage1_nd_archive.mat is inconsistent with stage1_archive.mat")

        reference_set = self.checkpoints.load_reference_set(result_dir / "reference_set.mat")
        if reference_set.shape[1] != config.M:
            raise RuntimeError("reference_set.mat has the wrong objective dimension")
        bounds_path = result_dir / "problem_bounds.mat"
        if bounds_path.is_file():
            lower, upper = self.checkpoints.load_bounds(bounds_path)
        else:
            LOGGER.info(
                "problem_bounds.mat is absent; querying bounds from PlatEMO without consuming FE"
            )
            lower, upper = self.adapter.bounds(config)
            self.checkpoints.save_bounds(lower, upper)
        if lower.shape != (archive.X.shape[1],) or upper.shape != (archive.X.shape[1],):
            raise RuntimeError("PlatEMO decision bounds do not match the Stage 1 archive")
        if np.any(archive.X < lower - 1e-12) or np.any(archive.X > upper + 1e-12):
            raise RuntimeError("Stage 1 decisions lie outside the current PlatEMO problem bounds")

        hv_reference = (
            None
            if config.hv_reference_point is None
            else np.asarray(config.hv_reference_point, dtype=float)
        )
        stage1_metrics = run_metrics(reference_set, archive, hv_reference)
        trajectory = [{"FE": float(archive.last_fe), **stage1_metrics.to_dict()}]
        LOGGER.info(
            "Stage 2 resume: loaded %d Stage 1 evaluations and %d ND solutions",
            len(archive),
            len(recomputed_nd),
        )
        return self._run_stage2(
            archive, lower, upper, reference_set, stage1_metrics, trajectory
        )

    def _run_stage2(
        self,
        archive: EvaluationArchive,
        lower: np.ndarray,
        upper: np.ndarray,
        reference_set: np.ndarray,
        stage1_metrics: MetricValues,
        trajectory: list[dict[str, float]],
    ) -> ExperimentResult:
        config = self.config
        result_dir = Path(config.results_dir)
        hv_reference = (
            None
            if config.hv_reference_point is None
            else np.asarray(config.hv_reference_point, dtype=float)
        )
        self.checkpoints.save_json(
            "stage2_status.json",
            {"status": "training", "completed_FE": archive.last_fe, "iteration": 0},
        )

        dataset = run_build_psm_dataset(archive, reference_set)
        self.checkpoints.save_dataset(dataset)
        trained_model = run_train_psm(dataset, lower, upper, config)
        self.checkpoints.save_model(trained_model)
        LOGGER.info(
            "PSM trained on %d samples using %s; best epoch=%d",
            len(dataset.X),
            trained_model.device,
            trained_model.history.best_epoch,
        )

        traces: list[FillingTrace] = []
        model_evaluations = 0
        iteration = 0
        while model_evaluations < config.model_budget:
            iteration += 1
            remaining = config.model_budget - model_evaluations
            batch_size = min(config.num_holes, remaining)
            nd_before = run_extract_archive(archive)
            scores, selected = run_hole_detection(
                reference_set,
                nd_before.F,
                batch_size,
                config.hole_selection,
                config.hole_min_separation,
            )
            targets = reference_set[selected]
            candidates = run_generate_candidates(
                trained_model, targets, lower, upper
            )
            duplicate_candidates = np.asarray(
                [np.any(np.all(np.isclose(archive.X, x, rtol=0.0, atol=1e-12), axis=1)) for x in candidates],
                dtype=bool,
            )
            if np.any(duplicate_candidates):
                LOGGER.warning(
                    "iteration=%d generated %d candidate(s) already present in A_all; "
                    "they remain true evaluations under the fixed-model protocol",
                    iteration,
                    int(np.sum(duplicate_candidates)),
                )
            evaluated = run_true_evaluation(self.adapter, candidates, config)
            if len(evaluated.X) != batch_size:
                raise RuntimeError("true evaluator returned a different number of candidates")
            if np.any(evaluated.X < lower - 1e-12) or np.any(
                evaluated.X > upper + 1e-12
            ):
                raise RuntimeError("true evaluator returned decisions outside problem bounds")
            fe_indices = run_archive_update(archive, evaluated)
            model_evaluations += batch_size
            if archive.last_fe > config.maxFE:
                raise RuntimeError("FE accounting exceeded maxFE")

            nd_flags = nondominated_mask(archive.F, archive.C)[-batch_size:]
            target_distances = np.linalg.norm(targets - evaluated.F, axis=1)
            nd_after = run_extract_archive(archive)
            new_scores, _ = run_hole_detection(
                reference_set, nd_after.F, 1, config.hole_selection, config.hole_min_separation
            )
            metrics = run_metrics(reference_set, archive, hv_reference)
            trajectory.append({"FE": float(archive.last_fe), **metrics.to_dict()})
            trace = FillingTrace(
                iteration,
                int(fe_indices[0]),
                int(fe_indices[-1]),
                targets,
                scores.distances[selected],
                evaluated.X,
                evaluated.F,
                target_distances,
                duplicate_candidates,
                nd_flags,
                new_scores.distances[selected],
                scores.igd_infinity,
                new_scores.igd_infinity,
            )
            traces.append(trace)
            payload = {
                "iteration": iteration,
                "fe_start": trace.fe_start,
                "fe_end": trace.fe_end,
                "targets": targets,
                "old_h": trace.old_h,
                "x_hat": evaluated.X,
                "f_x_hat": evaluated.F,
                "target_distances": target_distances,
                "duplicate_candidates": duplicate_candidates.astype(np.uint8),
                "nondominated": nd_flags.astype(np.uint8),
                "new_h": trace.new_h,
                "old_igd_infinity": trace.old_igd_infinity,
                "new_igd_infinity": trace.new_igd_infinity,
            }
            self.checkpoints.save_iteration(iteration, payload)
            self.checkpoints.save_hole_scores(new_scores, selected)
            self.checkpoints.save_archive("stage2_progress_archive.mat", archive)
            self.checkpoints.save_trajectory(trajectory)
            self.checkpoints.save_json(
                "stage2_status.json",
                {
                    "status": "filling",
                    "completed_FE": archive.last_fe,
                    "stage2_evaluations": model_evaluations,
                    "iteration": iteration,
                },
            )
            LOGGER.debug(
                "iteration=%d FE=%d archive=%d ND=%d IGDinf=%.6g targets=%s x_hat=%s f=%s distance=%s",
                iteration,
                archive.last_fe,
                len(archive),
                len(nd_after),
                metrics.igd_infinity,
                np.array2string(targets, precision=4),
                np.array2string(evaluated.X, precision=4),
                np.array2string(evaluated.F, precision=4),
                np.array2string(target_distances, precision=4),
            )

            if config.retrain_model and iteration % config.retrain_interval == 0:
                dataset = run_build_psm_dataset(archive, reference_set)
                trained_model = run_train_psm(dataset, lower, upper, config)
                self.checkpoints.save_dataset(dataset)
                self.checkpoints.save_model(trained_model)

        if archive.last_fe != config.maxFE:
            raise RuntimeError(f"final FE is {archive.last_fe}, expected {config.maxFE}")
        final_nd = run_extract_archive(archive)
        stage2_metrics = run_metrics(reference_set, archive, hv_reference)
        self.checkpoints.save_archive("final_archive.mat", archive)
        self.checkpoints.save_archive("final_nd_archive.mat", final_nd)
        self.checkpoints.save_trajectory(trajectory)
        summary: dict[str, Any] = {
            "algorithm": config.algorithm,
            "problem": config.problem,
            "seed": config.seed,
            "stage1_FE": config.ea_budget,
            "stage2_FE": config.model_budget,
            "total_FE": archive.last_fe,
            "stage1_metrics": stage1_metrics.to_dict(),
            "stage2_metrics": stage2_metrics.to_dict(),
            "archive_size": len(archive),
            "nd_archive_size": len(final_nd),
            "filling_iterations": len(traces),
        }
        self.checkpoints.save_json("summary.json", summary)
        self.checkpoints.save_json(
            "stage2_status.json",
            {
                "status": "complete",
                "completed_FE": archive.last_fe,
                "stage2_evaluations": config.model_budget,
                "iteration": len(traces),
            },
        )
        LOGGER.info("Stage 2 complete: FE=%d, IGDinf=%.6g", archive.last_fe, stage2_metrics.igd_infinity)
        return ExperimentResult(
            archive,
            final_nd,
            reference_set,
            stage1_metrics,
            stage2_metrics,
            trajectory,
            traces,
            result_dir,
        )

"""MATLAB-readable checkpoints and experiment summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from .archive import EvaluationArchive
from .coverage import HoleScores
from .psm import PSMDataset, TrainedPSM


class CheckpointManager:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_archive(self, filename: str, archive: EvaluationArchive) -> Path:
        path = self.directory / filename
        source_code = np.where(archive.source == "model", 1, 0).astype(np.int8)
        savemat(
            path,
            {"X": archive.X, "F": archive.F, "C": archive.C, "FE": archive.FE, "source": source_code},
        )
        return path

    @staticmethod
    def load_archive(path: str | Path) -> EvaluationArchive:
        data = loadmat(path)
        FE = np.asarray(data["FE"]).reshape(-1).astype(np.int64)
        codes = np.asarray(data.get("source", np.zeros(len(FE)))).reshape(-1)
        source = np.where(codes == 1, "model", "EA")
        C = np.asarray(data.get("C", np.empty((len(FE), 0))), dtype=float)
        return EvaluationArchive(data["X"], data["F"], FE, source, C)

    def save_reference_set(self, reference_set: np.ndarray) -> None:
        savemat(self.directory / "reference_set.mat", {"Z": reference_set})

    def save_bounds(self, lower: np.ndarray, upper: np.ndarray) -> None:
        lower_array = np.asarray(lower, dtype=float).reshape(-1)
        upper_array = np.asarray(upper, dtype=float).reshape(-1)
        if lower_array.shape != upper_array.shape or np.any(upper_array <= lower_array):
            raise ValueError("cannot save invalid decision bounds")
        savemat(
            self.directory / "problem_bounds.mat",
            {"lower": lower_array, "upper": upper_array},
        )

    @staticmethod
    def load_bounds(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        data = loadmat(path)
        lower = np.asarray(data["lower"], dtype=float).reshape(-1)
        upper = np.asarray(data["upper"], dtype=float).reshape(-1)
        if lower.shape != upper.shape or np.any(upper <= lower):
            raise ValueError(f"invalid decision bounds checkpoint: {path}")
        return lower, upper

    @staticmethod
    def load_reference_set(path: str | Path) -> np.ndarray:
        data = loadmat(path)
        if "Z" not in data:
            raise KeyError(f"reference checkpoint does not contain Z: {path}")
        reference = np.asarray(data["Z"], dtype=float)
        if reference.ndim != 2 or not len(reference) or not np.all(np.isfinite(reference)):
            raise ValueError(f"invalid reference set checkpoint: {path}")
        return reference

    def save_hole_scores(self, scores: HoleScores, selected: np.ndarray) -> None:
        savemat(
            self.directory / "hole_scores.mat",
            {
                "Z": scores.reference_set,
                "h": scores.distances,
                "nearest_archive_indices": scores.nearest_archive_indices + 1,
                "selected_indices": selected + 1,
            },
        )

    def save_dataset(self, dataset: PSMDataset) -> None:
        savemat(
            self.directory / "psm_dataset.mat",
            {
                "Z_train": dataset.Z,
                "X_train": dataset.X,
                "reference_indices": dataset.reference_indices + 1,
                "matching_distances": dataset.matching_distances,
            },
        )

    def save_model(self, model: TrainedPSM) -> None:
        model.save(self.directory / "psm_model.pt")

    def save_iteration(self, iteration: int, payload: dict[str, Any]) -> None:
        serializable = {key: value for key, value in payload.items() if isinstance(value, np.ndarray)}
        serializable.update(
            {key: value for key, value in payload.items() if isinstance(value, (int, float, bool))}
        )
        savemat(self.directory / f"filling_iteration_{iteration:03d}.mat", serializable)

    def save_trajectory(self, rows: list[dict[str, float]]) -> None:
        if not rows:
            raise ValueError("cannot save an empty metrics trajectory")
        path = self.directory / "metrics_history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        savemat(
            self.directory / "metrics_history.mat",
            {key: np.asarray([row[key] for row in rows]) for key in rows[0]},
        )

    def save_json(self, filename: str, payload: dict[str, Any]) -> None:
        with (self.directory / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

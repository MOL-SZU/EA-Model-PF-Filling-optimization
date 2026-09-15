"""Validated experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    device: str = "auto"
    hidden_size: int = 128
    num_layers: int = 3
    learning_rate: float = 1e-3
    epochs: int = 300
    batch_size: int = 64
    validation_ratio: float = 0.2
    patience: int = 40
    weight_decay: float = 1e-6

    def validate(self) -> None:
        if self.device != "auto" and self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("model.device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("model.hidden_size and model.num_layers must be positive")
        if self.learning_rate <= 0 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("model learning rate, epochs, and batch size must be positive")
        if not 0 <= self.validation_ratio < 1:
            raise ValueError("model.validation_ratio must be in [0, 1)")
        if self.patience < 1 or self.weight_decay < 0:
            raise ValueError("model.patience must be positive and weight_decay non-negative")


@dataclass(slots=True)
class ExperimentConfig:
    algorithm: str = "NSGAII"
    problem: str = "DTLZ2"
    N: int = 100
    M: int = 3
    D: int | None = None
    maxFE: int = 10_000
    EA_ratio: float = 0.8
    reference_point_num: int = 2_000
    num_holes: int = 10
    hole_selection: str = "separated_topB"
    hole_min_separation: float | None = None
    candidates_per_hole: int = 1
    retrain_model: bool = False
    retrain_interval: int = 5
    validation_mode: bool = False
    seed: int = 1
    results_dir: str = "Results/default"
    platemo_path: str | None = None
    problem_parameters: list[float] = field(default_factory=list)
    hv_reference_point: list[float] | None = None
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> None:
        if not self.algorithm or not self.problem:
            raise ValueError("algorithm and problem must be non-empty PlatEMO class names")
        if self.N < 2 or self.M < 2 or (self.D is not None and self.D < 1):
            raise ValueError("N and M must be >= 2; D must be positive when specified")
        if self.maxFE < self.N:
            raise ValueError("maxFE must be at least one population")
        if not 0 < self.EA_ratio < 1:
            raise ValueError("EA_ratio must be strictly between 0 and 1")
        if int(self.maxFE * self.EA_ratio) < self.N:
            raise ValueError("EA_ratio allocates less than one population to Stage 1")
        if self.reference_point_num < self.M or self.num_holes < 1:
            raise ValueError("reference_point_num and num_holes are too small")
        if self.hole_selection not in {"topB", "separated_topB"}:
            raise ValueError("hole_selection must be 'topB' or 'separated_topB'")
        if self.hole_min_separation is not None and self.hole_min_separation < 0:
            raise ValueError("hole_min_separation must be non-negative")
        if self.candidates_per_hole != 1:
            raise ValueError("version 0.1 supports exactly one candidate per hole")
        if self.retrain_interval < 1:
            raise ValueError("retrain_interval must be positive")
        if self.hv_reference_point is not None and len(self.hv_reference_point) != self.M:
            raise ValueError("hv_reference_point length must equal M")
        self.model.validate()

    @property
    def ea_budget(self) -> int:
        """Population-aligned Stage 1 budget that cannot exceed EA_ratio*maxFE."""
        target = int(self.maxFE * self.EA_ratio)
        aligned = (target // self.N) * self.N
        return max(self.N, aligned)

    @property
    def model_budget(self) -> int:
        return self.maxFE - self.ea_budget

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        values = dict(data)
        values["model"] = ModelConfig(**values.get("model", {}))
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

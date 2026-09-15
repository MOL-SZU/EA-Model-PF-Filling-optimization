"""Single-head MLP Pareto Set Model implementing z -> x."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ModelConfig


def resolve_device(requested: str) -> torch.device:
    """Resolve a configured device without silently ignoring a CUDA request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid model device: {requested}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "model.device requests CUDA, but torch.cuda.is_available() is false"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"model.device requests cuda:{device.index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
    return device


@dataclass(frozen=True, slots=True)
class MinMaxScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def scale(self) -> np.ndarray:
        span = self.maximum - self.minimum
        return np.where(np.abs(span) > 1e-12, span, 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.minimum) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.minimum


@dataclass(frozen=True, slots=True)
class PSMDataset:
    Z: np.ndarray
    X: np.ndarray
    reference_indices: np.ndarray
    matching_distances: np.ndarray


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int


class ParetoSetMLP(nn.Module):
    """A deliberately simple, single-output-head MLP."""

    def __init__(self, n_obj: int, n_var: int, hidden_size: int = 128, num_layers: int = 3):
        super().__init__()
        if n_obj < 1 or n_var < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("all model dimensions must be positive")
        layers: list[nn.Module] = []
        input_size = n_obj
        for _ in range(num_layers):
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.extend((nn.Linear(input_size, n_var), nn.Sigmoid()))
        self.network = nn.Sequential(*layers)
        self.n_obj = n_obj
        self.n_var = n_var
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.n_obj:
            raise ValueError(f"model input must have shape (batch, {self.n_obj})")
        return self.network(z)


@dataclass(slots=True)
class TrainedPSM:
    model: ParetoSetMLP
    z_scaler: MinMaxScaler
    x_scaler: MinMaxScaler
    history: TrainingHistory

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def predict(self, targets: np.ndarray) -> np.ndarray:
        Z = np.asarray(targets, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != self.model.n_obj:
            raise ValueError("target PF locations have the wrong shape")
        if not np.all(np.isfinite(Z)):
            raise ValueError("target PF locations contain NaN or Inf")
        inputs = torch.as_tensor(
            self.z_scaler.transform(Z), dtype=torch.float32, device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            normalized = self.model(inputs).cpu().numpy()
        return self.x_scaler.inverse_transform(normalized)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_obj": self.model.n_obj,
                "n_var": self.model.n_var,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "training_device": str(self.device),
                "z_min": self.z_scaler.minimum,
                "z_max": self.z_scaler.maximum,
                "x_min": self.x_scaler.minimum,
                "x_max": self.x_scaler.maximum,
                "history": {
                    "train_loss": self.history.train_loss,
                    "validation_loss": self.history.validation_loss,
                    "best_epoch": self.history.best_epoch,
                },
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TrainedPSM":
        target_device = resolve_device(device)
        payload = torch.load(Path(path), map_location=target_device, weights_only=False)
        model = ParetoSetMLP(
            payload["n_obj"], payload["n_var"], payload["hidden_size"], payload["num_layers"]
        )
        model.load_state_dict(payload["state_dict"])
        model.to(target_device)
        history = TrainingHistory(**payload["history"])
        return cls(
            model,
            MinMaxScaler(np.asarray(payload["z_min"]), np.asarray(payload["z_max"])),
            MinMaxScaler(np.asarray(payload["x_min"]), np.asarray(payload["x_max"])),
            history,
        )


def build_psm_dataset(
    archive_X: np.ndarray,
    archive_F: np.ndarray,
    reference_set: np.ndarray,
) -> PSMDataset:
    X = np.asarray(archive_X, dtype=float)
    F = np.asarray(archive_F, dtype=float)
    Z = np.asarray(reference_set, dtype=float)
    if X.ndim != 2 or F.ndim != 2 or Z.ndim != 2 or len(X) != len(F):
        raise ValueError("X, F, and reference_set must be aligned two-dimensional arrays")
    if len(X) < 2:
        raise ValueError("at least two non-dominated samples are required to train the PSM")
    if F.shape[1] != Z.shape[1]:
        raise ValueError("objective and PF reference dimensions must match")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(F)) or not np.all(np.isfinite(Z)):
        raise ValueError("PSM dataset contains NaN or Inf")

    distances = cdist(F, Z)
    reference_indices = np.argmin(distances, axis=1)
    matched = Z[reference_indices]
    matching_distances = distances[np.arange(len(F)), reference_indices]
    key = np.hstack((matched, X))
    _, unique = np.unique(key, axis=0, return_index=True)
    unique = np.sort(unique)
    return PSMDataset(matched[unique], X[unique], reference_indices[unique], matching_distances[unique])


def train_psm(
    dataset: PSMDataset,
    lower: np.ndarray,
    upper: np.ndarray,
    config: ModelConfig,
    seed: int,
) -> TrainedPSM:
    config.validate()
    device = resolve_device(config.device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if len(lower) != dataset.X.shape[1] or np.any(upper <= lower):
        raise ValueError("invalid decision bounds")
    if np.any(dataset.X < lower - 1e-10) or np.any(dataset.X > upper + 1e-10):
        raise ValueError("training decisions lie outside the configured bounds")

    z_scaler = MinMaxScaler(np.min(dataset.Z, axis=0), np.max(dataset.Z, axis=0))
    x_scaler = MinMaxScaler(lower, upper)
    Z = torch.as_tensor(z_scaler.transform(dataset.Z), dtype=torch.float32)
    X = torch.as_tensor(x_scaler.transform(dataset.X), dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(Z), generator=generator).numpy()
    validation_count = int(round(len(Z) * config.validation_ratio))
    if config.validation_ratio > 0 and len(Z) >= 5:
        validation_count = max(1, min(validation_count, len(Z) - 1))
    else:
        validation_count = 0
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]
    loader = DataLoader(
        TensorDataset(Z[train_indices], X[train_indices]),
        batch_size=min(config.batch_size, len(train_indices)),
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    model = ParetoSetMLP(
        Z.shape[1], X.shape[1], config.hidden_size, config.num_layers
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    train_history: list[float] = []
    validation_history: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_Z, batch_X in loader:
            batch_Z = batch_Z.to(device, non_blocking=True)
            batch_X = batch_X.to(device, non_blocking=True)
            prediction = model(batch_Z)
            loss = criterion(prediction, batch_X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_Z)
            total_count += len(batch_Z)
        train_loss = total_loss / total_count
        train_history.append(train_loss)

        model.eval()
        with torch.no_grad():
            if validation_count:
                validation_Z = Z[validation_indices].to(device, non_blocking=True)
                validation_X = X[validation_indices].to(device, non_blocking=True)
                monitored = float(criterion(model(validation_Z), validation_X).cpu())
            else:
                monitored = train_loss
        validation_history.append(monitored)
        if monitored < best_loss - 1e-12:
            best_loss = monitored
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    if not all(np.isfinite(train_history)) or not all(np.isfinite(validation_history)):
        raise RuntimeError("non-finite loss encountered while training the PSM")
    return TrainedPSM(
        model,
        z_scaler,
        x_scaler,
        TrainingHistory(train_history, validation_history, best_epoch),
    )


def repair_candidates(candidates: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    X = np.asarray(candidates, dtype=float)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("candidates must be a finite two-dimensional array")
    return np.clip(X, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))

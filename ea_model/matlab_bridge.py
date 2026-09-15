"""MATLAB Engine adapter for generic PlatEMO algorithms and problems."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .archive import EvaluationArchive
from .config import ExperimentConfig
from .interfaces import EvaluationBatch, Stage1Output


class MatlabPlatEMOAdapter:
    """Use MATLAB Engine and the bundled recorder without modifying PlatEMO."""

    def __init__(self, platemo_path: str | Path, engine: Any | None = None):
        self.platemo_path = Path(platemo_path).resolve()
        if not self.platemo_path.exists():
            raise FileNotFoundError(f"PlatEMO path does not exist: {self.platemo_path}")
        self._engine = engine
        self._owns_engine = engine is None
        self._matlab_dir = Path(__file__).resolve().parent.parent / "matlab"

    def _start(self) -> Any:
        if self._engine is None:
            try:
                import matlab.engine
            except ImportError as error:
                raise RuntimeError(
                    "MATLAB Engine for Python is not installed. Install it from the matching "
                    "MATLAB release, then verify `import matlab.engine`."
                ) from error
            self._engine = matlab.engine.start_matlab()
        self._engine.addpath(str(self._matlab_dir), nargout=0)
        self._engine.addpath(self._engine.genpath(str(self.platemo_path)), nargout=0)
        return self._engine

    @staticmethod
    def _matlab_double(values: np.ndarray | list[float]) -> Any:
        import matlab

        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return matlab.double([])
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return matlab.double(array.tolist())

    @staticmethod
    def _array(value: Any, columns: int | None = None) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.size == 0 and columns is not None:
            return np.empty((0, columns))
        return array

    def run_stage1(self, config: ExperimentConfig) -> Stage1Output:
        engine = self._start()
        outputs = engine.run_platemo_ea(
            str(self.platemo_path),
            config.algorithm,
            config.problem,
            float(config.N),
            float(config.M),
            float(config.D or 0),
            float(config.ea_budget),
            float(config.seed),
            self._matlab_double(config.problem_parameters),
            nargout=6,
        )
        X, F, C, FE, lower, upper = outputs
        X_np = self._array(X)
        F_np = self._array(F)
        C_np = self._array(C)
        if C_np.size == 0:
            C_np = np.empty((len(X_np), 0))
        FE_np = np.asarray(FE, dtype=np.int64).reshape(-1)
        archive = EvaluationArchive(X_np, F_np, FE_np, np.full(len(X_np), "EA"), C_np)
        return Stage1Output(
            archive, np.asarray(lower, dtype=float).reshape(-1), np.asarray(upper, dtype=float).reshape(-1)
        )

    def evaluate(self, X: np.ndarray, config: ExperimentConfig) -> EvaluationBatch:
        engine = self._start()
        repaired, objectives, constraints = engine.evaluate_platemo_problem(
            str(self.platemo_path),
            config.problem,
            float(config.N),
            float(config.M),
            float(config.D or 0),
            self._matlab_double(config.problem_parameters),
            self._matlab_double(X),
            nargout=3,
        )
        X_np = self._array(repaired)
        F_np = self._array(objectives)
        C_np = self._array(constraints)
        if C_np.size == 0:
            C_np = np.empty((len(X_np), 0))
        return EvaluationBatch(X_np, F_np, C_np)

    def reference_set(self, count: int, config: ExperimentConfig) -> np.ndarray:
        engine = self._start()
        result = engine.get_platemo_reference_set(
            str(self.platemo_path),
            config.problem,
            float(config.N),
            float(config.M),
            float(config.D or 0),
            self._matlab_double(config.problem_parameters),
            float(count),
            nargout=1,
        )
        reference = self._array(result)
        if len(reference) < 2:
            raise RuntimeError(
                f"PlatEMO problem {config.problem} did not provide a dense true PF. "
                "Supply a problem with GetOptimum(N) support or a custom adapter."
            )
        return reference

    def bounds(self, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
        engine = self._start()
        lower, upper = engine.get_platemo_problem_bounds(
            str(self.platemo_path),
            config.problem,
            float(config.N),
            float(config.M),
            float(config.D or 0),
            self._matlab_double(config.problem_parameters),
            nargout=2,
        )
        lower_np = np.asarray(lower, dtype=float).reshape(-1)
        upper_np = np.asarray(upper, dtype=float).reshape(-1)
        if lower_np.shape != upper_np.shape or np.any(upper_np <= lower_np):
            raise RuntimeError("PlatEMO returned invalid decision bounds")
        return lower_np, upper_np

    def close(self) -> None:
        if self._engine is not None and self._owns_engine:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> "MatlabPlatEMOAdapter":
        self._start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

"""EA+Model Pareto-front hole filling framework."""

from .archive import EvaluationArchive
from .config import ExperimentConfig, ModelConfig
from .pipeline import EAModelPipeline, ExperimentResult

__all__ = [
    "EAModelPipeline",
    "EvaluationArchive",
    "ExperimentConfig",
    "ExperimentResult",
    "ModelConfig",
]

__version__ = "0.1.0"


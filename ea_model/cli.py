"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import ExperimentConfig
from .matlab_bridge import MatlabPlatEMOAdapter
from .pipeline import EAModelPipeline
from .stage1_experiment import run_stage1_experiment
from .synthetic import DeterministicDTLZ2Adapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EA+Model PF hole filling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a MATLAB/PlatEMO experiment")
    run.add_argument("--config", required=True, type=Path)
    stage1 = subparsers.add_parser(
        "stage1", help="run only the MATLAB/PlatEMO EA and save its archive"
    )
    stage1.add_argument("--config", required=True, type=Path)
    stage2 = subparsers.add_parser(
        "stage2", help="load Stage 1 checkpoints, train the MLP, and run PF filling"
    )
    stage2.add_argument("--config", required=True, type=Path)
    validate = subparsers.add_parser("validate-small", help="run the deterministic smoke experiment")
    validate.add_argument("--output", type=Path, default=Path("Results/validation-small"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command in {"run", "stage1", "stage2"}:
        config = ExperimentConfig.from_json(args.config)
        platemo_path = config.platemo_path or os.environ.get("PLATEMO_PATH")
        if not platemo_path:
            raise SystemExit("Set config.platemo_path or the PLATEMO_PATH environment variable.")
        with MatlabPlatEMOAdapter(platemo_path) as adapter:
            if args.command == "run":
                result = EAModelPipeline(config, adapter).run()
                output = {
                    "results": str(result.result_directory),
                    "stage1": result.stage1_metrics.to_dict(),
                    "stage2": result.stage2_metrics.to_dict(),
                    "total_FE": result.archive.last_fe,
                }
            elif args.command == "stage1":
                result = run_stage1_experiment(config, adapter)
                output = {
                    "results": str(result.result_directory),
                    "stage1": result.metrics.to_dict(),
                    "stage1_FE": result.archive.last_fe,
                    "archive_size": len(result.archive),
                    "nd_archive_size": len(result.nd_archive),
                    "model_trained": False,
                    "stage2_started": False,
                }
            else:
                result = EAModelPipeline(config, adapter).run_stage2_from_checkpoints()
                output = {
                    "results": str(result.result_directory),
                    "stage1": result.stage1_metrics.to_dict(),
                    "stage2": result.stage2_metrics.to_dict(),
                    "total_FE": result.archive.last_fe,
                    "model_trained": True,
                    "stage2_complete": True,
                }
    else:
        config = ExperimentConfig(
            algorithm="validation-LHS",
            problem="DTLZ2",
            N=20,
            M=3,
            D=12,
            maxFE=100,
            EA_ratio=0.8,
            reference_point_num=300,
            num_holes=4,
            validation_mode=True,
            results_dir=str(args.output),
        )
        config.model.hidden_size = 32
        config.model.num_layers = 2
        config.model.epochs = 60
        config.model.patience = 15
        result = EAModelPipeline(config, DeterministicDTLZ2Adapter()).run()
        output = {
            "results": str(result.result_directory),
            "stage1": result.stage1_metrics.to_dict(),
            "stage2": result.stage2_metrics.to_dict(),
            "total_FE": result.archive.last_fe,
        }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

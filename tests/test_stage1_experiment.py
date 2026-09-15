from pathlib import Path

from ea_model.config import ExperimentConfig
from ea_model.stage1_experiment import run_stage1_experiment
from ea_model.synthetic import DeterministicDTLZ2Adapter


def test_stage1_experiment_stops_before_model_training(tmp_path):
    config = ExperimentConfig(
        algorithm="validation-LHS",
        problem="DTLZ2",
        N=20,
        M=3,
        D=12,
        maxFE=60,
        EA_ratio=2 / 3,
        reference_point_num=120,
        num_holes=4,
        results_dir=str(tmp_path),
        seed=17,
    )

    result = run_stage1_experiment(config, DeterministicDTLZ2Adapter())

    assert result.archive.last_fe == 40
    assert len(result.archive) == 40
    assert len(result.nd_archive) >= 2
    assert all(
        (tmp_path / name).is_file()
        for name in (
            "config_resolved.json",
            "stage1_archive.mat",
            "stage1_nd_archive.mat",
            "reference_set.mat",
            "problem_bounds.mat",
            "stage1_summary.json",
        )
    )
    assert not (tmp_path / "psm_model.pt").exists()
    assert not list(Path(tmp_path).glob("filling_iteration_*.mat"))
    assert not (tmp_path / "final_archive.mat").exists()


def test_mansgaii_configs_match_nsgaii_stage1_protocol():
    root = Path(__file__).resolve().parents[1] / "configs"
    baseline = ExperimentConfig.from_json(root / "dtlz2_nsgaii.json")
    variants = (
        ExperimentConfig.from_json(root / "dtlz2_mansgaii.json"),
        ExperimentConfig.from_json(root / "dtlz2_mansgaii_norm.json"),
    )

    assert {config.algorithm for config in variants} == {"MaNSGAII", "MaNSGAII_Norm"}
    for config in variants:
        assert (config.problem, config.N, config.M, config.D) == (
            baseline.problem,
            baseline.N,
            baseline.M,
            baseline.D,
        )
        assert config.ea_budget == baseline.ea_budget == 8000
        assert config.seed == baseline.seed

import numpy as np

from ea_model.checkpoints import CheckpointManager
from ea_model.config import ExperimentConfig
from ea_model.pipeline import EAModelPipeline
from ea_model.stage1_experiment import run_stage1_experiment
from ea_model.synthetic import DeterministicDTLZ2Adapter


def small_config(tmp_path) -> ExperimentConfig:
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
        validation_mode=True,
        results_dir=str(tmp_path),
        seed=11,
    )
    config.model.hidden_size = 24
    config.model.num_layers = 2
    config.model.epochs = 30
    config.model.patience = 10
    return config


def test_fe_accounting_example():
    config = ExperimentConfig(N=100, maxFE=1000, EA_ratio=.8)
    assert config.ea_budget == 800
    assert config.model_budget == 200
    assert config.ea_budget + config.model_budget == config.maxFE


def test_one_step_filling_trace_and_dynamic_reassessment(tmp_path):
    config = small_config(tmp_path)
    config.maxFE = 24
    config.EA_ratio = 5 / 6
    result = EAModelPipeline(config, DeterministicDTLZ2Adapter()).run()
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.targets.shape == (4, 3)
    assert trace.candidates.shape == (4, 12)
    assert trace.objectives.shape == (4, 3)
    assert np.all(np.isfinite(trace.target_distances))
    assert trace.new_igd_infinity <= trace.old_igd_infinity + 1e-12


def test_small_end_to_end_exact_budget_outputs_and_restart(tmp_path):
    config = small_config(tmp_path)
    result = EAModelPipeline(config, DeterministicDTLZ2Adapter()).run()
    assert result.archive.last_fe == config.maxFE
    assert len(result.archive) == config.maxFE
    assert len(result.traces) == 5
    assert len(result.trajectory) == 6
    assert all(np.isfinite(list(result.stage2_metrics.to_dict().values())))
    expected = [
        "stage1_archive.mat",
        "stage1_nd_archive.mat",
        "reference_set.mat",
        "hole_scores.mat",
        "psm_dataset.mat",
        "psm_model.pt",
        "metrics_history.mat",
        "metrics_history.csv",
        "summary.json",
        "final_archive.mat",
    ]
    assert all((tmp_path / name).exists() for name in expected)
    restored = CheckpointManager.load_archive(tmp_path / "final_archive.mat")
    assert np.allclose(restored.X, result.archive.X)
    assert np.array_equal(restored.FE, result.archive.FE)


def test_reproducible_validation_mode(tmp_path):
    first = EAModelPipeline(small_config(tmp_path / "a"), DeterministicDTLZ2Adapter()).run()
    second = EAModelPipeline(small_config(tmp_path / "b"), DeterministicDTLZ2Adapter()).run()
    assert np.allclose(first.archive.X, second.archive.X)
    assert np.allclose(first.archive.F, second.archive.F)


def test_stage2_resumes_from_stage1_checkpoints_without_rerunning_ea(tmp_path):
    config = small_config(tmp_path)
    adapter = DeterministicDTLZ2Adapter()
    stage1 = run_stage1_experiment(config, adapter)
    stage1_X = stage1.archive.X.copy()

    result = EAModelPipeline(config, adapter).run_stage2_from_checkpoints()

    assert result.archive.last_fe == config.maxFE
    assert np.array_equal(result.archive.X[: config.ea_budget], stage1_X)
    assert len(result.traces) == config.model_budget // config.num_holes
    assert (tmp_path / "psm_model.pt").is_file()
    assert (tmp_path / "stage2_progress_archive.mat").is_file()
    assert (tmp_path / "final_archive.mat").is_file()
    assert (tmp_path / "stage2_status.json").is_file()

import numpy as np

from ea_model.config import ModelConfig
from ea_model.psm import build_psm_dataset, repair_candidates, resolve_device, train_psm


def test_psm_dataset_nearest_reference_matching():
    Z = np.array([[0, 1], [.5, .5], [1, 0]], dtype=float)
    F = np.array([[.1, .9], [.55, .45], [.9, .1]])
    X = np.array([[.1, .9], [.55, .45], [.9, .1]])
    dataset = build_psm_dataset(X, F, Z)
    assert np.array_equal(dataset.reference_indices, [0, 1, 2])
    assert np.allclose(dataset.Z, Z)


def test_simple_mlp_learns_synthetic_mapping_and_generates_batch(tmp_path):
    t = np.linspace(0, 1, 80)
    Z = np.column_stack((t, 1 - t))
    X = np.column_stack((t, 1 - t, .25 + .5 * t))
    dataset = build_psm_dataset(X, Z, Z)
    config = ModelConfig(
        hidden_size=32,
        num_layers=2,
        learning_rate=5e-3,
        epochs=250,
        batch_size=32,
        patience=50,
    )
    trained = train_psm(dataset, np.zeros(3), np.ones(3), config, seed=7)
    targets = np.array([[.2, .8], [.8, .2]])
    predictions = trained.predict(targets)
    expected = np.column_stack((targets[:, 0], targets[:, 1], .25 + .5 * targets[:, 0]))
    assert trained.history.train_loss[-1] < trained.history.train_loss[0]
    assert predictions.shape == (2, 3)
    assert np.all(np.isfinite(predictions))
    assert np.mean((predictions - expected) ** 2) < 0.02
    assert not np.allclose(predictions[0], predictions[1], atol=1e-3)
    path = tmp_path / "model.pt"
    trained.save(path)
    from ea_model.psm import TrainedPSM
    assert np.allclose(TrainedPSM.load(path).predict(targets), predictions)


def test_candidate_bounds_repair():
    values = np.array([[-1, .5, 2]])
    repaired = repair_candidates(values, np.zeros(3), np.ones(3))
    assert np.array_equal(repaired, [[0, .5, 1]])


def test_auto_device_resolves_to_available_backend():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda"}

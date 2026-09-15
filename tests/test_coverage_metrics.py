import numpy as np

from ea_model.coverage import calculate_hole_distances, select_multiple_holes
from ea_model.metrics import hypervolume, igd, igd_infinity
from ea_model.synthetic import DeterministicDTLZ2Adapter
from ea_model.config import ExperimentConfig


def test_reference_set_dtlz2_geometry():
    config = ExperimentConfig(M=3, reference_point_num=200)
    Z = DeterministicDTLZ2Adapter().reference_set(200, config)
    assert Z.shape == (200, 3)
    assert np.all(np.isfinite(Z))
    assert np.allclose(np.sum(Z**2, axis=1), 1.0)


def test_hole_distance_and_largest_hole_hand_calculation():
    Z = np.array([[0, 1], [.25, .75], [.5, .5], [.75, .25], [1, 0]], dtype=float)
    A = np.array([[0, 1], [1, 0]], dtype=float)
    scores = calculate_hole_distances(Z, A)
    expected = np.array([0, np.sqrt(.125), np.sqrt(.5), np.sqrt(.125), 0])
    assert np.allclose(scores.distances, expected)
    assert scores.largest_index == 2
    assert np.isclose(scores.igd_infinity, np.sqrt(.5))


def test_top_b_and_separated_top_b_are_valid():
    Z = np.column_stack((np.linspace(0, 1, 11), np.linspace(1, 0, 11)))
    scores = calculate_hole_distances(Z, Z[[0, -1]])
    top = select_multiple_holes(scores, 3, "topB")
    separated = select_multiple_holes(scores, 3, "separated_topB")
    assert len(np.unique(top)) == len(np.unique(separated)) == 3
    assert top[0] == separated[0] == 5


def test_igd_infinity_monotonic_and_exact_match():
    Z = np.array([[0, 1], [.5, .5], [1, 0]], dtype=float)
    endpoints = Z[[0, 2]]
    before = igd_infinity(Z, endpoints)
    after = igd_infinity(Z, np.vstack((endpoints, Z[1])))
    assert after <= before
    assert igd_infinity(Z, Z) == 0.0


def test_igd_hand_calculation():
    Z = np.array([[0, 1], [.5, .5], [1, 0]], dtype=float)
    endpoints = Z[[0, 2]]
    assert np.isclose(igd(Z, endpoints), np.sqrt(.5) / 3)


def test_hv_hand_calculation_and_dominated_invariance():
    F = np.array([[1, 4], [2, 2], [4, 1]], dtype=float)
    ref = np.array([5, 5], dtype=float)
    assert np.isclose(hypervolume(F, ref), 11.0)
    assert np.isclose(hypervolume(np.vstack((F, [3, 4])), ref), 11.0)
    assert hypervolume(F, np.array([6, 6])) > 11.0


import numpy as np

from ea_model.archive import EvaluationArchive


def _archive() -> EvaluationArchive:
    X = np.arange(10, dtype=float).reshape(5, 2)
    F = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [3, 4]], dtype=float)
    return EvaluationArchive(X, F, np.arange(1, 6), np.full(5, "EA"))


def test_archive_recorder_alignment_and_fe_indices():
    archive = _archive()
    assert len(archive) == 5
    assert np.array_equal(archive.FE, np.arange(1, 6))
    assert archive.X.shape[0] == archive.F.shape[0]


def test_nondominated_filter_and_duplicate_handling():
    archive = _archive()
    nd = archive.nondominated()
    assert np.array_equal(nd.F, np.array([[1, 4], [2, 3], [3, 2], [4, 1]]))
    archive.append(archive.X[:1], archive.F[:1], "model")
    assert len(archive) == 6
    assert len(archive.nondominated()) == 4


def test_dominated_candidate_stays_in_all_but_not_nd():
    archive = _archive()
    archive.append(np.array([[0.2, 0.2]]), np.array([[5.0, 5.0]]), "model")
    assert len(archive) == 6
    assert not np.any(np.all(archive.nondominated().F == [5.0, 5.0], axis=1))


def test_fe_indices_must_be_strictly_increasing():
    try:
        EvaluationArchive(np.zeros((2, 1)), np.zeros((2, 2)), [1, 1], ["EA", "EA"])
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("invalid FE indices were accepted")


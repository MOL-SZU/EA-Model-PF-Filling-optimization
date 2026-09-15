import os

import numpy as np
import pytest

from ea_model.config import ExperimentConfig
from ea_model.matlab_bridge import MatlabPlatEMOAdapter


@pytest.mark.matlab
@pytest.mark.skipif(not os.environ.get("PLATEMO_PATH"), reason="PLATEMO_PATH is not configured")
def test_true_evaluation_matches_dtlz2_formula():
    pytest.importorskip("matlab.engine")
    from ea_model.synthetic import dtlz2

    config = ExperimentConfig(problem="DTLZ2", M=3, D=12, N=20, maxFE=100)
    X = np.full((2, 12), .5)
    with MatlabPlatEMOAdapter(os.environ["PLATEMO_PATH"]) as adapter:
        evaluated = adapter.evaluate(X, config)
    assert np.allclose(evaluated.F, dtlz2(X, 3), atol=1e-12)

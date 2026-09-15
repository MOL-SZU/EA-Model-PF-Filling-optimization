import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.io import savemat


def test_complete_visualizer_supports_stage1_only_results(tmp_path):
    results = tmp_path / "run"
    figures = tmp_path / "figures"
    results.mkdir()

    rng = np.random.default_rng(7)
    directions = rng.random((40, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    all_objectives = np.vstack((1.4 * directions, directions))
    savemat(
        results / "stage1_archive.mat",
        {"F": all_objectives, "FE": np.arange(1, len(all_objectives) + 1)},
    )
    savemat(results / "stage1_nd_archive.mat", {"F": directions})
    savemat(results / "reference_set.mat", {"Z": directions})
    (results / "config_resolved.json").write_text(
        json.dumps(
            {
                "algorithm": "NSGAII",
                "problem": "DTLZ2",
                "num_holes": 4,
                "hole_selection": "separated_topB",
                "hole_min_separation": None,
            }
        ),
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "visualize_results.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--results-dir",
            str(results),
            "--output-dir",
            str(figures),
            "--formats",
            "png",
            "--dpi",
            "72",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expected = {
        "fig00_method_workflow.png",
        "fig01_stage1_search_history_3d.png",
        "fig02_stage1_pf_3d.png",
        "fig03_stage1_pairwise.png",
        "fig04_stage1_holes.png",
        "figure_manifest.json",
    }
    assert expected.issubset(path.name for path in figures.iterdir())
    manifest = json.loads((figures / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage1_evaluations"] == 80
    assert manifest["stage1_nd_size"] == 40
    assert len(manifest["skipped"]) == 5
    assert "Generated 5 file(s)" in completed.stdout

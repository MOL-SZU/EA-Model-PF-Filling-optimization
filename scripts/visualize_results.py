"""Generate the complete paper figure suite from one experiment directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.io import loadmat

from ea_model.coverage import calculate_hole_distances, select_multiple_holes


LABELS = (r"$f_1$", r"$f_2$", r"$f_3$")
BLUE = "#1756d1"
GREEN = "#16856b"
ORANGE = "#d97706"
MAGENTA = "#c0267e"
GRAY = "0.72"


def _matrix(path: Path, key: str, columns: int = 3) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint does not exist: {path}")
    payload = loadmat(path)
    if key not in payload:
        raise KeyError(f"{path} does not contain {key}")
    values = np.asarray(payload[key], dtype=float)
    if values.ndim != 2 or values.shape[1] != columns or not len(values):
        raise ValueError(f"{path}:{key} must have shape (n, {columns})")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}:{key} contains NaN or Inf")
    return values


def _archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = loadmat(path)
    F = np.asarray(payload["F"], dtype=float)
    FE = np.asarray(payload["FE"], dtype=float).reshape(-1)
    if F.ndim != 2 or F.shape[1] != 3 or len(F) != len(FE):
        raise ValueError(f"Invalid three-objective archive: {path}")
    if not np.all(np.isfinite(F)) or not np.all(np.isfinite(FE)):
        raise ValueError(f"Archive contains NaN or Inf: {path}")
    return F, FE


def _limit(*arrays: np.ndarray) -> float:
    maximum = max(float(np.max(values)) for values in arrays)
    return max(1.0, maximum) * 1.03


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )


def _axes_3d(axis: object, limit: float, title: str) -> None:
    axis.set_xlabel(LABELS[0], labelpad=5)
    axis.set_ylabel(LABELS[1], labelpad=5)
    axis.set_zlabel(LABELS[2], labelpad=5)
    axis.set_xlim(0.0, limit)
    axis.set_ylim(0.0, limit)
    axis.set_zlim(0.0, limit)
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")
    axis.view_init(elev=20, azim=45)
    axis.grid(True, alpha=0.2)
    axis.set_title(title, pad=8)


def _save(figure: plt.Figure, stem: Path, formats: list[str], dpi: int) -> list[Path]:
    paths: list[Path] = []
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_workflow(output: Path, formats: list[str], dpi: int) -> list[Path]:
    figure, axis = plt.subplots(figsize=(11.2, 3.0))
    axis.set_xlim(0, 11.2)
    axis.set_ylim(0, 3.0)
    axis.axis("off")

    boxes = (
        (0.2, 1.75, 1.45, 0.72, "PlatEMO EA\ntrue evaluations", BLUE),
        (2.0, 1.75, 1.45, 0.72, "Unbounded\narchive", BLUE),
        (3.8, 1.75, 1.45, 0.72, "Non-dominated\narchive", BLUE),
        (5.6, 1.75, 1.45, 0.72, "PF hole\nlocalization", ORANGE),
        (7.4, 1.75, 1.45, 0.72, "Single-head MLP\n$z^* \\mapsto \\hat{x}$", MAGENTA),
        (9.2, 1.75, 1.75, 0.72, "True evaluation\n$f(\\hat{x})$", GREEN),
    )
    for x, y, width, height, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center")

    for start, end in ((1.65, 2.0), (3.45, 3.8), (5.25, 5.6), (7.05, 7.4), (8.85, 9.2)):
        axis.annotate("", xy=(end, 2.11), xytext=(start, 2.11), arrowprops={"arrowstyle": "->"})

    axis.annotate(
        "archive update and dynamic hole reassessment",
        xy=(3.45, 1.75),
        xytext=(10.05, 0.65),
        ha="center",
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.22"},
    )
    axis.text(1.72, 2.66, "Stage 1", ha="center", color=BLUE, weight="bold")
    axis.text(7.65, 2.66, "Stage 2", ha="center", color=MAGENTA, weight="bold")
    return _save(figure, output / "fig00_method_workflow", formats, dpi)


def _plot_history(
    F: np.ndarray,
    FE: np.ndarray,
    reference: np.ndarray,
    output: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    limit = _limit(F, reference)
    figure = plt.figure(figsize=(6.4, 5.5))
    axis = figure.add_subplot(111, projection="3d")
    points = axis.scatter(
        F[:, 0], F[:, 1], F[:, 2], c=FE, cmap="viridis", s=2, alpha=0.28,
        depthshade=False, rasterized=True,
    )
    axis.scatter(
        reference[:, 0], reference[:, 1], reference[:, 2], color=GRAY, s=1,
        alpha=0.18, depthshade=False, rasterized=True,
    )
    _axes_3d(axis, limit, "All Stage 1 true evaluations")
    colorbar = figure.colorbar(points, ax=axis, fraction=0.035, pad=0.08)
    colorbar.set_label("Function evaluations (FE)")
    return _save(figure, output / "fig01_stage1_search_history_3d", formats, dpi)


def _plot_pf_3d(
    approximation: np.ndarray,
    reference: np.ndarray,
    title: str,
    output: Path,
    formats: list[str],
    dpi: int,
    stem: str = "fig02_stage1_pf_3d",
) -> list[Path]:
    limit = _limit(approximation, reference)
    figure = plt.figure(figsize=(6.4, 5.5))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        reference[:, 0], reference[:, 1], reference[:, 2], s=3, color=GRAY,
        alpha=0.32, depthshade=False, label="Reference PF", rasterized=True,
    )
    axis.scatter(
        approximation[:, 0], approximation[:, 1], approximation[:, 2], s=4,
        color=BLUE, alpha=0.62, depthshade=False, label="ND archive", rasterized=True,
    )
    _axes_3d(axis, limit, title)
    axis.legend(loc="upper right", frameon=False)
    return _save(figure, output / stem, formats, dpi)


def _plot_pairwise(
    approximation: np.ndarray,
    reference: np.ndarray,
    output: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    limit = _limit(approximation, reference)
    pairs = ((0, 1), (0, 2), (1, 2))
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    for axis, (first, second) in zip(axes, pairs, strict=True):
        axis.scatter(
            reference[:, first], reference[:, second], s=2, color=GRAY,
            alpha=0.28, rasterized=True,
        )
        axis.scatter(
            approximation[:, first], approximation[:, second], s=3, color=BLUE,
            alpha=0.55, rasterized=True,
        )
        axis.set_xlabel(LABELS[first])
        axis.set_ylabel(LABELS[second])
        axis.set_xlim(0.0, limit)
        axis.set_ylim(0.0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.18)
    return _save(figure, output / "fig03_stage1_pairwise", formats, dpi)


def _select_holes(reference: np.ndarray, approximation: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    scores = calculate_hole_distances(reference, approximation)
    count = int(config.get("num_holes", 10))
    method = str(config.get("hole_selection", "separated_topB"))
    separation = config.get("hole_min_separation")
    selected = select_multiple_holes(scores, count, method, separation)
    return scores.distances, selected


def _plot_holes(
    approximation: np.ndarray,
    reference: np.ndarray,
    config: dict,
    output: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    distances, selected = _select_holes(reference, approximation, config)
    limit = _limit(approximation, reference)
    figure = plt.figure(figsize=(6.5, 5.5))
    axis = figure.add_subplot(111, projection="3d")
    colored = axis.scatter(
        reference[:, 0], reference[:, 1], reference[:, 2], c=distances,
        cmap="magma", s=7, alpha=0.8, depthshade=False, rasterized=True,
    )
    axis.scatter(
        approximation[:, 0], approximation[:, 1], approximation[:, 2], s=3,
        color=BLUE, alpha=0.32, depthshade=False, label="Stage 1 ND archive",
        rasterized=True,
    )
    axis.scatter(
        reference[selected, 0], reference[selected, 1], reference[selected, 2],
        marker="*", s=75, color="#00a6a6", edgecolor="black", linewidth=0.35,
        depthshade=False, label="Selected holes",
    )
    _axes_3d(axis, limit, "Stage 1 PF hole localization")
    axis.legend(loc="upper right", frameon=False)
    colorbar = figure.colorbar(colored, ax=axis, fraction=0.035, pad=0.08)
    colorbar.set_label(r"$h(z)=\min_{a \in A_{ND}}\|z-f(a)\|_2$")
    return _save(figure, output / "fig04_stage1_holes", formats, dpi)


def _plot_psm_training(
    model_path: Path, output: Path, formats: list[str], dpi: int
) -> list[Path]:
    import torch

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    history = payload["history"]
    train = np.asarray(history["train_loss"], dtype=float)
    validation = np.asarray(history["validation_loss"], dtype=float)
    epochs = np.arange(1, len(train) + 1)
    best = int(history["best_epoch"]) + 1

    figure, axis = plt.subplots(figsize=(6.3, 3.8))
    axis.plot(epochs, train, color=BLUE, linewidth=1.3, label="Training")
    axis.plot(epochs, validation, color=ORANGE, linewidth=1.3, label="Validation")
    axis.axvline(best, color="0.35", linestyle="--", linewidth=1, label=f"Best epoch: {best}")
    if np.all(train > 0) and np.all(validation > 0):
        axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title("Single-head MLP training history")
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)
    return _save(figure, output / "fig05_psm_training", formats, dpi)


def _iteration_payloads(directory: Path) -> list[dict[str, np.ndarray]]:
    payloads: list[dict[str, np.ndarray]] = []
    for path in sorted(directory.glob("filling_iteration_*.mat")):
        raw = loadmat(path)
        payloads.append(
            {
                "iteration": np.asarray(raw["iteration"]).reshape(-1),
                "targets": np.asarray(raw["targets"], dtype=float),
                "objectives": np.asarray(raw["f_x_hat"], dtype=float),
                "distances": np.asarray(raw["target_distances"], dtype=float).reshape(-1),
                "old_igd": np.asarray(raw["old_igd_infinity"], dtype=float).reshape(-1),
                "new_igd": np.asarray(raw["new_igd_infinity"], dtype=float).reshape(-1),
            }
        )
    return payloads


def _plot_generated(
    payloads: list[dict[str, np.ndarray]],
    stage1: np.ndarray,
    reference: np.ndarray,
    output: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    targets = np.vstack([item["targets"] for item in payloads])
    objectives = np.vstack([item["objectives"] for item in payloads])
    limit = _limit(stage1, reference, objectives)
    figure = plt.figure(figsize=(6.5, 5.5))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        reference[:, 0], reference[:, 1], reference[:, 2], s=2, color=GRAY,
        alpha=0.2, depthshade=False, rasterized=True,
    )
    axis.scatter(
        stage1[:, 0], stage1[:, 1], stage1[:, 2], s=3, color=BLUE,
        alpha=0.25, depthshade=False, label="Stage 1 ND", rasterized=True,
    )
    axis.scatter(
        targets[:, 0], targets[:, 1], targets[:, 2], marker="*", s=24,
        color=ORANGE, alpha=0.75, depthshade=False, label="Hole targets",
        rasterized=True,
    )
    axis.scatter(
        objectives[:, 0], objectives[:, 1], objectives[:, 2], marker="^", s=14,
        color=MAGENTA, alpha=0.7, depthshade=False, label=r"True $f(\hat{x})$",
        rasterized=True,
    )
    line_indices = np.linspace(0, len(targets) - 1, min(150, len(targets)), dtype=int)
    for index in line_indices:
        axis.plot(
            [targets[index, 0], objectives[index, 0]],
            [targets[index, 1], objectives[index, 1]],
            [targets[index, 2], objectives[index, 2]],
            color="0.45", alpha=0.15, linewidth=0.45,
        )
    _axes_3d(axis, limit, "Generated candidates and true objective values")
    axis.legend(loc="upper right", frameon=False)
    return _save(figure, output / "fig06_generated_solutions", formats, dpi)


def _plot_before_after(
    stage1: np.ndarray,
    final: np.ndarray,
    reference: np.ndarray,
    output: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    limit = _limit(stage1, final, reference)
    figure = plt.figure(figsize=(10.5, 4.7))
    for position, (values, color, title) in enumerate(
        ((stage1, BLUE, "Before filling"), (final, GREEN, "After filling")), start=1
    ):
        axis = figure.add_subplot(1, 2, position, projection="3d")
        axis.scatter(
            reference[:, 0], reference[:, 1], reference[:, 2], s=2, color=GRAY,
            alpha=0.25, depthshade=False, rasterized=True,
        )
        axis.scatter(
            values[:, 0], values[:, 1], values[:, 2], s=3, color=color,
            alpha=0.58, depthshade=False, rasterized=True,
        )
        _axes_3d(axis, limit, title)
    figure.tight_layout()
    return _save(figure, output / "fig07_before_after_pf", formats, dpi)


def _plot_metrics(
    csv_path: Path, output: Path, formats: list[str], dpi: int
) -> list[Path]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    FE = np.asarray([float(row["FE"]) for row in rows])
    metrics = (
        ("igd_infinity", r"IGD$_\infty$", BLUE),
        ("igd", "IGD", ORANGE),
        ("hv", "HV", GREEN),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    for axis, (key, label, color) in zip(axes, metrics, strict=True):
        values = np.asarray([float(row[key]) for row in rows])
        axis.plot(FE, values, color=color, marker="o", markersize=2.5, linewidth=1.2)
        axis.set_xlabel("Function evaluations (FE)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.2)
    return _save(figure, output / "fig08_metrics_trajectory", formats, dpi)


def _plot_generation_quality(
    payloads: list[dict[str, np.ndarray]], output: Path, formats: list[str], dpi: int
) -> list[Path]:
    iterations = np.concatenate(
        [np.full(len(item["distances"]), int(item["iteration"][0])) for item in payloads]
    )
    distances = np.concatenate([item["distances"] for item in payloads])
    old_igd = np.asarray([item["old_igd"][0] for item in payloads])
    new_igd = np.asarray([item["new_igd"][0] for item in payloads])
    batch_iterations = np.asarray([int(item["iteration"][0]) for item in payloads])

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    axes[0].scatter(iterations, distances, s=6, color=MAGENTA, alpha=0.45, rasterized=True)
    axes[0].set_xlabel("Filling iteration")
    axes[0].set_ylabel(r"$\|z^*-f(\hat{x})\|_2$")
    axes[0].set_title("Target realization error")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(batch_iterations, old_igd, color=ORANGE, linewidth=1.1, label="Before batch")
    axes[1].plot(batch_iterations, new_igd, color=GREEN, linewidth=1.1, label="After batch")
    axes[1].set_xlabel("Filling iteration")
    axes[1].set_ylabel(r"IGD$_\infty$")
    axes[1].set_title("Largest-hole reassessment")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(frameon=False)
    return _save(figure, output / "fig09_generation_quality", formats, dpi)


def generate_figures(
    results_dir: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> tuple[list[Path], list[str]]:
    results_dir = results_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / "config_resolved.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}

    reference = _matrix(results_dir / "reference_set.mat", "Z")
    stage1_all, stage1_fe = _archive(results_dir / "stage1_archive.mat")
    stage1_nd = _matrix(results_dir / "stage1_nd_archive.mat", "F")
    algorithm = config.get("algorithm", "EA")
    problem = config.get("problem", "three-objective problem")

    generated: list[Path] = []
    skipped: list[str] = []
    generated.extend(_plot_workflow(output_dir, formats, dpi))
    generated.extend(_plot_history(stage1_all, stage1_fe, reference, output_dir, formats, dpi))
    generated.extend(
        _plot_pf_3d(
            stage1_nd, reference, f"{algorithm} on three-objective {problem}",
            output_dir, formats, dpi,
        )
    )
    generated.extend(_plot_pairwise(stage1_nd, reference, output_dir, formats, dpi))
    generated.extend(_plot_holes(stage1_nd, reference, config, output_dir, formats, dpi))

    model_path = results_dir / "psm_model.pt"
    if model_path.is_file():
        generated.extend(_plot_psm_training(model_path, output_dir, formats, dpi))
    else:
        skipped.append("fig05_psm_training (psm_model.pt is absent)")

    payloads = _iteration_payloads(results_dir)
    if payloads:
        generated.extend(_plot_generated(payloads, stage1_nd, reference, output_dir, formats, dpi))
        generated.extend(_plot_generation_quality(payloads, output_dir, formats, dpi))
    else:
        skipped.extend(
            (
                "fig06_generated_solutions (filling checkpoints are absent)",
                "fig09_generation_quality (filling checkpoints are absent)",
            )
        )

    final_path = results_dir / "final_nd_archive.mat"
    if final_path.is_file():
        final_nd = _matrix(final_path, "F")
        generated.extend(_plot_before_after(stage1_nd, final_nd, reference, output_dir, formats, dpi))
    else:
        skipped.append("fig07_before_after_pf (final_nd_archive.mat is absent)")

    metrics_path = results_dir / "metrics_history.csv"
    if metrics_path.is_file():
        generated.extend(_plot_metrics(metrics_path, output_dir, formats, dpi))
    else:
        skipped.append("fig08_metrics_trajectory (metrics_history.csv is absent)")

    stage1_h, selected = _select_holes(reference, stage1_nd, config)
    manifest = {
        "results_dir": str(results_dir),
        "algorithm": algorithm,
        "problem": problem,
        "stage1_evaluations": int(len(stage1_all)),
        "stage1_nd_size": int(len(stage1_nd)),
        "reference_size": int(len(reference)),
        "stage1_igd_infinity": float(np.max(stage1_h)),
        "selected_hole_indices_zero_based": selected.tolist(),
        "formats": formats,
        "dpi": dpi,
        "generated": [path.name for path in generated],
        "skipped": skipped,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return generated, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all publication figures available for an EA+Model run."
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("Results/NSGAII_DTLZ2_M3_seed1"),
        help="Experiment checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Figure directory (default: <results-dir>/figures).",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"),
        help="Output formats.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    output = args.output_dir or args.results_dir / "figures"
    generated, skipped = generate_figures(args.results_dir, output, list(args.formats), args.dpi)
    print(f"Generated {len(generated)} file(s) in {output.resolve()}")
    for path in generated:
        print(path)
    for item in skipped:
        print(f"Skipped: {item}")


if __name__ == "__main__":
    main()

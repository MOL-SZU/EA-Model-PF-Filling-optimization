# EA+Model Pareto-Front Hole Filling

A reproducible two-stage framework that combines a generic PlatEMO evolutionary
algorithm with a single-head MLP Pareto Set Model (PSM). Stage 1 performs the
evolutionary search and records every true evaluation. Stage 2 repeatedly finds
poorly covered locations on a reference Pareto front, maps each location
`z* -> MLP(z*) -> x_hat`, truly evaluates the candidates in PlatEMO, and updates
the unbounded archive.

The implementation follows
[`EA_Model_Implementation_and_Validation_Spec.md`](EA_Model_Implementation_and_Validation_Spec.md).
The former clustering and multi-head model are intentionally absent.

## Method

For a dense PF reference set `Z` and the current non-dominated archive `A_ND`,
the hole score is

```text
h(z) = min_{a in A_ND} ||z - f(a)||_2.
```

The largest score is `IGD-infinity`. Stage 2 selects either the top `B` scores
or a spatially separated top `B`, predicts one decision vector per target with
a plain MLP, and recomputes all scores after every evaluated batch. Model
training, inference, sorting, distances, and metrics do not consume FE.

## Repository Layout

```text
ea_model/                 Python package
  archive.py              unbounded archive and non-dominated filtering
  coverage.py             hole distances and top-B selection
  psm.py                  single-head MLP, dataset, training, inference
  metrics.py              IGD-infinity, IGD, HV
  stages.py               independently callable workflow stages
  pipeline.py             complete two-stage execution
  matlab_bridge.py        matlab.engine PlatEMO adapter
matlab/
  EA_ModelRecordedProblem.m   generic evaluation recorder
  run_platemo_ea.m            generic Stage 1 entry point
  evaluate_platemo_problem.m  Stage 2 true evaluation
  get_platemo_reference_set.m benchmark PF oracle
  get_platemo_problem_bounds.m decision bounds for Stage 2 resume
configs/                  reproducible JSON configurations
tests/                    module, integration, and FE-accounting tests
Results/                  generated checkpoints (ignored by Git)
```

## Installation

Python 3.10 or newer is required. Install the package and test dependencies:

```powershell
python -m pip install -e ".[test]"
```

Install MATLAB Engine for Python using the instructions for the installed
MATLAB release, then verify:

```powershell
python -c "import matlab.engine; print('MATLAB Engine available')"
```

PlatEMO is an external dependency and is not vendored. Set its root either in
the JSON config or through `PLATEMO_PATH`:

```powershell
$env:PLATEMO_PATH = 'D:\path\to\PlatEMO\PlatEMO'
```

The MATLAB recorder targets the current PlatEMO `PROBLEM`/`SOLUTION` API. It
decorates the problem evaluation path and does not modify individual EAs.

### Optional MaNSGA-II baseline

The author-provided MaNSGA-II implementation is an optional external
dependency. Install the revision pinned in
`third_party/MaNSGA-II.lock.json` into an existing PlatEMO checkout:

```bash
python scripts/install_mansgaii.py \
  --platemo-path "$PLATEMO_PATH" \
  --matlab /workspace/MATLAB/bin/matlab
```

This installs both `MaNSGAII` and `MaNSGAII_Norm`. The upstream repository
states compatibility with PlatEMO v4.12. It has no top-level license file, so
its source is fetched rather than vendored or relicensed under this project's
MIT license. `MaNSGA-II` is the 2025 algorithm by Pang et al.; it is not the
2014 `MO-NSGA-II` by Chen and Chiang.

## Running

Run NSGA-II + DTLZ2 with the supplied configuration:

```powershell
python -m ea_model.cli run --config configs/dtlz2_nsgaii.json
```

Run only Stage 1 and save its archive without training the MLP or starting PF
filling:

```bash
python -m ea_model.cli stage1 --config configs/dtlz2_nsgaii.json
```

The equivalent pinned MaNSGA-II baselines are:

```bash
python -m ea_model.cli stage1 --config configs/dtlz2_mansgaii.json
python -m ea_model.cli stage1 --config configs/dtlz2_mansgaii_norm.json
```

These configurations use the MaNSGA-II default tradeoff parameter `a=0.01`
and the same DTLZ2 dimensions, population size, 8000-FE Stage 1 budget, and
seed as the NSGA-II configuration.

Resume from the NSGA-II Stage 1 checkpoints, train the configured MLP, and use
the remaining 2000 true evaluations for iterative PF filling:

```bash
python -m ea_model.cli stage2 --config configs/dtlz2_nsgaii.json
```

The command verifies the saved configuration, archive FE count, non-dominated
archive, reference set, and current PlatEMO problem bounds before training. It
never reruns Stage 1. During filling it updates `stage2_status.json`,
`stage2_progress_archive.mat`, and `metrics_history.csv` after every evaluated
batch. It refuses to overwrite a completed or partially written Stage 2 run.

Generate the complete publication figure suite for every checkpoint currently
available in a result directory:

```powershell
python scripts/visualize_results.py --results-dir Results/NSGAII_DTLZ2_M3_seed1
```

PNG and vector PDF files are written to `<results-dir>/figures`, together with
`figure_manifest.json`. A Stage 1-only result produces the method workflow, EA
history, three-dimensional PF, pairwise projections, and hole localization.
After model training and filling, the same command also produces the training
curve, generated-solution view, before/after PF comparison, metric trajectory,
and generation-quality plots. Figures whose checkpoints do not exist are
reported as skipped rather than treated as errors.

Changing `algorithm` to `NSGAIII`, `MOEAD`, or `RVEA` requires no changes to
archive processing, model training, filling, or metrics. A deterministic
MATLAB-free smoke experiment is also available:

```powershell
python -m ea_model.cli validate-small --output Results/validation-small
```

The legacy command is retained as a thin launcher:

```powershell
python "Archive+Model.py" run --config configs/dtlz2_nsgaii.json
```

## Configuration

The main fields are:

| Field | Meaning |
|---|---|
| `algorithm`, `problem` | PlatEMO class names |
| `N`, `M`, `D` | population, objective, and decision dimensions |
| `maxFE`, `EA_ratio` | total budget and Stage 1 fraction |
| `reference_point_num` | requested true-PF oracle samples |
| `num_holes` | targets evaluated per Stage 2 batch |
| `hole_selection` | `topB` or `separated_topB` |
| `model.device` | `cuda`, `cuda:<index>`, `cpu`, or automatic selection |
| `model.*` | MLP width, depth, optimizer, and stopping settings |
| `retrain_model` | periodically retrain from the updated `A_ND` |
| `validation_mode` | deterministic detailed logging |
| `seed` | Python, PyTorch, and MATLAB random seed |

Stage 1 is aligned down to a whole population:
`FE_EA = floor(maxFE * EA_ratio / N) * N`; Stage 2 receives the exact remainder.
The pipeline rejects a PlatEMO run whose recorder count differs from this
budget, preventing silent FE overshoot.

## Outputs and Restartable Stages

Each run saves resolved configuration, Stage 1 and final archives, reference
set, decision bounds, hole scores, PSM dataset/model, every filling iteration,
metric trajectory, and a JSON summary. MATLAB-readable numerical checkpoints use `.mat`; the
PyTorch model uses `.pt`; the trajectory is also saved as `.csv`.

Individual operations are public functions in `ea_model.stages`, including
`run_hole_detection`, `run_build_psm_dataset`, `run_train_psm`,
`run_generate_candidates`, `run_true_evaluation`, `run_archive_update`, and
`run_metrics`. Archives can be restored with
`CheckpointManager.load_archive(...)`, so later stages do not require rerunning
the EA.

The supplied DTLZ2 configuration requires CUDA. Confirm the assigned GPU before
running it with `python -c "import torch; print(torch.cuda.is_available())"`.
Use `model.device: "cuda:1"` to select a specific visible GPU. A requested CUDA
device is never silently replaced by CPU.

## Validation

Run all MATLAB-independent checks:

```powershell
python scripts/run_validation.py
```

The suite covers archive completeness and updates, duplicate/non-dominated
filtering, DTLZ2 reference geometry, hand-computable hole distances,
IGD-infinity, IGD, HV, PSM matching/training/generation, bounds repair, exact FE
accounting, checkpoint reload, one-step filling, and a small end-to-end run.
The optional `matlab` test marker is reserved for environments with MATLAB and
PlatEMO installed.

## Scientific Scope and Limitations

- Using `Problem.GetOptimum(K)` for hole localization is a benchmark oracle.
  Unknown real-world PFs require a separately justified reference estimator.
- A deterministic single-output MLP cannot represent a genuinely one-to-many
  mapping from one PF location to disconnected Pareto-set branches. This first
  version prioritizes traceability over multimodal generative capacity.
- The archive stores constraints and applies feasibility-first filtering, but
  the initial validation experiments are unconstrained DTLZ problems.
- HV uses a fixed configured reference point when supplied. Cross-run HV
  comparisons are invalid if different reference points are used.
- Formal claims require independent runs and statistical analysis; the bundled
  smoke backend is for software validation, not paper results.

## References

1. L. M. Pang, Y. Nan, and H. Ishibuchi, "How to Find a Large Solution Set to
   Cover the Entire Pareto Front in Evolutionary Multi-Objective Optimization,"
   IEEE SMC, 2023.
2. Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB Platform for
   Evolutionary Multi-Objective Optimization," IEEE Computational Intelligence
   Magazine, 12(4), 73-87, 2017.
3. L. M. Pang, H. Ishibuchi, K. Deb, and K. Shang, "MaNSGA-II:
   Many-Objective NSGA-II," IEEE Transactions on Emerging Topics in
   Computational Intelligence, 2025, doi: 10.1109/TETCI.2025.3576105.

## License

MIT. PlatEMO and MATLAB are separate products under their own licenses.

# EA+Model Two-Stage Pareto-Front Hole Filling Framework
## Implementation and Validation Specification for MATLAB + PlatEMO

## 1. Goal

Implement a generic **EA+Model two-stage multi-objective optimization framework** in MATLAB/PlatEMO.

The framework must:

1. Support most PlatEMO algorithms through a generic algorithm interface rather than being tied to MOEA/D.
2. Record an unbounded search-history archive of all truly evaluated solutions.
3. Detect poorly covered regions on the Pareto Front using the distance-based hole definition inspired by Pang, Nan, and Ishibuchi (SMC 2023).
4. Train a Pareto Set Model (PSM) using archive data.
5. Use a **PF-location-conditioned mapping**

\[
\boxed{z^* \rightarrow M_\theta(z^*) \rightarrow \hat{x}}
\]

where:
- \(z^*\) is a poorly covered PF reference point,
- \(M_\theta\) is the learned Pareto Set Model,
- \(\hat{x}\) is a generated decision-space candidate.
6. Truly evaluate every model-generated candidate.
7. Add every truly evaluated candidate to the archive.
8. Reassess coverage during Stage 2 until the model-evaluation budget is exhausted.
9. Provide independent module-level tests, integration tests, saved intermediate states, and a validation mode so every critical step can be checked separately.

---

# 2. Overall Algorithm Logic

The method is a **two-stage framework**.

## Stage 1: Generic Evolutionary Search

\[
\text{Generic PlatEMO EA}
\rightarrow
\text{Unbounded Search-history Archive}
\rightarrow
A_{\mathrm{ND}}
\]

The EA is responsible for exploration.

The archive is responsible for retaining search history.

## Stage 2: Archive-based Model Filling

\[
A_{\mathrm{ND}}
\rightarrow
\text{PF Coverage Assessment}
\rightarrow
\text{PF Hole Localization}
\rightarrow
\text{PSM Training}
\rightarrow
\text{Targeted Generation}
\rightarrow
\text{True Evaluation}
\rightarrow
\text{Archive Update}
\rightarrow
\text{Coverage Reassessment}
\]

Within Stage 2, repeat:

\[
\boxed{
A_{\mathrm{ND}}
\rightarrow
z^*
\rightarrow
M_\theta(z^*)
\rightarrow
\hat{x}
\rightarrow
f(\hat{x})
\rightarrow
A'_{\mathrm{ND}}
}
\]

until the Stage 2 FE budget is exhausted.

Important:

- This is **not** the online PSM-MOEA/D closed loop.
- Model-generated candidates are used to update the archive.
- They are **not** fed back into the Stage 1 EA population for further evolutionary search.
- The EA finishes first; model-based filling happens afterward.

---

# 3. Generic PlatEMO Interface

Do not hard-code MOEA/D or any other specific optimizer.

The implementation should support calls conceptually like:

```matlab
result = run_EA_Model(@NSGAII, @DTLZ2, config);
result = run_EA_Model(@NSGAIII, @DTLZ2, config);
result = run_EA_Model(@MOEAD, @DTLZ2, config);
result = run_EA_Model(@RVEA, @DTLZ2, config);
```

Changing the base EA must not require changes to:

- archive processing,
- PF hole detection,
- model training,
- model generation,
- true evaluation,
- metrics,
- Stage 2 filling logic.

Use PlatEMO's normal calling pattern, parameterizing:

```matlab
'algorithm'
'problem'
'N'
'maxFE'
'draw'
'silent'
```

Prefer reusing PlatEMO's existing infrastructure rather than reimplementing optimizer internals.

---

# 4. Stage 1: EA Search and Unbounded Archive

Run the selected PlatEMO algorithm using the Stage 1 budget:

\[
FE_{\mathrm{EA}}
\]

Maintain:

\[
A_{\mathrm{all}}
\]

as an unbounded archive containing every truly evaluated solution:

\[
A_{\mathrm{all}}
=
\{(x_i,f(x_i),FE_i)\}
\]

Each record should contain at least:

- decision vector \(x_i\),
- objective vector \(f(x_i)\),
- evaluation index or FE number.

If PlatEMO does not expose all examined solutions directly, inspect its evaluation pathway and implement a generic, minimally intrusive recorder around the actual objective evaluation mechanism.

Do **not** separately modify NSGA-II, NSGA-III, MOEA/D, etc. unless absolutely necessary.

At the end of Stage 1:

1. remove duplicates,
2. compute the current non-dominated archive:

\[
A_{\mathrm{ND}}=ND(A_{\mathrm{all}})
\]

---

# 5. PF Reference Set

Construct a dense PF reference set:

\[
Z=\{z_1,z_2,\ldots,z_K\}
\]

where:

\[
z_j\in\mathbb{R}^{M}
\]

is a **PF location / PF reference point**.

Strictly distinguish:

```text
z, Z      : PF reference point / PF reference set
x         : decision vector
f(x)      : objective vector
A_all     : all truly evaluated solutions
A_ND      : current non-dominated archive
```

Do not use the previous main formulation based on:

```text
r*
W_d
diagnostic direction
angular coverage score
```

for the hole definition in this version.

For known benchmark problems such as DTLZ2 and DTLZ7, use a dense true-PF reference set when available.

Document clearly that using the true PF internally for hole localization is a **benchmark/oracle setting** and may not generalize directly to unknown real-world PFs.

---

# 6. PF Hole Definition

Use the distance-based hole definition inspired by:

**Pang, L. M., Nan, Y., & Ishibuchi, H.  
“How to Find a Large Solution Set to Cover the Entire Pareto Front in Evolutionary Multi-Objective Optimization,” IEEE SMC 2023.**

For every PF reference point \(z_j\in Z\), compute:

\[
h(z_j)
=
\min_{a_i\in A_{\mathrm{ND}}}
d(z_j,f(a_i))
\]

where \(d(\cdot,\cdot)\) is the objective-space distance.

Interpretation:

- small \(h(z_j)\): this PF location is well covered,
- large \(h(z_j)\): this PF location is poorly covered.

The largest hole is:

\[
\boxed{
z^*
=
\arg\max_{z_j\in Z} h(z_j)
}
\]

and:

\[
\boxed{
IGD^\infty_Z(A)
=
\max_{z_j\in Z}
\min_{a_i\in A}
d(z_j,a_i)
}
\]

This quantity measures the largest uncovered PF region.

---

# 7. Multiple-Hole Localization

The original \(IGD^\infty\) logic identifies the largest hole.

For model filling, extend this to multiple targets.

Compute:

\[
h(z_1),h(z_2),\ldots,h(z_K)
\]

Sort descending and select the top-\(B\) poorly covered PF locations:

\[
Z^*
=
\{z_1^*,z_2^*,\ldots,z_B^*\}
\]

Support:

```matlab
config.hole_selection = 'topB';
config.hole_selection = 'separated_topB';
```

`separated_topB` should optionally prevent several adjacent reference points from all representing the same physical hole.

Important:

- Do not claim that Pang et al. proposed top-\(B\) hole selection.
- Treat top-\(B\) as a project-specific extension built on the same distance-based hole definition.

---

# 8. Pareto Set Model Definition

The PSM must be defined as a **PF-location-conditioned model**:

\[
\boxed{
M_\theta:z\rightarrow x
}
\]

The key inference path is:

\[
\boxed{
z^*
\rightarrow
M_\theta(z^*)
\rightarrow
\hat{x}
}
\]

Do not use the old primary formulation:

\[
r^*\rightarrow M(r^*)\rightarrow x
\]

for this EA+Model version.

Use a simple MLP for the first implementation.

Do not initially add:

- CVAE,
- GAN,
- diffusion,
- complicated multi-head structures.

The first version should prioritize correctness, traceability, and reproducibility.

---

# 9. PSM Training Dataset

Use \(A_{\mathrm{ND}}\) to construct training pairs.

For every:

\[
(x_i,f(x_i))\in A_{\mathrm{ND}}
\]

find the nearest PF reference point:

\[
z_i
=
\arg\min_{z_j\in Z}
d(z_j,f(x_i))
\]

Construct:

\[
\boxed{
D=
\{(z_i,x_i)\}
}
\]

and train:

\[
M_\theta:z\rightarrow x
\]

Handle:

- PF-location normalization,
- decision-variable normalization,
- duplicate samples,
- train/validation split,
- invalid or non-finite values.

The model input dimension should equal the number of objectives \(M\).

The model output dimension should equal the number of decision variables \(D\).

---

# 10. Hole-Targeted Candidate Generation

Given:

\[
Z^*
=
\{z_1^*,\ldots,z_B^*\}
\]

support batch inference:

\[
\begin{bmatrix}
z_1^*\\
z_2^*\\
\vdots\\
z_B^*
\end{bmatrix}
\xrightarrow{M_\theta}
\begin{bmatrix}
\hat{x}_1\\
\hat{x}_2\\
\vdots\\
\hat{x}_B
\end{bmatrix}
\]

First version:

```text
one PF hole target -> one candidate
```

Reserve:

```matlab
config.candidates_per_hole
```

for future extension.

Do not assume that model output has successfully filled a hole.

Each output is only a candidate until it is truly evaluated.

---

# 11. True Objective Evaluation

For each generated candidate \(\hat{x}\):

1. repair decision-variable bounds,
2. call the true objective function of the selected PlatEMO problem,
3. obtain:

\[
f(\hat{x})
\]

4. increment FE exactly once.

Every model-generated solution must be truly evaluated.

Count as FE:

```text
true objective evaluation
```

Do not count as FE:

```text
model training
model inference
hole detection
distance calculation
non-dominated sorting
metric calculation
```

Every evaluated model candidate must first be added to:

\[
A_{\mathrm{all}}
\]

even if it is dominated.

Then recompute:

\[
A_{\mathrm{ND}}=ND(A_{\mathrm{all}})
\]

---

# 12. Verify Whether the Target Hole Was Reached

For each target \(z_i^*\), record:

\[
d(z_i^*,f(\hat{x}_i))
\]

Save:

```text
target PF location z*
generated decision vector x_hat
evaluated objective vector f(x_hat)
distance d(z*, f(x_hat))
dominance status
```

This is essential for checking whether the model actually generated a solution near the intended hole.

---

# 13. Stage 2 Iterative Filling

Within the Stage 2 FE budget:

```text
Extract A_ND
    ↓
Compute h(z)
    ↓
Select current hole targets Z*
    ↓
Generate candidates with Mθ
    ↓
True objective evaluation
    ↓
Update A_all
    ↓
Recompute A_ND
    ↓
Reassess coverage
    ↓
Repeat
```

Do not keep using the first detected \(Z^*\) forever.

After every evaluated batch, recompute the coverage state.

---

# 14. Model Retraining

Default first version:

```matlab
config.retrain_model = false;
```

Train once after Stage 1.

Reserve:

```matlab
config.retrain_model = true;
config.retrain_interval = ...;
```

for experiments that periodically rebuild:

\[
A_{\mathrm{ND}}
\rightarrow
D
\rightarrow
M_\theta
\]

during Stage 2.

This allows future comparison of:

```text
fixed model
vs.
periodically retrained model
```

---

# 15. FE Budget

Enforce:

\[
\boxed{
FE_{\max}
=
FE_{\mathrm{EA}}
+
FE_{\mathrm{Model}}
}
\]

Recommended default:

\[
FE_{\mathrm{EA}}=0.8FE_{\max}
\]

\[
FE_{\mathrm{Model}}=0.2FE_{\max}
\]

but expose this through config.

If fewer than \(B\) FEs remain, only generate and evaluate the number of candidates allowed by the remaining budget.

Never exceed:

\[
FE_{\max}
\]

---

# 16. Evaluation Metrics

At minimum compute:

\[
\boxed{
IGD^\infty,\quad IGD,\quad HV
}
\]

Record both:

```text
after Stage 1
after Stage 2
```

Interpretation:

### IGD∞

\[
IGD^\infty(A)
=
\max_{z\in Z}\min_{a\in A}d(z,a)
\]

Purpose:

```text
measure the largest remaining PF hole
```

Smaller is better.

### IGD

\[
IGD(A)
=
\frac{1}{|Z|}
\sum_{z\in Z}
\min_{a\in A}d(z,a)
\]

Purpose:

```text
measure average PF coverage quality
```

Smaller is better.

### HV

Purpose:

```text
measure the overall dominated objective-space volume of the non-dominated set
```

Larger is better.

Save a filling trajectory:

```text
FE
IGD∞
IGD
HV
```

---

# 17. Recommended Project Structure

```text
EA_Model/
│
├── run_EA_Model.m
├── config_EA_Model.m
│
├── EA/
│   ├── RunPlatEMOEA.m
│   └── ArchiveRecorder.m
│
├── Archive/
│   ├── UpdateArchive.m
│   ├── GetNonDominatedArchive.m
│   └── RemoveDuplicates.m
│
├── Coverage/
│   ├── GeneratePFReferenceSet.m
│   ├── CalculateHoleDistances.m
│   ├── FindLargestHole.m
│   └── SelectMultipleHoles.m
│
├── Model/
│   ├── BuildPSMDataset.m
│   ├── TrainPSM.m
│   └── PredictCandidates.m
│
├── Filling/
│   ├── RepairCandidate.m
│   ├── EvaluateCandidate.m
│   └── ModelFilling.m
│
├── Metrics/
│   ├── IGDInfinity.m
│   ├── IGD.m
│   └── HV.m
│
├── Validation/
│   ├── test_ArchiveRecorder.m
│   ├── test_NonDominatedArchive.m
│   ├── test_ReferenceSet.m
│   ├── test_HoleDistance.m
│   ├── test_IGDInfinity.m
│   ├── test_IGD.m
│   ├── test_HV.m
│   ├── test_PSMDataset.m
│   ├── test_PSMTraining.m
│   ├── test_ModelGeneration.m
│   ├── test_TrueEvaluation.m
│   ├── test_FEAccounting.m
│   ├── test_ArchiveUpdate.m
│   ├── test_OneStepFilling.m
│   ├── test_EndToEndSmall.m
│   └── run_all_validation_tests.m
│
└── Results/
```

Adapt this structure to the existing project if needed.

Do not mechanically duplicate functionality already available in PlatEMO.

---

# 18. Config Requirements

At minimum support:

```matlab
config.algorithm
config.problem
config.N
config.M
config.maxFE
config.EA_ratio

config.reference_point_num
config.num_holes
config.hole_selection

config.model.hidden_size
config.model.num_layers
config.model.learning_rate
config.model.epochs

config.candidates_per_hole
config.retrain_model
config.retrain_interval

config.validation_mode
config.seed
```

---

# 19. Independent Validation and Verification Layer

The project must not only run end-to-end.

Every critical module must be independently executable and independently verifiable.

The validation principle is:

\[
\boxed{
\text{each critical step must be testable in isolation}
}
\]

The purpose is to detect whether an error comes from:

```text
archive recording
non-dominated filtering
PF reference construction
hole localization
PSM dataset construction
model training
candidate generation
true objective evaluation
archive update
FE accounting
metrics
```

---

# 20. Archive Recorder Validation

Provide:

```matlab
test_ArchiveRecorder.m
```

Use a very small FE budget.

Verify:

```text
every true evaluation is recorded
X and F are one-to-one
FE indices increase correctly
number of records matches actual evaluations
no unexplained missing records
no unintended duplicate recording
```

Print or save all records for manual inspection.

---

# 21. Non-Dominated Filtering Validation

Provide:

```matlab
test_NonDominatedArchive.m
```

Use a hand-constructed small objective set such as:

\[
F=
\begin{bmatrix}
1&4\\
2&3\\
3&2\\
4&1\\
3&4
\end{bmatrix}
\]

Verify that the dominated point is correctly removed.

Also verify duplicate handling.

---

# 22. PF Reference Set Validation

Provide:

```matlab
test_ReferenceSet.m
```

For known benchmark PFs verify:

```text
size(Z,2) == M
no NaN
no Inf
reference points satisfy known PF constraints
distribution is sufficiently dense
```

For DTLZ2, verify approximately:

\[
\sum_{j=1}^{M}z_j^2\approx1
\]

for every reference point.

---

# 23. Hole Distance Validation

Provide:

```matlab
test_HoleDistance.m
```

Use a simple manually checkable example:

\[
Z=
\{
(0,1),
(0.25,0.75),
(0.5,0.5),
(0.75,0.25),
(1,0)
\}
\]

and:

\[
A=
\{
(0,1),
(1,0)
\}
\]

Manually verify:

\[
h(z_j)=\min_{a\in A}d(z_j,a)
\]

Output:

```text
z_j
nearest archive solution
h(z_j)
largest-hole location
IGD∞
```

The selected largest hole should match the hand calculation.

---

# 24. IGD∞ Validation

Provide:

```matlab
test_IGDInfinity.m
```

Verify the implementation against a manually computable case.

Also verify:

1. adding a solution near the largest hole should reduce or preserve \(IGD^\infty\),
2. if Archive equals the reference set, then:

\[
IGD^\infty\approx0
\]

---

# 25. IGD Validation

Provide:

```matlab
test_IGD.m
```

Verify:

\[
IGD(A)
=
\frac{1}{|Z|}
\sum_{z\in Z}
\min_{a\in A}d(z,a)
\]

against a hand-calculated example.

---

# 26. HV Validation

Provide:

```matlab
test_HV.m
```

Use a simple two-objective non-dominated set and a fixed reference point.

Verify:

```text
HV equals hand calculation
dominated points do not incorrectly increase HV
changing the HV reference point changes the result consistently
```

If reusing PlatEMO's HV implementation, still provide a wrapper-level sanity test.

---

# 27. PSM Dataset Validation

Provide:

```matlab
test_PSMDataset.m
```

For each archive sample verify:

\[
z_i
=
\arg\min_{z_j\in Z}
d(z_j,f(x_i))
\]

Output:

```text
x_i
f(x_i)
matched z_i
matching distance
```

For 2D or 3D cases, optionally visualize the matching for manual inspection.

---

# 28. PSM Training Validation

Provide:

```matlab
test_PSMTraining.m
```

Before using real archive data, construct a synthetic mapping:

\[
x=g(z)
\]

for example:

\[
x_1=z_1,\qquad x_2=z_2
\]

Train the MLP on the synthetic dataset.

Verify:

```text
training loss decreases
validation error is reasonable
input/output dimensions are correct
no NaN or Inf
predictions approximately match the known mapping
```

Only after this test passes should the same training pipeline be used on real archive data.

---

# 29. Model Generation Validation

Provide:

```matlab
test_ModelGeneration.m
```

For several target points:

\[
z_1^*,\ldots,z_B^*
\]

verify:

```text
batch size is correct
decision dimension is correct
outputs are finite
outputs can be repaired into valid bounds
different target z* values do not all collapse to identical outputs
```

If all outputs are nearly identical, print a warning.

---

# 30. True Objective Evaluation Validation

Provide:

```matlab
test_TrueEvaluation.m
```

For several known decision vectors \(x\):

1. evaluate through the EA+Model wrapper,
2. evaluate directly through the original PlatEMO problem logic.

Verify:

\[
f_{\mathrm{wrapper}}(x)
\approx
f_{\mathrm{PlatEMO}}(x)
\]

This must be numerically consistent.

---

# 31. FE Accounting Validation

Provide:

```matlab
test_FEAccounting.m
```

For example:

```text
maxFE = 1000
EA_ratio = 0.8
```

verify:

\[
FE_{\mathrm{EA}}=800
\]

\[
FE_{\mathrm{Model}}=200
\]

\[
FE_{\mathrm{total}}=1000
\]

Check that:

```text
model training does not increase FE
model inference does not increase FE
hole detection does not increase FE
metrics do not increase FE
each true candidate evaluation increments FE exactly once
final batch never exceeds maxFE
```

---

# 32. Archive Update Validation

Provide:

```matlab
test_ArchiveUpdate.m
```

Insert a generated candidate and verify:

```text
candidate enters A_all
A_all size increases correctly
A_ND is recomputed
dominated candidate remains in A_all
dominated candidate is excluded from A_ND
```

---

# 33. One-Step Filling Validation

Provide:

```matlab
test_OneStepFilling.m
```

Run exactly one full filling step:

\[
A_{\mathrm{ND}}
\rightarrow
z^*
\rightarrow
M_\theta(z^*)
\rightarrow
\hat{x}
\rightarrow
f(\hat{x})
\rightarrow
A'_{\mathrm{all}}
\]

Output a full trace:

```text
selected z*
old h(z*)
generated x_hat
true f(x_hat)
distance d(z*,f(x_hat))
dominance status
new h(z*)
old IGD∞
new IGD∞
```

This is the most important integration test before running the full framework.

---

# 34. Small End-to-End Test

Provide:

```matlab
test_EndToEndSmall.m
```

Use a very small setup such as:

```text
Problem      : DTLZ2
M            : 3
N            : 20
maxFE        : 500
EA_ratio     : 0.8
num_holes    : 2
```

Verify:

```text
Stage 1 completes
A_all is created
A_ND is extracted
PF reference set is valid
holes are localized
PSM is trained
candidates are generated
candidates are truly evaluated
archive is updated
Stage 2 completes
FE is exact
IGD∞, IGD, and HV are produced
```

This test is for regression and debugging, not for paper results.

---

# 35. Defensive Assertions

Use `assert`, `error`, or `warning` at critical interfaces.

Examples:

```matlab
assert(size(Z,2) == M);
assert(size(X,1) == size(F,1));
assert(all(isfinite(X(:))));
assert(all(isfinite(F(:))));
assert(FE <= maxFE);
```

Before true evaluation, verify decision bounds.

For the model, verify:

```text
input dimension = M
output dimension = D
```

Do not silently continue after critical inconsistencies.

---

# 36. Validation Mode

Support:

```matlab
config.validation_mode = true;
```

In validation mode:

```text
fix random seed
use small N
use small FE
enable detailed logging
save all intermediate variables
keep temporary files
```

This mode should make debugging deterministic and reproducible.

---

# 37. Debug Logging

Provide a unified logger or structured diagnostic output.

At minimum support:

```text
INFO
DEBUG
WARNING
ERROR
```

In validation mode print:

```text
current FE
archive size
ND archive size
current IGD∞
selected z*
h(z*)
generated x_hat
true f(x_hat)
distance to target
```

Detailed logs should be switchable off for formal experiments.

---

# 38. Intermediate Result Saving

Save reusable checkpoints such as:

```text
stage1_archive.mat
stage1_nd_archive.mat
reference_set.mat
hole_scores.mat
psm_dataset.mat
psm_model.mat
filling_iteration_001.mat
metrics_history.mat
```

The system must support restarting from saved intermediate results.

Examples:

```matlab
run_hole_detection('stage1_archive.mat');
run_train_psm('stage1_archive.mat');
run_generate_candidates('psm_model.mat', 'hole_scores.mat');
```

Do not require rerunning Stage 1 every time a later module is tested.

---

# 39. Step-by-Step Execution Mode

Besides:

```matlab
run_EA_Model(...)
```

provide modular entry points conceptually equivalent to:

```matlab
run_stage1_EA(...)
run_extract_archive(...)
run_generate_reference_set(...)
run_hole_detection(...)
run_build_psm_dataset(...)
run_train_psm(...)
run_generate_candidates(...)
run_true_evaluation(...)
run_archive_update(...)
run_metrics(...)
```

Each stage should accept saved outputs from the previous stage whenever practical.

---

# 40. Validation Report

Provide:

```matlab
run_all_validation_tests.m
```

The final report should clearly show:

```text
PASS Archive recorder
PASS Non-dominated filtering
PASS PF reference set
PASS Hole distance
PASS IGD∞
PASS IGD
PASS HV
PASS PSM dataset
PASS PSM synthetic training
PASS Candidate generation
PASS True objective evaluation
PASS FE accounting
PASS Archive update
PASS One-step filling
PASS Small end-to-end test
```

If a test fails, report:

```text
FAIL <test name>
expected = ...
actual   = ...
reason   = ...
```

Do not return only an uninformative generic failure message.

---

# 41. Initial Validation Experiments

First validate:

```text
NSGA-II  + DTLZ2
NSGA-III + DTLZ2
MOEA/D   + DTLZ2
```

Then:

```text
DTLZ7
```

Support at least:

\[
M=3
\]

and:

\[
M=5
\]

Only after the validation layer passes should larger formal experiments be run.

---

# 42. Agent Workflow Before Coding

Before implementing the full framework:

1. inspect the current MATLAB project,
2. inspect the current PlatEMO calling pattern,
3. locate the actual objective-evaluation entry point,
4. identify how to record all examined solutions generically,
5. identify how PF/reference data can be generated,
6. check whether PlatEMO already provides reliable IGD/HV implementations,
7. list all files that will be added or modified,
8. describe the data flow,
9. describe FE accounting,
10. then begin implementation.

Prefer reusing reliable PlatEMO components.

Do not duplicate existing functionality unnecessarily.

---

# 43. Final Acceptance Checklist

```text
[ ] Base EA can be replaced by config
[ ] MOEA/D is not hard-coded
[ ] Every true evaluation is recorded
[ ] A_all is correct
[ ] A_ND is correct
[ ] PF reference set Z is correct
[ ] Hole definition uses nearest-archive distance
[ ] Largest hole produces z*, not r*
[ ] Multiple holes can be selected
[ ] PSM implements z -> x
[ ] Batch hole-targeted generation works
[ ] Bounds repair works
[ ] Every generated candidate is truly evaluated
[ ] d(z*, f(x_hat)) is recorded
[ ] Every evaluated candidate enters A_all
[ ] A_ND is recomputed after filling
[ ] Stage 2 reassesses holes dynamically
[ ] FE never exceeds maxFE
[ ] IGD∞ is correct
[ ] IGD is correct
[ ] HV is correct
[ ] Fixed seed is supported
[ ] Validation mode works
[ ] Intermediate states can be saved and reloaded
[ ] Key modules can be run independently
[ ] One-step filling test passes
[ ] Small end-to-end test passes
```

---

# 44. Expected Final Deliverables

After implementation, provide:

1. final directory structure,
2. all added files,
3. all modified files,
4. main execution example,
5. config documentation,
6. a minimal runnable experiment,
7. Stage 1 FE,
8. Stage 2 FE,
9. total FE,
10. Stage 1 \(IGD^\infty\), IGD, HV,
11. Stage 2 \(IGD^\infty\), IGD, HV,
12. per-iteration selected \(z^*\),
13. per-iteration generated \(\hat{x}\),
14. per-iteration true \(f(\hat{x})\),
15. per-iteration target error \(d(z^*,f(\hat{x}))\),
16. filling trajectory,
17. validation report,
18. known limitations of the current implementation.

---

# 45. Core Design Summary

The final implemented logic should remain:

\[
\boxed{
\text{Generic PlatEMO EA Search}
\rightarrow
\text{Unbounded Archive}
\rightarrow
A_{\mathrm{ND}}
\rightarrow
\text{PF Hole Localization}
\rightarrow
z^*
\rightarrow
M_\theta(z^*)
\rightarrow
\hat{x}
\rightarrow
f(\hat{x})
\rightarrow
\text{Archive Update}
\rightarrow
\text{Coverage Reassessment}
}
\]

The critical modeling relation is:

\[
\boxed{
z^* \rightarrow M_\theta(z^*) \rightarrow \hat{x}
}
\]

The critical validation principle is:

\[
\boxed{
\text{Every major module must be independently executable and independently verifiable.}
}
\]

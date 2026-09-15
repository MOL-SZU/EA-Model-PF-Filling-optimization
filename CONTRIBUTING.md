# Contributing

Use a focused branch and include tests for behavioral changes. Before opening a
pull request, run:

```powershell
python -m compileall -q ea_model tests
python -m pytest -v -m "not matlab"
```

Changes to the MATLAB bridge should additionally be checked against the
supported PlatEMO revision with `PLATEMO_PATH` configured. Report the MATLAB
release, PlatEMO commit, algorithm, problem, dimensions, population size, FE
budget, and seed in bridge-related issues.

Do not commit generated files under `Results/`. New metrics or model variants
must retain exact FE accounting and the unbounded archive invariant.

Do not vendor MaNSGA-II source into this MIT-licensed repository unless the
upstream authors provide compatible redistribution terms. Update the pinned
commit and provenance in `third_party/MaNSGA-II.lock.json` deliberately.

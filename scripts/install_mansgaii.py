"""Install the author-provided MaNSGA-II code at a pinned commit into PlatEMO."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "third_party" / "MaNSGA-II.lock.json"
REQUIRED_FILES = (
    "MaNSGAII.m",
    "MaNSGAII_Norm.m",
    "EnvironmentalSelection.m",
    "EnvironmentalSelectionNorm.m",
    "DisSel.m",
    "README.md",
)


def _run(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def _load_lock() -> dict[str, object]:
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if not payload.get("repository") or not payload.get("commit"):
        raise ValueError(f"Invalid dependency lock: {LOCK_PATH}")
    return payload


def _destination(platemo_path: Path) -> Path:
    return (
        platemo_path
        / "Algorithms"
        / "Multi-objective optimization"
        / "MaNSGA-II"
    )


def _verify_checkout(destination: Path, expected_commit: str) -> str:
    missing = [name for name in REQUIRED_FILES if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"MaNSGA-II checkout is incomplete; missing: {', '.join(missing)}")
    if not (destination / ".git").exists():
        raise RuntimeError(
            f"{destination} is not the pinned Git checkout created by this installer"
        )
    actual_commit = _run(["git", "rev-parse", "HEAD"], cwd=destination)
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"MaNSGA-II commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
    return actual_commit


def _verify_matlab(matlab: str, platemo_path: Path) -> None:
    escaped = str(platemo_path).replace("'", "''")
    expression = (
        f"addpath(genpath('{escaped}')); "
        "a=which('MaNSGAII'); b=which('MaNSGAII_Norm'); "
        "disp(a); disp(b); assert(~isempty(a)); assert(~isempty(b));"
    )
    subprocess.run([matlab, "-batch", expression], check=True)


def install(platemo_path: Path, check_only: bool, matlab: str | None) -> Path:
    platemo_path = platemo_path.expanduser().resolve()
    if not (platemo_path / "platemo.m").is_file():
        raise FileNotFoundError(
            f"--platemo-path must be the directory containing platemo.m: {platemo_path}"
        )
    lock = _load_lock()
    repository = str(lock["repository"])
    commit = str(lock["commit"])
    destination = _destination(platemo_path)

    if not destination.exists():
        if check_only:
            raise FileNotFoundError(f"MaNSGA-II is not installed: {destination}")
        if shutil.which("git") is None:
            raise RuntimeError("git is required to install MaNSGA-II")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repository, str(destination)])
        _run(["git", "checkout", "--detach", commit], cwd=destination)

    actual_commit = _verify_checkout(destination, commit)
    print(f"MaNSGA-II directory: {destination}")
    print(f"MaNSGA-II commit: {actual_commit}")
    print(f"Upstream-tested PlatEMO version: {lock['upstream_supported_platemo']}")

    if matlab:
        resolved_matlab = shutil.which(matlab) if not Path(matlab).is_file() else matlab
        if not resolved_matlab:
            raise FileNotFoundError(f"MATLAB executable was not found: {matlab}")
        _verify_matlab(str(resolved_matlab), platemo_path)
        print("MATLAB discovery check passed for MaNSGAII and MaNSGAII_Norm")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the pinned author implementation of MaNSGA-II into PlatEMO."
    )
    parser.add_argument(
        "--platemo-path",
        type=Path,
        default=os.environ.get("PLATEMO_PATH"),
        required=not bool(os.environ.get("PLATEMO_PATH")),
        help="Directory containing platemo.m (or set PLATEMO_PATH).",
    )
    parser.add_argument(
        "--matlab",
        default=None,
        help="Optional MATLAB executable used to verify algorithm discovery.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the installed files and pinned commit without cloning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install(args.platemo_path, args.check_only, args.matlab)


if __name__ == "__main__":
    main()

"""Scheduling guards for the Makefile's lint targets.

Both properties asserted here are wall-clock only: reverting either one still
produces a correct lint run, just a slower one, so nothing but a test that
inspects the schedule itself will catch the regression.
"""

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"
GATE_SLOT_LOCK = ROOT / "scripts" / "gate_slot_lock.py"

BARRIER_HELPER = """barrier_sync() {
    touch "$STUB_BARRIER_DIR/$1.started"
    tries=0
    while [ ! -f "$STUB_BARRIER_DIR/$2.started" ]; do
        tries=$((tries + 1))
        if [ "$tries" -gt 100 ]; then
            echo "barrier timeout: $1 ran without $2" >&2
            exit 1
        fi
        sleep 0.1
    done
}
"""

UV_STUB = """#!/bin/sh
. "$STUB_BIN/barrier.sh"
case "$1" in
    sync) barrier_sync uv-sync git-fetch ;;
esac
exit 0
"""

GIT_STUB = """#!/bin/sh
. "$STUB_BIN/barrier.sh"
case "$1" in
    fetch) barrier_sync git-fetch uv-sync ;;
esac
exit 0
"""

LINT_CHECKS_LONGEST_FIRST = (
    ("lint-basedpyright", "scripts/type_check_gate.py"),
    ("lint-type-discipline", "scripts/type_discipline_gate.py"),
    ("lint-e2e-basedpyright", "basedpyright tests/e2e"),
    ("check-import-safety", "from litellm import *"),
    ("lint-gate", "scripts/ruff_strict_gate.py"),
    ("lint-ruff", "ruff check ."),
    ("check-circular-imports", "test_circular_imports.py"),
    ("lint-format-check-changed", "ruff format --check"),
)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _sandbox(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "litellm").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "tests" / "e2e").mkdir(parents=True)
    shutil.copy(MAKEFILE, repo / "Makefile")
    shutil.copy(GATE_SLOT_LOCK, repo / "scripts" / "gate_slot_lock.py")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "barrier.sh").write_text(BARRIER_HELPER)
    _write_executable(bin_dir / "uv", UV_STUB)
    _write_executable(bin_dir / "git", GIT_STUB)

    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    return repo, bin_dir, barrier_dir


def test_lint_overlaps_env_sync_with_base_fetch(tmp_path: Path) -> None:
    """`make lint` must start the env sync and the base fetch together.

    One waits on the disk and the other on the network, so running them in
    sequence pays their sum for no reason. The stubs deadlock and time out if
    either one is allowed to finish before the other starts.
    """
    repo, bin_dir, barrier_dir = _sandbox(tmp_path)
    result = subprocess.run(
        ["make", "lint"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "PATH": os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"]),
            "HOME": str(tmp_path),
            "STUB_BIN": str(bin_dir),
            "STUB_BARRIER_DIR": str(barrier_dir),
            "LITELLM_GATE_SLOTS": "0",
        },
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (barrier_dir / "uv-sync.started").exists()
    assert (barrier_dir / "git-fetch.started").exists()


def test_lint_checks_are_declared_longest_first() -> None:
    """The fan-out is bounded by its longest check, so that one must start first.

    make starts prerequisites in declared order as slots free, so a cheap check
    declared ahead of basedpyright delays the whole makespan by its own runtime
    at narrow -j. `make -n` walks the same order without running anything.
    """
    dry_run = subprocess.run(
        [
            "make",
            "-n",
            "LINT_DEP_INSTALL=",
            "LINT_E2E_DEP_INSTALL=",
            "LINT_DEP_BASE=",
            "lint-checks",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert dry_run.returncode == 0, dry_run.stderr

    positions = {name: dry_run.stdout.find(marker) for name, marker in LINT_CHECKS_LONGEST_FIRST}
    unmatched = sorted(name for name, at in positions.items() if at < 0)
    assert not unmatched, (
        f"these checks no longer match their marker, so the order is unverifiable: {unmatched}\n"
        f"{dry_run.stdout}"
    )

    expected = [name for name, _ in LINT_CHECKS_LONGEST_FIRST]
    assert sorted(positions, key=positions.__getitem__) == expected, (
        "lint-checks prerequisites are no longer ordered longest-first; declare the "
        "slowest check first so it is never the last thing left running"
    )

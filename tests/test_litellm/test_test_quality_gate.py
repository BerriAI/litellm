"""Tests for scripts/test_quality_gate.py.

The gate's whole value is that it blames a change only for what it adds. That lives
in pure functions, so they are tested directly: `lint_base_counts.evaluate` for the
blame rule and `parse_changed_lines` for the diff scan that turns a breach into
file:line. The base scan spawns a worktree and a checker subprocess, so its cleanup
on termination is driven end to end.
"""

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import NamedTuple

import lint_base_counts
import test_quality_gate as gate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "test_quality_gate.py"

_SCAN_BASE = (
    "import pathlib, sys\n"
    "import test_quality_gate as gate\n"
    "gate.base_counts('HEAD', repo_root=pathlib.Path(sys.argv[1]), checker=pathlib.Path(sys.argv[2]))\n"
)
_SCAN_BASE_WITH_SIGHUP_IGNORED = "import signal\nsignal.signal(signal.SIGHUP, signal.SIG_IGN)\n" + _SCAN_BASE


def test_a_rule_that_did_not_grow_is_not_a_breach():
    assert lint_base_counts.evaluate({"TQ001": 10}, {"TQ001": 10}, {}) == ()


def test_a_rule_the_change_grew_is_a_breach_for_exactly_what_it_added():
    breaches = lint_base_counts.evaluate({"TQ001": 12}, {"TQ001": 10}, {})
    assert [(b.rule, b.total, b.cap, b.added) for b in breaches] == [("TQ001", 12, 10, 2)]


def test_drift_already_in_the_base_is_not_blamed_on_the_change():
    assert lint_base_counts.evaluate({"TQ001": 14}, {"TQ001": 14}, {}) == ()


def test_a_change_that_reduces_a_rule_is_not_blamed():
    assert lint_base_counts.evaluate({"TQ001": 13}, {"TQ001": 14}, {}) == ()


def test_a_rule_absent_from_head_counts_as_zero():
    assert lint_base_counts.evaluate({}, {}, {}) == ()


def test_the_gate_grants_no_headroom_to_any_rule():
    assert dict(gate.HEADROOM) == {}


def test_the_checker_identity_is_keyed_on_the_checker_source():
    identity = gate.checker_identity()
    assert identity == lint_base_counts.Checker("test-quality", (lint_base_counts.sha256_of(gate.CHECKER),))


def test_parse_changed_lines_groups_hunks_under_their_own_file():
    diff = (
        "diff --git a/tests/a.py b/tests/a.py\n"
        "--- a/tests/a.py\n"
        "+++ b/tests/a.py\n"
        "@@ -0,0 +3,2 @@\n"
        "+one\n"
        "+two\n"
        "diff --git a/tests/b.py b/tests/b.py\n"
        "--- a/tests/b.py\n"
        "+++ b/tests/b.py\n"
        "@@ -0,0 +10 @@\n"
        "+only\n"
    )
    changed = gate.parse_changed_lines(diff)
    assert changed["tests/a.py"] == frozenset({3, 4})
    assert changed["tests/b.py"] == frozenset({10})


def test_parse_changed_lines_handles_several_hunks_in_one_file():
    diff = (
        "+++ b/tests/a.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+a\n"
        "@@ -9,0 +20,1 @@\n"
        "+b\n"
    )
    assert gate.parse_changed_lines(diff)["tests/a.py"] == frozenset({1, 2, 20})


def test_parse_changed_lines_on_an_empty_diff_is_empty():
    assert dict(gate.parse_changed_lines("")) == {}


def test_introduced_keeps_only_violations_on_changed_lines():
    violations = (
        gate.Violation("tests/a.py", 3, "TQ001"),
        gate.Violation("tests/a.py", 99, "TQ001"),
        gate.Violation("tests/b.py", 3, "TQ003"),
    )
    kept = gate.introduced(violations, {"tests/a.py": frozenset({3})})
    assert kept == (gate.Violation("tests/a.py", 3, "TQ001"),)


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _committed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_seed.py").write_text("def test_seed():\n    assert True\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "gate@example.com")
    _git(repo, "config", "user.name", "gate")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _wait_until(predicate: Callable[[], bool], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _reap(process: subprocess.Popen[bytes]) -> None:
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)
    if process.poll() is None:
        process.kill()
        process.wait(timeout=10)


def _registered_worktrees(repo: Path) -> int:
    listing = _git(repo, "worktree", "list", "--porcelain")
    return sum(line.startswith("worktree ") for line in listing.splitlines())


class _StalledScan(NamedTuple):
    process: subprocess.Popen[bytes]
    repo: Path
    release: Path
    temp_dir: Path


def _base_scan_stalled_in_its_checker(tmp_path: Path, driver: str) -> _StalledScan:
    repo = _committed_repo(tmp_path)
    scanning = tmp_path / "scanning"
    release = tmp_path / "release"
    slow_checker = tmp_path / "slow_checker.py"
    slow_checker.write_text(
        "import pathlib, time\n"
        f"pathlib.Path({str(scanning)!r}).touch()\n"
        f"while not pathlib.Path({str(release)!r}).exists():\n"
        "    time.sleep(0.05)\n"
    )
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    scan = subprocess.Popen(
        [sys.executable, "-c", driver, str(repo), str(slow_checker)],
        cwd=_MODULE_PATH.parent,
        env={**os.environ, "TMPDIR": str(temp_dir)},
    )
    if not _wait_until(scanning.exists, 30):
        _reap(scan)
        raise AssertionError("the base scan never reached the checker")
    return _StalledScan(scan, repo, release, temp_dir)


def test_a_terminated_base_scan_still_removes_its_worktree(tmp_path: Path) -> None:
    stalled = _base_scan_stalled_in_its_checker(tmp_path, _SCAN_BASE)
    try:
        stalled.process.send_signal(signal.SIGTERM)
        assert stalled.process.wait(timeout=30) == 128 + signal.SIGTERM
    finally:
        _reap(stalled.process)
    assert _registered_worktrees(stalled.repo) == 1
    assert list(stalled.temp_dir.iterdir()) == []


def test_a_base_scan_keeps_ignoring_the_hangup_its_parent_ignored(tmp_path: Path) -> None:
    stalled = _base_scan_stalled_in_its_checker(tmp_path, _SCAN_BASE_WITH_SIGHUP_IGNORED)
    try:
        stalled.process.send_signal(signal.SIGHUP)
        time.sleep(1)
        assert stalled.process.poll() is None, "a hangup the parent ignored killed the scan"
        stalled.release.touch()
        assert stalled.process.wait(timeout=30) == 0
    finally:
        _reap(stalled.process)
    assert _registered_worktrees(stalled.repo) == 1
    assert list(stalled.temp_dir.iterdir()) == []

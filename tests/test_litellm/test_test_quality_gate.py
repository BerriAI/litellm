"""Tests for scripts/test_quality_gate.py.

The gate's whole value is that it blames a change only for what it adds and that a
limit can never rise. Both live in pure functions, so they are tested directly:
`evaluate` for the blame rule, `ratcheted_budget` for the one-way ratchet, and
`parse_changed_lines` for the diff scan that turns a breach into file:line.
"""

import importlib.util
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "test_quality_gate.py"
_spec = importlib.util.spec_from_file_location("test_quality_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
# @dataclass(slots=True) rebuilds its class through sys.modules[__module__], so the
# module has to be registered before exec_module runs or Scope fails to construct.
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)

_BUDGET = {"TQ001": {"limit": 10}, "TQ003": {"limit": 5}}

_SCAN_BASE = (
    "import importlib.util, pathlib, sys\n"
    "spec = importlib.util.spec_from_file_location('test_quality_gate', sys.argv[1])\n"
    "gate = importlib.util.module_from_spec(spec)\n"
    "sys.modules[spec.name] = gate\n"
    "spec.loader.exec_module(gate)\n"
    "gate.base_counts('HEAD', repo_root=pathlib.Path(sys.argv[2]), checker=pathlib.Path(sys.argv[3]))\n"
)
_SCAN_BASE_WITH_SIGHUP_IGNORED = "import signal\nsignal.signal(signal.SIGHUP, signal.SIG_IGN)\n" + _SCAN_BASE


def test_a_rule_within_its_limit_is_not_a_breach():
    assert gate.evaluate({"TQ001": 10}, {"TQ001": 10}, _BUDGET) == ()


def test_a_rule_over_its_limit_that_the_change_added_is_a_breach():
    breaches = gate.evaluate({"TQ001": 12}, {"TQ001": 10}, _BUDGET)
    assert [(b.rule, b.total, b.cap, b.added) for b in breaches] == [("TQ001", 12, 10, 2)]


def test_drift_already_in_the_base_is_not_blamed_on_the_change():
    assert gate.evaluate({"TQ001": 14}, {"TQ001": 14}, _BUDGET) == ()


def test_a_change_that_reduces_an_over_limit_rule_is_not_blamed():
    assert gate.evaluate({"TQ001": 13}, {"TQ001": 14}, _BUDGET) == ()


def test_a_rule_absent_from_head_counts_as_zero():
    assert gate.evaluate({}, {}, _BUDGET) == ()


def test_over_ceiling_names_only_the_rules_above_their_limit():
    assert gate.over_ceiling({"TQ001": 11, "TQ003": 5}, _BUDGET) == frozenset({"TQ001"})


def test_over_ceiling_is_empty_when_everything_fits():
    assert gate.over_ceiling({"TQ001": 10, "TQ003": 4}, _BUDGET) == frozenset()


def test_ratchet_lowers_a_limit_by_what_the_branch_fixed():
    updated = gate.ratcheted_budget(_BUDGET, {"TQ001": 6}, {"TQ001": 10})
    assert updated["TQ001"]["limit"] == 6


def test_ratchet_never_raises_a_limit_when_violations_grew():
    updated = gate.ratcheted_budget(_BUDGET, {"TQ001": 20}, {"TQ001": 10})
    assert updated["TQ001"]["limit"] == 10


def test_ratchet_never_goes_below_zero():
    updated = gate.ratcheted_budget({"TQ001": {"limit": 2}}, {"TQ001": 0}, {"TQ001": 100})
    assert updated["TQ001"]["limit"] == 0


def test_ratchet_lowers_a_rule_introduced_on_this_branch_like_any_other():
    updated = gate.ratcheted_budget(_BUDGET, {"TQ001": 4}, {"TQ001": 10})
    assert updated["TQ001"]["limit"] == 4


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


def test_the_shipped_budget_covers_every_rule_the_checker_can_emit():
    import json

    budget = json.loads((_REPO_ROOT / "test-quality-budget.json").read_text())
    assert set(budget) == {"TQ001", "TQ002", "TQ003", "TQ004", "TQ005", "TQ006", "TQ007", "TQ008"}
    assert all(spec["limit"] >= 0 for spec in budget.values())


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
        [sys.executable, "-c", driver, str(_MODULE_PATH), str(repo), str(slow_checker)],
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

"""Tests for scripts/type_discipline_gate.py.

The gate's correctness lives in two pure functions: `over_ceiling` (which decides
whether the expensive base worktree scan is even needed) and `evaluate` (the
drift-safe breach check). Both are pinned here.
"""

import importlib.util
import subprocess
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "type_discipline_gate.py"
_spec = importlib.util.spec_from_file_location("type_discipline_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _budget(limit):
    return {"LIT006": {"limit": limit}}


def test_over_ceiling_flags_only_counts_above_the_limit():
    budget = _budget(12)
    assert gate.over_ceiling({"LIT006": 12}, budget) == frozenset()  # at limit
    assert gate.over_ceiling({"LIT006": 13}, budget) == frozenset({"LIT006"})  # over limit
    assert gate.over_ceiling({}, budget) == frozenset()  # missing rule counts as zero


def test_over_ceiling_is_independent_across_rules():
    budget = {"LIT001": {"limit": 5}, "LIT006": {"limit": 10}}
    assert gate.over_ceiling({"LIT001": 6, "LIT006": 10}, budget) == frozenset({"LIT001"})


def test_evaluate_blames_only_a_rule_over_limit_and_over_base():
    budget = _budget(10)
    # over limit and grown vs base -> breach
    assert [b.rule for b in gate.evaluate({"LIT006": 12}, {"LIT006": 9}, budget)] == ["LIT006"]
    # over limit but flat vs base (pre-existing drift) -> not blamed
    assert gate.evaluate({"LIT006": 12}, {"LIT006": 12}, budget) == []
    # within limit -> not blamed regardless of base
    assert gate.evaluate({"LIT006": 10}, {"LIT006": 0}, budget) == []


def test_update_ratchets_limit_down_by_what_the_branch_fixed_never_up():
    budget = {"LIT001": {"limit": 100}, "LIT006": {"limit": 10}}
    # LIT001 fixed 15 (60 -> 45) so its limit falls 100 -> 85; LIT006 grew, so its
    # limit holds flat at 10.
    current = {"LIT001": 45, "LIT006": 12}
    base = {"LIT001": 60, "LIT006": 9}
    assert gate.ratcheted_budget(budget, current, base) == {
        "LIT001": {"limit": 85},
        "LIT006": {"limit": 10},
    }


def test_update_leaves_rules_seeded_on_this_branch_untouched():
    # A rule absent from the base budget was seeded with grandfathered headroom on
    # this branch; the base tree predates the rule (e.g. no Final annotations yet),
    # so ratcheting against it would collapse the deliberate headroom.
    budget = {"LIT001": {"limit": 100}, "LIT010": {"limit": 24600}}
    current = {"LIT001": 45, "LIT010": 16400}
    base = {"LIT001": 60, "LIT010": 40000}
    assert gate.ratcheted_budget(budget, current, base, frozenset({"LIT010"})) == {
        "LIT001": {"limit": 85},
        "LIT010": {"limit": 24600},
    }


def _git(cwd, *args):
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _commit(cwd, name):
    (cwd / name).write_text(name)
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", "-m", name)
    return _git(cwd, "rev-parse", "HEAD")


def _branched_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "gate@example.com")
    _git(repo, "config", "user.name", "gate")
    _git(repo, "config", "commit.gpgsign", "false")
    branch_point = _commit(repo, "shared.txt")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feature.txt")
    _git(repo, "checkout", "-q", "main")
    base_tip = _commit(repo, "drift.txt")
    _git(repo, "checkout", "-q", "feature")
    return repo, branch_point, base_tip


def test_base_point_is_the_branch_point_when_no_merge_is_in_progress(tmp_path):
    repo, branch_point, _ = _branched_repo(tmp_path)
    assert gate.resolve_base_point("main", cwd=repo) == branch_point


def test_base_point_mid_merge_advances_to_the_merged_in_base_tip(tmp_path):
    repo, _, base_tip = _branched_repo(tmp_path)
    _git(repo, "merge", "--no-commit", "--no-ff", "main")
    assert gate.resolve_base_point("main", cwd=repo) == base_tip

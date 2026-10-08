"""Tests for scripts/type_discipline_gate.py.

Pins the fast-path ceiling check, drift-safe aggregate comparison, and budget
seeding and downward ratcheting, including LIT015 before its budget is seeded.
"""

import importlib.util
import subprocess
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest

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


def test_effective_budget_adds_only_lit015_without_changing_existing_specs() -> None:
    spec: Final = MappingProxyType({"limit": 12, "original_limit": 20})
    budget: Final = MappingProxyType({"LIT006": spec})
    effective: Final = gate.effective_budget(budget)
    assert effective == {"LIT006": spec, "LIT015": {"limit": 0}}
    assert effective["LIT006"] is spec
    assert budget == {"LIT006": spec}


@pytest.mark.parametrize("limit", (0, 7))
def test_effective_budget_preserves_an_explicit_lit015_spec(limit: int) -> None:
    spec: Final = MappingProxyType({"limit": limit, "original_limit": 9})
    budget: Final = MappingProxyType({"LIT015": spec, "LIT006": {"limit": 12}})
    effective: Final = gate.effective_budget(budget)
    assert effective == budget
    assert effective["LIT015"] is spec


@pytest.mark.parametrize(
    ("head_count", "base_count", "expected"),
    (
        (0, 0, ()),
        (1, 0, (("LIT015", 1, 0, 1),)),
        (4, 4, ()),
        (3, 4, ()),
        (6, 4, (("LIT015", 6, 0, 2),)),
    ),
)
def test_unbudgeted_lit015_grandfathers_only_nonincreasing_totals(
    head_count: int, base_count: int, expected: tuple[tuple[str, int, int, int], ...]
) -> None:
    assert gate.evaluate({"LIT015": head_count}, {"LIT015": base_count}, _budget(12)) == list(expected)


def test_unbudgeted_lit015_requires_base_scan_only_when_nonzero() -> None:
    assert gate.over_ceiling({"LIT015": 4}, _budget(12)) == frozenset({"LIT015"})
    assert gate.over_ceiling({"LIT015": 0}, {}) == frozenset()
    assert gate.over_ceiling({}, {}) == frozenset()
    assert gate.evaluate({"LIT015": 1}, {}, {}) == [gate.Breach("LIT015", 1, 0, 1)]
    assert gate.evaluate({}, {}, {}) == []


def test_lit015_addition_fails_against_the_lower_updated_base() -> None:
    budget: Final = _budget(12)
    assert gate.evaluate({"LIT015": 2}, {"LIT015": 3}, budget) == []
    assert gate.evaluate({"LIT015": 3}, {"LIT015": 2}, budget) == [gate.Breach("LIT015", 3, 0, 1)]


def test_lit015_grandfathering_allows_removals_to_offset_additions() -> None:
    base: Final = [gate.Violation("litellm/removed.py", 2, "LIT015")]
    head: Final = [gate.Violation("litellm/added.py", 2, "LIT015")]
    assert gate.evaluate(gate.count_by_rule(head), gate.count_by_rule(base), {}) == []


def test_lit015_fallback_does_not_gate_other_unregistered_rules() -> None:
    head: Final = {"LIT015": 1, "LIT999": 10}
    assert gate.over_ceiling(head, {}) == frozenset({"LIT015"})
    assert gate.evaluate(head, {}, {}) == [gate.Breach("LIT015", 1, 0, 1)]


def test_lit015_explicit_limit_is_used_by_both_check_paths() -> None:
    budget: Final = {"LIT015": {"limit": 5}}
    assert gate.over_ceiling({"LIT015": 5}, budget) == frozenset()
    assert gate.evaluate({"LIT015": 5}, {}, budget) == []
    assert gate.over_ceiling({"LIT015": 6}, budget) == frozenset({"LIT015"})
    assert gate.evaluate({"LIT015": 6}, {}, budget) == [gate.Breach("LIT015", 6, 5, 6)]


@pytest.mark.parametrize("current_count", (0, 4))
def test_update_seeds_missing_lit015_at_actual_count_and_preserves_other_ratchets(current_count: int) -> None:
    budget: Final = {"LIT006": {"limit": 12}, "LIT010": {"limit": 20}}
    current: Final = {"LIT006": 7, "LIT010": 10, "LIT015": current_count, "LIT999": 5}
    base: Final = {"LIT006": 9, "LIT010": 30, "LIT015": 10}
    starting_budget: Final = gate.seed_frozen_model_budget(budget, current)
    seeded: Final = frozenset(starting_budget) - frozenset({"LIT006"})
    assert starting_budget == {**budget, "LIT015": {"limit": current_count}}
    assert budget == {"LIT006": {"limit": 12}, "LIT010": {"limit": 20}}
    assert gate.ratcheted_budget(starting_budget, current, base, seeded) == {
        "LIT006": {"limit": 10},
        "LIT010": {"limit": 20},
        "LIT015": {"limit": current_count},
    }


def test_update_seeds_lit015_at_zero_when_checker_has_no_violations() -> None:
    assert gate.seed_frozen_model_budget(_budget(12), {}) == {
        "LIT006": {"limit": 12},
        "LIT015": {"limit": 0},
    }


@pytest.mark.parametrize("limit", (0, 7))
def test_update_does_not_reseed_existing_lit015_limits(limit: int) -> None:
    spec: Final = MappingProxyType({"limit": limit, "original_limit": 9})
    budget: Final = MappingProxyType({"LIT015": spec})
    starting_budget: Final = gate.seed_frozen_model_budget(budget, {"LIT015": 4})
    assert starting_budget == budget
    assert starting_budget["LIT015"] is spec


def test_seeded_lit015_limit_later_ratchets_down_but_never_back_up() -> None:
    seeded: Final = gate.seed_frozen_model_budget({}, {"LIT015": 4})
    lower: Final = gate.ratcheted_budget(seeded, {"LIT015": 3}, {"LIT015": 4})
    assert lower == {"LIT015": {"limit": 3}}
    unchanged: Final = gate.ratcheted_budget(lower, {"LIT015": 3}, {"LIT015": 3})
    assert unchanged == lower
    growing: Final = gate.ratcheted_budget(lower, {"LIT015": 4}, {"LIT015": 3})
    assert growing == lower
    assert gate.evaluate({"LIT015": 4}, {"LIT015": 3}, growing) == [gate.Breach("LIT015", 4, 3, 1)]
    assert gate.ratcheted_budget(lower, {}, {"LIT015": 3}) == {"LIT015": {"limit": 0}}


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

"""Tests for scripts/test_quality_gate.py.

The gate's whole value is that it blames a change only for what it adds, that a limit
can never rise, and that a limit cannot stay above a count the branch pushed below it.
All three live in pure functions, so they are tested directly: `evaluate` for the blame
rule, `ratcheted_budget` for the one-way ratchet, `unratcheted` for the ceiling a branch
left behind, and `parse_changed_lines` for the diff scan that turns a breach into
file:line.
"""

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "test_quality_gate.py"
_spec = importlib.util.spec_from_file_location("test_quality_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
# @dataclass(slots=True) rebuilds its class through sys.modules[__module__], so the
# module has to be registered before exec_module runs or Scope fails to construct.
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)

_BUDGET = {"TQ001": {"limit": 10}, "TQ003": {"limit": 5}}


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


def test_a_branch_that_cleared_violations_must_lower_the_ceiling():
    stale = gate.unratcheted({"TQ001": 6}, {"TQ001": 10}, _BUDGET)
    assert [(b.rule, b.total, b.cap, b.added) for b in stale] == [("TQ001", 6, 10, -4)]


def test_headroom_already_in_the_base_is_not_blamed_on_this_branch():
    assert gate.unratcheted({"TQ001": 6}, {"TQ001": 6}, _BUDGET) == ()


def test_a_branch_that_cleared_down_to_the_ceiling_exactly_is_clean():
    assert gate.unratcheted({"TQ001": 10}, {"TQ001": 12}, _BUDGET) == ()


def test_a_branch_that_added_violations_is_not_a_ratchet_finding():
    assert gate.unratcheted({"TQ001": 14}, {"TQ001": 10}, _BUDGET) == ()


def test_the_ratchet_finding_survives_the_update_that_answers_it():
    cleared = {"TQ001": 6}
    updated = gate.ratcheted_budget(_BUDGET, cleared, {"TQ001": 10})
    assert gate.unratcheted(cleared, {"TQ001": 10}, updated) == ()


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

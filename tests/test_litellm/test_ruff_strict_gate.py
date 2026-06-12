import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ruff_strict_gate.py"
_spec = importlib.util.spec_from_file_location("ruff_strict_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

Violation = gate.Violation


def rule(name, baseline, slack):
    return {name: {"baseline": baseline, "slack": slack}}


def test_under_ceiling_passes():
    assert gate.evaluate({"ANN001": 100}, {"ANN001": 100}, rule("ANN001", 90, 20)) == []


def test_ceiling_is_baseline_plus_slack_boundary():
    budget = rule("ANN001", 90, 20)  # cap 110
    at = gate.evaluate({"ANN001": 110}, {"ANN001": 90}, budget)
    over = gate.evaluate({"ANN001": 111}, {"ANN001": 90}, budget)
    assert at == []
    assert [b.rule for b in over] == ["ANN001"]
    assert over[0].cap == 110
    assert over[0].added == 21


def test_over_ceiling_and_change_added_fails():
    breaches = gate.evaluate({"C901": 11}, {"C901": 9}, rule("C901", 10, 0))
    assert [b.rule for b in breaches] == ["C901"]
    assert breaches[0].added == 2


def test_base_already_over_ceiling_change_added_nothing_is_not_blamed():
    # drift safety: base is over cap, this change leaves the count where it is
    assert gate.evaluate({"C901": 15}, {"C901": 15}, rule("C901", 10, 0)) == []


def test_change_that_reduces_an_over_ceiling_rule_is_not_blamed():
    # still over cap, but moving the right direction
    assert gate.evaluate({"C901": 14}, {"C901": 16}, rule("C901", 10, 0)) == []


def test_rules_are_independent():
    budget = {**rule("ANN001", 100, 50), **rule("C901", 10, 0)}
    breaches = gate.evaluate(
        {"ANN001": 130, "C901": 11}, {"ANN001": 100, "C901": 10}, budget
    )
    assert [b.rule for b in breaches] == ["C901"]  # ANN001 130 <= 150, C901 11 > 10


def test_missing_rule_counts_as_zero():
    assert gate.evaluate({}, {}, rule("C901", 0, 0)) == []


def test_parse_changed_lines_maps_added_lines_per_file():
    diff = (
        "+++ b/litellm/a.py\n"
        "@@ -10 +10,3 @@\n+x\n+y\n+z\n"
        "+++ b/litellm/b.py\n"
        "@@ -5,2 +7 @@\n+q\n"
    )
    changed = gate.parse_changed_lines(diff)
    assert changed["litellm/a.py"] == {10, 11, 12}
    assert changed["litellm/b.py"] == {7}


def test_introduced_keeps_only_violations_on_changed_lines():
    violations = [
        Violation("litellm/a.py", 10, "ANN001"),
        Violation("litellm/a.py", 99, "C901"),
    ]
    assert gate.introduced(violations, {"litellm/a.py": {10}}) == [
        Violation("litellm/a.py", 10, "ANN001")
    ]


@pytest.mark.parametrize("hunk", ["@@ -1 +1 @@", "@@ -1,0 +1,2 @@"])
def test_parse_changed_lines_handles_single_and_ranged_hunks(hunk):
    assert gate.parse_changed_lines(f"+++ b/litellm/a.py\n{hunk}\n")["litellm/a.py"]

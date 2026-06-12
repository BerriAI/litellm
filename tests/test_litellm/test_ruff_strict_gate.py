import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ruff_strict_gate.py"
_spec = importlib.util.spec_from_file_location("ruff_strict_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

Violation = gate.Violation


def v(file, line, code):
    return Violation(file, line, code)


def test_parse_changed_lines_maps_added_lines_per_file():
    diff = (
        "diff --git a/litellm/a.py b/litellm/a.py\n"
        "--- a/litellm/a.py\n"
        "+++ b/litellm/a.py\n"
        "@@ -10 +10,3 @@\n"
        "+x\n+y\n+z\n"
        "diff --git a/litellm/b.py b/litellm/b.py\n"
        "--- a/litellm/b.py\n"
        "+++ b/litellm/b.py\n"
        "@@ -5,2 +7 @@\n"
        "+q\n"
    )
    changed = gate.parse_changed_lines(diff)
    assert changed["litellm/a.py"] == {10, 11, 12}
    assert changed["litellm/b.py"] == {7}


def test_parse_changed_lines_pure_deletion_adds_nothing():
    diff = "+++ b/litellm/a.py\n@@ -4,3 +3,0 @@\n"
    assert gate.parse_changed_lines(diff).get("litellm/a.py", set()) == set()


def test_introduced_keeps_only_violations_on_changed_lines():
    violations = [
        v("litellm/a.py", 10, "ANN001"),
        v("litellm/a.py", 99, "C901"),
        v("litellm/b.py", 7, "PLR0913"),
    ]
    changed = {"litellm/a.py": {10, 11}, "litellm/b.py": {7}}
    assert gate.introduced(violations, changed) == [
        v("litellm/a.py", 10, "ANN001"),
        v("litellm/b.py", 7, "PLR0913"),
    ]


def test_base_drift_on_unchanged_lines_is_not_introduced():
    base_drift = [v("litellm/newrelic.py", 50, "ANN001")]
    changed = {"litellm/other.py": {1, 2, 3}}
    assert gate.introduced(base_drift, changed) == []


def test_evaluate_passes_at_allowance_and_fails_one_over():
    at = gate.evaluate([v("f.py", 1, "ANN401"), v("f.py", 2, "ANN401")], {"ANN401": 2})
    over = gate.evaluate(
        [v("f.py", 1, "ANN401"), v("f.py", 2, "ANN401"), v("f.py", 3, "ANN401")],
        {"ANN401": 2},
    )
    assert at.ok is True
    assert over.ok is False
    assert over.breaches == {"ANN401": 3}


def test_evaluate_missing_rule_defaults_to_zero_allowance():
    result = gate.evaluate([v("f.py", 1, "C901")], allowances={})
    assert result.ok is False
    assert result.breaches == {"C901": 1}


def test_evaluate_is_per_rule_one_breach_does_not_mask_a_passing_rule():
    result = gate.evaluate(
        [v("f.py", 1, "ANN001"), v("f.py", 2, "C901")],
        {"ANN001": 5, "C901": 0},
    )
    assert result.ok is False
    assert result.breaches == {"C901": 1}
    assert result.by_rule == {"ANN001": 1, "C901": 1}


def test_no_introduced_violations_passes():
    result = gate.evaluate([], {"ANN001": 0})
    assert result.ok is True
    assert result.by_rule == {}


@pytest.mark.parametrize("hunk", ["@@ -1 +1 @@", "@@ -1,0 +1,2 @@"])
def test_parse_changed_lines_handles_single_and_ranged_hunks(hunk):
    changed = gate.parse_changed_lines(f"+++ b/litellm/a.py\n{hunk}\n")
    assert changed["litellm/a.py"]

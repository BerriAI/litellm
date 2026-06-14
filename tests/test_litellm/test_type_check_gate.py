import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "type_check_gate.py"
_spec = importlib.util.spec_from_file_location("type_check_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

ROOT = gate.REPO_ROOT


def test_mypy_counts_errors_per_file_ignoring_line_numbers_notes_and_summary():
    lines = [
        f"{ROOT}/litellm/utils.py:10: error: missing annotation  [no-untyped-def]",
        f"{ROOT}/litellm/utils.py:9999: error: missing annotation  [no-untyped-def]",
        f"{ROOT}/litellm/main.py:5: error: missing annotation  [no-untyped-def]",
        f"{ROOT}/litellm/main.py:5: note: see here",
        "Found 3 errors in 2 files (checked 100 source files)",
    ]
    assert gate.count_errors(lines, gate.PATTERNS["mypy"]) == {
        "litellm/utils.py": 2,
        "litellm/main.py": 1,
    }


def test_basedpyright_counts_only_errors_not_warnings_or_summary():
    lines = [
        f"  {ROOT}/litellm/utils.py:6389:40 - error: Type is unknown (reportUnknownVariableType)",
        f"  {ROOT}/litellm/utils.py:6395:5 - error: Type is unknown (reportUnknownVariableType)",
        f"  {ROOT}/litellm/main.py:1:1 - warning: something (reportFoo)",
        "117033 errors, 5 warnings, 0 notes",
    ]
    assert gate.count_errors(lines, gate.PATTERNS["basedpyright"]) == {
        "litellm/utils.py": 2
    }


def test_paths_outside_repo_are_skipped():
    lines = ["/tmp/elsewhere.py:1: error: missing annotation  [no-untyped-def]"]
    assert gate.count_errors(lines, gate.PATTERNS["mypy"]) == {}


def test_at_or_under_ceiling_passes():
    assert gate.evaluate({"a.py": 5, "b.py": 3}, {"a.py": 5, "b.py": 10}, 0) == []


def test_one_more_error_than_ceiling_fails():
    breaches = gate.evaluate({"a.py": 6}, {"a.py": 5}, 0)
    assert breaches == [gate.Breach("a.py", 6, 5)]


def test_file_absent_from_budget_has_zero_ceiling():
    breaches = gate.evaluate({"new.py": 1}, {}, 0)
    assert breaches == [gate.Breach("new.py", 1, 0)]


def test_slack_absorbs_small_increase_then_fails_past_it():
    assert gate.evaluate({"a.py": 10}, {"a.py": 5}, 5) == []
    assert gate.evaluate({"a.py": 11}, {"a.py": 5}, 5) == [gate.Breach("a.py", 11, 10)]


def test_slack_applies_to_new_files_too():
    assert gate.evaluate({"new.py": 5}, {}, 5) == []
    assert gate.evaluate({"new.py": 6}, {}, 5) == [gate.Breach("new.py", 6, 5)]

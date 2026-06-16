import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "type_check_gate.py"
_spec = importlib.util.spec_from_file_location("type_check_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

ROOT = gate.REPO_ROOT


def test_mypy_counts_per_code_ignoring_lines_notes_and_summary():
    text = "\n".join(
        [
            f"{ROOT}/litellm/utils.py:10: error: missing annotation  [no-untyped-def]",
            f"{ROOT}/litellm/utils.py:9999: error: missing annotation  [no-untyped-def]",
            f"{ROOT}/litellm/main.py:5: error: Returning Any  [no-any-return]",
            f"{ROOT}/litellm/main.py:5: note: see here",
            "Found 3 errors in 2 files (checked 100 source files)",
        ]
    )
    assert gate.count_errors(text, "mypy") == {
        "no-untyped-def": 2,
        "no-any-return": 1,
    }


def _bpr(file, severity, rule):
    diag = {"file": str(file), "severity": severity, "message": "msg"}
    if rule is not None:
        diag["rule"] = rule
    return diag


def test_basedpyright_counts_per_rule_from_json_not_warnings():
    # basedpyright wraps long messages across lines, so the (reportRule) lands on
    # a continuation line away from the `- error:` marker; --outputjson avoids it.
    payload = json.dumps(
        {
            "generalDiagnostics": [
                _bpr(f"{ROOT}/litellm/utils.py", "error", "reportUnknownVariableType"),
                _bpr(f"{ROOT}/litellm/utils.py", "error", "reportUnknownVariableType"),
                _bpr(f"{ROOT}/litellm/main.py", "error", "reportArgumentType"),
                _bpr(f"{ROOT}/litellm/main.py", "warning", "reportUnusedImport"),
            ]
        }
    )
    assert gate.count_errors(payload, "basedpyright") == {
        "reportUnknownVariableType": 2,
        "reportArgumentType": 1,
    }


def test_basedpyright_error_without_a_rule_is_bucketed():
    payload = json.dumps(
        {"generalDiagnostics": [_bpr(f"{ROOT}/litellm/x.py", "error", None)]}
    )
    assert gate.count_errors(payload, "basedpyright") == {gate.UNCODED: 1}


def test_mypy_error_without_a_code_is_bucketed_so_it_is_still_gated():
    text = f"{ROOT}/litellm/x.py:1: error: something broke with no code"
    assert gate.count_errors(text, "mypy") == {gate.UNCODED: 1}


def test_paths_outside_repo_are_skipped():
    text = "/tmp/elsewhere.py:1: error: missing annotation  [no-untyped-def]"
    assert gate.count_errors(text, "mypy") == {}
    payload = json.dumps(
        {"generalDiagnostics": [_bpr("/tmp/elsewhere.py", "error", "reportArgumentType")]}
    )
    assert gate.count_errors(payload, "basedpyright") == {}


def test_at_or_under_ceiling_passes():
    budget = {"no-any-return": {"baseline": 5, "slack": 0}}
    assert gate.evaluate({"no-any-return": 5}, budget) == []


def test_one_more_error_than_ceiling_fails():
    budget = {"no-any-return": {"baseline": 5, "slack": 0}}
    assert gate.evaluate({"no-any-return": 6}, budget) == [
        gate.Breach("no-any-return", 6, 5)
    ]


def test_slack_absorbs_small_increase_then_fails_past_it():
    budget = {"arg-type": {"baseline": 5, "slack": 5}}
    assert gate.evaluate({"arg-type": 10}, budget) == []
    assert gate.evaluate({"arg-type": 11}, budget) == [gate.Breach("arg-type", 11, 10)]


def test_unbudgeted_new_code_uses_default_slack():
    assert gate.evaluate({"brand-new": gate.DEFAULT_SLACK}, {}) == []
    assert gate.evaluate({"brand-new": gate.DEFAULT_SLACK + 1}, {}) == [
        gate.Breach("brand-new", gate.DEFAULT_SLACK + 1, gate.DEFAULT_SLACK)
    ]

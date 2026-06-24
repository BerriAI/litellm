import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "type_check_gate.py"
_spec = importlib.util.spec_from_file_location("type_check_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

ROOT = gate.REPO_ROOT


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
    assert gate.count_basedpyright(payload) == {
        "reportUnknownVariableType": 2,
        "reportArgumentType": 1,
    }


def test_basedpyright_error_without_a_rule_is_bucketed():
    payload = json.dumps(
        {"generalDiagnostics": [_bpr(f"{ROOT}/litellm/x.py", "error", None)]}
    )
    assert gate.count_basedpyright(payload) == {gate.UNCODED: 1}


def test_paths_outside_repo_are_skipped():
    payload = json.dumps(
        {
            "generalDiagnostics": [
                _bpr("/tmp/elsewhere.py", "error", "reportArgumentType")
            ]
        }
    )
    assert gate.count_basedpyright(payload) == {}


def test_at_or_under_ceiling_passes():
    budget = {"no-any-return": {"baseline": 5, "slack": 0}}
    assert gate.evaluate({"no-any-return": 5}, {}, budget) == []


def test_one_more_error_than_ceiling_fails():
    budget = {"no-any-return": {"baseline": 5, "slack": 0}}
    assert gate.evaluate({"no-any-return": 6}, {}, budget) == [
        gate.Breach("no-any-return", 6, 5, 6)
    ]


def test_slack_absorbs_small_increase_then_fails_past_it():
    budget = {"arg-type": {"baseline": 5, "slack": 5}}
    assert gate.evaluate({"arg-type": 10}, {}, budget) == []
    assert gate.evaluate({"arg-type": 11}, {}, budget) == [
        gate.Breach("arg-type", 11, 10, 11)
    ]


def test_unbudgeted_new_code_uses_default_slack():
    assert gate.evaluate({"brand-new": gate.DEFAULT_SLACK}, {}, {}) == []
    assert gate.evaluate({"brand-new": gate.DEFAULT_SLACK + 1}, {}, {}) == [
        gate.Breach(
            "brand-new",
            gate.DEFAULT_SLACK + 1,
            gate.DEFAULT_SLACK,
            gate.DEFAULT_SLACK + 1,
        )
    ]


def test_drift_already_over_cap_in_base_is_not_blamed_on_a_flat_change():
    # The bystander case: a rule sits over its ceiling because two earlier PRs
    # summed past it. A PR that branches off that base and adds nothing must pass
    # -- total > cap but total == base, so the `> base` guard spares it.
    budget = {"arg-type": {"baseline": 5, "slack": 5}}
    assert gate.evaluate({"arg-type": 12}, {"arg-type": 12}, budget) == []


def test_change_that_grows_an_over_cap_rule_is_blamed_for_only_what_it_added():
    # Over cap AND above base: blamed, and `added` is the delta vs base, not the
    # whole overage, so the message points at this change's contribution.
    budget = {"arg-type": {"baseline": 5, "slack": 5}}
    assert gate.evaluate({"arg-type": 14}, {"arg-type": 12}, budget) == [
        gate.Breach("arg-type", 14, 10, 2)
    ]


def test_reducing_an_over_cap_rule_below_base_passes():
    budget = {"arg-type": {"baseline": 5, "slack": 5}}
    assert gate.evaluate({"arg-type": 11}, {"arg-type": 12}, budget) == []


def test_no_output_against_a_nonempty_budget_is_a_vacuous_run():
    # A crashed type checker emits nothing; the gate must not certify it as clean.
    budget = {"no-untyped-def": {"baseline": 4888, "slack": 10}}
    assert gate.is_vacuous_run({}, budget) is True


def test_genuine_zero_and_empty_budget_are_not_vacuous():
    assert gate.is_vacuous_run({}, {}) is False
    assert (
        gate.is_vacuous_run({}, {"no-untyped-def": {"baseline": 0, "slack": 3}})
        is False
    )
    assert (
        gate.is_vacuous_run({"arg-type": 1}, {"arg-type": {"baseline": 9, "slack": 1}})
        is False
    )


def test_malformed_basedpyright_json_exits_loudly_not_as_zero_errors():
    import pytest

    with pytest.raises(SystemExit):
        gate.count_basedpyright("startup warning\n{not json")


def test_empty_basedpyright_payload_counts_zero():
    # Empty (not malformed) output parses to zero; the vacuous-run guard, not the
    # parser, is what rejects an empty run.
    assert gate.count_basedpyright("") == {}

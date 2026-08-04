import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ruff_strict_gate.py"
_spec = importlib.util.spec_from_file_location("ruff_strict_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

Violation = gate.Violation


def rule(name, limit):
    return {name: {"limit": limit}}


def test_under_ceiling_passes():
    assert gate.evaluate({"ANN001": 100}, {"ANN001": 100}, rule("ANN001", 110)) == []


def test_ceiling_is_the_limit_boundary():
    budget = rule("ANN001", 110)
    at = gate.evaluate({"ANN001": 110}, {"ANN001": 90}, budget)
    over = gate.evaluate({"ANN001": 111}, {"ANN001": 90}, budget)
    assert at == []
    assert [b.rule for b in over] == ["ANN001"]
    assert over[0].cap == 110
    assert over[0].added == 21


def test_over_ceiling_and_change_added_fails():
    breaches = gate.evaluate({"C901": 11}, {"C901": 9}, rule("C901", 10))
    assert [b.rule for b in breaches] == ["C901"]
    assert breaches[0].added == 2


def test_base_already_over_ceiling_change_added_nothing_is_not_blamed():
    # drift safety: base is over limit, this change leaves the count where it is
    assert gate.evaluate({"C901": 15}, {"C901": 15}, rule("C901", 10)) == []


def test_change_that_reduces_an_over_ceiling_rule_is_not_blamed():
    # still over limit, but moving the right direction
    assert gate.evaluate({"C901": 14}, {"C901": 16}, rule("C901", 10)) == []


def test_rules_are_independent():
    budget = {**rule("ANN001", 150), **rule("C901", 10)}
    breaches = gate.evaluate(
        {"ANN001": 130, "C901": 11}, {"ANN001": 100, "C901": 10}, budget
    )
    assert [b.rule for b in breaches] == ["C901"]  # ANN001 130 <= 150, C901 11 > 10


def test_missing_rule_counts_as_zero():
    assert gate.evaluate({}, {}, rule("C901", 0)) == []


def test_update_ratchets_limit_down_by_what_the_branch_fixed_never_up():
    budget = {**rule("ANN001", 150), **rule("C901", 10)}
    # ANN001 fixed 20 (100 -> 80) so its limit falls 150 -> 130; C901 grew, so its
    # limit holds flat at 10 (a fix must never loosen a ceiling).
    current = {"ANN001": 80, "C901": 12}
    base = {"ANN001": 100, "C901": 9}
    assert gate.ratcheted_budget(budget, current, base) == {
        "ANN001": {"limit": 130},
        "C901": {"limit": 10},
    }


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


def test_over_ceiling_flags_only_counts_above_the_limit():
    budget = rule("C901", 10)
    assert gate.over_ceiling({"C901": 10}, budget) == frozenset()
    assert gate.over_ceiling({"C901": 11}, budget) == frozenset({"C901"})
    assert gate.over_ceiling({}, budget) == frozenset()


def test_over_ceiling_ignores_rules_missing_from_the_budget():
    assert gate.over_ceiling({"NEW99": 100}, rule("C901", 10)) == frozenset()


def test_over_ceiling_is_independent_across_rules():
    budget = {**rule("ANN001", 150), **rule("C901", 10)}
    assert gate.over_ceiling({"ANN001": 130, "C901": 11}, budget) == frozenset({"C901"})


def test_no_violations_against_a_nonempty_budget_is_vacuous():
    assert gate.is_vacuous_run({}, rule("ANN001", 110)) is True


def test_genuine_zero_counts_are_not_vacuous():
    assert gate.is_vacuous_run({}, {}) is False
    assert gate.is_vacuous_run({}, rule("ANN001", 0)) is False
    assert gate.is_vacuous_run({"ANN001": 1}, rule("ANN001", 110)) is False


def _violations(code, count):
    return [Violation("litellm/a.py", line, code) for line in range(1, count + 1)]


def _raise_if_called(*args):
    raise AssertionError("must not be called")


def _budget_file(tmp_path, limit):
    path = tmp_path / "budget.json"
    path.write_text(f'{{"ANN001": {{"limit": {limit}}}}}')
    return path


def test_check_rejects_a_vacuous_head_scan(tmp_path, capsys):
    with pytest.raises(SystemExit):
        gate.cmd_check(
            "origin/main",
            violations=lambda: [],
            base_counts_for=_raise_if_called,
            merge_base=_raise_if_called,
            budget_path=_budget_file(tmp_path, 110),
        )
    assert "vacuous" in capsys.readouterr().out


def test_check_within_ceiling_skips_the_base_scan(tmp_path, capsys):
    gate.cmd_check(
        "origin/main",
        violations=lambda: _violations("ANN001", 3),
        base_counts_for=_raise_if_called,
        merge_base=_raise_if_called,
        budget_path=_budget_file(tmp_path, 110),
    )
    assert capsys.readouterr().out.startswith("OK")


def test_check_refuses_to_blame_the_change_for_a_vacuous_base_scan(tmp_path, capsys):
    with pytest.raises(SystemExit):
        gate.cmd_check(
            "origin/main",
            violations=lambda: _violations("ANN001", 3),
            base_counts_for=lambda ref: {},
            merge_base=lambda base: "a" * 40,
            budget_path=_budget_file(tmp_path, 2),
        )
    assert "refusing to blame" in capsys.readouterr().out


def test_check_spares_a_bystander_whose_base_matches_head(tmp_path, capsys):
    gate.cmd_check(
        "origin/main",
        violations=lambda: _violations("ANN001", 3),
        base_counts_for=lambda ref: {"ANN001": 3},
        merge_base=lambda base: "a" * 40,
        budget_path=_budget_file(tmp_path, 2),
    )
    assert capsys.readouterr().out.startswith("OK")

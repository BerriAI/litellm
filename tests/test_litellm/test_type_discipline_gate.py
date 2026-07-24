"""Tests for scripts/type_discipline_gate.py.

The gate's correctness lives in two pure functions: `over_ceiling` (which decides
whether the expensive base worktree scan is even needed) and `evaluate` (the
drift-safe breach check). Both are pinned here.
"""

import importlib.util
from pathlib import Path

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


def test_no_violations_against_a_nonempty_budget_is_vacuous():
    assert gate.is_vacuous_run({}, _budget(12)) is True


def test_genuine_zero_counts_are_not_vacuous():
    assert gate.is_vacuous_run({}, {}) is False
    assert gate.is_vacuous_run({}, _budget(0)) is False
    assert gate.is_vacuous_run({"LIT006": 1}, _budget(12)) is False


def _violations(code, count):
    return [gate.Violation("litellm/a.py", line, code) for line in range(1, count + 1)]


def _raise_if_called(*args):
    raise AssertionError("must not be called")


def _budget_file(tmp_path, limit):
    path = tmp_path / "budget.json"
    path.write_text(f'{{"LIT006": {{"limit": {limit}}}}}')
    return path


def test_check_rejects_a_vacuous_head_pass(tmp_path, capsys):
    with pytest.raises(SystemExit):
        gate.cmd_check(
            "origin/main",
            violations=lambda: [],
            base_counts_for=_raise_if_called,
            merge_base=_raise_if_called,
            budget_path=_budget_file(tmp_path, 12),
        )
    assert "vacuous" in capsys.readouterr().out


def test_check_within_ceiling_skips_the_base_pass(tmp_path, capsys):
    gate.cmd_check(
        "origin/main",
        violations=lambda: _violations("LIT006", 3),
        base_counts_for=_raise_if_called,
        merge_base=_raise_if_called,
        budget_path=_budget_file(tmp_path, 12),
    )
    assert capsys.readouterr().out.startswith("OK")


def test_check_refuses_to_blame_the_change_for_a_vacuous_base_pass(tmp_path, capsys):
    with pytest.raises(SystemExit):
        gate.cmd_check(
            "origin/main",
            violations=lambda: _violations("LIT006", 3),
            base_counts_for=lambda ref: {},
            merge_base=lambda base: "a" * 40,
            budget_path=_budget_file(tmp_path, 2),
        )
    assert "refusing to blame" in capsys.readouterr().out


def test_check_spares_a_bystander_whose_base_matches_head(tmp_path, capsys):
    gate.cmd_check(
        "origin/main",
        violations=lambda: _violations("LIT006", 3),
        base_counts_for=lambda ref: {"LIT006": 3},
        merge_base=lambda base: "a" * 40,
        budget_path=_budget_file(tmp_path, 2),
    )
    assert capsys.readouterr().out.startswith("OK")

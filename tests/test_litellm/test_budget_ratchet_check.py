"""Tests for scripts/budget_ratchet_check.py.

The guard's whole contract is "ceilings may only fall": a raised ceiling, a dropped
rule, or a deleted file is a regression, while a lowered/equal ceiling, a brand-new
rule, or a brand-new budget file is fine. Each branch is pinned here.
"""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "budget_ratchet_check.py"
_spec = importlib.util.spec_from_file_location("budget_ratchet_check", _MODULE_PATH)
ratchet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ratchet)


def _spec_of(baseline, slack):
    return {"baseline": baseline, "slack": slack}


def test_caps_sum_baseline_and_slack_and_skip_malformed():
    caps = ratchet._caps({"LIT006": _spec_of(1013, 10), "junk": 5})
    assert caps == {"LIT006": 1023}  # malformed (non-dict) spec ignored


def test_raised_ceiling_is_a_regression():
    base = {"LIT006": _spec_of(1013, 10)}
    head = {"LIT006": _spec_of(1013, 11)}  # cap 1023 -> 1024
    regs = ratchet.regressions_for("b.json", base, head)
    assert [r.rule for r in regs] == ["LIT006"]
    assert "1023 -> 1024" in regs[0].detail


def test_lowered_or_equal_ceiling_is_clean():
    base = {"LIT006": _spec_of(1013, 10)}
    assert ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1000, 10)}) == []
    assert ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1013, 10)}) == []
    # slack traded for baseline at the same ceiling is fine
    assert ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1023, 0)}) == []


def test_dropped_rule_is_a_regression():
    regs = ratchet.regressions_for("b.json", {"LIT007": _spec_of(0, 0)}, {})
    assert [r.rule for r in regs] == ["LIT007"]
    assert "dropped" in regs[0].detail


def test_new_rule_in_head_is_clean():
    assert ratchet.regressions_for("b.json", {}, {"LIT009": _spec_of(5, 0)}) == []


def test_deleted_budget_file_is_a_regression():
    regs = ratchet.regressions_for("b.json", {"LIT006": _spec_of(1, 0)}, None)
    assert [r.rule for r in regs] == ["*"]
    assert "deleted" in regs[0].detail


def test_new_budget_file_has_nothing_to_ratchet():
    assert ratchet.regressions_for("b.json", None, {"LIT006": _spec_of(1, 0)}) == []

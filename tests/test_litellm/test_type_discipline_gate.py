"""Tests for scripts/type_discipline_gate.py.

The gate's correctness lives in two pure functions: `over_ceiling` (which decides
whether the expensive base worktree scan is even needed) and `evaluate` (the
drift-safe breach check). Both are pinned here.
"""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "type_discipline_gate.py"
_spec = importlib.util.spec_from_file_location("type_discipline_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _budget(baseline, slack):
    return {"LIT006": {"baseline": baseline, "slack": slack}}


def test_over_ceiling_flags_only_counts_above_baseline_plus_slack():
    budget = _budget(10, 2)  # cap 12
    assert gate.over_ceiling({"LIT006": 12}, budget) == frozenset()  # at cap
    assert gate.over_ceiling({"LIT006": 13}, budget) == frozenset({"LIT006"})  # over cap
    assert gate.over_ceiling({}, budget) == frozenset()  # missing rule counts as zero


def test_over_ceiling_is_independent_across_rules():
    budget = {"LIT001": {"baseline": 5, "slack": 0}, "LIT006": {"baseline": 10, "slack": 0}}
    assert gate.over_ceiling({"LIT001": 6, "LIT006": 10}, budget) == frozenset({"LIT001"})


def test_evaluate_blames_only_a_rule_over_cap_and_over_base():
    budget = _budget(10, 0)  # cap 10
    # over cap and grown vs base -> breach
    assert [b.rule for b in gate.evaluate({"LIT006": 12}, {"LIT006": 9}, budget)] == ["LIT006"]
    # over cap but flat vs base (pre-existing drift) -> not blamed
    assert gate.evaluate({"LIT006": 12}, {"LIT006": 12}, budget) == []
    # within cap -> not blamed regardless of base
    assert gate.evaluate({"LIT006": 10}, {"LIT006": 0}, budget) == []

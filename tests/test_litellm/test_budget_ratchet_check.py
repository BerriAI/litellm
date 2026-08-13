"""Tests for scripts/budget_ratchet_check.py.

The guard's contract is "limits may only fall": a raised limit, a dropped rule, or
a deleted file is a regression, while a lowered/equal limit, a brand-new rule, or a
brand-new budget file is fine. Each branch is pinned here.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "budget_ratchet_check.py"
)
_spec = importlib.util.spec_from_file_location("budget_ratchet_check", _MODULE_PATH)
ratchet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ratchet)


def _spec_of(limit):
    return {"limit": limit}


def test_limits_read_the_limit_and_skip_malformed():
    limits = ratchet._limits({"LIT006": _spec_of(1023), "junk": 5})
    assert limits == {"LIT006": 1023}  # malformed (non-dict) spec ignored


def test_limits_fall_back_to_legacy_baseline_plus_slack():
    # The base side of a diff can predate the `limit` migration; its ceiling is
    # baseline + slack, read on the same footing as a new-schema `limit`.
    assert ratchet._limits({"LIT006": {"baseline": 1013, "slack": 10}}) == {"LIT006": 1023}


def test_migration_from_legacy_schema_to_equal_limit_is_clean():
    # baseline+slack (1023) -> limit 1023 is the same ceiling, so no regression.
    base = {"LIT006": {"baseline": 1013, "slack": 10}}
    assert ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1023)}) == []
    # ...and a genuine raise across the migration is still caught.
    regs = ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1024)})
    assert [r.rule for r in regs] == ["LIT006"] and "1023 -> 1024" in regs[0].detail


def test_raised_limit_is_a_regression():
    base = {"LIT006": _spec_of(1023)}
    head = {"LIT006": _spec_of(1024)}
    regs = ratchet.regressions_for("b.json", base, head)
    assert [r.rule for r in regs] == ["LIT006"]
    assert "1023 -> 1024" in regs[0].detail


def test_lowered_or_equal_limit_is_clean():
    base = {"LIT006": _spec_of(1023)}
    # limit drops
    assert ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1000)}) == []
    # nothing changes
    assert ratchet.regressions_for("b.json", base, {"LIT006": _spec_of(1023)}) == []


def test_dropped_rule_is_a_regression():
    regs = ratchet.regressions_for("b.json", {"LIT007": _spec_of(0)}, {})
    assert [r.rule for r in regs] == ["LIT007"]
    assert "dropped" in regs[0].detail


def test_new_rule_in_head_is_clean():
    assert ratchet.regressions_for("b.json", {}, {"new-rule": _spec_of(5)}) == []


def test_dropped_rule_that_graduated_to_a_hard_failing_config_is_clean():
    base = {"UP006": _spec_of(0)}
    assert ratchet.regressions_for("b.json", base, {}, graduated=("UP006",)) == []


def test_graduation_matches_by_prefix_like_ruff_selectors_do():
    base = {"ANN202": _spec_of(865)}
    assert ratchet.regressions_for("b.json", base, {}, graduated=("ANN",)) == []


def test_an_unrelated_graduation_does_not_excuse_a_dropped_rule():
    base = {"C901": _spec_of(3)}
    regs = ratchet.regressions_for("b.json", base, {}, graduated=("UP006", "SIM118"))
    assert [r.rule for r in regs] == ["C901"]
    assert "dropped" in regs[0].detail


def test_graduation_never_excuses_a_raised_limit():
    base = {"UP006": _spec_of(0)}
    regs = ratchet.regressions_for("b.json", base, {"UP006": _spec_of(7)}, graduated=("UP006",))
    assert [r.rule for r in regs] == ["UP006"]
    assert "0 -> 7" in regs[0].detail


def test_graduated_selectors_come_from_the_paired_ruff_config():
    selectors = ratchet.graduated_selectors("ruff-strict-budget.json")
    assert "UP006" in selectors
    assert "ANN" not in selectors


def test_budgets_without_a_paired_config_can_never_graduate():
    assert ratchet.graduated_selectors("type-discipline-budget.json") == ()
    assert ratchet.graduated_selectors("basedpyright-code-budget.json") == ()


def test_a_selector_the_config_also_ignores_does_not_count_as_graduated():
    lint = {"ignore": ["UP006"], "extend-select": ["UP006", "SIM118"]}
    assert ratchet.selectors_hard_failed_by(lint) == ("SIM118",)


def test_selectors_hard_failed_by_reads_a_config_with_no_ignore_list():
    assert ratchet.selectors_hard_failed_by({"extend-select": ["UP006"]}) == ("UP006",)


def test_deleted_budget_file_is_a_regression():
    regs = ratchet.regressions_for("b.json", {"LIT006": _spec_of(1)}, None)
    assert [r.rule for r in regs] == ["*"]
    assert "deleted" in regs[0].detail


def test_new_budget_file_has_nothing_to_ratchet():
    assert ratchet.regressions_for("b.json", None, {"LIT006": _spec_of(1)}) == []


def test_default_budgets_watch_every_budget_file_in_the_repo():
    # This job is the repo's only ceiling-raise alarm, so every *-budget.json on disk must be
    # watched; a budget left out of DEFAULT_BUDGETS (e.g. basedpyright-code-budget.json) can be
    # loosened with no signal. Equality also catches a phantom entry that no longer exists.
    repo_root = _MODULE_PATH.parents[1]
    on_disk = frozenset(p.name for p in repo_root.glob("*budget*.json"))
    assert on_disk == frozenset(ratchet.DEFAULT_BUDGETS)


# --------------------------------------------------------------------------- #
# Base-ref resolution: a bad ref must fail loudly, never pass vacuously
# --------------------------------------------------------------------------- #


def test_ref_is_commit_distinguishes_real_from_bogus():
    assert ratchet._ref_is_commit("HEAD") is True
    assert ratchet._ref_is_commit("definitely-not-a-real-ref-zzz") is False


def test_load_base_reads_a_present_file_and_none_for_an_absent_one():
    # A real budget file exists at HEAD; a made-up path is absent at the same (valid) ref.
    assert ratchet._load_base("type-discipline-budget.json", "HEAD") is not None
    assert ratchet._load_base("scripts/no-such-budget-xyz.json", "HEAD") is None


def test_unresolvable_base_ref_exits_nonzero_instead_of_skipping():
    proc = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "--base", "definitely-not-a-real-ref-zzz"],
        cwd=_MODULE_PATH.parents[1],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "does not resolve to a commit" in proc.stderr

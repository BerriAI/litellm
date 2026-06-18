"""Tests for scripts/budget_ratchet_check.py.

The guard's whole contract is "ceilings may only fall": a raised ceiling, a dropped
rule, or a deleted file is a regression, while a lowered/equal ceiling, a brand-new
rule, or a brand-new budget file is fine. Each branch is pinned here.
"""

import importlib.util
import subprocess
import sys
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
    assert ratchet.regressions_for("b.json", {}, {"new-rule": _spec_of(5, 0)}) == []


def test_deleted_budget_file_is_a_regression():
    regs = ratchet.regressions_for("b.json", {"LIT006": _spec_of(1, 0)}, None)
    assert [r.rule for r in regs] == ["*"]
    assert "deleted" in regs[0].detail


def test_new_budget_file_has_nothing_to_ratchet():
    assert ratchet.regressions_for("b.json", None, {"LIT006": _spec_of(1, 0)}) == []


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

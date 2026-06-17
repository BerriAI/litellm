import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_any_discipline.py"
)
_spec = importlib.util.spec_from_file_location("check_any_discipline", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

Violation = mod.Violation


def _v(path="litellm/x.py", line=10, code="LIT009"):
    return Violation(Path(path), line, 0, code, "Any-typed value")


def test_violation_on_a_changed_line_is_in_scope():
    assert mod._in_scope(_v(line=10), {"litellm/x.py": {10, 11}}) is True


def test_violation_on_an_unchanged_line_of_a_changed_file_is_out_of_scope():
    assert mod._in_scope(_v(line=99), {"litellm/x.py": {10, 11}}) is False


def test_whole_new_file_puts_every_line_in_scope():
    assert mod._in_scope(_v(line=99999), {"litellm/x.py": mod.ALL_LINES}) is True


def test_file_absent_from_line_map_is_out_of_scope():
    # Regression: ALL_LINES is a distinct sentinel, so a path missing from the map
    # (line_map.get -> None) is NOT mistaken for "whole file in scope".
    assert mod._in_scope(_v(path="litellm/other.py"), {"litellm/x.py": {1}}) is False


def test_no_line_map_means_no_line_filtering():
    assert mod._in_scope(_v(line=12345), None) is True


def test_build_error_is_always_in_scope():
    assert mod._in_scope(_v(code="LIT000", line=1), {"litellm/x.py": {2}}) is True


# --- per-file Any budget ------------------------------------------------------


def test_slack_is_50_percent_rounded_up():
    assert mod._slack_for(0) == 0
    assert mod._slack_for(1) == 1  # ceil(0.5): even a 1-Any file gets a little room
    assert mod._slack_for(3) == 2  # ceil(1.5)
    assert mod._slack_for(20) == 10
    assert mod._slack_for(5145) == 2573


def test_ceiling_is_baseline_plus_slack():
    assert mod._ceiling({"baseline": 20, "slack": 10}) == 30
    assert mod._ceiling({}) == 0  # an absent/empty entry means a zero ceiling


def test_lit009_counts_groups_by_file_and_ignores_other_codes():
    violations = [
        _v(path="litellm/a.py", line=1, code="LIT009"),
        _v(path="litellm/a.py", line=2, code="LIT009"),
        _v(path="litellm/a.py", line=3, code="LIT005"),  # suppression hygiene, not an Any
        _v(path="litellm/b.py", line=1, code="LIT009"),
        _v(path="litellm/c.py", line=0, code="LIT000"),  # build error, not an Any
    ]
    assert mod.lit009_counts(violations) == {"litellm/a.py": 2, "litellm/b.py": 1}


def test_save_budget_omits_zero_count_files_and_round_trips(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "BUDGET_PATH", tmp_path / "any-discipline-budget.json")
    mod.save_budget({"litellm/a.py": 20, "litellm/b.py": 0, "litellm/c.py": 1})
    loaded = mod.load_budget()
    assert loaded == {
        "litellm/a.py": {"baseline": 20, "slack": 10},
        "litellm/c.py": {"baseline": 1, "slack": 1},
    }
    assert "litellm/b.py" not in loaded  # zero-Any files are never baselined


def test_load_budget_missing_file_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "BUDGET_PATH", tmp_path / "nope.json")
    assert mod.load_budget() == {}

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ruff_suppressions.py"
_spec = importlib.util.spec_from_file_location("ruff_suppressions", _MODULE_PATH)
ruff_suppressions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ruff_suppressions)

evaluate_budget = ruff_suppressions.evaluate_budget


def test_clean_tree_passes_with_headroom_equal_to_slack():
    baseline = {"a.py": {"ANN001": 1000}}
    result = evaluate_budget(baseline, baseline, slack_ratio=0.005)
    assert result.ok
    assert result.baseline == 1000
    assert result.current == 1000
    assert result.slack == 5
    assert result.ceiling == 1005
    assert result.regressions == []


def test_growth_exactly_at_ceiling_passes_one_over_fails():
    baseline = {"a.py": {"ANN001": 1000}}
    at_ceiling = evaluate_budget(
        baseline, {"a.py": {"ANN001": 1005}}, slack_ratio=0.005
    )
    over_ceiling = evaluate_budget(
        baseline, {"a.py": {"ANN001": 1006}}, slack_ratio=0.005
    )
    assert at_ceiling.ok is True
    assert over_ceiling.ok is False


def test_slack_rounds_up():
    baseline = {"a.py": {"ANN001": 8129}}
    result = evaluate_budget(baseline, baseline, slack_ratio=0.005)
    assert result.slack == 41
    assert result.ceiling == 8170


def test_only_grown_entries_are_reported_as_regressions():
    baseline = {"a.py": {"ANN001": 5}, "b.py": {"C901": 3}}
    counts = {"a.py": {"ANN001": 2}, "b.py": {"C901": 7}}
    result = evaluate_budget(baseline, counts, slack_ratio=0.005)
    assert result.regressions == [("b.py", "C901", 3, 7)]


def test_fix_here_add_there_stays_under_budget_but_still_lists_the_growth():
    baseline = {"a.py": {"ANN001": 5}, "b.py": {"C901": 3}}
    counts = {"a.py": {"ANN001": 2}, "b.py": {"C901": 6}}
    result = evaluate_budget(baseline, counts, slack_ratio=0.005)
    assert result.ok is True
    assert ("b.py", "C901", 3, 6) in result.regressions


def test_brand_new_file_counts_and_is_reported():
    result = evaluate_budget({}, {"new.py": {"ANN001": 2}}, slack_ratio=0.005)
    assert result.ok is False
    assert result.baseline == 0
    assert result.ceiling == 0
    assert result.regressions == [("new.py", "ANN001", 0, 2)]


@pytest.mark.parametrize(
    "counts, expected_total",
    [
        ({}, 0),
        ({"a.py": {"ANN001": 3, "C901": 2}}, 5),
        ({"a.py": {"ANN001": 3}, "b.py": {"PLR0913": 4}}, 7),
    ],
)
def test_total_sums_every_rule_in_every_file(counts, expected_total):
    assert ruff_suppressions.total(counts) == expected_total

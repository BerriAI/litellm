"""Tests for scripts/mutation_report.py.

The report is the only thing anyone reads after a mutation run, so the one thing it
must never do is describe a run that produced nothing as a run that killed everything.
`render` decides that wording and `get_survivors` supplies the evidence for it, so both
are tested directly.
"""

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "mutation_report.py"
_spec = importlib.util.spec_from_file_location("mutation_report", _MODULE_PATH)
report = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = report
_spec.loader.exec_module(report)

_CONFIG = {"paths_to_mutate": ["litellm/proxy/management_endpoints/"], "tests_dir": ["tests/"]}


def test_a_run_that_reported_nothing_is_not_a_clean_sweep():
    rendered = report.render(_CONFIG, report.MutmutResults(survivors=(), reported=0), None)

    assert "not a passing score" in rendered
    assert "caught every mutation" not in rendered


def test_a_run_that_killed_every_mutant_says_so():
    rendered = report.render(
        _CONFIG, report.MutmutResults(survivors=(), reported=0), {"killed": 48, "survived": 0}
    )

    assert "caught every mutation" in rendered
    assert "not a passing score" not in rendered


def test_stats_counting_survivors_results_never_listed_is_not_a_clean_sweep():
    rendered = report.render(
        _CONFIG, report.MutmutResults(survivors=(), reported=0), {"killed": 48, "survived": 3}
    )

    assert "not a passing score" in rendered
    assert "caught every mutation" not in rendered
    assert "3 surviving mutant(s)" in rendered


def test_mutants_that_never_reached_the_tests_are_not_a_clean_sweep():
    rendered = report.render(
        _CONFIG,
        report.MutmutResults(survivors=(), reported=0),
        {"killed": 48, "survived": 0, "no_tests": 4, "timeout": 1},
    )

    assert "not a passing score" in rendered
    assert "caught every mutation" not in rendered
    assert "4 no tests" in rendered
    assert "1 timeout" in rendered


def test_a_status_the_reporter_has_never_met_still_blocks_a_clean_sweep():
    rendered = report.render(
        _CONFIG,
        report.MutmutResults(survivors=(), reported=0),
        {"killed": 48, "survived": 0, "check_was_interrupted_by_user": 2},
    )

    assert "not a passing score" in rendered
    assert "caught every mutation" not in rendered
    assert "2 check was interrupted by user" in rendered


def test_no_survivors_without_a_kill_is_not_a_clean_sweep():
    rendered = report.render(
        _CONFIG, report.MutmutResults(survivors=(), reported=48), {"killed": 0, "survived": 0}
    )

    assert "not a passing score" in rendered
    assert "caught every mutation" not in rendered


def test_no_survivors_and_no_stats_cannot_claim_a_sweep():
    """`mutmut results` never lists killed mutants, so with the stats file missing an
    empty survivor list is equally consistent with a perfect run and a dead one."""
    rendered = report.render(_CONFIG, report.MutmutResults(survivors=(), reported=48), None)

    assert "not a passing score" in rendered
    assert "caught every mutation" not in rendered


def test_survivors_are_read_out_of_the_verdicts_they_came_with(monkeypatch):
    class _Proc:
        stdout = (
            "litellm.proxy.management_endpoints.key_management_endpoints.x_1: killed\n"
            "litellm.proxy.management_endpoints.key_management_endpoints.x_2: survived\n"
            "litellm.proxy.management_endpoints.key_management_endpoints.x_3: no tests\n"
            "not a verdict line at all\n"
        )

    monkeypatch.setattr(report.subprocess, "run", lambda *a, **k: _Proc())

    results = report.get_survivors()

    assert results.survivors == (
        "litellm.proxy.management_endpoints.key_management_endpoints.x_2",
    )
    assert results.reported == 3


def test_every_multi_word_verdict_mutmut_can_emit_still_counts(monkeypatch):
    class _Proc:
        stdout = "".join(
            f"litellm.proxy.management_endpoints.key_management_endpoints.x_{i}: {verdict}\n"
            for i, verdict in enumerate(
                (
                    "no tests",
                    "not checked",
                    "caught by type check",
                    "check was interrupted by user",
                )
            )
        )

    monkeypatch.setattr(report.subprocess, "run", lambda *a, **k: _Proc())

    results = report.get_survivors()

    assert results.survivors == ()
    assert results.reported == 4


def test_an_empty_mutmut_results_reports_nothing_rather_than_zero_survivors(monkeypatch):
    class _Proc:
        stdout = ""

    monkeypatch.setattr(report.subprocess, "run", lambda *a, **k: _Proc())

    assert report.get_survivors() == report.MutmutResults(survivors=(), reported=0)

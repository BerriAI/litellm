"""Regression tests for the mutation report the PR job posts as its summary.

The report is the only thing a reviewer reads, so two failures matter more than
the rest of the rendering: a run that executed nothing must not read as a clean
sweep, and a surviving mutant in a class method must resolve to that method
rather than to whatever else in the file shares its name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "mutation_report", _REPO_ROOT / "scripts" / "mutation_report.py"
)
report = importlib.util.module_from_spec(_spec)
# dataclasses resolves a frozen class's module through sys.modules at decoration time.
sys.modules[_spec.name] = report
_spec.loader.exec_module(report)


SOURCE = '''\
def handle(payload: str) -> str:
    return payload


class Router:
    def handle(self, payload: str) -> str:
        return payload.strip()
'''


def test_class_method_mutant_resolves_to_the_class_and_method():
    parsed = report.parse_mutant_name("litellm.router.xǁRouterǁ_get_client__mutmut_11")

    assert (parsed.module, parsed.class_name, parsed.function, parsed.number) == (
        "litellm.router",
        "Router",
        "_get_client",
        "11",
    )
    assert parsed.mangled == "xǁRouterǁ_get_client"
    assert parsed.qualified == "Router._get_client"


def test_module_level_mutant_keeps_leading_underscores():
    parsed = report.parse_mutant_name(
        "litellm.proxy.common_utils.x__is_user_team_admin__mutmut_2"
    )

    assert (parsed.class_name, parsed.function, parsed.mangled) == (
        None,
        "_is_user_team_admin",
        "x__is_user_team_admin",
    )


def test_unparseable_mutant_name_is_kept_verbatim():
    parsed = report.parse_mutant_name("not-a-mutant")

    assert parsed.function == "not-a-mutant"
    assert parsed.number == "?"


def test_class_context_picks_the_method_over_the_module_level_twin(tmp_path: Path):
    """Both definitions are named `handle`; only the class context tells them apart."""
    source = tmp_path / "router.py"
    source.write_text(SOURCE, encoding="utf-8")

    method = report.find_function_in_file(source, "handle", "Router")
    function = report.find_function_in_file(source, "handle", None)

    assert method is not None and "payload.strip()" in method[2]
    assert method[3] == [6]
    assert function is not None and "payload.strip()" not in function[2]


def test_summarize_ignores_mutants_the_run_never_checked():
    results = (
        ("mod.x_a__mutmut_1", "killed"),
        ("mod.x_a__mutmut_2", "survived"),
        ("mod.x_b__mutmut_1", "not checked"),
    )

    assert report.summarize(results) == {
        "total": 2,
        "killed": 1,
        "survived": 1,
        "no_tests": 0,
        "skipped": 0,
        "suspicious": 0,
        "timeout": 0,
    }


def test_summarize_returns_none_when_nothing_ran():
    assert report.summarize((("mod.x_a__mutmut_1", "not checked"),)) is None


def test_a_run_that_checked_nothing_is_not_reported_as_a_clean_sweep():
    """mutmut crashing must not render the same green summary as a killed-everything run."""
    rendered = report.render({}, (), None)

    assert "caught every mutation" not in rendered
    assert "failed run" in rendered


def test_a_run_with_no_survivors_still_reports_the_clean_sweep():
    rendered = report.render({}, (), {"total": 3, "killed": 3, "survived": 0})

    assert "caught every mutation" in rendered
    assert "Mutation score: **100.0%**" in rendered


def test_in_scope_mutants_left_unchecked_are_what_marks_a_run_unfinished():
    """A killed mutmut leaves the rest at `not checked`; only the requested ones count."""
    results = (
        ("litellm.router.xǁRouterǁpick__mutmut_1", "killed"),
        ("litellm.router.xǁRouterǁpick__mutmut_2", "not checked"),
        ("litellm.other.x_untouched__mutmut_1", "not checked"),
    )

    assert report.unchecked_in_scope(results, ("litellm.router.xǁRouterǁpick__mutmut_*",)) == (
        "litellm.router.xǁRouterǁpick__mutmut_2",
    )
    assert report.unchecked_in_scope(results, ()) == ()


def test_a_partial_run_is_not_reported_as_a_clean_sweep():
    """3 of 27 mutants killed is not a 100% score; it is a run that stopped early."""
    rendered = report.render(
        {},
        (),
        {"total": 3, "killed": 3, "survived": 0},
        ("litellm.router.xǁRouterǁpick__mutmut_4",),
    )

    assert "caught every mutation" not in rendered
    assert "stopped early" in rendered
    assert "Never checked: **1**" in rendered

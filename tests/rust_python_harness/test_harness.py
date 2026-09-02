from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.rust_python_harness.catalog import load_catalog
from tests.rust_python_harness.cli import _pick_values, _select
from tests.rust_python_harness.models import (
    CaseResult,
    Coverage,
    HarnessCase,
    HarnessRun,
    RunStatus,
    SDK_FUNCTIONS,
)
from tests.rust_python_harness.runner import (
    run_pytest,
    runnable_selectors,
    selector_matches_node,
)
from tests.rust_python_harness.ui import _format_duration, _rerun_command, _summary


def _case(
    *, selectors: tuple[str, ...] = (), coverage: Coverage = Coverage.COMPLETE
) -> HarnessCase:
    return HarnessCase(
        strategy_id="example",
        strategy_label="Example",
        sdk_function="messages",
        coverage=coverage,
        selectors=selectors,
    )


def _manifest() -> dict[str, object]:
    return {
        "order": 1,
        "id": "example",
        "label": "Example strategy",
        "description": "Example description",
        "functions": {
            function: {"coverage": "planned", "selectors": []}
            for function in SDK_FUNCTIONS
        },
    }


def test_should_load_every_tdd_strategy_in_order() -> None:
    strategies = load_catalog()

    assert [strategy.id for strategy in strategies] == [
        "end_to_end",
        "transform_request",
        "transform_response",
        "transform_stream",
        "cassettes",
        "callbacks",
        "manifest_coverage",
        "dual_build_suite",
        "shadow_mode",
    ]
    assert all(
        tuple(case.sdk_function for case in strategy.cases) == SDK_FUNCTIONS
        for strategy in strategies
    )


def test_should_reject_a_manifest_missing_an_sdk_function(tmp_path: Path) -> None:
    strategy_directory = tmp_path / "example"
    strategy_directory.mkdir()
    manifest = _manifest()
    del manifest["functions"]["count_tokens"]  # type: ignore[index]
    (strategy_directory / "strategy.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="functions must exactly match"):
        load_catalog(tmp_path)


@pytest.mark.parametrize(
    ("selector", "nodeid", "matches"),
    [
        ("tests/test_parity.py", "tests/test_parity.py::test_one", True),
        ("tests/test_parity.py::test_one", "tests/test_parity.py::test_one", True),
        (
            "tests/test_parity.py::test_one",
            "tests/test_parity.py::test_one[value]",
            True,
        ),
        ("tests/test_parity.py::test_one", "tests/test_parity.py::test_two", False),
    ],
)
def test_should_match_pytest_file_and_node_selectors(
    selector: str, nodeid: str, matches: bool
) -> None:
    assert selector_matches_node(selector, nodeid) is matches


def test_should_only_return_selectors_whose_files_exist(tmp_path: Path) -> None:
    existing = tmp_path / "tests" / "test_parity.py"
    existing.parent.mkdir()
    existing.write_text("", encoding="utf-8")
    case = _case(
        selectors=("tests/test_parity.py", "tests/test_missing.py::test_missing")
    )

    assert runnable_selectors((case,), tmp_path) == ("tests/test_parity.py",)


def test_should_mark_planned_and_not_applicable_cases_without_running() -> None:
    planned = CaseResult(case=_case(coverage=Coverage.PLANNED))
    not_applicable = CaseResult(case=_case(coverage=Coverage.NOT_APPLICABLE))

    planned.set_initial_status()
    not_applicable.set_initial_status()

    assert planned.status is RunStatus.PLANNED
    assert not_applicable.status is RunStatus.NOT_APPLICABLE


def test_should_treat_an_all_planned_filtered_run_as_success(tmp_path: Path) -> None:
    exit_code, run = run_pytest(
        cases=(_case(coverage=Coverage.PLANNED),),
        repo_root=tmp_path,
        on_update=lambda _: None,
    )

    assert exit_code == 0
    assert next(iter(run.results.values())).status is RunStatus.PLANNED


def test_should_finalize_a_fully_passing_case() -> None:
    result = CaseResult(case=_case(selectors=("tests/test_parity.py",)))
    result.set_initial_status()
    result.collected.update({"one", "two"})
    result.completed.update({"one", "two"})
    result.passed = 2

    result.finalize()

    assert result.status is RunStatus.PASSED


def test_should_replace_a_pass_with_a_teardown_error() -> None:
    result = CaseResult(case=_case(selectors=("tests/test_parity.py",)))
    result.set_initial_status()
    result.collected.add("one")

    result.record("one", RunStatus.PASSED, 0.1)
    result.record("one", RunStatus.ERROR, 0.2)

    assert result.status is RunStatus.ERROR
    assert result.passed == 0
    assert result.errors == 1
    assert result.duration == pytest.approx(0.3)


def test_should_filter_the_catalog_by_strategy_and_sdk_function() -> None:
    strategies = load_catalog()

    cases = _select(strategies, {"end_to_end"}, {"messages"})

    assert len(cases) == 1
    assert cases[0].key == "end_to_end:messages"


def test_should_reject_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        _select(load_catalog(), {"not-real"}, set())


def test_should_pick_multiple_interactive_filters() -> None:
    answers = iter(["nope", "1, 3"])

    selected = _pick_values(
        "Examples",
        (("one", "One"), ("two", "Two"), ("three", "Three")),
        input_fn=lambda _: next(answers),
    )

    assert selected == {"one", "three"}


def test_should_format_developer_facing_run_context() -> None:
    run = HarnessRun.from_cases((_case(selectors=("tests/test_parity.py",)),))
    result = next(iter(run.results.values()))
    result.collected.add("tests/test_parity.py::test_one")
    result.record("tests/test_parity.py::test_one", RunStatus.PASSED, 1.25)

    assert _summary(run) == (1, 0, 0, 0)
    assert _format_duration(1.25) == "1.2s"
    assert _rerun_command("tests/test_parity.py::test_one") == (
        "poetry run pytest tests/test_parity.py::test_one -q"
    )
    assert _rerun_command("tests/test_parity.py::test_one[value with spaces]") == (
        "poetry run pytest 'tests/test_parity.py::test_one[value with spaces]' -q"
    )

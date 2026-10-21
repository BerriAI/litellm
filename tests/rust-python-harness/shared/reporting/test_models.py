from __future__ import annotations

from typing import Final

import pytest

from .models import CaseResult, Coverage, HarnessCase, RunStatus
from .strategy import ModuleCaseSpec, NotImplementedCaseSpec, SkippedCaseSpec


def _case(spec: ModuleCaseSpec | NotImplementedCaseSpec | SkippedCaseSpec) -> HarnessCase:
    return HarnessCase(strategy_id="example", strategy_label="Example", sdk_function="messages", spec=spec)


def _runnable_case() -> HarnessCase:
    return _case(ModuleCaseSpec(coverage=Coverage.COMPLETE, module="tests.example"))


def test_should_mark_not_implemented_and_skipped_cases_without_running() -> None:
    not_implemented: Final = CaseResult(case=_case(NotImplementedCaseSpec(reason="No case is registered.")))
    skipped: Final = CaseResult(case=_case(SkippedCaseSpec(reason="The surface does not apply.")))

    not_implemented.set_initial_status()
    skipped.set_initial_status()

    assert not_implemented.status is RunStatus.NOT_IMPLEMENTED
    assert skipped.status is RunStatus.SKIPPED


def test_should_finalize_a_fully_passing_case() -> None:
    result: Final = CaseResult(case=_runnable_case())
    result.set_initial_status()
    result.collected.update({"one", "two"})
    result.completed.update({"one", "two"})
    result.passed = 2

    result.finalize()

    assert result.status is RunStatus.PASSED


def test_should_replace_a_pass_with_a_teardown_error() -> None:
    result: Final = CaseResult(case=_runnable_case())
    result.set_initial_status()
    result.collected.add("one")

    result.record("one", RunStatus.PASSED, 0.1)
    result.record("one", RunStatus.ERROR, 0.2)

    assert result.status is RunStatus.ERROR
    assert result.passed == 0
    assert result.errors == 1
    assert result.duration == pytest.approx(0.3)

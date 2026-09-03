from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from ..reporting.models import Coverage, HarnessCase, ResultArtifact, RunStatus
from ..reporting.strategy import CaseSpec, NotImplementedCaseSpec, SuiteCaseSpec
from .suite_runner import SuiteExecution, run_suites


class _Suite(BaseModel):
    problems: tuple[str, ...] = ()


def _execute(suite: _Suite, repo_root: Path, pytest_args: Sequence[str]) -> SuiteExecution:
    del repo_root, pytest_args
    return SuiteExecution(problems=suite.problems)


def _case(spec: CaseSpec) -> HarnessCase:
    return HarnessCase(
        strategy_id="example",
        strategy_label="Example",
        sdk_function="ocr",
        spec=spec,
    )


def test_not_implemented_cell_finalizes_without_running(tmp_path: Path) -> None:
    case = _case(NotImplementedCaseSpec(reason="No suite is registered."))

    code, report = run_suites((case,), tmp_path, lambda _: None, (), suites={}, execute=_execute)

    assert code == 0
    assert report.results[case.key].status is RunStatus.NOT_IMPLEMENTED
    assert not report.failures


def test_missing_registered_suite_marks_the_cell_as_error(tmp_path: Path) -> None:
    case = _case(SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"))

    code, report = run_suites((case,), tmp_path, lambda _: None, (), suites={}, execute=_execute)

    assert code == 1
    assert report.results[case.key].status is RunStatus.ERROR
    assert report.failures


def test_suite_problems_mark_the_cell_as_failed(tmp_path: Path) -> None:
    case = _case(SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"))

    code, report = run_suites(
        (case,), tmp_path, lambda _: None, (), suites={"ocr": _Suite(problems=("boom",))}, execute=_execute
    )

    assert code == 1
    assert report.results[case.key].status is RunStatus.FAILED
    assert ("suite:example:ocr:ocr", "boom") in report.failures


def test_suite_without_problems_passes(tmp_path: Path) -> None:
    case = _case(SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"))

    code, report = run_suites((case,), tmp_path, lambda _: None, (), suites={"ocr": _Suite()}, execute=_execute)

    assert code == 0
    assert report.results[case.key].status is RunStatus.PASSED
    assert not report.failures


def test_suite_attaches_artifacts_to_passing_and_failing_results(tmp_path: Path) -> None:
    case: Final = _case(SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="ocr"))
    artifact: Final = ResultArtifact("example", "body")

    def execute(suite: _Suite, repo_root: Path, pytest_args: Sequence[str]) -> SuiteExecution:
        del repo_root, pytest_args
        return SuiteExecution(problems=suite.problems, artifacts=(artifact,))

    _, passing = run_suites((case,), tmp_path, lambda _: None, suites={"ocr": _Suite()}, execute=execute)
    _, failing = run_suites(
        (case,), tmp_path, lambda _: None, suites={"ocr": _Suite(problems=("boom",))}, execute=execute
    )

    assert passing.results[case.key].artifacts == {"suite:example:ocr:ocr": (artifact,)}
    assert failing.results[case.key].artifacts == {"suite:example:ocr:ocr": (artifact,)}

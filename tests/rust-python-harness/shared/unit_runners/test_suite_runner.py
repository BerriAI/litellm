from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from ..reporting.models import Coverage, HarnessCase, RunStatus
from ..reporting.strategy import SuiteCaseSpec
from .suite_runner import run_suites


class _Suite(BaseModel):
    problems: tuple[str, ...] = ()


def _load(path: Path) -> _Suite:
    return _Suite.model_validate_json(path.read_text(encoding="utf-8"))


def _execute(
    suite: _Suite, repo_root: Path, pytest_args: Sequence[str]
) -> tuple[str, ...]:
    del repo_root, pytest_args
    return suite.problems


def _case(spec: SuiteCaseSpec) -> HarnessCase:
    return HarnessCase(
        strategy_id="example",
        strategy_label="Example",
        sdk_function="ocr",
        spec=spec,
    )


def test_planned_cell_finalizes_without_running(tmp_path: Path) -> None:
    case = _case(SuiteCaseSpec(coverage=Coverage.PLANNED))

    code, report = run_suites((case,), tmp_path, lambda _: None, (), load=_load, execute=_execute)

    assert code == 0
    assert report.results[case.key].status is RunStatus.PLANNED
    assert not report.failures


def test_suite_load_error_marks_the_cell_as_error(tmp_path: Path) -> None:
    (tmp_path / "suite.json").write_text("{not json", encoding="utf-8")
    case = _case(SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="suite.json"))

    code, report = run_suites((case,), tmp_path, lambda _: None, (), load=_load, execute=_execute)

    assert code == 1
    assert report.results[case.key].status is RunStatus.ERROR
    assert report.failures


def test_suite_problems_mark_the_cell_as_failed(tmp_path: Path) -> None:
    (tmp_path / "suite.json").write_text('{"problems": ["boom"]}', encoding="utf-8")
    case = _case(SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="suite.json"))

    code, report = run_suites((case,), tmp_path, lambda _: None, (), load=_load, execute=_execute)

    assert code == 1
    assert report.results[case.key].status is RunStatus.FAILED
    assert ("suite:example:ocr:suite.json", "boom") in report.failures


def test_suite_without_problems_passes(tmp_path: Path) -> None:
    (tmp_path / "suite.json").write_text('{"problems": []}', encoding="utf-8")
    case = _case(SuiteCaseSpec(coverage=Coverage.COMPLETE, suite="suite.json"))

    code, report = run_suites((case,), tmp_path, lambda _: None, (), load=_load, execute=_execute)

    assert code == 0
    assert report.results[case.key].status is RunStatus.PASSED
    assert not report.failures

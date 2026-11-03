from __future__ import annotations

import os
import re
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

RESULTS_PATH_ENV: Final = "CODEX_COMPAT_RESULTS_PATH"
DETAIL_LIMIT: Final = 500
_ROW_FILE_PREFIX: Final = "test_"
_ROW_FILE_SUFFIX: Final = "_e2e.py"
_ROW_CLASS_PREFIX: Final = "TestCodex"
_WORD_START: Final = re.compile(r"(?<!^)(?=[A-Z])")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Passed(_Frozen):
    status: Literal["pass"] = "pass"


class Failed(_Frozen):
    status: Literal["fail"] = "fail"
    error: str = Field(min_length=1)


class KnownGap(_Frozen):
    status: Literal["known_gap"] = "known_gap"
    reason: str = Field(min_length=1)


class NotTested(_Frozen):
    status: Literal["not_tested"] = "not_tested"
    reason: str | None = None


type Outcome = Annotated[Passed | Failed | KnownGap | NotTested, Field(discriminator="status")]


class CellResult(_Frozen):
    nodeid: str
    row: str
    row_name: str
    setup: str
    outcome: Outcome


_CELL_RESULT: Final = TypeAdapter(CellResult)


@dataclass(frozen=True, slots=True)
class _Cell:
    row: str
    row_name: str
    setup: str


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report: Final = yield
    destination: Final = os.environ.get(RESULTS_PATH_ENV)
    result: Final = cell_result(item, call, report)
    if destination and result is not None:
        append_result(Path(destination), result)
    return report


def cell_result(item: pytest.Item, call: pytest.CallInfo[None], report: pytest.TestReport) -> CellResult | None:
    cell: Final = _cell(item)
    outcome: Final = _outcome(call, report)
    if cell is None or outcome is None:
        return None
    return CellResult(nodeid=report.nodeid, row=cell.row, row_name=cell.row_name, setup=cell.setup, outcome=outcome)


def row_name(class_name: str) -> str:
    return _WORD_START.sub(" ", class_name.removeprefix(_ROW_CLASS_PREFIX)).capitalize()


def append_result(path: Path, result: CellResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as results:
        _ = results.write(result.model_dump_json(exclude_none=True).encode() + b"\n")


def read_results(path: Path) -> tuple[CellResult, ...]:
    return tuple(_CELL_RESULT.validate_json(line) for line in path.read_text().splitlines() if line.strip())


def _cell(item: pytest.Item) -> _Cell | None:
    name: Final = item.path.name
    row_class: Final = item.getparent(pytest.Class)
    is_row_file: Final = name.startswith(_ROW_FILE_PREFIX) and name.endswith(_ROW_FILE_SUFFIX)
    if not is_row_file or row_class is None or not isinstance(item, pytest.Function) or not hasattr(item, "callspec"):
        return None
    if "deployment" not in item.callspec.params:
        return None
    return _Cell(
        row=name.removeprefix(_ROW_FILE_PREFIX).removesuffix(_ROW_FILE_SUFFIX),
        row_name=row_name(row_class.name),
        setup=item.callspec.id,
    )


def _outcome(call: pytest.CallInfo[None], report: pytest.TestReport) -> Outcome | None:
    expected_failure: Final[object] = vars(report).get("wasxfail")
    match report.when, report.outcome:
        case "teardown", _:
            return None
        case _, "skipped" if isinstance(expected_failure, str):
            return KnownGap(reason=expected_failure or "marked as an expected failure")
        case _, "failed":
            return Failed(error=_failure(call, report))
        case _, "skipped":
            return NotTested(reason=_skip_reason(report))
        case "call", "passed":
            return Passed()
        case _:
            return None


def _failure(call: pytest.CallInfo[None], report: pytest.TestReport) -> str:
    text: Final = call.excinfo.exconly() if call.excinfo is not None else report.longreprtext
    first_line: Final = next((line.strip() for line in text.splitlines() if line.strip()), f"{report.when} failed")
    return first_line if len(first_line) <= DETAIL_LIMIT else f"{first_line[: DETAIL_LIMIT - 3]}..."


def _skip_reason(report: pytest.TestReport) -> str | None:
    match report.longrepr:
        case (str(), int(), str() as message):
            return message.removeprefix("Skipped: ")
        case _:
            return None

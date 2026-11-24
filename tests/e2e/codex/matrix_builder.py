from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, FilePath, TypeAdapter

from codex.compat_results import CellResult, Failed, KnownGap, NotTested, Outcome, Passed, read_results
from codex.setups import SETUPS

SCHEMA_VERSION: Final = "1"
REGRESSION_EXIT_CODE: Final = 3
MISSING: Final = "missing"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MatrixRow(_Frozen):
    id: str
    name: str
    providers: Mapping[str, Outcome]


class Matrix(_Frozen):
    schema_version: Literal["1"]
    generated_at: str
    litellm_version: str
    codex_version: str
    providers: tuple[str, ...]
    features: tuple[MatrixRow, ...]


class Regression(_Frozen):
    feature_id: str
    feature_name: str
    provider: str
    old_status: Literal["pass"]
    new_status: str
    error: str


class BuildArgs(_Frozen):
    command: Literal["build"]
    results: FilePath
    out: Path
    litellm_version: str
    codex_version: str


class RegressionArgs(_Frozen):
    command: Literal["regressions"]
    old: FilePath
    new: FilePath


CLI_ARGS: Final = TypeAdapter[BuildArgs | RegressionArgs](
    Annotated[BuildArgs | RegressionArgs, Field(discriminator="command")]
)
REGRESSIONS: Final = TypeAdapter(tuple[Regression, ...])


def setup_columns(results: tuple[CellResult, ...]) -> tuple[str, ...]:
    declared: Final = tuple(setup.id for setup in SETUPS if isinstance(setup.id, str))
    extra: Final = sorted(frozenset(result.setup for result in results) - frozenset(declared))
    return (*declared, *extra)


def latest_attempts(results: tuple[CellResult, ...]) -> tuple[CellResult, ...]:
    first_failures: Final = MappingProxyType(
        {result.nodeid: result.outcome.error for result in reversed(results) if isinstance(result.outcome, Failed)}
    )
    latest: Final = MappingProxyType({result.nodeid: result for result in results})
    return tuple(_note_rerun(result, first_failures.get(result.nodeid)) for result in latest.values())


def _note_rerun(result: CellResult, first_failure: str | None) -> CellResult:
    if first_failure is None or not isinstance(result.outcome, Passed | KnownGap):
        return result
    outcome: Final = result.outcome.model_copy(update={"first_attempt_error": first_failure})
    return result.model_copy(update={"outcome": outcome})


def aggregate(outcomes: tuple[Outcome, ...]) -> Outcome:
    failures: Final = tuple(outcome.error for outcome in outcomes if isinstance(outcome, Failed))
    if failures:
        return Failed(error="; ".join(failures))
    rerun_errors: Final = "; ".join(_first_attempt_errors(outcomes)) or None
    gaps: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, KnownGap))
    if gaps:
        return gaps[0].model_copy(update={"first_attempt_error": rerun_errors})
    if any(isinstance(outcome, Passed) for outcome in outcomes):
        return Passed(first_attempt_error=rerun_errors)
    skips: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, NotTested))
    return skips[0] if skips else NotTested()


def _first_attempt_errors(outcomes: tuple[Outcome, ...]) -> tuple[str, ...]:
    return tuple(
        outcome.first_attempt_error
        for outcome in outcomes
        if isinstance(outcome, Passed | KnownGap) and outcome.first_attempt_error
    )


def build_matrix(
    results: tuple[CellResult, ...],
    *,
    columns: tuple[str, ...],
    litellm_version: str,
    codex_version: str,
    generated_at: datetime,
) -> Matrix:
    attempts: Final = latest_attempts(results)
    names: Final = MappingProxyType({result.row: result.row_name for result in attempts})
    return Matrix(
        schema_version=SCHEMA_VERSION,
        generated_at=generated_at.isoformat(),
        litellm_version=litellm_version,
        codex_version=codex_version,
        providers=columns,
        features=tuple(
            MatrixRow(
                id=row,
                name=names[row],
                providers=MappingProxyType({column: _cell(attempts, row, column) for column in columns}),
            )
            for row in sorted(names)
        ),
    )


def _cell(attempts: tuple[CellResult, ...], row: str, column: str) -> Outcome:
    return aggregate(tuple(result.outcome for result in attempts if result.row == row and result.setup == column))


def _cells(matrix: Matrix) -> Iterator[tuple[MatrixRow, str, Outcome]]:
    for row in matrix.features:
        for column, outcome in row.providers.items():
            yield row, column, outcome


def find_regressions(old: Matrix, new: Matrix) -> tuple[Regression, ...]:
    new_cells: Final = MappingProxyType({(row.id, column): outcome for row, column, outcome in _cells(new)})
    return tuple(
        Regression(
            feature_id=row.id,
            feature_name=row.name,
            provider=column,
            old_status="pass",
            new_status=_status(new_cells.get((row.id, column))),
            error=_detail(new_cells.get((row.id, column))),
        )
        for row, column, outcome in _cells(old)
        if isinstance(outcome, Passed) and not isinstance(new_cells.get((row.id, column)), Passed)
    )


def _status(outcome: Outcome | None) -> str:
    return MISSING if outcome is None else outcome.status


def _detail(outcome: Outcome | None) -> str:
    match outcome:
        case Failed(error=error):
            return error
        case KnownGap(reason=reason) | NotTested(reason=str() as reason):
            return reason
        case Passed() | NotTested() | None:
            return ""


def _build(args: BuildArgs) -> int:
    results: Final = read_results(args.results)
    matrix: Final = build_matrix(
        results,
        columns=setup_columns(results),
        litellm_version=args.litellm_version,
        codex_version=args.codex_version,
        generated_at=datetime.now(timezone.utc),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _ = args.out.write_text(matrix.model_dump_json(indent=2, exclude_none=True) + "\n")
    return 0


def _regressions(args: RegressionArgs) -> int:
    regressions: Final = find_regressions(
        Matrix.model_validate_json(args.old.read_bytes()), Matrix.model_validate_json(args.new.read_bytes())
    )
    _ = sys.stdout.write(REGRESSIONS.dump_json(regressions, indent=2).decode() + "\n")
    return REGRESSION_EXIT_CODE if regressions else 0


def main(argv: tuple[str, ...]) -> int:
    parser: Final = ArgumentParser(prog="python -m codex.matrix_builder")
    commands: Final = parser.add_subparsers(dest="command", required=True)
    build: Final = commands.add_parser("build")
    _ = build.add_argument("--results", required=True)
    _ = build.add_argument("--out", required=True)
    _ = build.add_argument("--litellm-version", dest="litellm_version", required=True)
    _ = build.add_argument("--codex-version", dest="codex_version", required=True)
    regressions: Final = commands.add_parser("regressions")
    _ = regressions.add_argument("--old", required=True)
    _ = regressions.add_argument("--new", required=True)
    match CLI_ARGS.validate_python(vars(parser.parse_args(argv))):
        case BuildArgs() as build_args:
            return _build(build_args)
        case RegressionArgs() as regression_args:
            return _regressions(regression_args)


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))

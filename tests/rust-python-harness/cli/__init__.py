from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from typing import Annotated, Final, TypeAlias

import pytest
from pydantic import Field, TypeAdapter, ValidationError

from ..shared.reporting.models import SDK_FUNCTIONS, SdkFunction, Strategy
from .catalog import load_catalog
from .commands import CheckArgs, ListArgs, RunArgs, check_command, list_command, run_command
from .selection import interactive_filters, select

__all__ = ["load_catalog", "main"]


ParsedArgs: TypeAlias = Annotated[RunArgs | ListArgs | CheckArgs, Field(discriminator="verb")]
_ARGS_ADAPTER: Final[TypeAdapter[ParsedArgs]] = TypeAdapter(ParsedArgs)


def _selection_parent(strategy_ids: Sequence[str]) -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    selection = parent.add_argument_group("selection")
    selection.add_argument(
        "--strategy",
        action="append",
        default=[],
        choices=strategy_ids,
        help="run only this strategy",
    )
    selection.add_argument(
        "--function",
        action="append",
        default=[],
        dest="sdk_functions",
        choices=SDK_FUNCTIONS,
        help="run only this SDK function",
    )
    selection.add_argument(
        "--surface",
        choices=("sdk", "gateway"),
        help="run only this API surface",
    )
    selection.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="pick strategies and SDK functions in a guided terminal menu",
    )
    return parent


def _build_parser(strategies: Sequence[Strategy]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rust-python-harness",
        description="Run Rust/Python parity tests with a live strategy-by-SDK-function dashboard.",
    )
    subparsers = parser.add_subparsers(dest="verb", required=True)
    strategy_ids: Final = tuple(strategy.id for strategy in strategies)
    selection = _selection_parent(strategy_ids)

    run_parser = subparsers.add_parser(
        "run", parents=[selection], help="run the selected strategies (default dashboard)"
    )
    run_parser.add_argument(
        "--plain",
        action="store_true",
        help="disable the interactive terminal dashboard",
    )
    run_parser.add_argument(
        "--coverage",
        action="store_true",
        help="write Python reference LOC reports (HTML, JSON, and XML)",
    )
    run_parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="append an argument to pytest (repeatable, for example --pytest-arg=-x)",
    )

    subparsers.add_parser(
        "list", parents=[selection], help="show the selected catalog without running tests"
    )

    check_parser = subparsers.add_parser(
        "check", parents=[selection], help="run each selected strategy's consistency check"
    )
    check_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print a PASS/FAIL marker per check report",
    )
    return parser


_INTERRUPTED_EXIT_CODE: Final = 130


def _main(argv: Sequence[str] | None = None) -> int:
    strategies: Final = load_catalog()
    parser: Final = _build_parser(strategies)
    namespace: Final = parser.parse_args(argv)
    try:
        args: Final = _ARGS_ADAPTER.validate_python(vars(namespace))
    except ValidationError as error:
        parser.error(str(error))
    if args.verb == "run" and args.coverage and importlib.util.find_spec("pytest_cov") is None:
        parser.error(
            "--coverage requires the project's pytest-cov dependency; run with "
            "`poetry run python -m tests.rust-python-harness run --coverage`"
        )
    strategy_ids: Final = frozenset(args.strategy)
    sdk_functions: Final = frozenset(args.sdk_functions)
    picked: Final[tuple[frozenset[str], frozenset[SdkFunction]]] = (
        interactive_filters(strategies)
        if args.interactive
        else (frozenset[str](), frozenset[SdkFunction]())
    )
    selected_strategy_ids: Final = strategy_ids or picked[0]
    selected_sdk_functions: Final = sdk_functions or picked[1]
    try:
        cases: Final = select(
            strategies, selected_strategy_ids, selected_sdk_functions, args.surface
        )
    except ValueError as exc:
        parser.error(str(exc))
    match args.verb:
        case "run":
            return run_command(args, strategies, cases)
        case "list":
            return list_command(args, strategies, cases)
        case "check":
            return check_command(args, strategies, cases)
    raise AssertionError("unreachable")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        exit_code: Final = _main(argv)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted\n")
        return _INTERRUPTED_EXIT_CODE
    if exit_code == int(pytest.ExitCode.INTERRUPTED):
        sys.stderr.write("Interrupted\n")
        return _INTERRUPTED_EXIT_CODE
    return exit_code

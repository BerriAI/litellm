from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence

from ..shared.reporting.models import SDK_FUNCTIONS
from .catalog import load_catalog
from .commands import check_command, list_command, run_command
from .selection import interactive_filters, select

__all__ = ["load_catalog", "main"]


def _selection_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    selection = parent.add_argument_group("selection")
    selection.add_argument(
        "--strategy",
        action="append",
        default=[],
        metavar="ID",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rust-python-harness",
        description="Run Rust/Python parity tests with a live strategy-by-SDK-function dashboard.",
    )
    subparsers = parser.add_subparsers(dest="verb", required=True)
    selection = _selection_parent()

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


_PARSER = _build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    args = _PARSER.parse_args(argv)
    if args.verb == "run" and args.coverage and importlib.util.find_spec("pytest_cov") is None:
        _PARSER.error(
            "--coverage requires the project's pytest-cov dependency; run with "
            "`poetry run python -m tests.rust-python-harness run --coverage`"
        )
    strategies = load_catalog()
    strategy_ids = set(args.strategy)
    sdk_functions = set(args.sdk_functions)
    if args.interactive:
        picked_strategies, picked_functions = interactive_filters(strategies)
        strategy_ids = strategy_ids or picked_strategies
        sdk_functions = sdk_functions or picked_functions
    try:
        cases = select(strategies, strategy_ids, sdk_functions, args.surface)
    except ValueError as exc:
        _PARSER.error(str(exc))
    match args.verb:
        case "run":
            return run_command(args, strategies, cases)
        case "list":
            return list_command(args, strategies, cases)
        case "check":
            return check_command(args, strategies, cases)
        case _:
            _PARSER.error(f"Unknown verb: {args.verb}")
    raise AssertionError("unreachable")

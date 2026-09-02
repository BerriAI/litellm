from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence
from pathlib import Path

from .catalog import load_catalog
from .models import HarnessCase, Strategy
from .runner import run_pytest
from .ui import make_dashboard

REPO_ROOT = Path(__file__).resolve().parents[2]
COVERAGE_ROOT = REPO_ROOT / "target" / "rust-python-harness"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rust-python-harness",
        description="Run Rust/Python parity tests with a live strategy-by-SDK-function dashboard.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="pick strategies and SDK functions in a guided terminal menu",
    )
    parser.add_argument(
        "--list", action="store_true", help="show the catalog without running tests"
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        metavar="ID",
        help="run only this strategy",
    )
    parser.add_argument(
        "--function",
        action="append",
        default=[],
        dest="sdk_functions",
        choices=("ocr", "messages", "responses", "count_tokens"),
        help="run only this SDK function",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="disable the interactive terminal dashboard",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="write Python reference LOC reports (HTML, JSON, and XML)",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="append an argument to pytest (repeatable, for example --pytest-arg=-x)",
    )
    return parser


def _coverage_pytest_args(output_root: Path = COVERAGE_ROOT) -> tuple[str, ...]:
    output_root.mkdir(parents=True, exist_ok=True)
    return (
        "--cov=litellm",
        "--cov-context=test",
        f"--cov-report=json:{output_root / 'python.json'}",
        f"--cov-report=xml:{output_root / 'python.xml'}",
        f"--cov-report=html:{output_root / 'python-html'}",
    )


def _pick_values(
    title: str, options: Sequence[tuple[str, str]], input_fn=input
) -> set[str]:
    print(f"\n{title} (Enter = all)")
    for index, (value, label) in enumerate(options, start=1):
        print(f"  {index:>2}. {label}  [{value}]")
    while True:
        answer = input_fn("Choose numbers, comma-separated: ").strip()
        if not answer:
            return set()
        try:
            indexes = {int(part.strip()) for part in answer.split(",")}
        except ValueError:
            print("Please enter numbers separated by commas.")
            continue
        if indexes and all(1 <= index <= len(options) for index in indexes):
            return {options[index - 1][0] for index in indexes}
        print(f"Choose values from 1 to {len(options)}.")


def _interactive_filters(strategies: Sequence[Strategy]) -> tuple[set[str], set[str]]:
    strategy_ids = _pick_values(
        "Testing strategies", [(strategy.id, strategy.label) for strategy in strategies]
    )
    sdk_functions = _pick_values(
        "SDK functions",
        [(name, name) for name in ("ocr", "messages", "responses", "count_tokens")],
    )
    return strategy_ids, sdk_functions


def _select(
    strategies: Sequence[Strategy], strategy_ids: set[str], sdk_functions: set[str]
) -> tuple[HarnessCase, ...]:
    known_ids = {strategy.id for strategy in strategies}
    unknown = strategy_ids - known_ids
    if unknown:
        raise ValueError(f"Unknown strategy: {', '.join(sorted(unknown))}")
    return tuple(
        case
        for strategy in strategies
        if not strategy_ids or strategy.id in strategy_ids
        for case in strategy.cases
        if not sdk_functions or case.sdk_function in sdk_functions
    )


def _print_catalog(strategies: Sequence[Strategy]) -> None:
    for strategy in strategies:
        print(f"{strategy.id:20} {strategy.label}")
        for case in strategy.cases:
            selectors = (
                ", ".join(case.selectors) if case.selectors else "no test configured"
            )
            print(f"  {case.sdk_function:12} {case.coverage.value:14} {selectors}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.coverage and importlib.util.find_spec("pytest_cov") is None:
        _parser().error(
            "--coverage requires the project's pytest-cov dependency; run with "
            "`poetry run python -m tests.rust-python-harness --coverage`"
        )
    strategies = load_catalog()
    if args.list:
        _print_catalog(strategies)
        return 0

    strategy_ids = set(args.strategy)
    sdk_functions = set(args.sdk_functions)
    if args.interactive:
        picked_strategies, picked_functions = _interactive_filters(strategies)
        strategy_ids = strategy_ids or picked_strategies
        sdk_functions = sdk_functions or picked_functions

    try:
        cases = _select(strategies, strategy_ids, sdk_functions)
    except ValueError as exc:
        _parser().error(str(exc))
    selected_strategy_ids = {case.strategy_id for case in cases}
    visible_strategies = tuple(
        strategy for strategy in strategies if strategy.id in selected_strategy_ids
    )
    dashboard = make_dashboard(
        visible_strategies,
        plain=args.plain,
        confidence_strategies=strategies,
    )
    pytest_args = [*args.pytest_arg]
    if args.coverage:
        pytest_args.extend(_coverage_pytest_args())
    with dashboard:
        exit_code, run = run_pytest(
            cases=cases,
            repo_root=REPO_ROOT,
            on_update=dashboard.update,
            pytest_args=pytest_args,
        )
        dashboard.finish(run, exit_code)
    if args.coverage and (COVERAGE_ROOT / "python.json").exists():
        print(f"Python LOC heatmap: {COVERAGE_ROOT / 'python-html' / 'index.html'}")
        print(f"Machine-readable coverage: {COVERAGE_ROOT / 'python.json'}")
    return exit_code

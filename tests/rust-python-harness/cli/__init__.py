from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Final, cast

import click

from ..shared.reporting.models import SDK_FUNCTIONS, SdkFunction, Strategy, Surface
from .catalog import load_catalog
from .commands import run_command, select_cases

__all__ = ["load_catalog", "main"]

_INTERRUPTED_EXIT_CODE: Final = 130


def _function_option() -> click.Option:
    return click.Option(
        ("--function", "sdk_functions"),
        type=click.Choice(SDK_FUNCTIONS),
        multiple=True,
        help="run only this SDK function; repeat to select more than one",
    )


def _run_all_command(strategies: Sequence[Strategy]) -> click.Command:
    def run_all(sdk_functions: tuple[str, ...]) -> int:
        selected_functions: Final = cast(frozenset[SdkFunction], frozenset(sdk_functions))
        cases: Final = select_cases(strategies, selected_functions)
        return run_command(strategies, cases)

    return click.Command(
        "all",
        params=[_function_option()],
        callback=run_all,
        help="run every strategy",
    )


def _strategy_command(strategy: Strategy) -> click.Command:
    params: list[click.Parameter] = [_function_option()]
    if strategy.definition.surfaces:
        params.append(
            click.Option(
                ("--surface",),
                type=click.Choice(strategy.definition.surfaces),
                help="run only this API surface; omit to run every surface",
            )
        )
    runner_argument: Final = strategy.definition.runner_argument
    if runner_argument is not None:
        params.append(
            click.Option(
                (runner_argument.option, "runner_args"),
                multiple=True,
                metavar=runner_argument.metavar,
                help=runner_argument.help,
            )
        )

    def run_strategy(
        sdk_functions: tuple[str, ...],
        surface: str | None = None,
        runner_args: tuple[str, ...] = (),
    ) -> int:
        selected_functions: Final = cast(frozenset[SdkFunction], frozenset(sdk_functions))
        selected_surface: Final = cast(Surface | None, surface)
        cases: Final = select_cases((strategy,), selected_functions, selected_surface)
        return run_command((strategy,), cases, runner_args)

    return click.Command(
        strategy.id,
        params=params,
        callback=run_strategy,
        help=strategy.description,
    )


def _build_cli(strategies: Sequence[Strategy]) -> click.Group:
    root: Final = click.Group(
        "rust-python-harness",
        help="Run Rust/Python parity tests with raw progress and strategy reports.",
    )
    run: Final = click.Group("run", help="run one strategy or the complete harness")
    run.add_command(_run_all_command(strategies))
    for strategy in strategies:
        run.add_command(_strategy_command(strategy))
    root.add_command(run)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    try:
        strategies: Final = load_catalog()
        result: Final = _build_cli(strategies).main(
            args=None if argv is None else list(argv),
            prog_name="rust-python-harness",
            standalone_mode=False,
        )
        exit_code: Final = result if isinstance(result, int) else 0
    except click.ClickException as error:
        error.show()
        return error.exit_code
    except click.Abort:
        click.echo("Aborted!", err=True)
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted\n")
        return _INTERRUPTED_EXIT_CODE
    if exit_code == _INTERRUPTED_EXIT_CODE:
        sys.stderr.write("Interrupted\n")
    return exit_code

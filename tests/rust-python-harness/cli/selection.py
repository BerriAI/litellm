from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from ..shared.reporting.models import SDK_FUNCTIONS, HarnessCase, Strategy


def pick_values(
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


def interactive_filters(
    strategies: Sequence[Strategy],
) -> tuple[set[str], set[str]]:
    strategy_ids = pick_values(
        "Testing strategies", [(strategy.id, strategy.label) for strategy in strategies]
    )
    sdk_functions = pick_values(
        "SDK functions",
        [(name, name) for name in SDK_FUNCTIONS],
    )
    return strategy_ids, sdk_functions


def select(
    strategies: Sequence[Strategy],
    strategy_ids: set[str],
    sdk_functions: set[str],
    surface: str | None = None,
) -> tuple[HarnessCase, ...]:
    known_ids: Final = {strategy.id for strategy in strategies}
    unknown: Final = strategy_ids - known_ids
    if unknown:
        raise ValueError(f"Unknown strategy: {', '.join(sorted(unknown))}")
    return tuple(
        case
        for strategy in strategies
        if not strategy_ids or strategy.id in strategy_ids
        for case in strategy.cases
        if (not sdk_functions or case.sdk_function in sdk_functions)
        and (surface is None or case.surface == surface)
    )

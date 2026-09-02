from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Coverage, HarnessCase, SDK_FUNCTIONS, Strategy

STRATEGIES_ROOT = Path(__file__).parent / "strategies"


def _require_string(value: Any, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} must be a non-empty string")
    return value


def _load_strategy(source: Path) -> Strategy:
    with source.open(encoding="utf-8") as stream:
        data = json.load(stream)

    strategy_id = _require_string(data.get("id"), "id", source)
    label = _require_string(data.get("label"), "label", source)
    description = _require_string(data.get("description"), "description", source)
    order = data.get("order")
    if not isinstance(order, int):
        raise ValueError(f"{source}: order must be an integer")
    function_data = data.get("functions")
    if not isinstance(function_data, dict):
        raise ValueError(f"{source}: functions must be an object")

    missing = set(SDK_FUNCTIONS) - set(function_data)
    extra = set(function_data) - set(SDK_FUNCTIONS)
    if missing or extra:
        raise ValueError(
            f"{source}: functions must exactly match {SDK_FUNCTIONS}; missing={missing}, extra={extra}"
        )

    cases: list[HarnessCase] = []
    for sdk_function in SDK_FUNCTIONS:
        case_data = function_data[sdk_function]
        if not isinstance(case_data, dict):
            raise ValueError(f"{source}: functions.{sdk_function} must be an object")
        try:
            coverage = Coverage(case_data.get("coverage"))
        except ValueError as exc:
            raise ValueError(f"{source}: invalid coverage for {sdk_function}") from exc
        selectors = case_data.get("selectors", [])
        if not isinstance(selectors, list) or not all(
            isinstance(item, str) and item for item in selectors
        ):
            raise ValueError(
                f"{source}: selectors for {sdk_function} must be a list of strings"
            )
        if coverage is Coverage.NOT_APPLICABLE and selectors:
            raise ValueError(
                f"{source}: not_applicable case {sdk_function} cannot have selectors"
            )
        cases.append(
            HarnessCase(
                strategy_id=strategy_id,
                strategy_label=label,
                sdk_function=sdk_function,
                coverage=coverage,
                selectors=tuple(selectors),
                note=str(case_data.get("note", "")),
            )
        )

    return Strategy(
        order=order,
        id=strategy_id,
        label=label,
        description=description,
        directory=source.parent,
        cases=tuple(cases),
    )


def load_catalog(root: Path = STRATEGIES_ROOT) -> tuple[Strategy, ...]:
    sources = sorted(root.glob("*/strategy.json"))
    if not sources:
        raise ValueError(f"No strategy manifests found below {root}")
    strategies = tuple(
        sorted(
            (_load_strategy(source) for source in sources),
            key=lambda strategy: strategy.order,
        )
    )
    ids = [strategy.id for strategy in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate strategy id in {root}")
    return strategies

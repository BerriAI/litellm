from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Coverage, HarnessCase, SDK_FUNCTIONS, Strategy

STRATEGIES_ROOT = Path(__file__).parent


def _require_string(value: Any, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} must be a non-empty string")
    return value


def _load_variants(data: dict[str, Any], source: Path) -> tuple[tuple[str | None, dict[str, str]], ...]:
    """Parse the optional `variants` manifest key.

    Absent `variants` yields a single unnamed variant with no environment
    overrides, so a manifest that never declares variants keeps its plain
    strategy id (backward compatible with the original three strategies).
    A declared `variants` list always suffixes the resulting strategy ids
    with `__<name>`, even for a single entry, so a manifest opting into the
    mechanism is unambiguous about which cells came from which environment.
    """
    raw = data.get("variants")
    if raw is None:
        return ((None, {}),)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source}: variants must be a non-empty list")
    variants: list[tuple[str | None, dict[str, str]]] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: each variant must be an object")
        name = _require_string(entry.get("name"), "variants[].name", source)
        if name in seen:
            raise ValueError(f"{source}: duplicate variant name {name!r}")
        seen.add(name)
        env = entry.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError(
                f"{source}: variants[].env must be a string-to-string mapping"
            )
        variants.append((name, dict(env)))
    return tuple(variants)


def _load_strategies(source: Path) -> tuple[Strategy, ...]:
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

    parsed_functions: dict[str, tuple[Coverage, tuple[str, ...], str]] = {}
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
        parsed_functions[sdk_function] = (
            coverage,
            tuple(selectors),
            str(case_data.get("note", "")),
        )

    strategies: list[Strategy] = []
    for variant_name, env in _load_variants(data, source):
        variant_id = strategy_id if variant_name is None else f"{strategy_id}__{variant_name}"
        variant_label = label if variant_name is None else f"{label} [{variant_name}]"
        cases = tuple(
            HarnessCase(
                strategy_id=variant_id,
                strategy_label=variant_label,
                sdk_function=sdk_function,
                coverage=coverage,
                selectors=selectors,
                note=note,
            )
            for sdk_function, (coverage, selectors, note) in parsed_functions.items()
        )
        strategies.append(
            Strategy(
                order=order,
                id=variant_id,
                label=variant_label,
                description=description,
                directory=source.parent,
                cases=cases,
                env=env,
            )
        )
    return tuple(strategies)


def load_catalog(root: Path = STRATEGIES_ROOT) -> tuple[Strategy, ...]:
    sources = sorted(root.glob("*/strategy.json"))
    if not sources:
        raise ValueError(f"No strategy manifests found below {root}")
    strategies = tuple(
        sorted(
            (
                strategy
                for source in sources
                for strategy in _load_strategies(source)
            ),
            key=lambda strategy: strategy.order,
        )
    )
    ids = [strategy.id for strategy in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate strategy id in {root}")
    return strategies

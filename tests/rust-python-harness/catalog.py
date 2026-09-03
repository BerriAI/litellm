from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from .shared.reporting.models import Coverage, HarnessCase, SDK_FUNCTIONS, Strategy

STRATEGIES_ROOT: Final = Path(__file__).parent / "strategies"


class CaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: Coverage
    selectors: tuple[str, ...] = ()
    note: str = ""
    unit_suite: str | None = None


class StrategySpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    order: int
    id: str
    label: str
    description: str
    functions: dict[str, CaseSpec]
    gateway: dict[str, CaseSpec] = {}


def _load_strategy(source: Path) -> Strategy:
    data: Final = StrategySpec.model_validate_json(source.read_text(encoding="utf-8"))
    if set(data.functions) != set(SDK_FUNCTIONS):
        raise ValueError(f"{source}: functions must exactly match {SDK_FUNCTIONS}")
    cases: Final = tuple(
        HarnessCase(
            strategy_id=data.id,
            strategy_label=data.label,
            sdk_function=name,
            coverage=case.coverage,
            selectors=case.selectors,
            note=case.note,
            surface=surface,
            unit_suite=case.unit_suite,
        )
        for surface, functions in (("sdk", data.functions), ("gateway", data.gateway))
        for name in (SDK_FUNCTIONS if surface == "sdk" else functions)
        for case in (functions[name],)
    )
    for case in cases:
        if case.coverage in {Coverage.PLANNED, Coverage.NOT_APPLICABLE} and (case.selectors or case.unit_suite):
            raise ValueError(f"{source}: {case.coverage.value} case {case.key} cannot configure tests")
        if any(not selector.strip() for selector in case.selectors):
            raise ValueError(f"{source}: empty selector in {case.key}")
        if data.id == "unit_tests" and case.selectors:
            raise ValueError(f"{source}: unit_tests must configure unit_suite instead of pytest selectors")
        if data.id != "unit_tests" and case.unit_suite:
            raise ValueError(f"{source}: unit_suite is only valid for unit_tests")
    return Strategy(data.order, data.id, data.label, data.description, source.parent, cases)


def load_catalog(root: Path = STRATEGIES_ROOT) -> tuple[Strategy, ...]:
    sources: Final = tuple(sorted(root.glob("*/strategy.json")))
    if not sources:
        raise ValueError(f"No strategy manifests found below {root}")
    try:
        strategies: Final = tuple(sorted((_load_strategy(source) for source in sources), key=lambda item: item.order))
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError(str(error)) from error
    if len({strategy.id for strategy in strategies}) != len(strategies):
        raise ValueError(f"Duplicate strategy id in {root}")
    return strategies

from __future__ import annotations

from pathlib import Path
from typing import Final

from ...shared.reporting.models import SDK_FUNCTIONS, Coverage
from ...shared.reporting.strategy import (
    CaseDefinition,
    NotImplementedCaseSpec,
    StrategyDefinition,
    SuiteCaseSpec,
)
from .mappings import MAPPING_SUITES
from .reporting import render_mapping_results
from .runner import run_mapping_cases


CASES: Final[tuple[CaseDefinition, ...]] = (
    *(
        CaseDefinition(
            sdk_function,
            SuiteCaseSpec(coverage=Coverage.COMPLETE, suite=sdk_function)
            if sdk_function in MAPPING_SUITES
            else NotImplementedCaseSpec(reason=f"No {sdk_function} unit-test mapping is registered."),
        )
        for sdk_function in SDK_FUNCTIONS
    ),
)

STRATEGY: Final = StrategyDefinition(
    id="unit_tests_mapping",
    order=30,
    label="Unit test mapping",
    description="Validate Python/Rust unit-test mappings against collected test inventories.",
    directory=Path(__file__).parent,
    runnable_spec=SuiteCaseSpec,
    cases=CASES,
    run=run_mapping_cases,
    render=render_mapping_results,
)

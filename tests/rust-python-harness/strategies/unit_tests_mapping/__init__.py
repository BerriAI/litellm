from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final

from ...shared.reporting.models import SDK_FUNCTIONS, Coverage
from ...shared.reporting.strategy import (
    CaseDefinition,
    NotImplementedCaseSpec,
    RunnerArgumentDefinition,
    StrategyDefinition,
    SuiteCaseSpec,
)
from ...shared.unit_runners.suite_runner import run_suites
from .mappings import UNIT_TEST_CONTRACTS
from .reporting import render_mapping_results
from .runner import run_suite


CASES: Final[tuple[CaseDefinition, ...]] = (
    *(
        CaseDefinition(
            sdk_function,
            SuiteCaseSpec(coverage=Coverage.COMPLETE, suite=sdk_function)
            if sdk_function in UNIT_TEST_CONTRACTS
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
    run=partial(run_suites, suites=UNIT_TEST_CONTRACTS, execute=run_suite),
    render=render_mapping_results,
    runner_argument=RunnerArgumentDefinition(
        option="--detail",
        metavar="MODE",
        help="show individual test names; any value enables full detail",
    ),
)

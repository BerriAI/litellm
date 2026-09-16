from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final

from ...shared.reporting.models import SDK_FUNCTIONS, Coverage, SdkFunction
from ...shared.reporting.strategy import (
    CaseDefinition,
    NotImplementedCaseSpec,
    RunnerArgumentDefinition,
    StrategyDefinition,
    SuiteCaseSpec,
)
from ...shared.unit_runners.suite_runner import run_suites
from .reporting import render_unit_parity_results
from .runner import UnitParityExclusion, UnitParitySuite, run_suite

UNIT_PARITY_SUITES: Final[Mapping[SdkFunction, UnitParitySuite]] = MappingProxyType(
    {
        "ocr": UnitParitySuite(
            python_selectors=(
                "tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py",
                "tests/test_litellm/llms/mistral/ocr",
                "tests/test_litellm/llms/ocr",
                "tests/test_litellm/ocr",
            ),
            exclusions=(
                UnitParityExclusion(
                    nodeid="tests/test_litellm/ocr/test_rust_bridge.py::test_rust_toggles_flag",
                    reason="This test asserts the process-level backend flag selected by the parity runner.",
                ),
            ),
        ),
    }
)


CASES: Final[tuple[CaseDefinition, ...]] = (
    *(
        CaseDefinition(
            sdk_function,
            SuiteCaseSpec(coverage=Coverage.COMPLETE, suite=sdk_function)
            if sdk_function in UNIT_PARITY_SUITES
            else NotImplementedCaseSpec(reason=f"No {sdk_function} unit-test parity suite is registered."),
        )
        for sdk_function in SDK_FUNCTIONS
    ),
)

STRATEGY: Final = StrategyDefinition(
    id="unit_tests_parity",
    order=31,
    label="Unit test parity",
    description=(
        "Run existing Python unit tests with LITELLM_RUST disabled and enabled and require matching outcomes."
    ),
    directory=Path(__file__).parent,
    runnable_spec=SuiteCaseSpec,
    cases=CASES,
    run=partial(run_suites, suites=UNIT_PARITY_SUITES, execute=run_suite),
    render=render_unit_parity_results,
    runner_argument=RunnerArgumentDefinition(
        option="--pytest-arg",
        help="append an argument to both Python and Rust-backed pytest runs",
    ),
)

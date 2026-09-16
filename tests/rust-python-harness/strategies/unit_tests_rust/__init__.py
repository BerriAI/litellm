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
    StrategyDefinition,
    SuiteCaseSpec,
)
from ...shared.unit_runners.contracts import UNIT_TEST_CONTRACTS
from ...shared.unit_runners.suite_runner import run_suites
from .reporting import render_rust_unit_results
from .runner import RustSuite, run_suite

RUST_SUITES: Final[Mapping[SdkFunction, RustSuite]] = MappingProxyType(
    {"ocr": RustSuite(cargo_manifest="litellm-rust/Cargo.toml", cargo_filter="ocr")}
)


CASES: Final[tuple[CaseDefinition, ...]] = (
    *(
        CaseDefinition(
            sdk_function,
            SuiteCaseSpec(coverage=Coverage.COMPLETE, suite=sdk_function)
            if sdk_function in RUST_SUITES
            else NotImplementedCaseSpec(reason=f"No focused {sdk_function} Rust unit suite is registered."),
        )
        for sdk_function in SDK_FUNCTIONS
    ),
)

STRATEGY: Final = StrategyDefinition(
    id="unit_tests_rust",
    order=32,
    label="Unit test Rust",
    description="Run the focused native Cargo test suite for each mapped API.",
    directory=Path(__file__).parent,
    runnable_spec=SuiteCaseSpec,
    cases=CASES,
    run=partial(run_suites, suites=RUST_SUITES, execute=run_suite),
    render=render_rust_unit_results,
)

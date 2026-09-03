from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final

from ...shared.reporting.rendering import render_outcomes
from ...shared.reporting.strategy import (
    CaseDefinition,
    NotImplementedCaseSpec,
    SkippedCaseSpec,
    StrategyDefinition,
    SuiteCaseSpec,
)
from ...shared.unit_runners.suite_runner import run_suites
from .runner import RustSuite, run_suite


def _load_suite(path: Path) -> RustSuite:
    return RustSuite.model_validate_json(path.read_text(encoding="utf-8"))


CASES: Final[tuple[CaseDefinition, ...]] = (
    CaseDefinition("sdk", "ocr", NotImplementedCaseSpec(reason="No focused OCR Rust unit suite is registered.")),
    CaseDefinition(
        "sdk", "messages", NotImplementedCaseSpec(reason="No focused Messages Rust unit suite is registered.")
    ),
    CaseDefinition(
        "sdk", "responses", NotImplementedCaseSpec(reason="No focused Responses Rust unit suite is registered.")
    ),
    CaseDefinition(
        "sdk", "count_tokens", NotImplementedCaseSpec(reason="No focused token-count Rust unit suite is registered.")
    ),
    CaseDefinition(
        "sdk",
        "chat_completions",
        NotImplementedCaseSpec(reason="No focused chat-completions Rust unit suite is registered."),
    ),
    CaseDefinition(
        "sdk", "transcription", NotImplementedCaseSpec(reason="No focused transcription Rust unit suite is registered.")
    ),
    CaseDefinition("gateway", "ocr", SkippedCaseSpec(reason="Native Rust unit tests are not gateway execution.")),
    CaseDefinition("gateway", "messages", SkippedCaseSpec(reason="Native Rust unit tests are not gateway execution.")),
    CaseDefinition("gateway", "responses", SkippedCaseSpec(reason="Native Rust unit tests are not gateway execution.")),
    CaseDefinition(
        "gateway", "count_tokens", SkippedCaseSpec(reason="Native Rust unit tests are not gateway execution.")
    ),
    CaseDefinition(
        "gateway", "chat_completions", SkippedCaseSpec(reason="Native Rust unit tests are not gateway execution.")
    ),
    CaseDefinition(
        "gateway", "transcription", SkippedCaseSpec(reason="Native Rust unit tests are not gateway execution.")
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
    run=partial(run_suites, load=_load_suite, execute=run_suite),
    render=render_outcomes,
)

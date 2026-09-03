from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final

from ...shared.reporting.strategy import (
    CaseDefinition,
    NotImplementedCaseSpec,
    SkippedCaseSpec,
    StrategyDefinition,
    SuiteCaseSpec,
)
from ...shared.unit_runners.suite_runner import run_suites
from .reporting import render_unit_parity_results
from .runner import UnitParitySuite, run_suite


def _load_suite(path: Path) -> UnitParitySuite:
    return UnitParitySuite.model_validate_json(path.read_text(encoding="utf-8"))


CASES: Final[tuple[CaseDefinition, ...]] = (
    CaseDefinition("sdk", "ocr", NotImplementedCaseSpec(reason="No OCR unit-test parity suite is registered.")),
    CaseDefinition(
        "sdk", "messages", NotImplementedCaseSpec(reason="No Messages unit-test parity suite is registered.")
    ),
    CaseDefinition(
        "sdk", "responses", NotImplementedCaseSpec(reason="No Responses unit-test parity suite is registered.")
    ),
    CaseDefinition(
        "sdk", "count_tokens", NotImplementedCaseSpec(reason="No token-count unit-test parity suite is registered.")
    ),
    CaseDefinition(
        "sdk",
        "chat_completions",
        NotImplementedCaseSpec(reason="No chat-completions unit-test parity suite is registered."),
    ),
    CaseDefinition(
        "sdk", "transcription", NotImplementedCaseSpec(reason="No transcription unit-test parity suite is registered.")
    ),
    CaseDefinition("gateway", "ocr", SkippedCaseSpec(reason="Unit-test parity is not gateway-surface execution.")),
    CaseDefinition(
        "gateway", "messages", SkippedCaseSpec(reason="Unit-test parity is not gateway-surface execution.")
    ),
    CaseDefinition(
        "gateway", "responses", SkippedCaseSpec(reason="Unit-test parity is not gateway-surface execution.")
    ),
    CaseDefinition(
        "gateway", "count_tokens", SkippedCaseSpec(reason="Unit-test parity is not gateway-surface execution.")
    ),
    CaseDefinition(
        "gateway", "chat_completions", SkippedCaseSpec(reason="Unit-test parity is not gateway-surface execution.")
    ),
    CaseDefinition(
        "gateway", "transcription", SkippedCaseSpec(reason="Unit-test parity is not gateway-surface execution.")
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
    run=partial(run_suites, load=_load_suite, execute=run_suite),
    render=render_unit_parity_results,
)

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
from .ledger_report import check_ledgers
from .runner import UnitSuite, run_suite


def _load_suite(path: Path) -> UnitSuite:
    return UnitSuite.model_validate_json(path.read_text(encoding="utf-8"))


CASES: Final[tuple[CaseDefinition, ...]] = (
    CaseDefinition("sdk", "ocr", NotImplementedCaseSpec(reason="No OCR unit-test mapping suite is registered.")),
    CaseDefinition(
        "sdk", "messages", NotImplementedCaseSpec(reason="No Messages unit-test mapping suite is registered.")
    ),
    CaseDefinition(
        "sdk", "responses", NotImplementedCaseSpec(reason="No Responses unit-test mapping suite is registered.")
    ),
    CaseDefinition(
        "sdk", "count_tokens", NotImplementedCaseSpec(reason="No token-count unit-test mapping suite is registered.")
    ),
    CaseDefinition(
        "sdk",
        "chat_completions",
        NotImplementedCaseSpec(reason="No chat-completions unit-test mapping suite is registered."),
    ),
    CaseDefinition(
        "sdk", "transcription", NotImplementedCaseSpec(reason="No transcription unit-test mapping suite is registered.")
    ),
    CaseDefinition("gateway", "ocr", SkippedCaseSpec(reason="Unit-test mapping is not gateway-surface execution.")),
    CaseDefinition(
        "gateway", "messages", SkippedCaseSpec(reason="Unit-test mapping is not gateway-surface execution.")
    ),
    CaseDefinition(
        "gateway", "responses", SkippedCaseSpec(reason="Unit-test mapping is not gateway-surface execution.")
    ),
    CaseDefinition(
        "gateway", "count_tokens", SkippedCaseSpec(reason="Unit-test mapping is not gateway-surface execution.")
    ),
    CaseDefinition(
        "gateway", "chat_completions", SkippedCaseSpec(reason="Unit-test mapping is not gateway-surface execution.")
    ),
    CaseDefinition(
        "gateway", "transcription", SkippedCaseSpec(reason="Unit-test mapping is not gateway-surface execution.")
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
    run=partial(run_suites, load=_load_suite, execute=run_suite),
    render=render_outcomes,
    check=check_ledgers,
)

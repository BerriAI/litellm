from pathlib import Path
from typing import Final

from ...shared.reporting.models import Coverage
from ...shared.reporting.strategy import (
    CaseDefinition,
    ModuleCaseSpec,
    NotImplementedCaseSpec,
    StrategyDefinition,
)
from .reporting import render_e2e_results
from .runner import run_e2e_cases

CASES: Final[tuple[CaseDefinition, ...]] = (
    CaseDefinition(
        "sdk",
        "ocr",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.e2e_parity.sdk.ocr.test_sdk_parity",
            note=(
                "Recorded sync/async SDK parity; invalid-model provider errors differ, "
                "and Reducto lacks a Rust contract."
            ),
        ),
    ),
    CaseDefinition(
        "sdk",
        "messages",
        NotImplementedCaseSpec(
            reason="Bridge unit tests exist, but no standalone end-to-end parity case is registered."
        ),
    ),
    CaseDefinition(
        "sdk",
        "responses",
        NotImplementedCaseSpec(
            reason="Bridge unit tests exist, but no standalone end-to-end parity case is registered."
        ),
    ),
    CaseDefinition(
        "sdk",
        "count_tokens",
        NotImplementedCaseSpec(reason="No Rust count_tokens parity test is present yet."),
    ),
    CaseDefinition(
        "sdk",
        "chat_completions",
        NotImplementedCaseSpec(
            reason="Bridge unit tests exist, but no standalone end-to-end parity case is registered."
        ),
    ),
    CaseDefinition(
        "sdk",
        "transcription",
        NotImplementedCaseSpec(
            reason="Bridge unit tests exist, but no standalone end-to-end parity case is registered."
        ),
    ),
    CaseDefinition(
        "gateway",
        "ocr",
        NotImplementedCaseSpec(reason="No gateway end-to-end OCR parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "messages",
        NotImplementedCaseSpec(reason="No gateway end-to-end Messages parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "responses",
        NotImplementedCaseSpec(reason="No gateway end-to-end Responses parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "count_tokens",
        NotImplementedCaseSpec(reason="No gateway end-to-end token-count parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "chat_completions",
        NotImplementedCaseSpec(reason="No gateway end-to-end chat parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "transcription",
        NotImplementedCaseSpec(reason="No gateway end-to-end transcription parity case is registered."),
    ),
)

STRATEGY: Final = StrategyDefinition(
    id="e2e_parity",
    order=10,
    label="End-to-end parity",
    description="Compare observable Python and Rust behavior over generated and recorded inputs.",
    directory=Path(__file__).parent,
    runnable_spec=ModuleCaseSpec,
    cases=CASES,
    run=run_e2e_cases,
    render=render_e2e_results,
)

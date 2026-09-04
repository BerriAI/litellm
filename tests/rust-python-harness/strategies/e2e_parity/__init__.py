from pathlib import Path
from typing import Final

from ...shared.reporting.models import SURFACES, Coverage
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
        "ocr",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.e2e_parity.sdk.ocr.test_sdk_parity",
            note=(
                "Recorded sync/async SDK parity; invalid-model provider errors differ, "
                "and Reducto lacks a Rust contract."
            ),
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "messages",
        NotImplementedCaseSpec(
            reason="Bridge unit tests exist, but no standalone end-to-end parity case is registered."
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "responses",
        NotImplementedCaseSpec(
            reason="Bridge unit tests exist, but no standalone end-to-end parity case is registered."
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "count_tokens",
        NotImplementedCaseSpec(reason="No Rust count_tokens parity test is present yet."),
        surface="sdk",
    ),
    CaseDefinition(
        "chat_completions",
        NotImplementedCaseSpec(
            reason="Bridge unit tests exist, but no standalone end-to-end parity case is registered."
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "transcription",
        NotImplementedCaseSpec(
            reason="Bridge unit tests exist, but no standalone end-to-end parity case is registered."
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "ocr",
        NotImplementedCaseSpec(reason="No gateway end-to-end OCR parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "messages",
        NotImplementedCaseSpec(reason="No gateway end-to-end Messages parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "responses",
        NotImplementedCaseSpec(reason="No gateway end-to-end Responses parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "count_tokens",
        NotImplementedCaseSpec(reason="No gateway end-to-end token-count parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "chat_completions",
        NotImplementedCaseSpec(reason="No gateway end-to-end chat parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "transcription",
        NotImplementedCaseSpec(reason="No gateway end-to-end transcription parity case is registered."),
        surface="gateway",
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
    surfaces=SURFACES,
)

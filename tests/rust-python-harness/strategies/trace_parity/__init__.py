from pathlib import Path
from typing import Final

from ...shared.reporting.models import Coverage
from ...shared.reporting.strategy import (
    CaseDefinition,
    ModuleCaseSpec,
    NotImplementedCaseSpec,
    StrategyDefinition,
)
from .reporting import render_trace_results
from .runner import run_trace_cases

CASES: Final[tuple[CaseDefinition, ...]] = (
    CaseDefinition(
        "sdk",
        "ocr",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.sdk.ocr.test_trace_parity",
        ),
    ),
    CaseDefinition(
        "sdk",
        "messages",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.sdk.messages.test_trace_parity",
            note="Async only until anthropic_messages_handler supports sync calls.",
        ),
    ),
    CaseDefinition(
        "sdk",
        "responses",
        NotImplementedCaseSpec(reason="No Responses trace-parity case is registered."),
    ),
    CaseDefinition(
        "sdk",
        "count_tokens",
        NotImplementedCaseSpec(reason="No token-count trace-parity case is registered."),
    ),
    CaseDefinition(
        "sdk",
        "chat_completions",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.sdk.chat_completions.test_trace_parity",
        ),
    ),
    CaseDefinition(
        "sdk",
        "transcription",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.sdk.transcription.test_trace_parity",
            note=(
                "The Python SDK delegates this provider to the Rust pipeline, so only dispatch is visible "
                "to the Python profiler."
            ),
        ),
    ),
    CaseDefinition(
        "gateway",
        "ocr",
        NotImplementedCaseSpec(reason="No gateway OCR trace-parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "messages",
        NotImplementedCaseSpec(reason="No gateway Messages trace-parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "responses",
        NotImplementedCaseSpec(reason="No gateway Responses trace-parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "count_tokens",
        NotImplementedCaseSpec(reason="No gateway token-count trace-parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "chat_completions",
        NotImplementedCaseSpec(reason="No gateway chat trace-parity case is registered."),
    ),
    CaseDefinition(
        "gateway",
        "transcription",
        NotImplementedCaseSpec(reason="No gateway transcription trace-parity case is registered."),
    ),
)

STRATEGY: Final = StrategyDefinition(
    id="trace_parity",
    order=20,
    label="Trace parity",
    description="Compare mapped operations, call counts, and required execution ordering.",
    directory=Path(__file__).parent,
    runnable_spec=ModuleCaseSpec,
    cases=CASES,
    run=run_trace_cases,
    render=render_trace_results,
)

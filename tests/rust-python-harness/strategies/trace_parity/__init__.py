from pathlib import Path
from typing import Final

from ...shared.reporting.models import SURFACES, Coverage
from ...shared.reporting.strategy import (
    CaseDefinition,
    ModuleCaseSpec,
    NotImplementedCaseSpec,
    RunnerArgumentDefinition,
    StrategyDefinition,
)
from .reporting import render_trace_results
from .runner import run_trace_cases

CASES: Final[tuple[CaseDefinition, ...]] = (
    CaseDefinition(
        "ocr",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.sdk.ocr.case",
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "messages",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.sdk.messages.case",
            note="Async only until anthropic_messages_handler supports sync calls.",
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "responses",
        NotImplementedCaseSpec(reason="No Responses trace-parity case is registered."),
        surface="sdk",
    ),
    CaseDefinition(
        "count_tokens",
        NotImplementedCaseSpec(reason="No token-count trace-parity case is registered."),
        surface="sdk",
    ),
    CaseDefinition(
        "chat_completions",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.sdk.chat_completions.case",
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "transcription",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.sdk.transcription.case",
            note=(
                "The Python SDK delegates this provider to the Rust pipeline, so only dispatch is visible "
                "to the Python profiler."
            ),
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "ocr",
        NotImplementedCaseSpec(reason="No gateway OCR trace-parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "messages",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.gateway.messages.case",
            note="Non-streaming success paths only.",
        ),
        surface="gateway",
    ),
    CaseDefinition(
        "responses",
        NotImplementedCaseSpec(reason="No gateway Responses trace-parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "count_tokens",
        NotImplementedCaseSpec(reason="No gateway token-count trace-parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "chat_completions",
        NotImplementedCaseSpec(reason="No gateway chat trace-parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "transcription",
        NotImplementedCaseSpec(reason="No gateway transcription trace-parity case is registered."),
        surface="gateway",
    ),
)

STRATEGY: Final = StrategyDefinition(
    id="trace_parity",
    order=20,
    label="Trace parity",
    description="Compare pipeline steps, order, and nesting between Python profiler frames and Rust spans via an explicit mapping.",
    directory=Path(__file__).parent,
    runnable_spec=ModuleCaseSpec,
    cases=CASES,
    run=run_trace_cases,
    render=render_trace_results,
    surfaces=SURFACES,
    runner_argument=RunnerArgumentDefinition(
        option="--scenario",
        metavar="NAME",
        help="run only this named trace scenario; repeat to select more than one",
    ),
)

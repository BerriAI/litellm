from pathlib import Path
from typing import Final

from ...shared.reporting.models import SURFACES, Coverage
from ...shared.reporting.strategy import (
    CaseDefinition,
    ModuleCaseSpec,
    NotImplementedCaseSpec,
    RunnerArgumentDefinition,
    RunnerOptionDefinition,
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
            note="Success paths are async; sync tracing captures the currently unsupported behavior.",
        ),
        surface="sdk",
    ),
    CaseDefinition(
        "responses",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.sdk.responses.case",
            note="Core create paths: native, streaming, provider error, Azure override, and chat bridge.",
        ),
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
            note="Anthropic/Azure provider routes plus a fully consumed downstream streaming path.",
        ),
        surface="gateway",
    ),
    CaseDefinition(
        "responses",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.gateway.responses.case",
            note="Native OpenAI non-streaming and fully consumed downstream streaming paths.",
        ),
        surface="gateway",
    ),
    CaseDefinition(
        "count_tokens",
        NotImplementedCaseSpec(reason="No gateway token-count trace-parity case is registered."),
        surface="gateway",
    ),
    CaseDefinition(
        "chat_completions",
        ModuleCaseSpec(
            coverage=Coverage.PARTIAL,
            module="tests.rust-python-harness.strategies.trace_parity.gateway.chat_completions.case",
            note="Anthropic non-streaming and fully consumed downstream streaming paths.",
        ),
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
    label="Traces",
    description="Print Python profiler frames and Rust spans for representative pipeline scenarios.",
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
    runner_options=(
        RunnerOptionDefinition(
            option="--engine",
            choices=("python", "rust"),
            help="show only this engine's trace; omit to print both engines",
        ),
    ),
)

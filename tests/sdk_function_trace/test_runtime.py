from __future__ import annotations

from typing import Final

import pytest

from tests.sdk_function_trace.runtime import (
    TraceFailed,
    TraceSkipped,
    attempt_trace,
    run_trace,
    trace_diff,
)
from tests.sdk_function_trace.steps import pipeline_issues, pipeline_steps


def test_sync_messages_records_the_known_python_limitation() -> None:
    result: Final = attempt_trace("messages", engine="python", asynchronous=False)

    assert isinstance(result, TraceSkipped)
    assert result.reason == "ValueError: anthropic_messages_handler is not implemented for sync calls"


def test_unexpected_call_failure_is_not_skipped() -> None:
    result: Final = attempt_trace("unknown", engine="python", asynchronous=False)

    assert isinstance(result, TraceFailed)
    assert result.reason == "ValueError: Unknown route: unknown"


@pytest.mark.parametrize(
    ("route", "asynchronous"),
    (("chat_completions", False), ("chat_completions", True), ("messages", True), ("ocr", False), ("ocr", True)),
)
def test_compiled_routes_match_python_steps(route: str, asynchronous: bool) -> None:
    from litellm.rust_bridge import get_native_bridge

    if get_native_bridge() is None:
        pytest.skip("build the native bridge to run executed route parity")
    python: Final = pipeline_steps(route, "python", run_trace(route, engine="python", asynchronous=asynchronous))
    rust: Final = pipeline_steps(route, "rust", run_trace(route, engine="rust", asynchronous=asynchronous))

    assert pipeline_issues(route, "python", python) == ()
    assert pipeline_issues(route, "rust", rust) == ()
    assert trace_diff(python, rust).matches
    if route != "messages":
        assert python == rust

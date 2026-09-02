from __future__ import annotations

from typing import Final

from tests.sdk_function_trace.runtime import attempt_trace


def test_sync_messages_records_the_known_python_limitation() -> None:
    result: Final = attempt_trace("messages", engine="python", asynchronous=False)

    assert result.skipped
    assert result.error == "ValueError: anthropic_messages_handler is not implemented for sync calls"
    assert result.events == ()


def test_unexpected_call_failure_is_not_skipped() -> None:
    result: Final = attempt_trace("unknown", engine="python", asynchronous=False)

    assert not result.skipped
    assert result.error == "ValueError: Unknown route: unknown"
    assert result.events == ()

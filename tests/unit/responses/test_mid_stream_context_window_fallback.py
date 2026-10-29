"""Regression tests for #48887.

A context-window overflow reported *inside* a stream arrives after HTTP 200, so it is not
covered by any status-code-based fallback gate. It is still satisfiable by a deployment
with a larger window, so both the Responses iterator and the chat stream wrapper must let
it through to the Router instead of raising it straight to the caller.
"""

from __future__ import annotations

import pytest

from litellm.exceptions import BadRequestError, ContextWindowExceededError
from litellm.litellm_core_utils.streaming_handler import _mid_stream_error_is_terminal
from litellm.responses.streaming_iterator import _mid_stream_fallback_eligible


def _context_window_error() -> ContextWindowExceededError:
    return ContextWindowExceededError(
        message="Your input exceeds the context window of this model.",
        model="gpt-6-luna",
        llm_provider="openai",
    )


def _bad_request(status_code: int | None = 400) -> BadRequestError:
    error = BadRequestError(
        message="Invalid value for 'temperature'.",
        model="gpt-6-luna",
        llm_provider="openai",
    )
    error.status_code = status_code  # type: ignore[misc]
    return error


# --- Responses API streams (litellm/responses/streaming_iterator.py) -------------------


def test_context_window_error_is_fallback_eligible_on_responses_stream() -> None:
    assert _mid_stream_fallback_eligible(_context_window_error()) is True


def test_plain_bad_request_still_skips_responses_fallback() -> None:
    assert _mid_stream_fallback_eligible(_bad_request()) is False


@pytest.mark.parametrize("status_code", [None, 429, 500, 503])
def test_transient_and_server_errors_stay_fallback_eligible(status_code: int | None) -> None:
    assert _mid_stream_fallback_eligible(_bad_request(status_code)) is True


# --- Chat completion streams (litellm/litellm_core_utils/streaming_handler.py) ---------


def test_context_window_error_is_not_terminal_on_chat_stream() -> None:
    error = _context_window_error()
    assert _mid_stream_error_is_terminal(error, error.status_code, error.status_code) is False


def test_context_window_error_is_not_terminal_even_when_envelope_reports_400() -> None:
    """The original exception can be the only one carrying the context-window type."""
    assert _mid_stream_error_is_terminal(_bad_request(400), 400, 400) is True
    assert _mid_stream_error_is_terminal(_context_window_error(), 400, 400) is False


def test_plain_bad_request_is_terminal_on_chat_stream() -> None:
    assert _mid_stream_error_is_terminal(_bad_request(), 400, 400) is True


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retriable_status_codes_are_not_terminal_on_chat_stream(status_code: int) -> None:
    error = _bad_request(status_code)
    assert _mid_stream_error_is_terminal(error, status_code, status_code) is False

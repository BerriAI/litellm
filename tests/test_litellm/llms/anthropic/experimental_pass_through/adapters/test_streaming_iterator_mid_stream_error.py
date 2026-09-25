"""
Regression tests for the ``/v1/messages`` async adapter dropping the socket on a
mid-stream provider error.

When a non-Anthropic model (e.g. Bedrock Converse) is served through
``/v1/messages``, the proxy hands Starlette the async SSE iterator directly. If
the upstream provider stream raises while being pulled (Bedrock raises
``BedrockError`` when a ConverseStream ends without a terminal ``messageStop``
event, common on cross-region inference profiles), the exception escaped the
request handler's try/except and tore down the connection. Clients like Claude
Code then showed a bare "Connection closed mid-response".

The async SSE wrapper must instead surface the failure as a well-formed
Anthropic ``error`` event so the stream stays valid and the client can retry.
"""

import json
import os
import sys
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.exceptions import MidStreamFallbackError
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
    _mid_stream_error_sse_event,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.utils import Delta, StreamingChoices


def _make_chunk(delta: Delta, finish_reason: Optional[str] = None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(finish_reason=finish_reason, index=0, delta=delta, logprobs=None)
    ]
    chunk.usage = None
    chunk._hidden_params = {}
    return chunk


class _AsyncStreamThenRaise:
    """Yields the given chunks, then raises ``exc`` (mimics a provider stream
    that terminates mid-response)."""

    def __init__(self, items: List[MagicMock], exc: BaseException):
        self._it = iter(items)
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise self._exc


def _parse_sse(raw: bytes) -> tuple[str, dict]:
    text = raw.decode()
    event_line, data_line = text.strip().split("\n", 1)
    return event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))


async def _drain_sse(wrapper: AnthropicStreamWrapper) -> List[bytes]:
    return [event async for event in wrapper.async_anthropic_sse_wrapper()]


@pytest.mark.asyncio
async def test_mid_stream_bedrock_error_becomes_anthropic_error_event():
    """A ``BedrockError`` raised after partial content must be surfaced as a
    terminal Anthropic ``error`` event, not propagated (which drops the socket
    and yields "Connection closed mid-response")."""
    chunks = [_make_chunk(Delta(content="Creating a file"))]
    bedrock_err = BedrockError(
        status_code=500,
        message="Bedrock ConverseStream ended without a terminal 'messageStop' event",
    )
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStreamThenRaise(chunks, bedrock_err),
        model="bedrock-converse-sonnet-4-6",
    )

    events = await _drain_sse(wrapper)

    parsed = [_parse_sse(e) for e in events]
    event_types = [name for name, _ in parsed]
    assert "message_start" in event_types
    assert event_types[-1] == "error"
    _, error_payload = parsed[-1]
    assert error_payload["type"] == "error"
    assert error_payload["error"]["type"] == "api_error"
    assert "messageStop" in error_payload["error"]["message"]


@pytest.mark.asyncio
async def test_mid_stream_error_does_not_raise_out_of_wrapper():
    """The async wrapper must fully drain without letting the upstream exception
    escape — escaping is exactly what tore down the connection before the fix."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStreamThenRaise([], BedrockError(status_code=500, message="boom")),
        model="claude-x",
    )
    events = await _drain_sse(wrapper)
    assert _parse_sse(events[-1])[0] == "error"


@pytest.mark.parametrize(
    "status_code, expected_type",
    [(500, "api_error"), (529, "overloaded_error"), (429, "rate_limit_error")],
)
def test_error_event_maps_status_code_to_anthropic_type(status_code, expected_type):
    raw = _mid_stream_error_sse_event(BedrockError(status_code=status_code, message="upstream failed"))
    name, payload = _parse_sse(raw)
    assert name == "error"
    assert payload["error"]["type"] == expected_type
    assert payload["error"]["message"] == "upstream failed"


def test_error_event_defaults_to_500_when_status_missing():
    raw = _mid_stream_error_sse_event(ValueError("no status here"))
    _, payload = _parse_sse(raw)
    assert payload["error"]["type"] == "api_error"
    assert payload["error"]["message"] == "no status here"


def test_error_event_preserves_midstream_fallback_error():
    exc = MidStreamFallbackError(
        message="BedrockException - internalServerException",
        model="bedrock-converse-sonnet-4-6",
        llm_provider="bedrock",
        original_exception=BedrockError(status_code=500, message="internalServerException"),
    )
    name, payload = _parse_sse(_mid_stream_error_sse_event(exc))
    assert name == "error"
    assert payload["error"]["type"] == "api_error"
    assert "internalServerException" in payload["error"]["message"]

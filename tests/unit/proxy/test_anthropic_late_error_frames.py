"""The stream framing a /v1/messages caller gets when the upstream fails after a keepalive ping.

The status line is already on the wire by then, so the failure can only reach the client as
an SSE event. Anthropic clients select events by name, so a bare `data:` line carrying the
upstream error body is skipped and the truncated stream reads as a completed turn.
"""

import asyncio
import json
from typing import Final

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from litellm.proxy.common_request_processing import (
    ANTHROPIC_LATE_SSE_FRAMING,
    OPENAI_LATE_SSE_FRAMING,
    open_sse_before_first_byte,
)

TTFT_PING: Final = b": ping\n\n"


async def _drain(response: Response) -> list[bytes]:
    assert isinstance(response, StreamingResponse)
    return [chunk async for chunk in response.body_iterator]


async def _slow_failure(message: str) -> Response:
    await asyncio.sleep(0.2)
    raise HTTPException(status_code=503, detail=message)


async def _slow_error_status(status_code: int, content: dict) -> Response:
    await asyncio.sleep(0.2)
    return JSONResponse(status_code=status_code, content=content)


def _error_event(collected: list[bytes]) -> dict:
    frame: Final = collected[-1].decode()
    assert frame.startswith("event: error\ndata: ")
    return json.loads(frame.removeprefix("event: error\ndata: ").strip())


@pytest.mark.asyncio
async def test_a_late_anthropic_failure_reaches_the_client_as_an_event_error():
    """The failure arrives after a ping, so it has to travel as an `event: error` frame
    the Anthropic client raises on, not as a bare data line it skips."""
    response = await open_sse_before_first_byte(
        _slow_failure("upstream unavailable"), ping_interval_seconds=0.05, late_framing=ANTHROPIC_LATE_SSE_FRAMING
    )
    collected: Final = await _drain(response)

    assert collected[0] == TTFT_PING
    assert _error_event(collected) == {
        "type": "error",
        "error": {"type": "api_error", "message": "upstream unavailable"},
    }


@pytest.mark.asyncio
async def test_a_late_anthropic_failure_redacts_internal_details():
    """A transport or callback exception can carry a filesystem path or an internal
    hostname. The message leaves the process, so it goes through the same redactor the
    non-keepalive error path applies."""
    response = await open_sse_before_first_byte(
        _slow_failure("could not read /etc/litellm/master.key"),
        ping_interval_seconds=0.05,
        late_framing=ANTHROPIC_LATE_SSE_FRAMING,
    )
    collected: Final = await _drain(response)

    message: Final = _error_event(collected)["error"]["message"]
    assert "/etc/litellm/master.key" not in message
    assert "REDACTED" in message


@pytest.mark.asyncio
async def test_a_late_error_status_is_framed_as_an_anthropic_error_event():
    """A JSON error body that arrives after the keepalive cannot be relayed verbatim:
    the caller reads events by name and would take the truncated stream as a whole turn."""
    response = await open_sse_before_first_byte(
        _slow_error_status(400, {"error": {"message": "model not found", "type": "invalid_request_error"}}),
        ping_interval_seconds=0.05,
        late_framing=ANTHROPIC_LATE_SSE_FRAMING,
    )
    collected: Final = await _drain(response)

    assert collected[0] == TTFT_PING
    assert _error_event(collected) == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "model not found"},
    }


@pytest.mark.asyncio
async def test_a_late_error_status_without_a_message_falls_back_to_the_status():
    response = await open_sse_before_first_byte(
        _slow_error_status(502, {"detail": "bad gateway"}),
        ping_interval_seconds=0.05,
        late_framing=ANTHROPIC_LATE_SSE_FRAMING,
    )
    collected: Final = await _drain(response)

    assert _error_event(collected)["error"]["message"] == '{"detail":"bad gateway"}'


@pytest.mark.asyncio
async def test_the_openai_framing_still_relays_a_late_error_body_verbatim():
    """Routes that answer as OpenAI keep the framing they had: the body is relayed as
    it is and the stream ends with [DONE]."""
    response = await open_sse_before_first_byte(
        _slow_error_status(400, {"error": {"message": "bad request"}}),
        ping_interval_seconds=0.05,
        late_framing=OPENAI_LATE_SSE_FRAMING,
    )
    collected: Final = await _drain(response)

    assert collected[0] == TTFT_PING
    assert json.loads(collected[-2].decode().removeprefix("data: ").strip()) == {"error": {"message": "bad request"}}
    assert collected[-1] == b"data: [DONE]\n\n"

"""A passthrough stream that fails after its keepalive has committed a 200.

The Anthropic `/anthropic/*` passthrough relays upstream bytes. Once the keepalive
has written `HTTP/1.1 200`, an upstream 4xx/5xx can no longer be expressed as a
status, so those bytes would reach the caller as a bare JSON chunk inside an event
stream. Anthropic clients pick events by name and read that as a normal end.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from starlette.responses import StreamingResponse

from litellm.proxy._types import ProxyException
from litellm.proxy.common_request_processing import open_sse_before_first_byte

PING = ": ping\n\n"


def _error_stream(chunks: list[bytes], status_code: int = 500) -> StreamingResponse:
    async def _body() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    return StreamingResponse(_body(), status_code=status_code, media_type="application/json")


def _json_error(payload: dict[str, object], status_code: int = 500) -> StreamingResponse:
    return _error_stream([json.dumps(payload).encode("utf-8")], status_code=status_code)


async def _drain(response: StreamingResponse) -> str:
    chunks = [chunk async for chunk in response.body_iterator]
    return b"".join(chunks).decode("utf-8")


async def _relay(produced: StreamingResponse, *, answers_as_anthropic: bool = False) -> str:
    async def deliver() -> StreamingResponse:
        await asyncio.sleep(0.05)
        return produced

    response = await open_sse_before_first_byte(
        deliver(), ping_interval_seconds=0.02, answers_as_anthropic=answers_as_anthropic
    )
    assert isinstance(response, StreamingResponse)
    return await _drain(response)


@pytest.mark.asyncio
async def test_an_anthropic_route_replaces_a_late_error_body_with_an_event():
    body = await _relay(
        _json_error({"error": {"message": "upstream said no", "type": "server_error"}}),
        answers_as_anthropic=True,
    )

    assert PING in body
    assert body.count("event: error") == 1
    assert "upstream said no" in body
    assert "[DONE]" not in body


@pytest.mark.asyncio
async def test_a_route_without_the_flag_still_relays_the_error_body_unchanged():
    body = await _relay(_json_error({"error": {"message": "relay me as is"}}))

    assert "event: error" not in body
    assert '"message": "relay me as is"' in body


@pytest.mark.asyncio
async def test_the_upstream_message_is_what_the_client_reads():
    body = await _relay(
        _json_error({"error": {"message": "model overloaded", "type": "server_error"}}, status_code=529),
        answers_as_anthropic=True,
    )

    assert "model overloaded" in body
    assert "overloaded_error" in body


@pytest.mark.asyncio
async def test_a_body_that_is_not_json_still_produces_an_error_frame():
    body = await _relay(_error_stream([b"<html>gateway timeout</html>"], status_code=504), answers_as_anthropic=True)

    assert body.count("event: error") == 1
    assert "gateway timeout" in body


@pytest.mark.asyncio
async def test_an_exception_before_the_first_byte_uses_an_anthropic_frame():
    async def boom():
        await asyncio.sleep(0.05)
        raise RuntimeError("upstream died after accepting the request")

    response = await open_sse_before_first_byte(boom(), ping_interval_seconds=0.02, answers_as_anthropic=True)

    body = await _drain(response)
    assert body.count("event: error") == 1
    assert "[DONE]" not in body


@pytest.mark.asyncio
async def test_a_body_that_fails_mid_read_still_produces_an_error_frame():
    async def _broken() -> AsyncIterator[bytes]:
        raise RuntimeError("upstream connection reset")
        yield b""

    produced = StreamingResponse(_broken(), status_code=500, media_type="application/json")

    body = await _relay(produced, answers_as_anthropic=True)

    assert body.count("event: error") == 1
    assert "upstream returned HTTP 500" in body


@pytest.mark.asyncio
async def test_a_large_error_body_is_read_with_a_bound():
    padding = "x" * (256 * 1024)
    produced = _error_stream([padding.encode("utf-8")], status_code=500)

    body = await _relay(produced, answers_as_anthropic=True)

    assert body.count("event: error") == 1
    assert len(body) < 128 * 1024


FAKE_KEY = "sk-test-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
INTERNAL_PATH = "/home/runner/litellm/proxy/internal_state.py"


@pytest.mark.asyncio
async def test_a_late_failure_exception_is_redacted_before_framing():
    async def boom():
        await asyncio.sleep(0.05)
        raise RuntimeError(f"upstream failed with key {FAKE_KEY} at {INTERNAL_PATH}")

    response = await open_sse_before_first_byte(boom(), ping_interval_seconds=0.02, answers_as_anthropic=True)
    body = await _drain(response)

    assert body.count("event: error") == 1
    assert FAKE_KEY not in body
    assert INTERNAL_PATH not in body
    assert "REDACTED" in body


@pytest.mark.asyncio
async def test_a_late_failure_that_is_already_a_proxy_exception_is_redacted():
    async def boom():
        await asyncio.sleep(0.05)
        raise ProxyException(
            message=f"upstream failed with key {FAKE_KEY} at {INTERNAL_PATH}",
            type="api_error",
            param=None,
            code=500,
        )

    response = await open_sse_before_first_byte(boom(), ping_interval_seconds=0.02, answers_as_anthropic=True)
    body = await _drain(response)

    assert body.count("event: error") == 1
    assert FAKE_KEY not in body
    assert INTERNAL_PATH not in body
    assert "REDACTED" in body


@pytest.mark.asyncio
async def test_a_late_error_body_is_redacted_before_framing():
    body = await _relay(
        _json_error({"error": {"message": f"failed with key {FAKE_KEY} at {INTERNAL_PATH}", "type": "server_error"}}),
        answers_as_anthropic=True,
    )

    assert body.count("event: error") == 1
    assert FAKE_KEY not in body
    assert INTERNAL_PATH not in body
    assert "REDACTED" in body


@pytest.mark.asyncio
async def test_an_ordinary_error_message_is_not_redacted():
    body = await _relay(
        _json_error({"error": {"message": "upstream said no", "type": "server_error"}}),
        answers_as_anthropic=True,
    )

    assert "upstream said no" in body
    assert "REDACTED" not in body

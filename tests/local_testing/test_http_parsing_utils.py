from collections.abc import Awaitable, Callable

import pytest
from fastapi import Request
from starlette.types import Message

from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body


def _request(receive: Callable[[], Awaitable[Message]]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


def _request_with_body(body: bytes) -> Request:
    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    return _request(receive)


@pytest.mark.asyncio
async def test_read_request_body_valid_json():
    result = await _read_request_body(_request_with_body(b'{"key": "value"}'))
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_read_request_body_empty_body():
    result = await _read_request_body(_request_with_body(b""))
    assert result == {}


@pytest.mark.asyncio
async def test_read_request_body_invalid_json():
    with pytest.raises(ProxyException):
        await _read_request_body(_request_with_body(b'{"key": value}'))


@pytest.mark.asyncio
async def test_read_request_body_large_payload():
    large_payload = '{"key":' + '"a"' * 10**6 + "}"
    with pytest.raises(ProxyException):
        await _read_request_body(_request_with_body(large_payload.encode()))


@pytest.mark.asyncio
async def test_read_request_body_unexpected_error():
    async def receive() -> Message:
        raise ValueError("Unexpected error")

    result = await _read_request_body(_request(receive))
    assert result == {}

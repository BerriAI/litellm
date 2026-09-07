import asyncio
from collections.abc import AsyncGenerator
from typing import Final

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Message

from litellm.constants import LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED
from litellm.proxy.common_request_processing import create_response
from litellm.proxy.response_polling.background_streaming import detach_request_from_client


def _request_whose_client_already_left() -> Request:
    async def receive() -> Message:
        return {"type": "http.disconnect"}

    scope: Final = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [(b"x-litellm-call-id", b"call-123")],
        "query_string": b"",
    }
    return Request(scope, receive)


async def _slow_first_chunk() -> AsyncGenerator[str, None]:
    await asyncio.sleep(0.05)
    yield 'data: {"type": "response.created"}\n\n'


@pytest.mark.asyncio
async def test_detached_request_survives_client_disconnect_before_first_chunk():
    original: Final = _request_whose_client_already_left()

    cancelled: Final = await create_response(_slow_first_chunk(), "text/event-stream", {}, request=original)
    assert isinstance(cancelled, JSONResponse)
    assert cancelled.status_code == LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED

    detached: Final = detach_request_from_client(original)
    kept_alive: Final = await create_response(_slow_first_chunk(), "text/event-stream", {}, request=detached)
    assert isinstance(kept_alive, StreamingResponse)
    assert kept_alive.status_code == 200


def test_detached_request_keeps_scope():
    original: Final = _request_whose_client_already_left()
    detached: Final = detach_request_from_client(original)
    assert detached.headers["x-litellm-call-id"] == "call-123"
    assert detached.scope is original.scope

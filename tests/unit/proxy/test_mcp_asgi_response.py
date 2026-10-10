import asyncio

import pytest
from fastapi import HTTPException

from litellm.proxy.proxy_server import _stream_mcp_asgi_response


@pytest.mark.asyncio
async def test_stream_mcp_asgi_response_propagates_pre_header_http_exception():
    async def handle_fn(_scope, _receive, _send):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={
                "WWW-Authenticate": "Bearer authorization_uri=https://example.test/auth"
            },
        )

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(
            _stream_mcp_asgi_response(
                handle_fn,
                {"type": "http", "method": "POST", "path": "/mcp", "headers": []},
                receive,
            ),
            timeout=1.0,
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {
        "WWW-Authenticate": "Bearer authorization_uri=https://example.test/auth"
    }

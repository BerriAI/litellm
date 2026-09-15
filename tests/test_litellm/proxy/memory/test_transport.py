import asyncio
from typing import Final

import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.memory.transport import gateway_round


@pytest.mark.asyncio
async def test_stream_reaches_client_before_model_finishes_and_disconnect_cancels_the_model() -> None:
    continuing: Final = asyncio.Event()
    cancelled: Final = asyncio.Event()

    async def app(inner: Request, data: dict[str, object], auth: UserAPIKeyAuth) -> StreamingResponse:
        scope = inner.scope
        assert scope["client"] == ("192.0.2.3", 12345)
        assert scope["query_string"] == b"api-version=test"
        body: Final = await _read_request_body(inner)
        assert body["model"] == "test"
        assert body["stream"] is True
        assert body["extra_headers"] == {"anthropic-beta": "test-beta"}
        assert body["headers"] == {"x-custom": "preserved"}
        assert not Request(scope).headers.get("idempotency-key")

        async def content():
            yield b"first delta"
            try:
                await continuing.wait()
            finally:
                cancelled.set()

        return StreamingResponse(content(), media_type="text/event-stream")

    request: Final = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/v1/messages",
            "raw_path": b"/v1/messages",
            "query_string": b"api-version=test",
            "headers": [(b"idempotency-key", b"outer-request")],
            "client": ("192.0.2.3", 12345),
            "server": ("gateway.example", 443),
            "parsed_body": (("model", "stream"), {"model": "original-body", "stream": False}),
            "state": {"_cached_headers": {"content-type": "application/x-www-form-urlencoded"}},
        }
    )
    async with gateway_round(
        app,
        request,
        {
            "model": "test",
            "stream": True,
            "extra_headers": {"Idempotency-Key": "outer-request", "anthropic-beta": "test-beta"},
            "headers": {"X-Request-ID": "outer-request", "x-custom": "preserved"},
        },
        UserAPIKeyAuth(),
    ) as call:
        stream: Final = call.chunks()
        assert await asyncio.wait_for(anext(stream), timeout=1) == b"first delta"
        assert not continuing.is_set()
    assert cancelled.is_set()
    assert call.task is not None and call.task.cancelled()
    await stream.aclose()


@pytest.mark.asyncio
async def test_memory_rounds_share_one_rpm_admission_but_keep_token_limits():
    from fastapi import HTTPException
    from starlette.responses import Response

    from litellm.caching.caching import DualCache
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3 as _PROXY_MaxParallelRequestsHandler,
    )
    from litellm.proxy.memory.transport import gateway_round
    from litellm.proxy.utils import InternalUsageCache, hash_token

    cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(cache))
    auth = UserAPIKeyAuth(api_key=hash_token("memory-round-limits"), rpm_limit=1, tpm_limit=100)
    request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})

    async def execute(inner, body, round_auth):
        await handler.async_pre_call_hook(round_auth, cache, body, "")
        return Response(b"accepted")

    for index in (0, 1):
        async with gateway_round(execute, request, {"model": "test"}, auth, index) as round:
            assert await round.read() == b"accepted"
    with pytest.raises(HTTPException) as rate_limited:
        async with gateway_round(execute, request, {"model": "test"}, auth, 0):
            pass
    assert rate_limited.value.status_code == 429
    assert "requests" in rate_limited.value.detail
    await cache.async_set_cache(key=f"{{api_key:{auth.api_key}}}:tokens", value=101, ttl=60)
    with pytest.raises(HTTPException) as token_limited:
        async with gateway_round(execute, request, {"model": "test"}, auth, 1):
            pass
    assert token_limited.value.status_code == 429
    assert "tokens" in token_limited.value.detail

import pytest
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from starlette.types import Message
from unittest.mock import AsyncMock

from litellm.proxy.auth.auth_utils import check_if_request_size_is_safe
from litellm.proxy.middleware.request_size_limit_middleware import (
    RequestSizeLimitMiddleware,
)


# ---------------------------------------------------------------------------
# RequestSizeLimitMiddleware -- ASGI streaming guard
# ---------------------------------------------------------------------------


def test_request_size_limit_middleware_rejects_content_length_before_body_read():
    downstream_called = False

    async def app(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True
        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    client = TestClient(
        RequestSizeLimitMiddleware(
            app,
            get_max_request_size_mb=lambda: 1,
            is_request_size_limit_enabled=lambda: True,
        )
    )

    response = client.post(
        "/chat/completions",
        content=b"x" * (1024 * 1024 + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"error": "Request size is too large. Max size is 1 MB"}
    assert response.headers["content-length"] == str(len(response.content))
    assert downstream_called is False


def test_request_size_limit_middleware_zero_limit_disables_guard():
    downstream_called = False

    async def app(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True
        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    client = TestClient(
        RequestSizeLimitMiddleware(
            app,
            get_max_request_size_mb=lambda: 0,
            is_request_size_limit_enabled=lambda: True,
        )
    )

    response = client.post(
        "/chat/completions",
        content=b"x" * (1024 * 1024 + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert downstream_called is True


# ---------------------------------------------------------------------------
# check_if_request_size_is_safe -- auth-layer guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_if_request_size_is_safe_skips_body_read_for_chunked_multipart(
    monkeypatch,
):
    """
    Regression test for #42224: request.body() must never be awaited for a
    chunked multipart upload without Content-Length.

    Asserting that body() is *not called* -- rather than only that a RuntimeError
    is caught -- is the minimal condition that prevents recurrence: a test that
    merely catches the error would still pass without the multipart skip.
    """
    from unittest.mock import patch

    body_mock = AsyncMock(side_effect=RuntimeError("Stream consumed"))

    class MockHeaders:
        def get(self, key, default=None):
            return {
                "content-type": "multipart/form-data; boundary=----boundary",
                "content-length": None,
            }.get(key, default)

    class MockChunkedMultipartRequest:
        headers = MockHeaders()
        body = body_mock

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"max_request_size_mb": 1},
    ), patch(
        "litellm.proxy.proxy_server.premium_user",
        True,
    ):
        result = await check_if_request_size_is_safe(request=MockChunkedMultipartRequest())

    assert result is True
    body_mock.assert_not_called()


@pytest.mark.asyncio
async def test_check_if_request_size_is_safe_uses_cached_body():
    """
    When request._body is already populated the size check uses the cached
    bytes and must not call request.body() again.
    """
    from unittest.mock import patch

    body_mock = AsyncMock(
        side_effect=AssertionError("body() must not be called when _body is cached")
    )

    class MockHeaders:
        def get(self, key, default=None):
            return None

    class MockCachedRequest:
        headers = MockHeaders()
        body = body_mock
        _body = b"x" * 512  # 512 bytes -- well under 1 MB

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"max_request_size_mb": 1},
    ), patch(
        "litellm.proxy.proxy_server.premium_user",
        True,
    ):
        result = await check_if_request_size_is_safe(request=MockCachedRequest())

    assert result is True
    body_mock.assert_not_called()


@pytest.mark.asyncio
async def test_check_if_request_size_is_safe_raises_413_on_oversized_cached_body():
    """Cached body that exceeds the limit must raise ProxyException with code 413."""
    from unittest.mock import patch
    from litellm.proxy._types import ProxyException

    body_mock = AsyncMock(
        side_effect=AssertionError("body() must not be called when _body is cached")
    )

    class MockHeaders:
        def get(self, key, default=None):
            return None

    class MockOversizedCachedRequest:
        headers = MockHeaders()
        body = body_mock
        _body = b"x" * (2 * 1024 * 1024)  # 2 MB -- over the 1 MB limit

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"max_request_size_mb": 1},
    ), patch(
        "litellm.proxy.proxy_server.premium_user",
        True,
    ):
        with pytest.raises(ProxyException) as exc_info:
            await check_if_request_size_is_safe(request=MockOversizedCachedRequest())

    assert exc_info.value.code == 413
    body_mock.assert_not_called()


@pytest.mark.asyncio
async def test_check_if_request_size_is_safe_handles_stream_consumed_gracefully():
    """
    If request.body() raises RuntimeError (stream consumed by a non-multipart
    handler), check_if_request_size_is_safe must return True rather than
    surface an opaque error or mask the failure as 401.
    """
    from unittest.mock import patch

    class MockHeaders:
        def get(self, key, default=None):
            return {"content-type": "application/json"}.get(key, default)

    class MockConsumedRequest:
        headers = MockHeaders()
        body = AsyncMock(side_effect=RuntimeError("Stream consumed"))

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"max_request_size_mb": 1},
    ), patch(
        "litellm.proxy.proxy_server.premium_user",
        True,
    ):
        result = await check_if_request_size_is_safe(request=MockConsumedRequest())

    assert result is True

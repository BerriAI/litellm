"""Unit tests for mavvrik._http.http_request — shared retry transport."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik._http import http_request


def _mock_response(status_code: int, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    return resp


def _mock_shared_client(return_value=None, side_effect=None):
    """Mock get_async_httpx_client returning a handler whose .client.request is stubbed."""
    inner_client = MagicMock()
    if side_effect:
        inner_client.request = AsyncMock(side_effect=side_effect)
    else:
        inner_client.request = AsyncMock(return_value=return_value)

    handler = MagicMock()
    handler.client = inner_client
    return handler, inner_client


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


class TestHttpRequestSuccess:
    @pytest.mark.asyncio
    async def test_returns_response_on_2xx(self):
        handler, _ = _mock_shared_client(return_value=_mock_response(200))
        with patch(
            "litellm.integrations.mavvrik._http.get_async_httpx_client",
            return_value=handler,
        ):
            resp = await http_request("GET", "https://example.com")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_4xx_without_retry(self):
        handler, http = _mock_shared_client(
            return_value=_mock_response(401, "Unauthorized")
        )
        with (
            patch(
                "litellm.integrations.mavvrik._http.get_async_httpx_client",
                return_value=handler,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            resp = await http_request("GET", "https://example.com")
        assert resp.status_code == 401
        assert http.request.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_headers(self):
        captured = []

        async def fake_request(method, url, headers=None, **kwargs):
            captured.append(headers)
            return _mock_response(200)

        handler = MagicMock()
        handler.client.request = fake_request

        with patch(
            "litellm.integrations.mavvrik._http.get_async_httpx_client",
            return_value=handler,
        ):
            await http_request(
                "POST", "https://example.com", headers={"x-api-key": "secret"}
            )

        assert captured[0]["x-api-key"] == "secret"

    @pytest.mark.asyncio
    async def test_passes_json_and_params(self):
        captured = []

        async def fake_request(method, url, json=None, params=None, **kwargs):
            captured.append({"json": json, "params": params})
            return _mock_response(200)

        handler = MagicMock()
        handler.client.request = fake_request

        with patch(
            "litellm.integrations.mavvrik._http.get_async_httpx_client",
            return_value=handler,
        ):
            await http_request(
                "GET", "https://example.com", json={"key": "val"}, params={"q": "1"}
            )

        assert captured[0]["json"] == {"key": "val"}
        assert captured[0]["params"] == {"q": "1"}

    @pytest.mark.asyncio
    async def test_passes_content(self):
        captured = []

        async def fake_request(method, url, content=None, **kwargs):
            captured.append(content)
            return _mock_response(201)

        handler = MagicMock()
        handler.client.request = fake_request

        with patch(
            "litellm.integrations.mavvrik._http.get_async_httpx_client",
            return_value=handler,
        ):
            await http_request("PUT", "https://example.com", content=b"gzip-data")

        assert captured[0] == b"gzip-data"


# ---------------------------------------------------------------------------
# Retry on 5xx
# ---------------------------------------------------------------------------


class TestHttpRequestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_5xx_then_raises(self):
        handler, http = _mock_shared_client(
            return_value=_mock_response(503, "unavailable")
        )
        with (
            patch(
                "litellm.integrations.mavvrik._http.get_async_httpx_client",
                return_value=handler,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="failed after"):
                await http_request("GET", "https://example.com")
        assert http.request.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_network_error_then_raises(self):
        handler, http = _mock_shared_client(side_effect=httpx.ConnectError("timeout"))
        with (
            patch(
                "litellm.integrations.mavvrik._http.get_async_httpx_client",
                return_value=handler,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="failed after"):
                await http_request("GET", "https://example.com")
        assert http.request.call_count == 3

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self):
        fail = _mock_response(503, "err")
        ok = _mock_response(200)
        handler, http = _mock_shared_client()
        http.request = AsyncMock(side_effect=[fail, ok])
        with (
            patch(
                "litellm.integrations.mavvrik._http.get_async_httpx_client",
                return_value=handler,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            resp = await http_request("GET", "https://example.com")
        assert resp.status_code == 200
        assert http.request.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_exponential_backoff(self):
        handler, http = _mock_shared_client(return_value=_mock_response(503, "err"))
        sleep_calls = []
        with (
            patch(
                "litellm.integrations.mavvrik._http.get_async_httpx_client",
                return_value=handler,
            ),
            patch(
                "asyncio.sleep",
                new_callable=AsyncMock,
                side_effect=lambda s: sleep_calls.append(s),
            ),
        ):
            with pytest.raises(RuntimeError):
                await http_request("GET", "https://example.com")
        # 3 attempts → 2 sleeps: 1.0s and 2.0s
        assert sleep_calls == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_error_message_contains_label(self):
        handler, _ = _mock_shared_client(return_value=_mock_response(503, "err"))
        with (
            patch(
                "litellm.integrations.mavvrik._http.get_async_httpx_client",
                return_value=handler,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="initiate"):
                await http_request("POST", "https://example.com", label="initiate")

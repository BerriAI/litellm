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


def _mock_http(return_value=None, side_effect=None):
    """Return a patched httpx.AsyncClient context manager."""
    http = MagicMock()
    if side_effect:
        http.request = AsyncMock(side_effect=side_effect)
    else:
        http.request = AsyncMock(return_value=return_value)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=http)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, http


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


class TestHttpRequestSuccess:
    @pytest.mark.asyncio
    async def test_returns_response_on_2xx(self):
        ctx, _ = _mock_http(return_value=_mock_response(200))
        with patch("httpx.AsyncClient", return_value=ctx):
            resp = await http_request("GET", "https://example.com")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_4xx_without_retry(self):
        ctx, http = _mock_http(return_value=_mock_response(401, "Unauthorized"))
        with patch("httpx.AsyncClient", return_value=ctx), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
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

        ctx = MagicMock()
        http = MagicMock()
        http.request = fake_request
        ctx.__aenter__ = AsyncMock(return_value=http)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=ctx):
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

        ctx = MagicMock()
        http = MagicMock()
        http.request = fake_request
        ctx.__aenter__ = AsyncMock(return_value=http)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=ctx):
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

        ctx = MagicMock()
        http = MagicMock()
        http.request = fake_request
        ctx.__aenter__ = AsyncMock(return_value=http)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=ctx):
            await http_request("PUT", "https://example.com", content=b"gzip-data")

        assert captured[0] == b"gzip-data"


# ---------------------------------------------------------------------------
# Retry on 5xx
# ---------------------------------------------------------------------------


class TestHttpRequestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_5xx_then_raises(self):
        ctx, http = _mock_http(return_value=_mock_response(503, "unavailable"))
        with patch("httpx.AsyncClient", return_value=ctx), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            with pytest.raises(RuntimeError, match="failed after"):
                await http_request("GET", "https://example.com")
        assert http.request.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_network_error_then_raises(self):
        ctx, http = _mock_http(side_effect=httpx.ConnectError("timeout"))
        with patch("httpx.AsyncClient", return_value=ctx), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            with pytest.raises(RuntimeError, match="failed after"):
                await http_request("GET", "https://example.com")
        assert http.request.call_count == 3

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self):
        fail = _mock_response(503, "err")
        ok = _mock_response(200)
        ctx, http = _mock_http()
        http.request = AsyncMock(side_effect=[fail, ok])
        with patch("httpx.AsyncClient", return_value=ctx), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            resp = await http_request("GET", "https://example.com")
        assert resp.status_code == 200
        assert http.request.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_exponential_backoff(self):
        ctx, http = _mock_http(return_value=_mock_response(503, "err"))
        sleep_calls = []
        with patch("httpx.AsyncClient", return_value=ctx), patch(
            "asyncio.sleep",
            new_callable=AsyncMock,
            side_effect=lambda s: sleep_calls.append(s),
        ):
            with pytest.raises(RuntimeError):
                await http_request("GET", "https://example.com")
        # 3 attempts → 2 sleeps: 1.0s and 2.0s
        assert sleep_calls == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_error_message_contains_label(self):
        ctx, _ = _mock_http(return_value=_mock_response(503, "err"))
        with patch("httpx.AsyncClient", return_value=ctx), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            with pytest.raises(RuntimeError, match="initiate"):
                await http_request("POST", "https://example.com", label="initiate")

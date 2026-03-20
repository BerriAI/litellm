import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath("../.."))

import httpx
import pytest

from litellm.proxy.proxy_server import app


@pytest.mark.asyncio
async def test_get_favicon_default():
    """Test that get_favicon returns the default favicon when no URL set."""
    os.environ.pop("LITELLM_FAVICON_URL", None)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        response = await ac.get("/get_favicon")

    assert response.status_code in [200, 404]
    if response.status_code == 200:
        assert response.headers["content-type"] == "image/x-icon"


@pytest.mark.asyncio
async def test_get_favicon_with_custom_url():
    """Test that get_favicon fetches from a custom URL."""
    os.environ["LITELLM_FAVICON_URL"] = "https://example.com/favicon.ico"

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b"\x00\x00\x01\x00"
    mock_response.headers = {"content-type": "image/x-icon"}

    try:
        with mock.patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
        ) as mock_get:
            mock_get.return_value = mock_response

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as ac:
                response = await ac.get("/get_favicon")

            assert response.status_code == 200
            assert response.headers["content-type"] == "image/x-icon"
    finally:
        os.environ.pop("LITELLM_FAVICON_URL", None)


@pytest.mark.asyncio
async def test_get_favicon_url_error_fallback():
    """Test that get_favicon falls back to default on error."""
    os.environ["LITELLM_FAVICON_URL"] = "https://invalid.com/favicon.ico"

    try:
        with mock.patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
        ) as mock_get:
            mock_get.side_effect = httpx.ConnectError("unreachable")

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as ac:
                response = await ac.get("/get_favicon")

            assert response.status_code in [200, 404]
    finally:
        os.environ.pop("LITELLM_FAVICON_URL", None)

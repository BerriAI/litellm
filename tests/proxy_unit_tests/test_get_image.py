import os
import sys
from unittest import mock

# Standard path insertion
sys.path.insert(0, os.path.abspath("../.."))

import pytest
import httpx
from litellm.proxy.proxy_server import app


@pytest.mark.asyncio
async def test_get_image_error_handling():
    """
    Test that get_image handles network errors gracefully and doesn't hang.
    """
    # Set an unreachable URL
    os.environ["UI_LOGO_PATH"] = "http://invalid-url-12345.com/logo.jpg"

    # Clear cache
    parent_dir = os.path.dirname(
        os.path.dirname(
            app.__file__
            if hasattr(app, "__file__")
            else "litellm/proxy/proxy_server.py"
        )
    )
    cache_path = os.path.join(parent_dir, "proxy", "cached_logo.jpg")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # Mock AsyncHTTPHandler to simulate a timeout or connection error
    with mock.patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
    ) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Network is unreachable")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            response = await ac.get("/get_image")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"


def _get_cache_dir(proxy_dir: str) -> str:
    return os.path.join(proxy_dir, "proxy")


@pytest.mark.asyncio
async def test_get_image_cache_logic():
    """
    Test that once cached, get_image doesn't hit the network.
    """
    os.environ["UI_LOGO_PATH"] = "http://example.com/logo.jpg"

    # Clear cache
    parent_dir = os.path.dirname(
        os.path.dirname(
            app.__file__
            if hasattr(app, "__file__")
            else "litellm/proxy/proxy_server.py"
        )
    )
    cache_dir = _get_cache_dir(parent_dir)
    cache_path = os.path.join(cache_dir, "cached_logo.jpg")
    cache_type_path = os.path.join(cache_dir, "cached_logo_type.txt")
    for p in (cache_path, cache_type_path):
        if os.path.exists(p):
            os.remove(p)

    # Mock response
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b"fake image data"
    mock_response.headers = {"content-type": "image/jpeg"}

    with mock.patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
    ) as mock_get:
        mock_get.return_value = mock_response

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            # First call - should hit download logic
            response1 = await ac.get("/get_image")
            assert response1.status_code == 200
            assert mock_get.call_count == 1

            # Second call - should hit cache
            response2 = await ac.get("/get_image")
            assert response2.status_code == 200
            # If cache works, mock_get shouldn't be called again
            assert mock_get.call_count == 1


@pytest.mark.asyncio
async def test_get_image_preserves_content_type_from_http_url():
    """
    When downloading a logo from an HTTP URL, the Content-Type from the upstream
    response should be persisted and used when serving from cache, not hardcoded
    to image/jpeg. This matters for PNG/SVG logos served through strict enterprise
    proxies that validate Content-Type.
    """
    os.environ["UI_LOGO_PATH"] = "http://example.com/logo.png"

    parent_dir = os.path.dirname(
        os.path.dirname(
            app.__file__
            if hasattr(app, "__file__")
            else "litellm/proxy/proxy_server.py"
        )
    )
    cache_dir = _get_cache_dir(parent_dir)
    cache_path = os.path.join(cache_dir, "cached_logo.jpg")
    cache_type_path = os.path.join(cache_dir, "cached_logo_type.txt")
    for p in (cache_path, cache_type_path):
        if os.path.exists(p):
            os.remove(p)

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
    mock_response.headers = {"content-type": "image/png"}

    with mock.patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
    ) as mock_get:
        mock_get.return_value = mock_response

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            # First request: download and cache
            response1 = await ac.get("/get_image")
            assert response1.status_code == 200
            assert response1.headers["content-type"] == "image/png"

            # Second request: served from cache with the persisted content-type
            response2 = await ac.get("/get_image")
            assert response2.status_code == 200
            assert response2.headers["content-type"] == "image/png"
            # Upstream should only have been hit once
            assert mock_get.call_count == 1

    # Clean up
    for p in (cache_path, cache_type_path):
        if os.path.exists(p):
            os.remove(p)

import os
import sys

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
async def test_get_favicon_with_custom_url(monkeypatch):
    """Test that get_favicon redirects browser-loaded custom URLs."""
    monkeypatch.setenv("LITELLM_FAVICON_URL", "https://example.com/favicon.ico")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        response = await ac.get("/get_favicon")

    assert response.status_code == 307
    assert response.headers["location"] == "https://example.com/favicon.ico"


@pytest.mark.asyncio
async def test_get_favicon_remote_url_is_not_server_fetched(monkeypatch):
    """Test that get_favicon does not validate remote URLs server-side."""
    monkeypatch.setenv("LITELLM_FAVICON_URL", "https://invalid.com/favicon.ico")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        response = await ac.get("/get_favicon")

    assert response.status_code == 307
    assert response.headers["location"] == "https://invalid.com/favicon.ico"

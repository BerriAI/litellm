import os
import sys
from unittest import mock

# Standard path insertion
sys.path.insert(0, os.path.abspath("../.."))

import httpx
import pytest
from litellm.proxy.proxy_server import app


@pytest.mark.asyncio
async def test_get_image_redirects_remote_logo_without_server_fetch(monkeypatch):
    """
    Remote logo URLs should be loaded by the browser, not fetched by the proxy.
    """
    monkeypatch.setenv("UI_LOGO_PATH", "http://invalid-url-12345.com/logo.jpg")

    with mock.patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
    ) as mock_get:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            response = await ac.get("/get_image")

    assert response.status_code == 307
    assert response.headers["location"] == "http://invalid-url-12345.com/logo.jpg"
    mock_get.assert_not_called()


@pytest.mark.asyncio
async def test_get_image_remote_logo_does_not_use_stale_cache(monkeypatch, tmp_path):
    """
    A stale pre-fix cache file should not mask a configured remote logo URL.
    """
    monkeypatch.setenv("UI_LOGO_PATH", "http://example.com/logo.jpg")
    monkeypatch.setenv("LITELLM_ASSETS_PATH", str(tmp_path))
    (tmp_path / "cached_logo.jpg").write_bytes(b"\xff\xd8\xff cached logo")

    with mock.patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
    ) as mock_get:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            response = await ac.get("/get_image")

    assert response.status_code == 307
    assert response.headers["location"] == "http://example.com/logo.jpg"
    mock_get.assert_not_called()

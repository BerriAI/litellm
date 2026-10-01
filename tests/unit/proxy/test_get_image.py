import os
import shutil
from unittest import mock

# Standard path insertion
import httpx
import pytest

from litellm.proxy import proxy_server
from litellm.proxy.proxy_server import app


@pytest.mark.asyncio
async def test_get_image_redirects_remote_logo_without_server_fetch(monkeypatch):
    """
    Remote logo URLs should be loaded by the browser, not fetched by the proxy.
    """
    monkeypatch.setenv("UI_LOGO_PATH", "http://invalid-url-12345.com/logo.jpg")

    with mock.patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get") as mock_get:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as ac:
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

    with mock.patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get") as mock_get:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as ac:
            response = await ac.get("/get_image")

    assert response.status_code == 307
    assert response.headers["location"] == "http://example.com/logo.jpg"
    mock_get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["", "?theme=dark", "?variant=monogram", "?theme=dark&variant=monogram"],
)
async def test_get_image_sends_no_cache_header(monkeypatch, query):
    """
    Browsers should revalidate the bundled logo instead of applying heuristic freshness.
    """
    monkeypatch.delenv("UI_LOGO_PATH", raising=False)
    monkeypatch.delenv("UI_LOGO_PATH_DARK", raising=False)
    monkeypatch.delenv("LITELLM_ASSETS_PATH", raising=False)
    monkeypatch.delenv("LITELLM_NON_ROOT", raising=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as ac:
        response = await ac.get(f"/get_image{query}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers.get("etag")


@pytest.mark.asyncio
async def test_get_image_custom_logo_sends_no_cache_header(monkeypatch, tmp_path):
    """
    An admin-configured local logo file should also be revalidated by the browser.
    """
    custom_logo = tmp_path / "custom_logo.png"
    shutil.copy(os.path.join(os.path.dirname(proxy_server.__file__), "logo.png"), custom_logo)
    monkeypatch.setenv("UI_LOGO_PATH", str(custom_logo))
    monkeypatch.delenv("UI_LOGO_PATH_DARK", raising=False)
    monkeypatch.delenv("LITELLM_ASSETS_PATH", raising=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as ac:
        response = await ac.get("/get_image")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"


@pytest.mark.asyncio
async def test_get_favicon_sends_no_cache_header(monkeypatch):
    """
    The default favicon should be revalidated by the browser too.
    """
    monkeypatch.delenv("LITELLM_FAVICON_URL", raising=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as ac:
        response = await ac.get("/get_favicon")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"

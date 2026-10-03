import os
import sys
from pathlib import Path
from typing import Final
from unittest.mock import patch

import httpx
import pytest

from litellm.proxy.proxy_server import app


@pytest.fixture
def packaged_favicon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
    payload: Final = b"\x00\x00\x01\x00packaged-icon"
    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "favicon.ico").write_bytes(payload)
    monkeypatch.setattr("litellm.proxy.common_utils.static_asset_utils.package_files", lambda package: tmp_path)
    return payload


@pytest.mark.asyncio
async def test_get_favicon_default(packaged_favicon: bytes):
    """Test that get_favicon returns the default favicon when no URL set."""
    os.environ.pop("LITELLM_FAVICON_URL", None)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        response = await ac.get("/get_favicon")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/x-icon"
    assert response.content == packaged_favicon


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


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", (True, False))
async def test_get_favicon_custom_file_or_packaged_fallback(
    valid: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packaged_favicon: bytes
) -> None:
    custom: Final = tmp_path / "favicon.ico"
    packaged: Final = packaged_favicon
    custom.write_bytes(packaged + b"custom" if valid else b"not an image")
    monkeypatch.setenv("LITELLM_FAVICON_URL", str(custom))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response: Final = await client.get("/get_favicon")
    assert response.status_code == 200
    assert response.content == (packaged + b"custom" if valid else packaged)
    assert response.headers["content-type"] == "image/x-icon"


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", ("", "missing.ico"))
async def test_missing_packaged_favicon_returns_not_found(
    custom: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_FAVICON_URL", custom)
    with patch("litellm.proxy.common_utils.static_asset_utils.package_files", return_value=tmp_path):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response: Final = await client.get("/get_favicon")
    assert response.status_code == 404
    assert "favicon" in response.json()["detail"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", ("remote", "local", "missing"))
async def test_custom_favicon_without_dashboard_package(
    custom: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    favicon: Final = tmp_path / "custom.ico"
    payload: Final = b"\x00\x00\x01\x00custom-icon"
    favicon.write_bytes(payload)
    url: Final = "https://example.com/custom.ico"
    monkeypatch.setenv("LITELLM_FAVICON_URL", url if custom == "remote" else str(favicon) if custom == "local" else "")
    monkeypatch.setitem(sys.modules, "litellm_proxy_extras", None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response: Final = await client.get("/get_favicon")
    if custom == "remote":
        assert response.status_code == 307
        assert response.headers["location"] == url
    elif custom == "local":
        assert response.status_code == 200
        assert response.content == payload
    else:
        assert response.status_code == 404

"""SSRF guards for GigaChat multimodal image_url downloads."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.litellm_core_utils.url_utils import SSRFError
from litellm.llms.gigachat import file_handler


def test_download_image_sync_blocks_private_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", True)
    mock_client = MagicMock()

    with patch.object(file_handler, "_get_httpx_client", return_value=mock_client):
        with pytest.raises(SSRFError):
            file_handler._download_image_sync("http://127.0.0.1/secret.png")

    mock_client.get.assert_not_called()


def test_upload_file_sync_propagates_ssrf_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", True)
    file_handler._file_cache.clear()

    with patch.object(
        file_handler,
        "_download_image_sync",
        side_effect=SSRFError("blocked"),
    ):
        with pytest.raises(SSRFError, match="blocked"):
            file_handler.upload_file_sync("http://169.254.169.254/latest/meta-data/")


@pytest.mark.asyncio
async def test_download_image_async_blocks_private_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", True)
    mock_client = MagicMock()
    mock_client.get = AsyncMock()

    with patch.object(file_handler, "get_async_httpx_client", return_value=mock_client):
        with pytest.raises(SSRFError):
            await file_handler._download_image_async("http://10.0.0.5/img.png")

    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_upload_file_async_propagates_ssrf_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", True)
    file_handler._file_cache.clear()

    with patch.object(
        file_handler,
        "_download_image_async",
        side_effect=SSRFError("blocked"),
    ):
        with pytest.raises(SSRFError, match="blocked"):
            await file_handler.upload_file_async(
                "http://169.254.169.254/latest/meta-data/"
            )

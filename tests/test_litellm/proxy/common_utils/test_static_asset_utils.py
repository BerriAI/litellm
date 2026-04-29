"""
Unit tests for the unauthenticated logo / favicon endpoint helpers.

Closes the LFI half of GHSA-3pcp-536p-ghjc and the SSRF half of
GHSA-pjc9-2hw6-78rr — both endpoints accept an admin-set env var and
return its contents unauthenticated, so the helpers must reject:

* local paths outside the allowed asset roots (LFI)
* HTTP URLs resolving to private / cloud-metadata addresses (SSRF)
* non-image responses (smuggling JSON / credentials through the
  ``image/jpeg`` response wrapper)
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.litellm_core_utils.url_utils import SSRFError
from litellm.proxy.common_utils.static_asset_utils import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    fetch_validated_image_bytes,
    resolve_local_asset_path,
)


class TestResolveLocalAssetPath:
    @pytest.fixture
    def assets_dir(self, tmp_path):
        d = tmp_path / "assets"
        d.mkdir()
        return d

    def test_returns_resolved_path_for_file_inside_allowed_root(self, assets_dir):
        logo = assets_dir / "logo.jpg"
        logo.write_bytes(b"\xff\xd8\xff")  # JPEG header

        result = resolve_local_asset_path(str(logo), [str(assets_dir)])
        assert result == str(logo.resolve())

    def test_rejects_path_outside_allowed_roots(self, tmp_path, assets_dir):
        outside = tmp_path / "secret.txt"
        outside.write_text("password=hunter2")

        result = resolve_local_asset_path(str(outside), [str(assets_dir)])
        assert result is None

    def test_rejects_etc_passwd(self, assets_dir):
        # The canonical LFI shape from GHSA-3pcp-536p-ghjc.
        result = resolve_local_asset_path("/etc/passwd", [str(assets_dir)])
        assert result is None

    def test_rejects_proc_self_environ(self, assets_dir):
        # Process environment exfil — same shape as /etc/passwd attack.
        result = resolve_local_asset_path("/proc/self/environ", [str(assets_dir)])
        assert result is None

    def test_rejects_symlink_pointing_outside_allowed_roots(self, tmp_path, assets_dir):
        secret = tmp_path / "secret.txt"
        secret.write_text("password=hunter2")
        sneaky = assets_dir / "logo.jpg"
        os.symlink(str(secret), str(sneaky))

        result = resolve_local_asset_path(str(sneaky), [str(assets_dir)])
        assert result is None

    def test_rejects_path_traversal_with_dotdot(self, tmp_path, assets_dir):
        outside = tmp_path / "secret.txt"
        outside.write_text("nope")
        traversal = str(assets_dir / ".." / "secret.txt")

        result = resolve_local_asset_path(traversal, [str(assets_dir)])
        assert result is None

    def test_rejects_directory(self, assets_dir):
        # Path containment requires the resolved entry to be a regular file.
        result = resolve_local_asset_path(str(assets_dir), [str(assets_dir)])
        assert result is None

    def test_rejects_nonexistent_file_inside_allowed_root(self, assets_dir):
        # Even a path that *would* be inside the allowed root must point at
        # an existing file — otherwise we shouldn't pretend it resolves.
        result = resolve_local_asset_path(
            str(assets_dir / "missing.jpg"), [str(assets_dir)]
        )
        assert result is None

    def test_rejects_empty_or_none(self, assets_dir):
        assert resolve_local_asset_path("", [str(assets_dir)]) is None

    def test_skips_empty_or_invalid_roots(self, assets_dir):
        logo = assets_dir / "logo.jpg"
        logo.write_bytes(b"\xff\xd8\xff")
        result = resolve_local_asset_path(
            str(logo), ["", str(assets_dir), "/nonexistent/root"]
        )
        assert result == str(logo.resolve())


class TestFetchValidatedImageBytes:
    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock()
        client.get = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_blocks_private_ip_via_validate_url(self, mock_async_client):
        # The SSRF half of GHSA-pjc9-2hw6-78rr — admin sets logo URL to
        # http://169.254.169.254/iam, attacker hits /get_image, exfils creds.
        with (
            patch(
                "litellm.proxy.common_utils.static_asset_utils.validate_url",
                side_effect=SSRFError("blocked: 169.254.169.254"),
            ),
            patch(
                "litellm.proxy.common_utils.static_asset_utils.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            result = await fetch_validated_image_bytes("http://169.254.169.254/iam")

        assert result is None
        # The fetch must not be attempted when the URL is rejected.
        mock_async_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_non_image_content_type(self, mock_async_client):
        # Even when the URL passes SSRF, the upstream response must be an
        # image. Otherwise an attacker could redirect to an upstream that
        # returns ``application/json`` AWS creds and have them tunneled
        # through the ``image/jpeg`` response wrapper.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"AccessKeyId": "..."}'
        mock_async_client.get.return_value = mock_response

        with (
            patch(
                "litellm.proxy.common_utils.static_asset_utils.validate_url",
                return_value=("http://cdn.example/logo", "cdn.example"),
            ),
            patch(
                "litellm.proxy.common_utils.static_asset_utils.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            result = await fetch_validated_image_bytes("http://cdn.example/logo")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_bytes_for_valid_image_response(self, mock_async_client):
        png_bytes = b"\x89PNG\r\n\x1a\nfake png body"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/png; charset=binary"}
        mock_response.content = png_bytes
        mock_async_client.get.return_value = mock_response

        with (
            patch(
                "litellm.proxy.common_utils.static_asset_utils.validate_url",
                return_value=(
                    "https://cdn.example/logo.png",
                    "cdn.example",
                ),
            ),
            patch(
                "litellm.proxy.common_utils.static_asset_utils.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            result = await fetch_validated_image_bytes("https://cdn.example/logo.png")

        assert result == png_bytes

    @pytest.mark.asyncio
    async def test_returns_none_on_non_200_response(self, mock_async_client):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {"content-type": "image/png"}
        mock_async_client.get.return_value = mock_response

        with (
            patch(
                "litellm.proxy.common_utils.static_asset_utils.validate_url",
                return_value=("https://cdn.example/logo", "cdn.example"),
            ),
            patch(
                "litellm.proxy.common_utils.static_asset_utils.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            result = await fetch_validated_image_bytes("https://cdn.example/logo")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_fetch_exception(self, mock_async_client):
        mock_async_client.get.side_effect = Exception("connection reset")

        with (
            patch(
                "litellm.proxy.common_utils.static_asset_utils.validate_url",
                return_value=("https://cdn.example/logo", "cdn.example"),
            ),
            patch(
                "litellm.proxy.common_utils.static_asset_utils.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            result = await fetch_validated_image_bytes("https://cdn.example/logo")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_url(self):
        result = await fetch_validated_image_bytes("")
        assert result is None

    @pytest.mark.parametrize(
        "content_type",
        sorted(ALLOWED_IMAGE_CONTENT_TYPES),
    )
    @pytest.mark.asyncio
    async def test_accepts_each_allowed_image_content_type(
        self, mock_async_client, content_type
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": content_type}
        mock_response.content = b"image-bytes"
        mock_async_client.get.return_value = mock_response

        with (
            patch(
                "litellm.proxy.common_utils.static_asset_utils.validate_url",
                return_value=("https://cdn.example/logo", "cdn.example"),
            ),
            patch(
                "litellm.proxy.common_utils.static_asset_utils.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            result = await fetch_validated_image_bytes("https://cdn.example/logo")

        assert result == b"image-bytes"

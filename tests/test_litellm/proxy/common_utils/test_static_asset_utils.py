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


def _image_response(*, status_code=200, content_type="image/png", body=b"image-bytes"):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": content_type}
    response.content = body
    return response


def _patch_async_safe_get(*, return_value=None, side_effect=None):
    return patch(
        "litellm.proxy.common_utils.static_asset_utils.async_safe_get",
        new_callable=AsyncMock,
        return_value=return_value,
        side_effect=side_effect,
    )


@pytest.fixture(autouse=True)
def _patch_httpx_client():
    # The helper builds the client first, then hands it to async_safe_get
    # — patch it once for every test so we never accidentally instantiate
    # a real client.
    with patch(
        "litellm.proxy.common_utils.static_asset_utils.get_async_httpx_client",
        return_value=MagicMock(),
    ):
        yield


class TestFetchValidatedImageBytes:
    """
    The helper delegates to ``async_safe_get`` for the SSRF guard +
    redirect handling. Tests mock ``async_safe_get`` directly so they
    exercise the helper's contract (Content-Type validation, status code
    handling, exception fallthrough) without depending on the SSRF
    primitive's internals.
    """

    @pytest.mark.asyncio
    async def test_blocks_ssrf_target(self):
        # ``async_safe_get`` raises SSRFError on private/metadata targets
        # and on redirect hops to those targets — closes the SSRF half of
        # GHSA-pjc9-2hw6-78rr including the redirect-bypass variant.
        with _patch_async_safe_get(side_effect=SSRFError("blocked: 169.254.169.254")):
            result = await fetch_validated_image_bytes("http://169.254.169.254/iam")
        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_non_image_content_type(self):
        # Without this, an upstream that returns ``application/json`` AWS
        # creds would be tunneled through the ``image/jpeg`` response
        # wrapper.
        with _patch_async_safe_get(
            return_value=_image_response(
                content_type="application/json", body=b'{"AccessKeyId": "..."}'
            ),
        ):
            result = await fetch_validated_image_bytes("http://cdn.example/logo")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_bytes_for_valid_image_response(self):
        png_bytes = b"\x89PNG\r\n\x1a\nfake png body"
        with _patch_async_safe_get(
            return_value=_image_response(
                content_type="image/png; charset=binary", body=png_bytes
            ),
        ):
            result = await fetch_validated_image_bytes("https://cdn.example/logo.png")
        assert result == png_bytes

    @pytest.mark.asyncio
    async def test_returns_none_on_non_200_response(self):
        with _patch_async_safe_get(return_value=_image_response(status_code=404)):
            result = await fetch_validated_image_bytes("https://cdn.example/logo")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_fetch_exception(self):
        with _patch_async_safe_get(side_effect=Exception("connection reset")):
            result = await fetch_validated_image_bytes("https://cdn.example/logo")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_url(self):
        result = await fetch_validated_image_bytes("")
        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_svg_content_type(self):
        # ``image/svg+xml`` is intentionally NOT in the allowlist for
        # unauthenticated endpoints — SVG is the only common image
        # format that can embed JavaScript.
        with _patch_async_safe_get(
            return_value=_image_response(
                content_type="image/svg+xml",
                body=b"<svg><script>alert(1)</script></svg>",
            ),
        ):
            result = await fetch_validated_image_bytes("https://cdn.example/x.svg")
        assert result is None

    @pytest.mark.parametrize(
        "content_type",
        sorted(ALLOWED_IMAGE_CONTENT_TYPES),
    )
    @pytest.mark.asyncio
    async def test_accepts_each_allowed_image_content_type(self, content_type):
        with _patch_async_safe_get(
            return_value=_image_response(content_type=content_type),
        ):
            result = await fetch_validated_image_bytes("https://cdn.example/logo")
        assert result == b"image-bytes"

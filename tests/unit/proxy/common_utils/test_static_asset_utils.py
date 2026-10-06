"""
Unit tests for unauthenticated logo / favicon endpoint helpers.

Local image paths are an existing deployment workflow, so the helper keeps
arbitrary local image paths working while refusing non-image files like
``/etc/passwd`` or ``/proc/self/environ``.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from litellm.proxy.common_utils.static_asset_utils import (
    detect_local_image_media_type,
    get_packaged_ui_directory,
    resolve_validated_local_image_path,
)


def test_dashboard_package_is_optional(tmp_path: Path) -> None:
    missing = ModuleNotFoundError("No companion installed", name="litellm_proxy_extras")
    with patch("litellm.proxy.common_utils.static_asset_utils.package_files", side_effect=[tmp_path, missing]):
        assert get_packaged_ui_directory() is None


@pytest.mark.parametrize("package", ("sdk", "companion"))
def test_dashboard_lookup_preserves_unrelated_import_failures(tmp_path: Path, package: str) -> None:
    error = ModuleNotFoundError("unrelated import failed", name="unrelated_dependency")
    with patch(
        "litellm.proxy.common_utils.static_asset_utils.package_files",
        side_effect=[tmp_path, error] if package == "companion" else error,
    ):
        with pytest.raises(ModuleNotFoundError) as raised:
            get_packaged_ui_directory()
    assert raised.value is error


@pytest.mark.parametrize("bundled", (True, False))
def test_dashboard_lookup_preserves_legacy_assets_with_an_old_companion(tmp_path: Path, bundled: bool) -> None:
    sdk = tmp_path / "litellm"
    companion = tmp_path / "litellm_proxy_extras"
    legacy_ui = sdk / "proxy/_experimental/out"
    if bundled:
        legacy_ui.mkdir(parents=True)
        (legacy_ui / "index.html").write_text("<html>legacy dashboard</html>")
    companion.mkdir()
    with patch(
        "litellm.proxy.common_utils.static_asset_utils.package_files",
        side_effect={"litellm": sdk, "litellm_proxy_extras": companion}.__getitem__,
    ):
        assert get_packaged_ui_directory() == str(legacy_ui if bundled else companion / "ui")


@pytest.mark.parametrize(
    ("body", "media_type"),
    [
        (b"\x89PNG\r\n\x1a\nfake png body", "image/png"),
        (b"GIF89a fake gif body", "image/gif"),
        (b"\xff\xd8\xff fake jpeg body", "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBP fake webp body", "image/webp"),
        (b"\x00\x00\x01\x00 fake ico body", "image/x-icon"),
    ],
)
def test_detect_local_image_media_type_accepts_supported_images(body, media_type):
    assert detect_local_image_media_type(body) == media_type


def test_detect_local_image_media_type_rejects_non_images():
    assert detect_local_image_media_type(b"root:x:0:0:root:/root:/bin/bash") is None


class TestResolveValidatedLocalImagePath:
    def test_returns_resolved_path_for_arbitrary_local_image(self, tmp_path):
        logo = tmp_path / "logo.png"
        logo.write_bytes(b"\x89PNG\r\n\x1a\nfake png body")

        result = resolve_validated_local_image_path(str(logo))

        assert result == (str(logo.resolve()), "image/png")

    def test_rejects_etc_passwd(self):
        result = resolve_validated_local_image_path("/etc/passwd")
        assert result is None

    def test_rejects_proc_self_environ(self):
        result = resolve_validated_local_image_path("/proc/self/environ")
        assert result is None

    def test_rejects_symlink_pointing_to_non_image(self, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("password=hunter2")
        symlink = tmp_path / "logo.png"
        os.symlink(str(secret), str(symlink))

        result = resolve_validated_local_image_path(str(symlink))

        assert result is None

    def test_accepts_symlink_pointing_to_image(self, tmp_path):
        logo = tmp_path / "real_logo.png"
        logo.write_bytes(b"\x89PNG\r\n\x1a\nfake png body")
        symlink = tmp_path / "logo.png"
        os.symlink(str(logo), str(symlink))

        result = resolve_validated_local_image_path(str(symlink))

        assert result == (str(logo.resolve()), "image/png")

    def test_rejects_path_traversal_to_non_image(self, tmp_path):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("nope")
        traversal = str(assets_dir / ".." / "secret.txt")

        result = resolve_validated_local_image_path(traversal)

        assert result is None

    def test_rejects_directory(self, tmp_path):
        result = resolve_validated_local_image_path(str(tmp_path))
        assert result is None

    def test_rejects_nonexistent_file(self, tmp_path):
        result = resolve_validated_local_image_path(str(tmp_path / "missing.jpg"))
        assert result is None

    def test_rejects_empty_path(self):
        assert resolve_validated_local_image_path("") is None

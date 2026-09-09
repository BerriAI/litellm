"""Tests for assert_gemini_video_download_url (Veo video.uri credentialed fetch)."""

import pytest

from litellm.llms.gemini.common_utils import (
    GeminiError,
    assert_gemini_video_download_url,
)

API_BASE = "https://generativelanguage.googleapis.com"


class TestAssertGeminiVideoDownloadUrl:
    def test_accepts_same_origin_absolute_uri(self):
        url = assert_gemini_video_download_url(
            "https://generativelanguage.googleapis.com/v1beta/files/abc:download?alt=media",
            API_BASE,
        )
        assert url.startswith("https://generativelanguage.googleapis.com/")

    def test_resolves_relative_file_uri_against_api_base(self):
        url = assert_gemini_video_download_url("files/abc123xyz", API_BASE)
        assert url == "https://generativelanguage.googleapis.com/files/abc123xyz"

    def test_rejects_off_origin_absolute_uri(self):
        with pytest.raises(GeminiError, match="Rejected video download URL"):
            assert_gemini_video_download_url("https://evil.com/steal-key", API_BASE)

    def test_rejects_http_scheme(self):
        with pytest.raises(GeminiError, match="Rejected video download URL"):
            assert_gemini_video_download_url(
                "http://generativelanguage.googleapis.com/v1beta/files/abc",
                API_BASE,
            )

    def test_rejects_missing_uri(self):
        with pytest.raises(GeminiError, match="missing uri"):
            assert_gemini_video_download_url("", API_BASE)

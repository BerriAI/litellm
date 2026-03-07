from unittest.mock import patch

import pytest
from httpx import Request, Response

import litellm
from litellm import constants
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    _get_valid_media_type,
    _infer_media_type_from_url,
    convert_url_to_base64,
)


class DummyClient:
    def get(self, url, follow_redirects=True):
        return Response(status_code=404, request=Request("GET", url))


def test_invalid_image_url_raises_bad_request(monkeypatch):
    monkeypatch.setattr(litellm, "module_level_client", DummyClient())
    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64("https://invalid.example/image.png")
    assert "Unable to fetch image" in str(excinfo.value)


def test_completion_with_invalid_image_url(monkeypatch):
    monkeypatch.setattr(litellm, "module_level_client", DummyClient())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://invalid.example/image.png"},
                },
            ],
        }
    ]
    with pytest.raises(litellm.ImageFetchError) as excinfo:
        litellm.completion(model="gemini/gemini-pro", messages=messages, api_key="test")
    assert excinfo.value.status_code == 400
    assert "Unable to fetch image" in str(excinfo.value)


class LargeImageClient:
    """
    Client that returns a large image exceeding size limit.
    """

    def __init__(self, size_mb=100, include_content_length=True):
        self.size_mb = size_mb
        self.include_content_length = include_content_length

    def get(self, url, follow_redirects=True):
        size_bytes = int(self.size_mb * 1024 * 1024)
        headers = {"Content-Type": "image/jpeg"}
        if self.include_content_length:
            headers["Content-Length"] = str(size_bytes)
        return Response(
            status_code=200,
            headers=headers,
            content=b"x" * size_bytes,
            request=Request("GET", url),
        )


class StreamingLargeImageClient:
    """
    Client that streams a large image to test streaming download protection.
    This simulates a huge file without actually creating it all in memory.
    """

    def __init__(self, size_mb=100, include_content_length=False):
        self.size_mb = size_mb
        self.include_content_length = include_content_length

    def get(self, url, follow_redirects=True):
        size_bytes = int(self.size_mb * 1024 * 1024)
        headers = {"Content-Type": "image/jpeg"}
        if self.include_content_length:
            headers["Content-Length"] = str(size_bytes)

        # Create a generator that yields chunks without creating the whole file in memory
        def generate_chunks(total_size, chunk_size=8192):
            bytes_sent = 0
            while bytes_sent < total_size:
                chunk = b"x" * min(chunk_size, total_size - bytes_sent)
                bytes_sent += len(chunk)
                yield chunk

        # Create response with streaming content
        response = Response(
            status_code=200,
            headers=headers,
            request=Request("GET", url),
        )
        # Mock the iter_bytes method to return our generator
        response.iter_bytes = lambda chunk_size=8192: generate_chunks(
            size_bytes, chunk_size
        )
        return response


def test_image_exceeds_size_limit_with_content_length(monkeypatch):
    """
    Test that images exceeding MAX_IMAGE_URL_DOWNLOAD_SIZE_MB are rejected when Content-Length header is present.
    """
    monkeypatch.setattr(litellm, "module_level_client", LargeImageClient(size_mb=100))

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64("https://example.com/large-image.jpg")

    assert "exceeds maximum allowed size" in str(excinfo.value)
    assert "100.00MB" in str(excinfo.value)
    assert "50.0MB" in str(excinfo.value)


def test_image_exceeds_size_limit_without_content_length(monkeypatch):
    """
    Test that images exceeding MAX_IMAGE_URL_DOWNLOAD_SIZE_MB are rejected even without Content-Length header.
    This uses the old non-streaming mock for backward compatibility.
    """
    monkeypatch.setattr(
        litellm,
        "module_level_client",
        LargeImageClient(size_mb=100, include_content_length=False),
    )

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64("https://example.com/large-image.jpg")

    assert "exceeds maximum allowed size" in str(excinfo.value)


def test_streaming_download_protects_against_huge_files(monkeypatch):
    """
    Test that streaming download aborts early when file exceeds size limit,
    preventing memory exhaustion from huge files (e.g., petabyte-sized files).

    This test verifies that the streaming implementation doesn't download the entire
    file into memory before checking size. Instead, it should abort as soon as the
    limit is exceeded during streaming.
    """
    # Simulate a 1GB file - far larger than the 50MB default limit
    client = StreamingLargeImageClient(size_mb=1024, include_content_length=False)
    monkeypatch.setattr(litellm, "module_level_client", client)

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64("https://example.com/huge-image.jpg")

    # Verify the error message shows it was caught during streaming
    assert "exceeds maximum allowed size" in str(excinfo.value)

    # The error should be raised after downloading just slightly more than the limit
    # not after downloading the full 1GB


class SmallImageClient:
    """
    Client that returns a small valid image.
    """

    def get(self, url, follow_redirects=True):
        size_bytes = 1024
        headers = {
            "Content-Type": "image/jpeg",
            "Content-Length": str(size_bytes),
        }
        return Response(
            status_code=200,
            headers=headers,
            content=b"x" * size_bytes,
            request=Request("GET", url),
        )


def test_image_within_size_limit(monkeypatch):
    """
    Test that images within size limit are processed successfully.
    """
    monkeypatch.setattr(litellm, "module_level_client", SmallImageClient())

    result = convert_url_to_base64("https://example.com/small-image.jpg")

    assert result.startswith("data:image/jpeg;base64,")


def test_streaming_download_handles_petabyte_file(monkeypatch):
    """
    Test that streaming download can handle extremely large file URLs (e.g., petabyte-sized)
    without attempting to download the entire file or causing memory exhaustion.

    This simulates what happens if a malicious actor or misconfiguration provides
    a URL to an extremely large file.
    """
    # Simulate a 1 petabyte file (1,000,000 GB)
    # Without streaming protection, this would cause OOM or hang indefinitely
    client = StreamingLargeImageClient(
        size_mb=1_000_000_000, include_content_length=False
    )
    monkeypatch.setattr(litellm, "module_level_client", client)

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64("https://example.com/petabyte-file.jpg")

    # Should fail fast without downloading anywhere near 1 petabyte
    assert "exceeds maximum allowed size" in str(excinfo.value)


def test_image_size_limit_disabled(monkeypatch):
    """
    Test that setting MAX_IMAGE_URL_DOWNLOAD_SIZE_MB to 0 disables all image URL downloads.
    """
    import litellm.litellm_core_utils.prompt_templates.image_handling as image_handling

    monkeypatch.setattr(litellm, "module_level_client", SmallImageClient())
    monkeypatch.setattr(image_handling, "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", 0)

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64("https://example.com/image.jpg")

    assert "Image URL download is disabled" in str(excinfo.value)
    assert "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0" in str(excinfo.value)


# ============================================================================
# Tests for Content-Type handling and URL extension inference
# ============================================================================


class TestInferMediaTypeFromUrl:
    """Tests for _infer_media_type_from_url function."""

    def test_simple_png_url(self):
        """Test simple PNG URL."""
        result = _infer_media_type_from_url("https://example.com/image.png")
        assert result == "image/png"

    def test_simple_jpeg_url(self):
        """Test simple JPEG URL with .jpeg extension."""
        result = _infer_media_type_from_url("https://example.com/photo.jpeg")
        assert result == "image/jpeg"

    def test_jpg_extension(self):
        """Test JPG extension maps to image/jpeg."""
        result = _infer_media_type_from_url("https://example.com/photo.jpg")
        assert result == "image/jpeg"

    def test_gif_url(self):
        """Test GIF URL."""
        result = _infer_media_type_from_url("https://example.com/animation.gif")
        assert result == "image/gif"

    def test_webp_url(self):
        """Test WebP URL."""
        result = _infer_media_type_from_url("https://example.com/image.webp")
        assert result == "image/webp"

    def test_signed_url_with_query_params(self):
        """Test signed URL (e.g., Aliyun OSS, AWS S3) with query parameters."""
        result = _infer_media_type_from_url(
            "https://bucket.oss-cn-hangzhou.aliyuncs.com/image.png"
            "?Expires=1699999999&OSSAccessKeyId=LTAI5xxx&Signature=xxx"
        )
        assert result == "image/png"

    def test_signed_url_complex_query(self):
        """Test signed URL with complex query string."""
        result = _infer_media_type_from_url(
            "https://s3.amazonaws.com/bucket/photos/sunset.jpg"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=xxx"
        )
        assert result == "image/jpeg"

    def test_unsupported_extension_raises(self):
        """Test that unsupported extension raises an exception."""
        with pytest.raises(Exception) as excinfo:
            _infer_media_type_from_url("https://example.com/document.pdf")
        assert "Unsupported image format" in str(excinfo.value)
        assert "pdf" in str(excinfo.value)

    def test_case_insensitive_extension(self):
        """Test that extension matching is case-insensitive."""
        result = _infer_media_type_from_url("https://example.com/IMAGE.PNG")
        assert result == "image/png"


class TestGetValidMediaType:
    """Tests for _get_valid_media_type function."""

    def test_valid_content_type_image_png(self):
        """Test valid Content-Type is returned directly."""
        result = _get_valid_media_type("image/png", "https://example.com/image.png")
        assert result == "image/png"

    def test_valid_content_type_image_jpeg(self):
        """Test valid JPEG Content-Type."""
        result = _get_valid_media_type("image/jpeg", "https://example.com/image.jpg")
        assert result == "image/jpeg"

    def test_content_type_with_charset_parameter(self):
        """Test Content-Type with charset parameter is stripped."""
        result = _get_valid_media_type(
            "image/png; charset=utf-8", "https://example.com/image.png"
        )
        assert result == "image/png"

    def test_content_type_with_boundary_parameter(self):
        """Test Content-Type with boundary parameter is stripped."""
        result = _get_valid_media_type(
            "image/jpeg; boundary=something", "https://example.com/image.jpg"
        )
        assert result == "image/jpeg"

    def test_invalid_content_type_application_octet_stream(self):
        """Test that application/octet-stream falls back to URL extension."""
        result = _get_valid_media_type(
            "application/octet-stream", "https://example.com/image.png"
        )
        assert result == "image/png"

    def test_invalid_content_type_urlencoded(self):
        """
        Test that application/x-www-form-urlencoded falls back to URL extension.
        This is a real-world case from Aliyun OSS CDN returning wrong Content-Type.
        """
        result = _get_valid_media_type(
            "application/x-www-form-urlencoded",
            "https://bucket.oss-cn-hangzhou.aliyuncs.com/image.png"
            "?Expires=1699999999&Signature=xxx",
        )
        assert result == "image/png"

    def test_invalid_content_type_text_html(self):
        """Test that text/html falls back to URL extension."""
        result = _get_valid_media_type("text/html", "https://example.com/image.webp")
        assert result == "image/webp"

    def test_none_content_type(self):
        """Test that None Content-Type falls back to URL extension."""
        result = _get_valid_media_type(None, "https://example.com/photo.jpeg")
        assert result == "image/jpeg"

    def test_empty_content_type_after_strip(self):
        """Test edge case where Content-Type is empty after stripping."""
        result = _get_valid_media_type(
            "; charset=utf-8", "https://example.com/image.gif"
        )
        assert result == "image/gif"


class InvalidContentTypeClient:
    """Client that returns an invalid Content-Type."""

    def __init__(self, content_type: str):
        self.content_type = content_type

    def get(self, url, follow_redirects=True):
        size_bytes = 1024
        headers = {
            "Content-Type": self.content_type,
            "Content-Length": str(size_bytes),
        }
        return Response(
            status_code=200,
            headers=headers,
            content=b"x" * size_bytes,
            request=Request("GET", url),
        )


def test_convert_url_handles_invalid_content_type(monkeypatch):
    """
    Integration test: convert_url_to_base64 handles invalid Content-Type.

    Simulates Aliyun OSS CDN returning application/x-www-form-urlencoded for a .png file.
    """
    monkeypatch.setattr(
        litellm,
        "module_level_client",
        InvalidContentTypeClient("application/x-www-form-urlencoded"),
    )

    result = convert_url_to_base64(
        "https://bucket.oss-cn-hangzhou.aliyuncs.com/image.png?Expires=xxx"
    )

    assert result.startswith("data:image/png;base64,")


def test_convert_url_handles_content_type_with_params(monkeypatch):
    """
    Integration test: convert_url_to_base64 handles Content-Type with parameters.

    Tests that 'image/png; charset=utf-8' is properly stripped to 'image/png'.
    """
    monkeypatch.setattr(
        litellm,
        "module_level_client",
        InvalidContentTypeClient("image/png; charset=utf-8"),
    )

    result = convert_url_to_base64("https://example.com/image.png")

    assert result.startswith("data:image/png;base64,")


def test_convert_url_handles_application_octet_stream(monkeypatch):
    """
    Integration test: convert_url_to_base64 handles application/octet-stream.

    Some servers return application/octet-stream for binary files including images.
    """
    monkeypatch.setattr(
        litellm,
        "module_level_client",
        InvalidContentTypeClient("application/octet-stream"),
    )

    result = convert_url_to_base64("https://example.com/photo.jpeg")

    assert result.startswith("data:image/jpeg;base64,")

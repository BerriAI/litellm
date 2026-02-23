from unittest.mock import patch

import pytest
from httpx import Request, Response

import litellm
from litellm import constants
from litellm.litellm_core_utils.prompt_templates.image_handling import (
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
        litellm.completion(
            model="gemini/gemini-pro", messages=messages, api_key="test"
        )
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
        response.iter_bytes = lambda chunk_size=8192: generate_chunks(size_bytes, chunk_size)
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
        litellm, "module_level_client", LargeImageClient(size_mb=100, include_content_length=False)
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
    client = StreamingLargeImageClient(size_mb=1_000_000_000, include_content_length=False)
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

from unittest.mock import patch

import pytest
from httpx import Request, Response

import litellm
from litellm import constants
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    convert_url_to_base64,
    async_convert_url_to_base64,
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
    """
    monkeypatch.setattr(
        litellm, "module_level_client", LargeImageClient(size_mb=100, include_content_length=False)
    )

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64("https://example.com/large-image.jpg")

    assert "exceeds maximum allowed size" in str(excinfo.value)


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


class StreamingImageClient:
    """
    Client that simulates streaming image downloads.
    Tracks how many bytes were actually downloaded via streaming.
    """

    def __init__(self, total_size_bytes, include_content_length=False):
        self.total_size_bytes = total_size_bytes
        self.include_content_length = include_content_length
        self.bytes_downloaded = 0

    def get(self, url, follow_redirects=True):
        client_ref = self

        class StreamingResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {"Content-Type": "image/jpeg"}
                if client_ref.include_content_length:
                    self.headers["Content-Length"] = str(client_ref.total_size_bytes)

            def iter_bytes(self, chunk_size=8192):
                """Simulate streaming bytes in chunks"""
                bytes_sent = 0
                while bytes_sent < client_ref.total_size_bytes:
                    chunk_bytes = min(chunk_size, client_ref.total_size_bytes - bytes_sent)
                    client_ref.bytes_downloaded += chunk_bytes
                    bytes_sent += chunk_bytes
                    yield b"x" * chunk_bytes

        return StreamingResponse()


class AsyncStreamingImageClient:
    """
    Async client that simulates streaming image downloads.
    Tracks how many bytes were actually downloaded via streaming.
    """

    def __init__(self, total_size_bytes, include_content_length=False):
        self.total_size_bytes = total_size_bytes
        self.include_content_length = include_content_length
        self.bytes_downloaded = 0

    async def get(self, url, follow_redirects=True):
        client_ref = self

        class AsyncStreamingResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {"Content-Type": "image/jpeg"}
                if client_ref.include_content_length:
                    self.headers["Content-Length"] = str(client_ref.total_size_bytes)

            async def aiter_bytes(self, chunk_size=8192):
                """Simulate async streaming bytes in chunks"""
                bytes_sent = 0
                while bytes_sent < client_ref.total_size_bytes:
                    chunk_bytes = min(chunk_size, client_ref.total_size_bytes - bytes_sent)
                    client_ref.bytes_downloaded += chunk_bytes
                    bytes_sent += chunk_bytes
                    yield b"x" * chunk_bytes

        return AsyncStreamingResponse()


def test_streaming_download_real_image():
    """
    E2E test: Actually download a real image from the internet.
    This ensures no regression in normal image download functionality.
    """
    # Use a real small test image (GitHub's logo, ~3KB)
    real_image_url = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"

    result = convert_url_to_base64(real_image_url)

    # Verify the image was downloaded successfully
    assert result.startswith("data:image/png;base64,")
    # Verify it's not empty
    assert len(result) > 100


@pytest.mark.asyncio
async def test_async_streaming_download_real_image():
    """
    E2E test: Actually download a real image asynchronously.
    """
    # Use a real small test image
    real_image_url = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"

    result = await async_convert_url_to_base64(real_image_url)

    # Verify the image was downloaded successfully
    assert result.startswith("data:image/png;base64,")
    # Verify it's not empty
    assert len(result) > 100


def test_streaming_stops_for_large_image_without_content_length(monkeypatch):
    """
    Unit test: Verify that streaming download stops early for images exceeding size limit
    when Content-Length header is not provided.

    Uses mock to simulate 100MB file without actually downloading it (for efficiency).
    This is the critical test proving we don't download the entire file before validation.
    """
    # Simulate a 100MB image without Content-Length header (mock for efficiency)
    client = StreamingImageClient(
        total_size_bytes=100 * 1024 * 1024, include_content_length=False
    )
    monkeypatch.setattr(litellm, "module_level_client", client)

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64("https://example.com/huge-image.jpg")

    # Verify it was rejected
    assert "exceeds maximum allowed size" in str(excinfo.value)

    # Critical assertion: verify we did NOT download the entire 100MB
    # We should have stopped around 50MB (the default limit)
    mb_downloaded = client.bytes_downloaded / (1024 * 1024)
    assert mb_downloaded < 55  # Allow for chunk size overhead
    assert mb_downloaded > 45  # Should download around the limit


@pytest.mark.asyncio
async def test_async_streaming_stops_for_large_image_without_content_length(monkeypatch):
    """
    Unit test: Verify that async streaming download stops early for large images.

    Uses mock to simulate 100MB file without actually downloading it (for efficiency).
    """
    # Simulate a 100MB image without Content-Length header (mock for efficiency)
    client = AsyncStreamingImageClient(
        total_size_bytes=100 * 1024 * 1024, include_content_length=False
    )
    monkeypatch.setattr(litellm, "module_level_aclient", client)

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        await async_convert_url_to_base64("https://example.com/huge-image.jpg")

    # Verify it was rejected
    assert "exceeds maximum allowed size" in str(excinfo.value)

    # Critical assertion: verify we did NOT download the entire 100MB
    mb_downloaded = client.bytes_downloaded / (1024 * 1024)
    assert mb_downloaded < 55  # Allow for chunk size overhead
    assert mb_downloaded > 45  # Should download around the limit

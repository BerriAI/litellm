import asyncio
import copy
import time
import uuid
from unittest.mock import patch

import pytest
from httpx import Request, Response

import litellm
from litellm import constants
from litellm.litellm_core_utils.prompt_templates import image_handling
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    MAX_CONCURRENT_REMOTE_MEDIA_FETCHES,
    RemoteMedia,
    async_convert_url_to_base64,
    async_inline_remote_media,
    convert_url_to_base64,
)
from litellm.litellm_core_utils.url_utils import SSRFError


@pytest.fixture(autouse=True)
def _bypass_ssrf(monkeypatch):
    """Bypass SSRF validation in image handling tests — tests use fake URLs."""
    monkeypatch.setattr(
        image_handling,
        "safe_get",
        lambda client, url, **kw: client.get(url, follow_redirects=True),
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
    client = StreamingLargeImageClient(size_mb=1_000_000_000, include_content_length=False)
    monkeypatch.setattr(litellm, "module_level_client", client)

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64("https://example.com/petabyte-file.jpg")

    # Should fail fast without downloading anywhere near 1 petabyte
    assert "exceeds maximum allowed size" in str(excinfo.value)


def test_data_url_is_returned_unchanged_without_fetch(monkeypatch):
    """
    A data URL is already inline base64 image data, so convert_url_to_base64
    must return it as-is instead of attempting an HTTP fetch.
    """

    class ExplodingClient:
        def get(self, url, follow_redirects=True):
            raise AssertionError("data URLs must not trigger an HTTP fetch")

    monkeypatch.setattr(litellm, "module_level_client", ExplodingClient())

    data_url = "data:image/png;base64,iVBORw0KGgo="

    assert convert_url_to_base64(data_url) == data_url


@pytest.mark.asyncio
async def test_async_data_url_is_returned_unchanged_without_fetch(monkeypatch):
    """
    The async path must short-circuit data URLs identically to the sync path,
    otherwise async OCR flows would attempt an impossible HTTP fetch.
    """

    class ExplodingAsyncClient:
        async def get(self, url, follow_redirects=True):
            raise AssertionError("data URLs must not trigger an HTTP fetch")

    monkeypatch.setattr(litellm, "module_level_aclient", ExplodingAsyncClient())

    data_url = "data:image/png;base64,iVBORw0KGgo="

    assert await async_convert_url_to_base64(data_url) == data_url


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


async def test_async_inline_remote_media_inlines_every_remote_part_shape(async_only_image_fetch):
    image_url = f"http://img.example/{uuid.uuid4()}.png"
    pdf_url = f"http://docs.example/{uuid.uuid4()}.pdf"
    messages = [
        {"role": "system", "content": "be terse"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                {"type": "image_url", "image_url": image_url},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                {"type": "file", "file": {"file_id": pdf_url}},
                {"type": "file", "file": {"file_id": image_url, "format": "image/png"}},
                {"type": "document", "source": {"type": "url", "url": pdf_url}, "title": "the doc"},
                {"type": "image", "source": {"type": "url", "url": image_url}},
                {"type": "document", "source": {"type": "file", "file_id": "file_abc"}},
            ],
        },
    ]
    snapshot = copy.deepcopy(messages)

    inlined = await async_inline_remote_media(messages)

    data_url = async_only_image_fetch.data_url
    base64_png = async_only_image_fetch.base64_png
    assert inlined[0] == {"role": "system", "content": "be terse"}
    assert inlined[1]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
        {"type": "image_url", "image_url": data_url},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        {"type": "file", "file": {"format": "application/pdf", "file_data": data_url}},
        {"type": "file", "file": {"format": "image/png", "file_data": data_url}},
        {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": base64_png},
            "title": "the doc",
        },
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_png}},
        {"type": "document", "source": {"type": "file", "file_id": "file_abc"}},
    ]
    assert sorted(async_only_image_fetch.fetched) == sorted([image_url, pdf_url])
    assert messages == snapshot


async def test_async_inline_remote_media_inlines_only_the_parts_the_predicate_accepts(async_only_image_fetch):
    files_api_prefix = "https://generativelanguage.googleapis.com/v1beta/files/"
    files_api_pdf = f"{files_api_prefix}{uuid.uuid4().hex}"
    hinted_image = f"https://img.example/{uuid.uuid4()}.png"
    plain_image = f"https://img.example/{uuid.uuid4()}.png"
    hinted_document = f"https://docs.example/{uuid.uuid4()}.pdf"
    seen = []

    def inline_unhinted_outside_files_api(media: RemoteMedia) -> bool:
        seen.append(media)
        return not media.url.startswith(files_api_prefix) and "format" not in media.fields

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": {"file_id": files_api_pdf}},
                {"type": "image_url", "image_url": {"url": hinted_image, "format": "image/png"}},
                {"type": "image_url", "image_url": {"url": plain_image}},
                {"type": "image_url", "image_url": plain_image},
                {"type": "document", "source": {"type": "url", "url": hinted_document, "format": "application/pdf"}},
            ],
        }
    ]
    snapshot = copy.deepcopy(messages)

    inlined = await async_inline_remote_media(messages, should_inline=inline_unhinted_outside_files_api)

    assert inlined[0]["content"] == [
        {"type": "file", "file": {"file_id": files_api_pdf}},
        {"type": "image_url", "image_url": {"url": hinted_image, "format": "image/png"}},
        {"type": "image_url", "image_url": {"url": async_only_image_fetch.data_url}},
        {"type": "image_url", "image_url": async_only_image_fetch.data_url},
        {"type": "document", "source": {"type": "url", "url": hinted_document, "format": "application/pdf"}},
    ]
    assert async_only_image_fetch.fetched == [plain_image]
    assert [(media.url, dict(media.fields)) for media in seen[:5]] == [
        (files_api_pdf, {"file_id": files_api_pdf}),
        (hinted_image, {"url": hinted_image, "format": "image/png"}),
        (plain_image, {"url": plain_image}),
        (plain_image, {}),
        (hinted_document, {"type": "url", "url": hinted_document, "format": "application/pdf"}),
    ]
    assert messages == snapshot


async def test_async_inline_remote_media_inlines_a_shared_url_only_where_the_predicate_accepts_it(
    async_only_image_fetch,
):
    shared = f"https://img.example/{uuid.uuid4()}.png"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": shared, "format": "image/png"}},
                {"type": "image_url", "image_url": {"url": shared}},
            ],
        }
    ]

    inlined = await async_inline_remote_media(messages, should_inline=lambda media: "format" not in media.fields)

    assert inlined[0]["content"] == [
        {"type": "image_url", "image_url": {"url": shared, "format": "image/png"}},
        {"type": "image_url", "image_url": {"url": async_only_image_fetch.data_url}},
    ]
    assert async_only_image_fetch.fetched == [shared]


async def test_async_inline_remote_media_cancels_the_other_fetches_when_one_fails(monkeypatch):
    missing = f"http://img.example/{uuid.uuid4()}-missing.png"
    slow = f"http://img.example/{uuid.uuid4()}-slow.png"
    slow_fetch_outcomes = []

    async def serve(client, url, **kwargs):
        if url == missing:
            return Response(404, request=Request("GET", url))
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            slow_fetch_outcomes.append("cancelled")
            raise
        slow_fetch_outcomes.append("finished")
        return Response(200, content=b"\x89PNG", headers={"content-type": "image/png"}, request=Request("GET", url))

    monkeypatch.setattr(image_handling, "async_safe_get", serve)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": missing}},
                {"type": "image_url", "image_url": {"url": slow}},
            ],
        }
    ]
    started = time.perf_counter()

    with pytest.raises(litellm.ImageFetchError, match="Status code: 404"):
        await async_inline_remote_media(messages)

    assert slow_fetch_outcomes == ["cancelled"]
    assert time.perf_counter() - started < 1


_SSRF_VERDICTS = (
    SSRFError(
        "URL targets a blocked address (10.0.0.8). If this is a legitimate internal service, "
        "add the host to `user_url_allowed_hosts` in general_settings."
    ),
    SSRFError("DNS resolution failed for 'internal.example': [Errno 8] nodename nor servname provided, or not known"),
    SSRFError("No addresses found for 'internal.example'"),
)


def _assert_verdict_free_messages(messages, url):
    assert len(messages) == len(_SSRF_VERDICTS)
    assert len(set(messages)) == 1, "a caller must not be able to tell a blocked host from one that does not resolve"
    message = messages[0]
    assert "The proxy could not resolve this host or its URL policy rejected it" in message
    assert "user_url_allowed_hosts" in message
    assert url in message
    assert "10.0.0.8" not in message
    assert "DNS" not in message
    assert "No addresses" not in message


async def test_async_convert_url_to_base64_hides_the_ssrf_verdict_and_does_not_retry(monkeypatch):
    attempts = []
    messages = []
    url = f"http://internal.example/{uuid.uuid4()}.png"

    for verdict in _SSRF_VERDICTS:

        async def block(client, fetched_url, verdict=verdict, **kwargs):
            attempts.append(fetched_url)
            raise verdict

        monkeypatch.setattr(image_handling, "async_safe_get", block)
        with pytest.raises(litellm.ImageFetchError) as raised:
            await async_convert_url_to_base64(url)
        messages.append(raised.value.message)

    assert attempts == [url] * len(_SSRF_VERDICTS)
    _assert_verdict_free_messages(messages, url)


def test_convert_url_to_base64_hides_the_ssrf_verdict_and_does_not_retry(monkeypatch):
    attempts = []
    messages = []
    url = f"http://internal.example/{uuid.uuid4()}.png"

    for verdict in _SSRF_VERDICTS:

        def block(client, fetched_url, verdict=verdict, **kwargs):
            attempts.append(fetched_url)
            raise verdict

        monkeypatch.setattr(image_handling, "safe_get", block)
        with pytest.raises(litellm.ImageFetchError) as raised:
            convert_url_to_base64(url)
        messages.append(raised.value.message)

    assert attempts == [url] * len(_SSRF_VERDICTS)
    _assert_verdict_free_messages(messages, url)


async def test_async_inline_remote_media_caps_in_flight_fetches_per_request(monkeypatch):
    in_flight = {"now": 0, "peak": 0}

    async def serve_png_slowly(client, url, **kwargs):
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1
        return Response(200, content=b"\x89PNG", headers={"content-type": "image/png"}, request=Request("GET", url))

    monkeypatch.setattr(image_handling, "async_safe_get", serve_png_slowly)
    urls = [f"https://img.example/{uuid.uuid4()}.png" for _ in range(MAX_CONCURRENT_REMOTE_MEDIA_FETCHES + 5)]
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}} for url in urls]}]

    inlined = await async_inline_remote_media(messages)

    assert in_flight["peak"] == MAX_CONCURRENT_REMOTE_MEDIA_FETCHES
    assert all(part["image_url"]["url"].startswith("data:image/png;base64,") for part in inlined[0]["content"])


async def test_async_inline_remote_media_leaves_messages_without_remote_parts_alone(async_only_image_fetch):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}],
        },
    ]

    assert await async_inline_remote_media(messages) is messages
    assert async_only_image_fetch.fetched == []


async def test_async_inline_remote_media_raises_image_fetch_error_when_the_fetch_fails(monkeypatch):
    async def serve_404(client, url, **kwargs):
        return Response(404, request=Request("GET", url))

    monkeypatch.setattr(image_handling, "async_safe_get", serve_404)
    url = f"http://img.example/{uuid.uuid4()}.png"

    with pytest.raises(litellm.ImageFetchError, match="Status code: 404"):
        await async_inline_remote_media([{"role": "user", "content": [{"type": "image_url", "image_url": url}]}])

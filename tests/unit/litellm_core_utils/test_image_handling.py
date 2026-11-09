import asyncio
import base64
import copy
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from unittest.mock import MagicMock

import pytest
from httpx import Request, Response

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils import url_utils
from litellm.litellm_core_utils.prompt_templates import image_handling
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    MAX_CONCURRENT_REMOTE_MEDIA_FETCHES,
    RemoteMedia,
    async_convert_url_to_base64,
    async_format_pdf_links,
    async_inline_remote_media,
    async_serves_pdf,
    convert_url_to_base64,
    format_pdf_links,
    inline_remote_media,
    is_unformatted_pdf_link_image,
    serves_pdf,
)
from litellm.litellm_core_utils.url_utils import SSRFError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


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


REAL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _stub_image_client(content, content_type):
    class _Client:
        def get(self, url, follow_redirects=True):
            headers = {} if content_type is None else {"Content-Type": content_type}
            return Response(200, content=content, headers=headers, request=Request("GET", url))

    return _Client()


def test_convert_url_to_base64_infers_the_type_when_the_server_sends_octet_stream(monkeypatch):
    monkeypatch.setattr(litellm, "module_level_client", _stub_image_client(REAL_PNG_BYTES, "application/octet-stream"))

    result = convert_url_to_base64(f"http://img.example/{uuid.uuid4()}")

    assert result.startswith("data:image/png;base64,")


def test_convert_url_to_base64_keeps_a_real_content_type(monkeypatch):
    monkeypatch.setattr(litellm, "module_level_client", _stub_image_client(REAL_PNG_BYTES, "image/jpeg"))

    result = convert_url_to_base64(f"http://img.example/{uuid.uuid4()}.png")

    assert result.startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize(
    ("url", "content_type", "expected"),
    [
        (
            "https://docs.example/document.pdf",
            "application/pdf; qs=0.001",
            "data:application/pdf;base64,cGRm",
        ),
        (
            "https://img.example/image.png",
            "image/png; charset=binary",
            "data:image/png;base64,cGRm",
        ),
        (
            "https://docs.example/document.pdf",
            "application/octet-stream; charset=binary",
            "data:application/pdf;base64,cGRm",
        ),
    ],
)
def test_convert_url_to_base64_uses_bare_content_type(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    content_type: str,
    expected: str,
) -> None:
    monkeypatch.setattr(litellm, "module_level_client", _stub_image_client(b"pdf", content_type))

    assert convert_url_to_base64(url) == expected


def test_convert_url_to_base64_raises_when_no_content_type_is_determinable(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "module_level_client",
        _stub_image_client(b"\x00\x01\x02\x03not-an-image", "application/octet-stream"),
    )
    url = f"http://img.example/{uuid.uuid4()}"

    with pytest.raises(litellm.ImageFetchError) as excinfo:
        convert_url_to_base64(url)

    assert url in str(excinfo.value)


def test_image_size_limit_disabled(monkeypatch):
    """
    Test that setting MAX_IMAGE_URL_DOWNLOAD_SIZE_MB to 0 disables all image URL downloads.
    """
    from litellm.litellm_core_utils.prompt_templates import image_handling

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


def test_inline_remote_media_inlines_every_remote_part_shape(monkeypatch):
    image_url = f"http://img.example/{uuid.uuid4()}.png"
    pdf_url = f"http://docs.example/{uuid.uuid4()}.pdf"
    fetched = []

    def fake_convert(url):
        fetched.append(url)
        return f"data:image/png;base64,{url}"

    monkeypatch.setattr(image_handling, "convert_url_to_base64", fake_convert)
    messages = [
        {"role": "system", "content": "be terse"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                {"type": "image_url", "image_url": image_url},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                {"type": "image_url", "image_url": {"url": "s3://bucket/key.png"}},
                {"type": "file", "file": {"file_id": pdf_url}},
                {"type": "document", "source": {"type": "url", "url": pdf_url}, "title": "the doc"},
            ],
        },
    ]
    snapshot = copy.deepcopy(messages)

    inlined = inline_remote_media(messages, should_inline=image_handling.inline_remote_image_urls)

    data_url = f"data:image/png;base64,{image_url}"
    assert inlined[0] == {"role": "system", "content": "be terse"}
    assert inlined[1]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
        {"type": "image_url", "image_url": data_url},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        {"type": "image_url", "image_url": {"url": "s3://bucket/key.png"}},
        {"type": "file", "file": {"file_id": pdf_url}},
        {"type": "document", "source": {"type": "url", "url": pdf_url}, "title": "the doc"},
    ]
    assert fetched == [image_url]
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
        "add the host to `user_url_allowed_hosts` in litellm_settings."
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


def test_convert_url_to_base64_retries_within_one_deadline(monkeypatch):
    timeouts = []

    def time_out(client, url, timeout=None, **kwargs):
        timeouts.append(timeout)
        time.sleep(timeout)
        raise litellm.Timeout(message="timed out", model="", llm_provider="")

    monkeypatch.setattr(image_handling, "safe_get", time_out)
    started = time.monotonic()
    with pytest.raises(litellm.ImageFetchError, match="after 3 attempts"):
        convert_url_to_base64(f"https://slow.example/{uuid.uuid4()}.png", timeout=0.3)

    assert time.monotonic() - started < 0.6
    assert len(timeouts) == 1 and 0 < timeouts[0] <= 0.3


async def test_async_convert_url_to_base64_retries_within_one_deadline(monkeypatch):
    timeouts = []

    async def time_out(client, url, timeout=None, **kwargs):
        timeouts.append(timeout)
        await asyncio.sleep(timeout)
        raise litellm.Timeout(message="timed out", model="", llm_provider="")

    monkeypatch.setattr(image_handling, "async_safe_get", time_out)
    started = time.monotonic()
    with pytest.raises(litellm.ImageFetchError, match="after 3 attempts"):
        await async_convert_url_to_base64(f"https://slow.example/{uuid.uuid4()}.png", timeout=0.3)

    assert time.monotonic() - started < 0.6
    assert len(timeouts) == 1 and 0 < timeouts[0] <= 0.3


def test_convert_url_to_base64_still_retries_fast_failures_three_times(monkeypatch):
    attempts = []

    def refuse(client, url, timeout=None, **kwargs):
        attempts.append(timeout)
        raise ConnectionError("refused")

    monkeypatch.setattr(image_handling, "safe_get", refuse)
    with pytest.raises(litellm.ImageFetchError, match="after 3 attempts"):
        convert_url_to_base64(f"https://down.example/{uuid.uuid4()}.png", timeout=30.0)

    assert len(attempts) == 3 and all(0 < timeout <= 30.0 for timeout in attempts)


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


PDF_BODY: Final = b"%PDF-1.7\n" + b"0" * 4096
PNG_BODY: Final = b"\x89PNG\r\n\x1a\n" + b"0" * 64


@dataclass(frozen=True, slots=True)
class _Served:
    status: int = 200
    content_type: str | None = "application/octet-stream"
    body: bytes = PDF_BODY
    honors_range: bool = True
    barrier: threading.Barrier | None = None
    declared_length: int | None = None
    hold_open_until: threading.Event | None = None
    respond_after: threading.Event | None = None
    drip_every: float | None = None


@dataclass(frozen=True, slots=True)
class _ProbeServer:
    base_url: str
    routes: dict[str, _Served]
    seen: list[tuple[str, str | None]]

    def serve(self, path: str, served: _Served) -> str:
        self.routes[path] = served
        return f"{self.base_url}{path}"


@pytest.fixture
def probe_server(monkeypatch):
    monkeypatch.setattr(litellm, "user_url_validation", False)
    monkeypatch.setattr(image_handling, "safe_get", url_utils.safe_get)
    monkeypatch.setattr(litellm, "module_level_client", HTTPHandler())
    monkeypatch.setattr(litellm, "module_level_aclient", AsyncHTTPHandler())
    routes: Final[dict[str, _Served]] = {}
    seen: Final[list[tuple[str, str | None]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            served = routes[self.path]
            requested_range = self.headers.get("Range")
            seen.append((self.path, requested_range))
            if served.barrier is not None:
                served.barrier.wait()
            if served.respond_after is not None:
                served.respond_after.wait(timeout=30)
            ranged = served.honors_range and requested_range == "bytes=0-1023"
            body = served.body[:1024] if ranged else served.body
            self.send_response(206 if ranged and served.status == 200 else served.status)
            if served.content_type is not None:
                self.send_header("Content-Type", served.content_type)
            self.send_header("Content-Length", str(served.declared_length or len(body)))
            self.end_headers()
            if served.drip_every is None:
                self.wfile.write(body)
            else:
                for byte in body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(served.drip_every)
            if served.hold_open_until is not None:
                served.hold_open_until.wait(timeout=30)

        def log_message(self, *args):
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield _ProbeServer(f"http://127.0.0.1:{server.server_port}", routes, seen)
    server.shutdown()
    server.server_close()


async def _serves_pdf(mode: str, url: str, **kwargs) -> bool:
    return serves_pdf(url, **kwargs) if mode == "sync" else await async_serves_pdf(url, **kwargs)


async def _format_pdf_links(mode, messages):
    if mode == "sync":
        return format_pdf_links(messages, should_probe=is_unformatted_pdf_link_image)
    return await async_format_pdf_links(messages, should_probe=is_unformatted_pdf_link_image)


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_serves_pdf_asks_for_only_the_leading_kilobyte(probe_server, mode):
    url = probe_server.serve("/files/report.pdf", _Served())

    assert await _serves_pdf(mode, url) is True
    assert probe_server.seen == [("/files/report.pdf", "bytes=0-1023")]


@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize(
    ("served", "expected"),
    (
        (_Served(content_type="application/pdf", body=PNG_BODY), True),
        (_Served(content_type="Application/X-PDF; charset=binary", body=b""), True),
        (_Served(content_type="binary/octet-stream", body=PDF_BODY), True),
        (_Served(content_type=None, body=PDF_BODY), True),
        (_Served(content_type="application/octet-stream", body=b"\n" * 1000 + PDF_BODY), True),
        (_Served(content_type="application/octet-stream", body=b"\n" * 1024 + PDF_BODY), False),
        (_Served(content_type="binary/octet-stream", body=PNG_BODY), False),
        (_Served(content_type="image/png", body=PDF_BODY), False),
        (_Served(content_type="text/html", body=b"<html>%PDF-</html>"), False),
        (_Served(status=404, content_type="application/pdf"), False),
        (_Served(honors_range=False), True),
    ),
    ids=(
        "served-pdf",
        "served-x-pdf",
        "generic-type-pdf-bytes",
        "no-type-pdf-bytes",
        "signature-inside-the-first-kilobyte",
        "signature-past-the-first-kilobyte",
        "generic-type-png-bytes",
        "served-png",
        "served-html",
        "not-found",
        "range-ignored",
    ),
)
async def test_serves_pdf_trusts_a_specific_served_type_and_sniffs_a_generic_one(probe_server, mode, served, expected):
    url = probe_server.serve(f"/{uuid.uuid4()}/report.pdf", served)

    assert await _serves_pdf(mode, url) is expected


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_serves_pdf_makes_no_request_when_url_downloads_are_disabled(probe_server, monkeypatch, mode):
    monkeypatch.setattr(image_handling, "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", 0)
    url = probe_server.serve("/report.pdf", _Served(content_type="application/pdf"))

    assert await _serves_pdf(mode, url) is False
    assert probe_server.seen == []


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_serves_pdf_is_false_when_the_host_refuses_the_connection(probe_server, mode):
    assert await _serves_pdf(mode, "http://127.0.0.1:1/report.pdf") is False


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_serves_pdf_is_false_without_a_request_when_the_ssrf_check_blocks_the_link(probe_server, monkeypatch, mode):
    monkeypatch.setattr(litellm, "user_url_validation", True)
    url = probe_server.serve("/report.pdf", _Served(content_type="application/pdf"))

    assert await _serves_pdf(mode, url) is False
    assert probe_server.seen == []


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_serves_pdf_stops_reading_after_the_leading_kilobyte_when_the_host_ignores_range(probe_server, mode):
    rest_of_body_sent = threading.Event()
    url = probe_server.serve(
        "/report.pdf", _Served(honors_range=False, declared_length=1024**3, hold_open_until=rest_of_body_sent)
    )

    try:
        assert await _serves_pdf(mode, url) is True
    finally:
        rest_of_body_sent.set()


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_serves_pdf_gives_up_within_its_timeout_when_the_host_sends_one_byte_at_a_time(probe_server, mode):
    url = probe_server.serve("/report.pdf", _Served(body=b"0" * 1024, honors_range=False, drip_every=0.02))
    started = time.monotonic()

    assert await _serves_pdf(mode, url, timeout=0.5) is False
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_serves_pdf_reuses_a_link_verdict_instead_of_probing_the_link_again(probe_server, mode):
    pdf_link = probe_server.serve("/a/report.pdf", _Served())
    png_link = probe_server.serve("/a/page.pdf", _Served(content_type="image/png", body=PNG_BODY))

    verdicts = [await _serves_pdf(mode, url) for url in (pdf_link, png_link, pdf_link, png_link)]

    assert verdicts == [True, False, True, False]
    assert [path for path, _ in probe_server.seen] == ["/a/report.pdf", "/a/page.pdf"]


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_serves_pdf_remembers_a_timed_out_probe_so_the_next_turn_does_not_wait_again(probe_server, mode):
    host_answers = threading.Event()
    url = probe_server.serve("/report.pdf", _Served(respond_after=host_answers))

    try:
        first = await _serves_pdf(mode, url, timeout=0.3)
        started = time.monotonic()
        second = await _serves_pdf(mode, url, timeout=0.3)
        second_took = time.monotonic() - started
    finally:
        host_answers.set()

    assert (first, second) == (False, False)
    assert second_took < 0.1
    assert probe_server.seen == [("/report.pdf", "bytes=0-1023")]


@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("status", (404, 503))
async def test_serves_pdf_probes_a_link_again_a_minute_after_its_host_answered_an_error(
    probe_server, monkeypatch, mode, status
):
    clock: Final = MagicMock(return_value=1_000.0)
    monkeypatch.setattr(image_handling, "pdf_probe_verdicts", InMemoryCache(clock=clock))
    url = probe_server.serve("/report.pdf", _Served(status=status))

    while_down = await _serves_pdf(mode, url)
    probe_server.serve("/report.pdf", _Served())
    clock.return_value = 1_030.0
    within_the_minute = await _serves_pdf(mode, url)
    clock.return_value = 1_061.0
    after_the_minute = await _serves_pdf(mode, url)

    assert (while_down, within_the_minute, after_the_minute) == (False, False, True)
    assert len(probe_server.seen) == 2


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_format_pdf_links_marks_only_links_that_serve_a_pdf(probe_server, mode):
    pdf_link = probe_server.serve("/a/report.pdf", _Served())
    query_link = probe_server.serve("/a/download?file=summary.pdf", _Served(content_type="application/pdf"))
    png_at_pdf_link = probe_server.serve("/a/page.pdf?fm=png", _Served(content_type="image/png", body=PNG_BODY))
    png_link = f"{probe_server.base_url}/a/photo.png"
    messages = [
        {"role": "system", "content": "read the files"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image_url", "image_url": {"url": pdf_link, "detail": "high"}, "cache_control": {"type": "ephemeral"}},
                {"type": "image_url", "image_url": query_link},
                {"type": "image_url", "image_url": {"url": png_at_pdf_link}},
                {"type": "image_url", "image_url": {"url": png_link}},
                {"type": "image_url", "image_url": {"url": pdf_link, "format": "image/png"}},
            ],
        },
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": pdf_link}}]},
    ]
    original = copy.deepcopy(messages)

    formatted = await _format_pdf_links(mode, messages)

    assert messages == original
    assert formatted == (
        messages[0],
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {
                    "type": "image_url",
                    "image_url": {"url": pdf_link, "detail": "high", "format": "application/pdf"},
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "image_url", "image_url": {"url": query_link, "format": "application/pdf"}},
                {"type": "image_url", "image_url": {"url": png_at_pdf_link}},
                {"type": "image_url", "image_url": {"url": png_link}},
                {"type": "image_url", "image_url": {"url": pdf_link, "format": "image/png"}},
            ],
        },
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": pdf_link, "format": "application/pdf"}}]},
    )
    assert sorted(path for path, _ in probe_server.seen) == ["/a/download?file=summary.pdf", "/a/page.pdf?fm=png", "/a/report.pdf"]


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_format_pdf_links_probes_every_link_at_once(probe_server, mode):
    both_in_flight = threading.Barrier(2, timeout=5)
    links = [probe_server.serve(f"/{name}.pdf", _Served(barrier=both_in_flight)) for name in ("first", "second")]
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": link}} for link in links]}]

    formatted = await _format_pdf_links(mode, messages)

    assert [part["image_url"]["format"] for part in formatted[0]["content"]] == ["application/pdf", "application/pdf"]


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_format_pdf_links_leaves_messages_without_pdf_links_alone(probe_server, mode):
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"{probe_server.base_url}/a.png"}}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:application/pdf;base64,JVBERi0="}}]},
    ]

    assert await _format_pdf_links(mode, messages) == tuple(messages)
    assert probe_server.seen == []

"""
Tests for credential leak prevention in HTTP handlers.

Covers:
- MaskedHTTPStatusError construction and masking behavior
- _safe_get_response_text, _safe_aread_response, _safe_read_response helpers
- _raise_masked_sync_error and _raise_masked_async_error
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    MaskedHTTPStatusError,
    _raise_masked_async_error,
    _raise_masked_sync_error,
    _safe_aread_response,
    _safe_get_response_text,
    _safe_read_response,
)


def _make_httpx_status_error(
    status_code: int = 400,
    url: str = "https://example.com/v1/models?key=SECRET_KEY_123",
    body: str = "Bad Request",
) -> httpx.HTTPStatusError:
    """Create a real httpx.HTTPStatusError for testing."""
    request = httpx.Request("POST", url)
    response = httpx.Response(status_code, request=request, content=body.encode())
    return httpx.HTTPStatusError(
        message=f"Client error '{status_code}' for url '{url}'",
        request=request,
        response=response,
    )


class TestMaskedHTTPStatusError:
    def test_masks_url_in_request(self):
        orig = _make_httpx_status_error(url="https://api.example.com?key=MY_SECRET")
        masked = MaskedHTTPStatusError(orig)

        assert "MY_SECRET" not in str(masked.request.url)
        assert "[REDACTED_API_KEY]" in str(masked.request.url)

    def test_masks_original_message(self):
        orig = _make_httpx_status_error(url="https://api.example.com?key=SUPER_SECRET")
        masked = MaskedHTTPStatusError(orig)

        assert "SUPER_SECRET" not in str(masked)
        assert "[REDACTED_API_KEY]" in str(masked)

    def test_preserves_status_code(self):
        orig = _make_httpx_status_error(status_code=403)
        masked = MaskedHTTPStatusError(orig)

        assert masked.status_code == 403
        assert masked.response.status_code == 403

    def test_preserves_message_and_text_attrs(self):
        orig = _make_httpx_status_error()
        masked = MaskedHTTPStatusError(orig, message="custom msg", text="custom text")

        assert masked.message == "custom msg"
        assert masked.text == "custom text"

    def test_handles_response_content_decompression_failure(self):
        """If response.content raises (e.g. zlib error), should fall back to b''."""
        orig = _make_httpx_status_error()

        with patch.object(
            type(orig.response),
            "content",
            new_callable=lambda: property(
                lambda self: (_ for _ in ()).throw(Exception("zlib error"))
            ),
        ):
            masked = MaskedHTTPStatusError(orig)

        assert masked.response.content == b""
        assert masked.status_code == 400

    def test_response_request_is_set(self):
        """response.request must be set so downstream code can read it safely.

        Regression: if the inner httpx.Response is constructed without
        request=..., accessing masked.response.request raises
        RuntimeError("The .request property has not been set.").
        """
        orig = _make_httpx_status_error(url="https://api.example.com?key=KEY_X")
        masked = MaskedHTTPStatusError(orig)

        # Must not raise RuntimeError.
        req = masked.response.request
        assert req is not None
        # The attached request must be the masked one, not the original.
        assert "KEY_X" not in str(req.url)

    def test_strips_content_encoding_to_avoid_double_decode(self):
        """If the upstream response declared Content-Encoding (e.g. gzip),
        the rebuilt Response must not carry that header over — otherwise httpx
        tries to decode the already-decoded bytes again and raises DecodingError.
        """
        # Build a gzipped upstream response so .content decodes once cleanly.
        import gzip

        body = b'{"error": "bad request"}'
        gzipped = gzip.compress(body)
        request = httpx.Request("POST", "https://api.example.com?key=KEY")
        response = httpx.Response(
            status_code=400,
            content=gzipped,
            headers={
                "content-encoding": "gzip",
                "content-length": str(len(gzipped)),
                "content-type": "application/json",
            },
            request=request,
        )
        orig = httpx.HTTPStatusError("400", request=request, response=response)

        # Previously this raised httpx.DecodingError; must now succeed.
        masked = MaskedHTTPStatusError(orig)

        # Content must be the once-decoded bytes, not a double-decode attempt.
        assert masked.response.content == body
        # Content-Encoding must have been stripped from the rebuilt headers.
        assert "content-encoding" not in {k.lower() for k in masked.response.headers}


class TestSafeResponseHelpers:
    def test_safe_get_response_text_normal(self):
        response = httpx.Response(200, content=b"hello world")
        assert _safe_get_response_text(response) == "hello world"

    def test_safe_get_response_text_error(self):
        response = MagicMock(spec=httpx.Response)
        type(response).text = property(
            lambda self: (_ for _ in ()).throw(
                UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
            )
        )
        assert _safe_get_response_text(response) == ""

    def test_safe_read_response_normal(self):
        response = httpx.Response(200, content=b"raw bytes")
        result = _safe_read_response(response)
        assert result == b"raw bytes"

    def test_safe_read_response_error(self):
        response = MagicMock(spec=httpx.Response)
        response.read.side_effect = Exception("read failure")
        assert _safe_read_response(response) == b""

    @pytest.mark.asyncio
    async def test_safe_aread_response_normal(self):
        response = MagicMock(spec=httpx.Response)
        response.aread = AsyncMock(return_value=b"async bytes")
        result = await _safe_aread_response(response)
        assert result == b"async bytes"

    @pytest.mark.asyncio
    async def test_safe_aread_response_error(self):
        response = MagicMock(spec=httpx.Response)
        response.aread = AsyncMock(side_effect=Exception("async read failure"))
        result = await _safe_aread_response(response)
        assert result == b""


class TestRaiseMaskedError:
    def test_sync_non_stream(self):
        orig = _make_httpx_status_error(
            url="https://api.example.com?key=LEAKED_KEY", body="error body"
        )
        with pytest.raises(MaskedHTTPStatusError) as exc_info:
            _raise_masked_sync_error(orig, stream=False)

        err = exc_info.value
        assert "LEAKED_KEY" not in str(err.request.url)
        assert err.status_code == 400
        assert err.text == "error body"

    def test_sync_stream(self):
        orig = _make_httpx_status_error(
            url="https://api.example.com?key=LEAKED_KEY", body="stream body"
        )
        with pytest.raises(MaskedHTTPStatusError) as exc_info:
            _raise_masked_sync_error(orig, stream=True)

        err = exc_info.value
        assert "LEAKED_KEY" not in str(err.request.url)
        assert err.message is not None

    def test_sync_breaks_exception_chain(self):
        orig = _make_httpx_status_error()
        with pytest.raises(MaskedHTTPStatusError) as exc_info:
            _raise_masked_sync_error(orig, stream=False)

        assert exc_info.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_async_non_stream(self):
        orig = _make_httpx_status_error(
            url="https://api.example.com?key=LEAKED_KEY", body="async error"
        )
        with pytest.raises(MaskedHTTPStatusError) as exc_info:
            await _raise_masked_async_error(orig, stream=False)

        err = exc_info.value
        assert "LEAKED_KEY" not in str(err.request.url)
        assert err.status_code == 400
        assert err.text == "async error"

    @pytest.mark.asyncio
    async def test_async_stream(self):
        orig = _make_httpx_status_error(
            url="https://api.example.com?key=LEAKED_KEY", body="async stream"
        )
        with pytest.raises(MaskedHTTPStatusError) as exc_info:
            await _raise_masked_async_error(orig, stream=True)

        err = exc_info.value
        assert "LEAKED_KEY" not in str(err.request.url)
        assert err.message is not None

    @pytest.mark.asyncio
    async def test_async_breaks_chain(self):
        orig = _make_httpx_status_error()
        with pytest.raises(MaskedHTTPStatusError) as exc_info:
            await _raise_masked_async_error(orig, stream=False)

        assert exc_info.value.__cause__ is None


class TestHTTPHandlerErrorPaths:
    """Test that HTTP handler methods raise MaskedHTTPStatusError on HTTPStatusError."""

    @pytest.fixture
    def sync_handler(self):
        handler = HTTPHandler()
        yield handler
        handler.close()

    @pytest.fixture
    async def async_handler(self):
        handler = AsyncHTTPHandler()
        yield handler
        await handler.close()

    @pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
    def test_sync_raises_masked_error(self, sync_handler, method):
        with patch.object(
            sync_handler.client,
            "send",
            side_effect=_make_httpx_status_error(url="https://api.test.com?key=SECRET"),
        ):
            with pytest.raises(MaskedHTTPStatusError) as exc_info:
                kwargs = {"url": "https://api.test.com?key=SECRET"}
                if method != "delete":
                    kwargs["data"] = {"test": 1}
                getattr(sync_handler, method)(**kwargs)

            assert "SECRET" not in str(exc_info.value.request.url)

    @pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
    @pytest.mark.asyncio
    async def test_async_raises_masked_error(self, async_handler, method):
        with patch.object(
            async_handler.client,
            "send",
            new_callable=AsyncMock,
            side_effect=_make_httpx_status_error(url="https://api.test.com?key=SECRET"),
        ):
            with pytest.raises(MaskedHTTPStatusError) as exc_info:
                kwargs = {"url": "https://api.test.com?key=SECRET"}
                if method != "delete":
                    kwargs["data"] = {"test": 1}
                await getattr(async_handler, method)(**kwargs)

            assert "SECRET" not in str(exc_info.value.request.url)

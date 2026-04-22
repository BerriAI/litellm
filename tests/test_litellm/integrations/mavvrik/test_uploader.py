"""Unit tests for Mavvrik Uploader — GCS resumable upload protocol."""

import gzip
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.uploader import Uploader
from litellm.integrations.mavvrik.client import Client


def _make_client(**kwargs) -> Client:
    defaults = dict(
        api_key="test-key",
        api_endpoint="https://api.mavvrik.dev/acme",
        connection_id="litellm-001",
    )
    defaults.update(kwargs)
    return Client(**defaults)


def _make_uploader(**kwargs) -> Uploader:
    return Uploader(client=_make_client(**kwargs))


def _mock_http_response(status_code: int, text="", headers=None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.headers = headers or {}
    return resp


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestUploaderInit:
    def test_accepts_client(self):
        client = _make_client()
        u = Uploader(client=client)
        assert u.client is client

    def test_raises_without_client(self):
        with pytest.raises(TypeError):
            Uploader()  # client is required


# ---------------------------------------------------------------------------
# _compress
# ---------------------------------------------------------------------------


class TestCompress:
    def test_returns_bytes(self):
        u = _make_uploader()
        result = u._compress("hello,world\n")
        assert isinstance(result, bytes)

    def test_gzip_decompresses_back_to_original(self):
        u = _make_uploader()
        original = "date,model,spend\n2025-01-15,gpt-4o,1.5\n"
        assert gzip.decompress(u._compress(original)) == original.encode("utf-8")

    def test_empty_string_compresses(self):
        u = _make_uploader()
        assert isinstance(u._compress(""), bytes)


# ---------------------------------------------------------------------------
# _initiate_resumable_upload
# ---------------------------------------------------------------------------


class TestInitiateResumableUpload:
    @pytest.mark.asyncio
    async def test_returns_location_header_on_201(self):
        u = _make_uploader()
        resp = _mock_http_response(
            201, headers={"Location": "https://gcs.example.com/session"}
        )

        with patch("httpx.AsyncClient") as mock_cls:
            http = MagicMock()
            http.post = AsyncMock(return_value=resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_uri = await u._initiate_resumable_upload("https://signed-url")

        assert session_uri == "https://gcs.example.com/session"

    @pytest.mark.asyncio
    async def test_raises_on_missing_location_header(self):
        u = _make_uploader()
        resp = _mock_http_response(201, headers={})

        with patch("httpx.AsyncClient") as mock_cls:
            http = MagicMock()
            http.post = AsyncMock(return_value=resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="Location"):
                await u._initiate_resumable_upload("https://signed-url")

    @pytest.mark.asyncio
    async def test_raises_on_non_201(self):
        u = _make_uploader()
        resp = _mock_http_response(403, text="Forbidden")

        with patch("httpx.AsyncClient") as mock_cls:
            http = MagicMock()
            http.post = AsyncMock(return_value=resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="initiate"):
                await u._initiate_resumable_upload("https://signed-url")

    @pytest.mark.asyncio
    async def test_sends_gzip_content_type(self):
        u = _make_uploader()
        resp = _mock_http_response(
            201, headers={"Location": "https://gcs.example.com/session"}
        )
        captured = []

        async def fake_post(url, headers=None, **kwargs):
            captured.append(headers)
            return resp

        with patch("httpx.AsyncClient") as mock_cls:
            http = MagicMock()
            http.post = fake_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await u._initiate_resumable_upload("https://signed-url")

        assert captured[0]["Content-Type"] == "application/gzip"
        assert captured[0]["x-goog-resumable"] == "start"

    @pytest.mark.asyncio
    async def test_retries_on_5xx_then_raises(self):
        u = _make_uploader()
        resp = _mock_http_response(503, text="unavailable")

        with patch("httpx.AsyncClient") as mock_cls, patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            http = MagicMock()
            http.post = AsyncMock(return_value=resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="failed after"):
                await u._initiate_resumable_upload("https://signed-url")

        assert http.post.call_count == 3


# ---------------------------------------------------------------------------
# _finalize_upload
# ---------------------------------------------------------------------------


class TestFinalizeUpload:
    @pytest.mark.asyncio
    async def test_accepts_200(self):
        u = _make_uploader()
        resp = _mock_http_response(200)

        with patch("httpx.AsyncClient") as mock_cls:
            http = MagicMock()
            http.put = AsyncMock(return_value=resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await u._finalize_upload("https://session-uri", b"gzip-bytes")

    @pytest.mark.asyncio
    async def test_accepts_201(self):
        u = _make_uploader()
        resp = _mock_http_response(201)

        with patch("httpx.AsyncClient") as mock_cls:
            http = MagicMock()
            http.put = AsyncMock(return_value=resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await u._finalize_upload("https://session-uri", b"gzip-bytes")

    @pytest.mark.asyncio
    async def test_raises_on_error_status(self):
        u = _make_uploader()
        resp = _mock_http_response(500, text="Server Error")

        with patch("httpx.AsyncClient") as mock_cls:
            http = MagicMock()
            http.put = AsyncMock(return_value=resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="finalize"):
                await u._finalize_upload("https://session-uri", b"gzip-bytes")

    @pytest.mark.asyncio
    async def test_sends_gzip_bytes_as_body(self):
        u = _make_uploader()
        resp = _mock_http_response(200)
        captured = []

        async def fake_put(url, content=None, **kwargs):
            captured.append(content)
            return resp

        with patch("httpx.AsyncClient") as mock_cls:
            http = MagicMock()
            http.put = fake_put
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await u._finalize_upload("https://session-uri", b"my-gzip-data")

        assert captured[0] == b"my-gzip-data"

    @pytest.mark.asyncio
    async def test_retries_on_5xx_then_raises(self):
        u = _make_uploader()
        resp = _mock_http_response(503, text="unavailable")

        with patch("httpx.AsyncClient") as mock_cls, patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            http = MagicMock()
            http.put = AsyncMock(return_value=resp)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="failed after"):
                await u._finalize_upload("https://session-uri", b"data")

        assert http.put.call_count == 3


# ---------------------------------------------------------------------------
# upload()
# ---------------------------------------------------------------------------


class TestUpload:
    @pytest.mark.asyncio
    async def test_skips_all_steps_on_empty_payload(self):
        u = _make_uploader()
        with patch.object(
            u.client, "get_signed_url", new_callable=AsyncMock
        ) as mock_url, patch.object(
            u, "_initiate_resumable_upload", new_callable=AsyncMock
        ) as mock_init, patch.object(
            u, "_finalize_upload", new_callable=AsyncMock
        ) as mock_fin:
            await u.upload("   ", date_str="2025-01-15")

        mock_url.assert_not_called()
        mock_init.assert_not_called()
        mock_fin.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_all_three_gcs_steps_in_order(self):
        u = _make_uploader()
        csv = "date,model,spend\n2025-01-15,gpt-4o,1.5"
        call_order = []

        async def fake_get_signed_url(date_str):
            call_order.append("get_signed_url")
            return "https://signed"

        async def fake_initiate(signed_url):
            call_order.append("initiate")
            assert signed_url == "https://signed"
            return "https://session"

        async def fake_finalize(session_uri, data):
            call_order.append("finalize")
            assert session_uri == "https://session"

        with patch.object(
            u.client, "get_signed_url", side_effect=fake_get_signed_url
        ), patch.object(
            u, "_initiate_resumable_upload", side_effect=fake_initiate
        ), patch.object(
            u, "_finalize_upload", side_effect=fake_finalize
        ):
            await u.upload(csv, date_str="2025-01-15")

        assert call_order == ["get_signed_url", "initiate", "finalize"]

    @pytest.mark.asyncio
    async def test_uploads_gzip_compressed_bytes(self):
        u = _make_uploader()
        csv = "date,model,spend\n2025-01-15,gpt-4o,1.5"
        captured_bytes = []

        async def fake_finalize(session_uri, data):
            captured_bytes.append(data)

        with patch.object(
            u.client,
            "get_signed_url",
            new_callable=AsyncMock,
            return_value="https://signed",
        ), patch.object(
            u,
            "_initiate_resumable_upload",
            new_callable=AsyncMock,
            return_value="https://session",
        ), patch.object(
            u, "_finalize_upload", side_effect=fake_finalize
        ):
            await u.upload(csv, date_str="2025-01-15")

        assert len(captured_bytes) == 1
        assert gzip.decompress(captured_bytes[0]) == csv.encode("utf-8")

    @pytest.mark.asyncio
    async def test_passes_date_str_to_get_signed_url(self):
        u = _make_uploader()
        captured_dates = []

        async def fake_get_signed_url(date_str):
            captured_dates.append(date_str)
            return "https://signed"

        with patch.object(
            u.client, "get_signed_url", side_effect=fake_get_signed_url
        ), patch.object(
            u,
            "_initiate_resumable_upload",
            new_callable=AsyncMock,
            return_value="https://session",
        ), patch.object(
            u, "_finalize_upload", new_callable=AsyncMock
        ):
            await u.upload("col\nval", date_str="2025-03-10")

        assert captured_dates[0] == "2025-03-10"

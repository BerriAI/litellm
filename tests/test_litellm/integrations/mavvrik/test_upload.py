"""Unit tests for the Mavvrik API streaming layer (signed URL upload)."""

import gzip
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.client import MavvrikClient


def _make_streamer(**kwargs) -> MavvrikClient:
    defaults = dict(
        api_key="test-key",
        api_endpoint="https://api.mavvrik.dev/acme",
        connection_id="litellm-001",
    )
    defaults.update(kwargs)
    return MavvrikClient(**defaults)


def _mock_async_client(method: str, mock_resp: MagicMock):
    """Return a context-manager mock for httpx.AsyncClient with one stubbed method."""
    client_mock = MagicMock()
    async_method = AsyncMock(return_value=mock_resp)
    setattr(client_mock, method, async_method)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client_mock)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, client_mock, async_method


class TestMavvrikClientInit:
    def test_init_strips_trailing_slash(self):
        streamer = MavvrikClient(
            api_key="key",
            api_endpoint="https://api.mavvrik.dev/acme/",
            connection_id="litellm-001",
        )
        assert not streamer.api_endpoint.endswith("/")
        assert streamer.api_endpoint == "https://api.mavvrik.dev/acme"

    def test_init_stores_attributes(self):
        streamer = MavvrikClient(
            api_key="mvk-key",
            api_endpoint="https://api.mavvrik.dev/my-tenant",
            connection_id="inst-123",
        )
        assert streamer.api_key == "mvk-key"
        assert streamer.connection_id == "inst-123"


class TestMavvrikClientGetSignedUrl:
    @pytest.mark.asyncio
    async def test_returns_signed_url_on_200(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "url": "https://storage.example.com/signed?token=abc"
        }

        cm, _, _ = _mock_async_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            url = await streamer._get_signed_url("2025-01-15")

        assert url == "https://storage.example.com/signed?token=abc"

    @pytest.mark.asyncio
    async def test_raises_on_missing_url_field(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}  # no 'url' key

        cm, _, _ = _mock_async_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            with pytest.raises(Exception, match="missing 'url' field"):
                await streamer._get_signed_url("2025-01-15")

    @pytest.mark.asyncio
    async def test_raises_immediately_on_4xx(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        cm, _, _ = _mock_async_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            with pytest.raises(Exception, match="401"):
                await streamer._get_signed_url("2025-01-15")
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_on_5xx_then_raises(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"

        cm, _, _ = _mock_async_client("get", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            with pytest.raises(Exception, match="503|failed after"):
                await streamer._get_signed_url("2025-01-15")

    @pytest.mark.asyncio
    async def test_builds_correct_url_with_connection_id(self):
        streamer = MavvrikClient(
            api_key="key",
            api_endpoint="https://api.example.com/my-org",
            connection_id="prod-001",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"url": "https://example.com/signed"}

        captured_url = []

        async def fake_get(url, **kwargs):
            captured_url.append(url)
            return mock_resp

        client_mock = MagicMock()
        client_mock.get = fake_get
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            await streamer._get_signed_url("2025-01-15")

        assert "prod-001" in captured_url[0]

    @pytest.mark.asyncio
    async def test_sends_x_api_key_header(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"url": "https://example.com/signed"}

        captured_headers = []

        async def fake_get(url, headers=None, **kwargs):
            captured_headers.append(headers)
            return mock_resp

        client_mock = MagicMock()
        client_mock.get = fake_get
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            await streamer._get_signed_url("2025-01-15")

        assert captured_headers[0].get("x-api-key") == "test-key"


class TestMavvrikClientInitiateResumable:
    @pytest.mark.asyncio
    async def test_returns_location_header_on_201(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.headers = {"Location": "https://example.com/session-uri"}

        cm, _, _ = _mock_async_client("post", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            session_uri = await streamer._initiate_resumable_upload(
                "https://signed-url"
            )

        assert session_uri == "https://example.com/session-uri"

    @pytest.mark.asyncio
    async def test_raises_on_non_201(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        cm, _, _ = _mock_async_client("post", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            with pytest.raises(Exception, match="initiate upload failed"):
                await streamer._initiate_resumable_upload("https://signed-url")

    @pytest.mark.asyncio
    async def test_raises_on_missing_location_header(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.headers = {}  # no Location

        cm, _, _ = _mock_async_client("post", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            with pytest.raises(Exception, match="missing Location header"):
                await streamer._initiate_resumable_upload("https://signed-url")


class TestMavvrikClientFinalizeUpload:
    @pytest.mark.asyncio
    async def test_accepts_200(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        cm, _, _ = _mock_async_client("put", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            await streamer._finalize_upload("https://session-uri", b"gzip-bytes")

    @pytest.mark.asyncio
    async def test_accepts_201(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 201

        cm, _, _ = _mock_async_client("put", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            await streamer._finalize_upload("https://session-uri", b"gzip-bytes")

    @pytest.mark.asyncio
    async def test_raises_on_error_status(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        cm, _, _ = _mock_async_client("put", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            with pytest.raises(Exception, match="finalize upload failed"):
                await streamer._finalize_upload("https://session-uri", b"gzip-bytes")


class TestMavvrikClientUpload:
    @pytest.mark.asyncio
    async def test_upload_empty_payload_skips_all_steps(self):
        streamer = _make_streamer()
        with patch.object(
            streamer, "_get_signed_url", new_callable=AsyncMock
        ) as mock_url, patch.object(
            streamer, "_initiate_resumable_upload", new_callable=AsyncMock
        ) as mock_init, patch.object(
            streamer, "_finalize_upload", new_callable=AsyncMock
        ) as mock_fin:
            await streamer.upload("   ", date_str="2025-01-15")
        mock_url.assert_not_called()
        mock_init.assert_not_called()
        mock_fin.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_calls_all_three_steps(self):
        streamer = _make_streamer()
        csv_payload = "date,model,spend\n2025-01-15,gpt-4o,1.5"
        with patch.object(
            streamer,
            "_get_signed_url",
            new_callable=AsyncMock,
            return_value="https://signed",
        ) as mock_url, patch.object(
            streamer,
            "_initiate_resumable_upload",
            new_callable=AsyncMock,
            return_value="https://session",
        ) as mock_init, patch.object(
            streamer, "_finalize_upload", new_callable=AsyncMock
        ) as mock_fin:
            await streamer.upload(csv_payload, date_str="2025-01-15")

        # Methods now receive client= kwarg — check positional args only
        mock_url.assert_called_once()
        assert mock_url.call_args.args[0] == "2025-01-15"
        mock_init.assert_called_once()
        assert mock_init.call_args.args[0] == "https://signed"
        mock_fin.assert_called_once()
        upload_bytes = mock_fin.call_args.args[1]
        assert isinstance(upload_bytes, bytes)
        assert gzip.decompress(upload_bytes) == csv_payload.encode("utf-8")


class TestMavvrikClientAdvanceMarker:
    @pytest.mark.asyncio
    async def test_accepts_204(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        cm, _, _ = _mock_async_client("patch", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            await streamer.advance_marker(1737000000)

    @pytest.mark.asyncio
    async def test_accepts_200(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        cm, _, _ = _mock_async_client("patch", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            await streamer.advance_marker(1737000000)

    @pytest.mark.asyncio
    async def test_raises_on_error_status(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        cm, _, _ = _mock_async_client("patch", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            with pytest.raises(Exception, match="advance_marker failed"):
                await streamer.advance_marker(1737000000)

    @pytest.mark.asyncio
    async def test_sends_correct_body(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        captured = []

        async def fake_patch(url, headers=None, json=None, **kwargs):
            captured.append({"url": url, "json": json})
            return mock_resp

        client_mock = MagicMock()
        client_mock.patch = fake_patch
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            await streamer.advance_marker(1737000000)

        assert captured[0]["json"] == {"metricsMarker": 1737000000}

    @pytest.mark.asyncio
    async def test_patches_agent_base_path_with_connection_id(self):
        streamer = MavvrikClient(
            api_key="key",
            api_endpoint="https://api.example.com/my-org",
            connection_id="prod-001",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        captured_url = []

        async def fake_patch(url, **kwargs):
            captured_url.append(url)
            return mock_resp

        client_mock = MagicMock()
        client_mock.patch = fake_patch
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            await streamer.advance_marker(1737000000)

        assert "prod-001" in captured_url[0]

    @pytest.mark.asyncio
    async def test_sends_x_api_key_header(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        captured_headers = []

        async def fake_patch(url, headers=None, **kwargs):
            captured_headers.append(headers)
            return mock_resp

        client_mock = MagicMock()
        client_mock.patch = fake_patch
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            await streamer.advance_marker(1737000000)

        assert captured_headers[0].get("x-api-key") == "test-key"


class TestMavvrikClientRegister:
    @pytest.mark.asyncio
    async def test_register_returns_iso_string_from_epoch(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "litellm-001", "metricsMarker": 1737000000}

        cm, _, _ = _mock_async_client("post", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            marker = await streamer.register()

        assert "2025-01-16" in marker
        assert "+00:00" in marker or "UTC" in marker or "Z" in marker or "+00" in marker

    @pytest.mark.asyncio
    async def test_register_returns_none_when_marker_zero(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "litellm-001", "metricsMarker": 0}

        cm, _, _ = _mock_async_client("post", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            marker = await streamer.register()

        assert marker is None

    @pytest.mark.asyncio
    async def test_register_returns_none_when_marker_absent(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "litellm-001"}  # no metricsMarker

        cm, _, _ = _mock_async_client("post", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            marker = await streamer.register()

        assert marker is None

    @pytest.mark.asyncio
    async def test_register_raises_on_non_200(self):
        streamer = _make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        cm, _, _ = _mock_async_client("post", mock_resp)
        with patch("httpx.AsyncClient", return_value=cm):
            with pytest.raises(Exception, match="registration failed"):
                await streamer.register()

    @pytest.mark.asyncio
    async def test_register_posts_to_agent_base_path(self):
        streamer = MavvrikClient(
            api_key="key",
            api_endpoint="https://api.example.com/my-org",
            connection_id="prod-001",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "prod-001", "metricsMarker": 1700000000}

        captured_url = []

        async def fake_post(url, **kwargs):
            captured_url.append(url)
            return mock_resp

        client_mock = MagicMock()
        client_mock.post = fake_post
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client_mock)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=cm):
            await streamer.register()

        assert "prod-001" in captured_url[0]

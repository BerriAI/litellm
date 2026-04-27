"""Integration tests for the Mavvrik upload pipeline — fully mock-based.

Tests the full Client → Uploader → Orchestrator flow without real network
calls, verifying that components wire together correctly end-to-end.
"""

import gzip
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.uploader import Uploader

_TEST_CSV = (
    "date,user_id,api_key,model,spend\n" "2026-01-01,user-1,sk-test,gpt-4o,0.0025\n"
)
_TEST_DATE = "2026-01-01"


def _make_client() -> Client:
    return Client(
        api_key="test-key",
        api_endpoint="https://api.mavvrik.dev/test",
        connection_id="litellm-test",
    )


def _make_uploader(client=None) -> Uploader:
    return Uploader(client=client or _make_client())


def _mock_response(
    status_code: int, json_body=None, headers=None, text=""
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_body or {}
    resp.headers = headers or {}
    return resp


# ---------------------------------------------------------------------------
# Client integration — each public method wired through _request
# ---------------------------------------------------------------------------


class TestClientIntegration:
    @pytest.mark.asyncio
    async def test_register_returns_iso_marker(self):
        """register() parses metricsMarker epoch into ISO-8601 string."""
        client = _make_client()
        with patch.object(
            client,
            "_request",
            return_value=_mock_response(200, {"metricsMarker": 1737000000}),
        ):
            marker = await client.register()
        assert marker is not None
        dt = datetime.fromisoformat(marker)
        assert dt.year == 2025

    @pytest.mark.asyncio
    async def test_register_returns_none_on_first_run(self):
        """register() returns None when metricsMarker is 0 (first run)."""
        client = _make_client()
        with patch.object(
            client,
            "_request",
            return_value=_mock_response(200, {"metricsMarker": 0}),
        ):
            marker = await client.register()
        assert marker is None

    @pytest.mark.asyncio
    async def test_advance_marker_sends_correct_epoch(self):
        """advance_marker() sends metricsMarker in body."""
        client = _make_client()
        captured = []

        async def fake_request(method, url, *, json=None, **kwargs):
            captured.append({"method": method, "json": json})
            return _mock_response(204)

        with patch.object(client, "_request", side_effect=fake_request):
            await client.advance_marker(1775001600)

        assert captured[0]["method"] == "PATCH"
        assert captured[0]["json"] == {"metricsMarker": 1775001600}

    @pytest.mark.asyncio
    async def test_get_signed_url_returns_url(self):
        """get_signed_url() extracts url field from response."""
        client = _make_client()
        with patch.object(
            client,
            "_request",
            return_value=_mock_response(
                200, {"url": "https://storage.googleapis.com/signed"}
            ),
        ):
            url = await client.get_signed_url(_TEST_DATE)
        assert url == "https://storage.googleapis.com/signed"

    @pytest.mark.asyncio
    async def test_get_signed_url_sends_name_and_datetime(self):
        """get_signed_url() passes name and datetime params."""
        client = _make_client()
        captured = []

        async def fake_request(method, url, *, params=None, **kwargs):
            captured.append(params)
            return _mock_response(200, {"url": "https://example.com/signed"})

        with patch.object(client, "_request", side_effect=fake_request):
            await client.get_signed_url("2026-04-01")

        assert captured[0]["name"] == "2026-04-01"
        assert captured[0]["datetime"] == "2026-04-01"
        assert captured[0]["type"] == "metrics"

    @pytest.mark.asyncio
    async def test_report_error_swallows_exceptions(self):
        """report_error() never raises even when the request fails."""
        client = _make_client()
        with patch.object(client, "_request", side_effect=RuntimeError("network down")):
            await client.report_error("something broke")  # must not raise


# ---------------------------------------------------------------------------
# Uploader integration — bulk path wires Client + GCS steps
# ---------------------------------------------------------------------------


class TestUploaderIntegration:
    @pytest.mark.asyncio
    async def test_upload_calls_signed_url_then_gcs(self):
        """upload() gets signed URL from Client then initiates and finalises GCS session."""
        uploader = _make_uploader()
        call_order = []

        async def fake_get_signed_url(date_str):
            call_order.append("get_signed_url")
            return "https://signed"

        async def fake_initiate(signed_url):
            call_order.append("initiate")
            assert signed_url == "https://signed"
            return "https://session"

        async def fake_finalize(session_uri, gzip_bytes):
            call_order.append("finalize")
            assert session_uri == "https://session"
            assert isinstance(gzip_bytes, bytes)
            assert gzip.decompress(gzip_bytes) == _TEST_CSV.encode("utf-8")

        with patch.object(
            uploader.client, "get_signed_url", side_effect=fake_get_signed_url
        ), patch.object(
            uploader, "_initiate_resumable_upload", side_effect=fake_initiate
        ), patch.object(
            uploader, "_finalize_upload", side_effect=fake_finalize
        ):
            await uploader.upload(_TEST_CSV, date_str=_TEST_DATE)

        assert call_order == ["get_signed_url", "initiate", "finalize"]

    @pytest.mark.asyncio
    async def test_upload_skips_all_gcs_on_empty_payload(self):
        """upload() makes no network calls when payload is blank."""
        uploader = _make_uploader()
        with patch.object(
            uploader.client, "get_signed_url", new_callable=AsyncMock
        ) as mock_url:
            await uploader.upload("   ", date_str=_TEST_DATE)
        mock_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_is_idempotent(self):
        """Calling upload() twice for the same date succeeds both times."""
        uploader = _make_uploader()
        call_count = [0]

        async def fake_get_signed_url(date_str):
            call_count[0] += 1
            return "https://signed"

        with patch.object(
            uploader.client, "get_signed_url", side_effect=fake_get_signed_url
        ), patch.object(
            uploader,
            "_initiate_resumable_upload",
            new_callable=AsyncMock,
            return_value="https://session",
        ), patch.object(
            uploader, "_finalize_upload", new_callable=AsyncMock
        ):
            await uploader.upload(_TEST_CSV, date_str=_TEST_DATE)
            await uploader.upload(_TEST_CSV, date_str=_TEST_DATE)

        assert call_count[0] == 2  # both calls went through


# ---------------------------------------------------------------------------
# Full pipeline — Client + Uploader wired together
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    @pytest.mark.asyncio
    async def test_register_then_upload_then_advance(self):
        """Simulate one complete export cycle: register → upload → advance."""
        client = _make_client()
        uploader = Uploader(client=client)
        advance_calls = []

        with patch.object(
            client,
            "_request",
            side_effect=[
                # register() → metricsMarker
                _mock_response(200, {"metricsMarker": 1775001600}),
                # get_signed_url() → url
                _mock_response(200, {"url": "https://signed"}),
                # advance_marker() → 204
                _mock_response(204),
            ],
        ), patch.object(
            uploader,
            "_initiate_resumable_upload",
            new_callable=AsyncMock,
            return_value="https://session",
        ), patch.object(
            uploader, "_finalize_upload", new_callable=AsyncMock
        ):
            marker = await client.register()
            assert marker is not None

            await uploader.upload(_TEST_CSV, date_str=_TEST_DATE)

            await client.advance_marker(1775088000)

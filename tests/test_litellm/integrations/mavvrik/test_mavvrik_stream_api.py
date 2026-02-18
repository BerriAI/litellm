"""Unit tests for the Mavvrik API streaming layer (signed URL + GCS upload)."""

import gzip
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.mavvrik_stream_api import MavvrikStreamer


class TestMavvrikStreamerInit:
    def test_init_strips_trailing_slash(self):
        streamer = MavvrikStreamer(
            api_key="key",
            api_endpoint="https://api.mavvrik.dev/",
            tenant="acme",
            instance_id="litellm-001",
        )
        assert not streamer.api_endpoint.endswith("/")
        assert streamer.api_endpoint == "https://api.mavvrik.dev"

    def test_init_stores_attributes(self):
        streamer = MavvrikStreamer(
            api_key="mvk-key",
            api_endpoint="https://api.mavvrik.dev",
            tenant="my-tenant",
            instance_id="inst-123",
        )
        assert streamer.api_key == "mvk-key"
        assert streamer.tenant == "my-tenant"
        assert streamer.instance_id == "inst-123"


class TestMavvrikStreamerGetSignedUrl:
    def _make_streamer(self):
        return MavvrikStreamer(
            api_key="test-key",
            api_endpoint="https://api.mavvrik.dev",
            tenant="acme",
            instance_id="litellm-001",
        )

    def test_returns_signed_url_on_200(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "url": "https://storage.googleapis.com/signed?token=abc"
        }

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            url = streamer._get_signed_url("2025-01-15T14:00:00Z")

        assert url == "https://storage.googleapis.com/signed?token=abc"

    def test_raises_on_missing_url_field(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}  # no 'url' key

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            with pytest.raises(Exception, match="missing 'url' field"):
                streamer._get_signed_url("2025-01-15T14:00:00Z")

    def test_raises_immediately_on_4xx(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with patch("httpx.Client") as MockClient, patch("time.sleep") as mock_sleep:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            with pytest.raises(Exception, match="401"):
                streamer._get_signed_url("2025-01-15T14:00:00Z")
        # No sleep on 4xx (no retries)
        mock_sleep.assert_not_called()

    def test_retries_on_5xx_then_raises(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"

        with patch("httpx.Client") as MockClient, patch("time.sleep"):
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            with pytest.raises(Exception, match="503|failed after"):
                streamer._get_signed_url("2025-01-15T14:00:00Z")

    def test_builds_correct_url_with_tenant_and_instance(self):
        streamer = MavvrikStreamer(
            api_key="key",
            api_endpoint="https://api.example.com",
            tenant="my-org",
            instance_id="prod-001",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"url": "https://gcs.example.com/signed"}

        captured_url = []

        def fake_get(url, **kwargs):
            captured_url.append(url)
            return mock_resp

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = fake_get
            streamer._get_signed_url("2025-01-15T14:00:00Z")

        assert "my-org" in captured_url[0]
        assert "prod-001" in captured_url[0]

    def test_sends_x_api_key_header(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"url": "https://gcs.example.com/signed"}

        captured_headers = []

        def fake_get(url, headers=None, **kwargs):
            captured_headers.append(headers)
            return mock_resp

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = fake_get
            streamer._get_signed_url("2025-01-15T14:00:00Z")

        assert captured_headers[0].get("x-api-key") == "test-key"


class TestMavvrikStreamerInitiateResumable:
    def _make_streamer(self):
        return MavvrikStreamer(
            api_key="key",
            api_endpoint="https://api.mavvrik.dev",
            tenant="acme",
            instance_id="litellm-001",
        )

    def test_returns_location_header_on_201(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.headers = {"Location": "https://storage.googleapis.com/session-uri"}

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            session_uri = streamer._initiate_resumable_upload("https://signed-url")

        assert session_uri == "https://storage.googleapis.com/session-uri"

    def test_raises_on_non_201(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            with pytest.raises(Exception, match="initiate upload failed"):
                streamer._initiate_resumable_upload("https://signed-url")

    def test_raises_on_missing_location_header(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.headers = {}  # no Location

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            with pytest.raises(Exception, match="missing Location header"):
                streamer._initiate_resumable_upload("https://signed-url")


class TestMavvrikStreamerFinalizeUpload:
    def _make_streamer(self):
        return MavvrikStreamer(
            api_key="key",
            api_endpoint="https://api.mavvrik.dev",
            tenant="acme",
            instance_id="litellm-001",
        )

    def test_accepts_200(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.put.return_value = mock_resp
            # Should not raise
            streamer._finalize_upload("https://session-uri", b"gzip-bytes")

    def test_accepts_201(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 201

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.put.return_value = mock_resp
            streamer._finalize_upload("https://session-uri", b"gzip-bytes")

    def test_raises_on_error_status(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.put.return_value = mock_resp
            with pytest.raises(Exception, match="finalize upload failed"):
                streamer._finalize_upload("https://session-uri", b"gzip-bytes")


class TestMavvrikStreamerUpload:
    def _make_streamer(self):
        return MavvrikStreamer(
            api_key="key",
            api_endpoint="https://api.mavvrik.dev",
            tenant="acme",
            instance_id="litellm-001",
        )

    def test_upload_empty_payload_skips_all_steps(self):
        streamer = self._make_streamer()
        with patch.object(streamer, "_get_signed_url") as mock_url, patch.object(
            streamer, "_initiate_resumable_upload"
        ) as mock_init, patch.object(streamer, "_finalize_upload") as mock_fin:
            streamer.upload("   ", "2025-01-15T14:00:00Z")
        mock_url.assert_not_called()
        mock_init.assert_not_called()
        mock_fin.assert_not_called()

    def test_upload_calls_all_three_steps(self):
        streamer = self._make_streamer()
        csv_payload = "date,model,spend\n2025-01-15,gpt-4o,1.5"
        with patch.object(
            streamer, "_get_signed_url", return_value="https://signed"
        ) as mock_url, patch.object(
            streamer, "_initiate_resumable_upload", return_value="https://session"
        ) as mock_init, patch.object(
            streamer, "_finalize_upload"
        ) as mock_fin:
            streamer.upload(csv_payload, "2025-01-15T14:00:00Z")

        mock_url.assert_called_once_with("2025-01-15T14:00:00Z")
        mock_init.assert_called_once_with("https://signed")
        mock_fin.assert_called_once()
        # Second arg to _finalize_upload is gzip-compressed CSV bytes
        upload_bytes = mock_fin.call_args[0][1]
        assert isinstance(upload_bytes, bytes)
        assert gzip.decompress(upload_bytes) == csv_payload.encode("utf-8")


class TestMavvrikStreamerAdvanceMarker:
    def _make_streamer(self):
        return MavvrikStreamer(
            api_key="test-key",
            api_endpoint="https://api.mavvrik.dev",
            tenant="acme",
            instance_id="litellm-001",
        )

    def test_accepts_204(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.patch.return_value = mock_resp
            # Should not raise
            streamer.advance_marker(1737000000)

    def test_accepts_200(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.patch.return_value = mock_resp
            streamer.advance_marker(1737000000)

    def test_raises_on_error_status(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.patch.return_value = mock_resp
            with pytest.raises(Exception, match="advance_marker failed"):
                streamer.advance_marker(1737000000)

    def test_sends_correct_body(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        captured = []

        def fake_patch(url, headers=None, json=None, **kwargs):
            captured.append({"url": url, "json": json})
            return mock_resp

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.patch.side_effect = fake_patch
            streamer.advance_marker(1737000000)

        assert captured[0]["json"] == {"metricsMarker": 1737000000}

    def test_patches_agent_base_path(self):
        streamer = MavvrikStreamer(
            api_key="key",
            api_endpoint="https://api.example.com",
            tenant="my-org",
            instance_id="prod-001",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        captured_url = []

        def fake_patch(url, **kwargs):
            captured_url.append(url)
            return mock_resp

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.patch.side_effect = fake_patch
            streamer.advance_marker(1737000000)

        assert "my-org" in captured_url[0]
        assert "prod-001" in captured_url[0]

    def test_sends_x_api_key_header(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        captured_headers = []

        def fake_patch(url, headers=None, **kwargs):
            captured_headers.append(headers)
            return mock_resp

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.patch.side_effect = fake_patch
            streamer.advance_marker(1737000000)

        assert captured_headers[0].get("x-api-key") == "test-key"


class TestMavvrikStreamerRegister:
    def _make_streamer(self):
        return MavvrikStreamer(
            api_key="test-key",
            api_endpoint="https://api.mavvrik.dev",
            tenant="acme",
            instance_id="litellm-001",
        )

    def test_register_returns_iso_string_from_epoch(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # epoch 1737000000 → 2025-01-16T08:00:00+00:00
        mock_resp.json.return_value = {"id": "litellm-001", "metricsMarker": 1737000000}

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            marker = streamer.register()

        assert "2025-01-16" in marker
        assert "+00:00" in marker or "UTC" in marker or "Z" in marker or "+00" in marker

    def test_register_defaults_to_first_of_month_when_marker_zero(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "litellm-001", "metricsMarker": 0}

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            marker = streamer.register()

        # Should be first day of month
        from datetime import datetime, timezone

        marker_dt = datetime.fromisoformat(marker)
        assert marker_dt.day == 1
        assert marker_dt.hour == 0

    def test_register_defaults_to_first_of_month_when_marker_absent(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "litellm-001"}  # no metricsMarker

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            marker = streamer.register()

        from datetime import datetime, timezone

        marker_dt = datetime.fromisoformat(marker)
        assert marker_dt.day == 1

    def test_register_raises_on_non_200(self):
        streamer = self._make_streamer()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            with pytest.raises(Exception, match="registration failed"):
                streamer.register()

    def test_register_posts_to_agent_base_path(self):
        streamer = MavvrikStreamer(
            api_key="key",
            api_endpoint="https://api.example.com",
            tenant="my-org",
            instance_id="prod-001",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "prod-001", "metricsMarker": 1700000000}

        captured_url = []

        def fake_post(url, **kwargs):
            captured_url.append(url)
            return mock_resp

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.side_effect = fake_post
            streamer.register()

        assert "my-org" in captured_url[0]
        assert "prod-001" in captured_url[0]

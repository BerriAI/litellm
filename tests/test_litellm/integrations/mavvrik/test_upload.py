"""Unit tests for Mavvrik Client — Mavvrik API HTTP calls."""

import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.client import Client


def _make_client(**kwargs) -> Client:
    defaults = dict(
        api_key="test-key",
        api_endpoint="https://api.mavvrik.dev/acme",
        connection_id="litellm-001",
    )
    defaults.update(kwargs)
    return Client(**defaults)


def _mock_response(
    status_code: int, json_body=None, text="", headers=None
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_body or {}
    resp.headers = headers or {}
    return resp


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_strips_trailing_slash(self):
        c = Client(
            api_key="k", api_endpoint="https://api.mavvrik.dev/acme/", connection_id="x"
        )
        assert c.api_endpoint == "https://api.mavvrik.dev/acme"

    def test_stores_attributes(self):
        c = Client(
            api_key="mvk-key",
            api_endpoint="https://api.mavvrik.dev/t",
            connection_id="inst-1",
        )
        assert c.api_key == "mvk-key"
        assert c.connection_id == "inst-1"

    def test_agent_url_includes_connection_id(self):
        c = _make_client(connection_id="prod-001")
        assert "prod-001" in c.agent_url

    def test_upload_url_includes_connection_id(self):
        c = _make_client(connection_id="prod-001")
        assert "prod-001" in c.upload_url
        assert "upload-url" in c.upload_url


# ---------------------------------------------------------------------------
# _request — delegates to http_request (retry behaviour tested in test_http.py)
# ---------------------------------------------------------------------------


class TestClientRequest:
    @pytest.mark.asyncio
    async def test_delegates_to_http_request(self):
        """_request() delegates to the shared http_request transport."""
        c = _make_client()
        mock_resp = _mock_response(200)

        with patch(
            "litellm.integrations.mavvrik.client.http_request",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_http:
            resp = await c._request(
                "GET",
                "https://example.com",
                headers={"x-api-key": "k"},
                json={"a": 1},
                label="test",
            )

        assert resp.status_code == 200
        mock_http.assert_called_once_with(
            "GET",
            "https://example.com",
            headers={"x-api-key": "k"},
            json={"a": 1},
            params=None,
            content=None,
            timeout=30.0,
            label="test",
        )


# ---------------------------------------------------------------------------
# _assert_ok — status checker
# ---------------------------------------------------------------------------


class TestAssertOk:
    def test_passes_on_expected_status(self):
        resp = _mock_response(200)
        Client._assert_ok(resp, expected={200, 201})  # no raise

    def test_raises_on_unexpected_status(self):
        resp = _mock_response(403, text="Forbidden")
        with pytest.raises(RuntimeError, match="403"):
            Client._assert_ok(resp, expected={200})

    def test_accepts_any_code_in_set(self):
        for code in (200, 201, 204):
            Client._assert_ok(_mock_response(code), expected={200, 201, 204})


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


class TestClientRegister:
    @pytest.mark.asyncio
    async def test_returns_iso_string_from_epoch(self):
        c = _make_client()
        with patch.object(
            c,
            "_request",
            return_value=_mock_response(200, {"metricsMarker": 1737000000}),
        ):
            marker = await c.register()
        assert "2025-01-16" in marker
        assert "+00:00" in marker or "Z" in marker

    @pytest.mark.asyncio
    async def test_returns_none_when_marker_zero(self):
        c = _make_client()
        with patch.object(
            c, "_request", return_value=_mock_response(200, {"metricsMarker": 0})
        ):
            assert await c.register() is None

    @pytest.mark.asyncio
    async def test_returns_none_when_marker_absent(self):
        c = _make_client()
        with patch.object(c, "_request", return_value=_mock_response(200, {"id": "x"})):
            assert await c.register() is None

    @pytest.mark.asyncio
    async def test_raises_on_non_200(self):
        c = _make_client()
        with patch.object(
            c, "_request", return_value=_mock_response(401, text="Unauthorized")
        ):
            with pytest.raises(RuntimeError, match="401"):
                await c.register()

    @pytest.mark.asyncio
    async def test_posts_to_agent_url(self):
        c = _make_client(connection_id="prod-001")
        calls = []

        async def fake_request(method, url, **kwargs):
            calls.append((method, url))
            return _mock_response(200, {"metricsMarker": 1700000000})

        with patch.object(c, "_request", side_effect=fake_request):
            await c.register()

        assert calls[0] == ("POST", c.agent_url)
        assert "prod-001" in calls[0][1]

    @pytest.mark.asyncio
    async def test_sends_auth_header(self):
        c = _make_client()
        captured = []

        async def fake_request(method, url, *, headers=None, **kwargs):
            captured.append(headers)
            return _mock_response(200, {"metricsMarker": 1700000000})

        with patch.object(c, "_request", side_effect=fake_request):
            await c.register()

        assert captured[0].get("x-api-key") == "test-key"


# ---------------------------------------------------------------------------
# advance_marker()
# ---------------------------------------------------------------------------


class TestClientAdvanceMarker:
    @pytest.mark.asyncio
    async def test_accepts_204(self):
        c = _make_client()
        with patch.object(c, "_request", return_value=_mock_response(204)):
            await c.advance_marker(1737000000)

    @pytest.mark.asyncio
    async def test_accepts_200(self):
        c = _make_client()
        with patch.object(c, "_request", return_value=_mock_response(200)):
            await c.advance_marker(1737000000)

    @pytest.mark.asyncio
    async def test_raises_on_error_status(self):
        c = _make_client()
        with patch.object(
            c, "_request", return_value=_mock_response(403, text="Forbidden")
        ):
            with pytest.raises(RuntimeError, match="403"):
                await c.advance_marker(1737000000)

    @pytest.mark.asyncio
    async def test_sends_correct_body(self):
        c = _make_client()
        captured = []

        async def fake_request(method, url, *, json=None, **kwargs):
            captured.append(json)
            return _mock_response(204)

        with patch.object(c, "_request", side_effect=fake_request):
            await c.advance_marker(1737000000)

        assert captured[0] == {"metricsMarker": 1737000000}

    @pytest.mark.asyncio
    async def test_patches_agent_url(self):
        c = _make_client(connection_id="prod-001")
        calls = []

        async def fake_request(method, url, **kwargs):
            calls.append((method, url))
            return _mock_response(204)

        with patch.object(c, "_request", side_effect=fake_request):
            await c.advance_marker(1737000000)

        assert calls[0][0] == "PATCH"
        assert "prod-001" in calls[0][1]

    @pytest.mark.asyncio
    async def test_sends_auth_header(self):
        c = _make_client()
        captured = []

        async def fake_request(method, url, *, headers=None, **kwargs):
            captured.append(headers)
            return _mock_response(204)

        with patch.object(c, "_request", side_effect=fake_request):
            await c.advance_marker(1737000000)

        assert captured[0].get("x-api-key") == "test-key"


# ---------------------------------------------------------------------------
# report_error()
# ---------------------------------------------------------------------------


class TestClientReportError:
    @pytest.mark.asyncio
    async def test_swallows_exception(self):
        c = _make_client()
        with patch.object(c, "_request", side_effect=RuntimeError("network down")):
            await c.report_error("something went wrong")  # must not raise

    @pytest.mark.asyncio
    async def test_truncates_message_to_500_chars(self):
        c = _make_client()
        captured = []

        async def fake_request(method, url, *, json=None, **kwargs):
            captured.append(json)
            return _mock_response(204)

        with patch.object(c, "_request", side_effect=fake_request):
            await c.report_error("x" * 600)

        assert len(captured[0]["error"]) == 500

    @pytest.mark.asyncio
    async def test_sends_error_field_in_body(self):
        c = _make_client()
        captured = []

        async def fake_request(method, url, *, json=None, **kwargs):
            captured.append(json)
            return _mock_response(204)

        with patch.object(c, "_request", side_effect=fake_request):
            await c.report_error("export failed")

        assert captured[0] == {"error": "export failed"}

    @pytest.mark.asyncio
    async def test_logs_warning_on_unexpected_status(self):
        """report_error logs a warning when response status is not 200/204."""
        c = _make_client()
        with patch.object(c, "_request", return_value=_mock_response(500, text="err")):
            await c.report_error("something broke")  # must not raise


# ---------------------------------------------------------------------------
# get_signed_url()
# ---------------------------------------------------------------------------


class TestClientGetSignedUrl:
    @pytest.mark.asyncio
    async def test_returns_signed_url_on_200(self):
        c = _make_client()
        with patch.object(
            c,
            "_request",
            return_value=_mock_response(
                200, {"url": "https://storage.example.com/signed"}
            ),
        ):
            url = await c.get_signed_url("2025-01-15")
        assert url == "https://storage.example.com/signed"

    @pytest.mark.asyncio
    async def test_raises_on_missing_url_field(self):
        c = _make_client()
        with patch.object(c, "_request", return_value=_mock_response(200, {})):
            with pytest.raises(RuntimeError, match="missing 'url' field"):
                await c.get_signed_url("2025-01-15")

    @pytest.mark.asyncio
    async def test_raises_on_non_200(self):
        c = _make_client()
        with patch.object(
            c, "_request", return_value=_mock_response(403, text="Forbidden")
        ):
            with pytest.raises(RuntimeError, match="403"):
                await c.get_signed_url("2025-01-15")

    @pytest.mark.asyncio
    async def test_sends_date_as_name_param(self):
        c = _make_client()
        captured = []

        async def fake_request(method, url, *, params=None, **kwargs):
            captured.append(params)
            return _mock_response(200, {"url": "https://example.com/signed"})

        with patch.object(c, "_request", side_effect=fake_request):
            await c.get_signed_url("2025-01-15")

        assert captured[0]["name"] == "2025-01-15"
        assert captured[0]["type"] == "metrics"
        assert captured[0]["datetime"] == "2025-01-15"

    @pytest.mark.asyncio
    async def test_sends_auth_header(self):
        c = _make_client()
        captured = []

        async def fake_request(method, url, *, headers=None, **kwargs):
            captured.append(headers)
            return _mock_response(200, {"url": "https://example.com/signed"})

        with patch.object(c, "_request", side_effect=fake_request):
            await c.get_signed_url("2025-01-15")

        assert captured[0].get("x-api-key") == "test-key"

    @pytest.mark.asyncio
    async def test_gets_upload_url_with_connection_id(self):
        c = _make_client(connection_id="prod-001")
        calls = []

        async def fake_request(method, url, **kwargs):
            calls.append(url)
            return _mock_response(200, {"url": "https://example.com/signed"})

        with patch.object(c, "_request", side_effect=fake_request):
            await c.get_signed_url("2025-01-15")

        assert "prod-001" in calls[0]
        assert "upload-url" in calls[0]

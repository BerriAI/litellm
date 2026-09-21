import gzip
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import MavvrikFocusLogger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy import proxy_server

_WINDOW_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
_WINDOW_END = datetime(2026, 1, 2, tzinfo=timezone.utc)
_SIGNED_URL = "https://storage.googleapis.com/mavvrik-bucket/metrics/2026-01-01?sig=abc"
_SESSION_URI = "https://storage.googleapis.com/upload/mavvrik-bucket/session-1"

_USAGE_ROW = {
    "id": "row-1",
    "date": "2026-01-01",
    "user_id": "user-1",
    "api_key": "hashed-key",
    "model": "gpt-4o",
    "model_group": "gpt-4o",
    "custom_llm_provider": "openai",
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "spend": 0.25,
    "api_requests": 1,
    "successful_requests": 1,
    "failed_requests": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "created_at": _WINDOW_START,
    "updated_at": _WINDOW_START,
    "team_id": "team-1",
    "api_key_alias": "key-alias",
    "team_alias": "team-alias",
    "user_email": "user@example.com",
    "organization_id": None,
    "organization_alias": None,
}


class _MavvrikWire:
    """Plays the Mavvrik API and GCS at the httpx boundary and records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/metrics/agent/ai/conn-1"):
            return httpx.Response(200, json={"metricsMarker": 0})
        if request.method == "GET" and request.url.path.endswith("/upload-url"):
            return httpx.Response(200, json={"url": _SIGNED_URL})
        if request.method == "POST" and request.url.host == "storage.googleapis.com":
            return httpx.Response(201, headers={"Location": _SESSION_URI})
        if request.method == "PUT" and request.url.host == "storage.googleapis.com":
            return httpx.Response(200)
        if request.method == "PATCH" and request.url.path.endswith("/metrics/agent/ai/conn-1"):
            return httpx.Response(200, json={})
        return httpx.Response(500, text=f"unexpected {request.method} {request.url}")

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, str(r.url).split("?")[0]) for r in self.requests]


@pytest.fixture
def mavvrik_env(monkeypatch) -> None:
    monkeypatch.setenv("MAVVRIK_API_KEY", "mavvrik-key")
    monkeypatch.setenv("MAVVRIK_API_ENDPOINT", "https://api.mavvrik.dev/tenant-1")
    monkeypatch.setenv("MAVVRIK_CONNECTION_ID", "conn-1")
    monkeypatch.delenv("MAVVRIK_FOCUS_MAX_ROWS", raising=False)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())


async def _seed_mavvrik_wire() -> _MavvrikWire:
    wire = _MavvrikWire()
    handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
    assert isinstance(handler, AsyncHTTPHandler)
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(wire))
    return wire


def _install_usage_rows(monkeypatch, rows: list[dict]) -> AsyncMock:
    query_raw = AsyncMock(return_value=rows)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=SimpleNamespace(query_raw=query_raw)))
    return query_raw


@pytest.mark.asyncio
async def test_export_window_delivers_empty_payload_for_empty_export(monkeypatch, mavvrik_env) -> None:
    mavvrik_wire = await _seed_mavvrik_wire()
    _install_usage_rows(monkeypatch, [])

    await MavvrikFocusLogger().export_usage_data(start_time_utc=_WINDOW_START, end_time_utc=_WINDOW_END)

    assert mavvrik_wire.calls() == [
        ("POST", "https://api.mavvrik.dev/tenant-1/metrics/agent/ai/conn-1"),
        ("PATCH", "https://api.mavvrik.dev/tenant-1/metrics/agent/ai/conn-1"),
    ]
    marker_update = mavvrik_wire.requests[-1]
    assert marker_update.headers["x-api-key"] == "mavvrik-key"
    assert marker_update.read() == b'{"metricsMarker":%d}' % int(_WINDOW_START.timestamp())


@pytest.mark.asyncio
async def test_export_window_uploads_gzipped_csv_and_advances_marker(monkeypatch, mavvrik_env) -> None:
    mavvrik_wire = await _seed_mavvrik_wire()
    query_raw = _install_usage_rows(monkeypatch, [_USAGE_ROW])

    await MavvrikFocusLogger().export_usage_data(start_time_utc=_WINDOW_START, end_time_utc=_WINDOW_END)

    assert query_raw.await_args.args[1:] == (_WINDOW_START, _WINDOW_END, 500_000)
    assert mavvrik_wire.calls() == [
        ("POST", "https://api.mavvrik.dev/tenant-1/metrics/agent/ai/conn-1"),
        ("GET", "https://api.mavvrik.dev/tenant-1/metrics/agent/ai/conn-1/upload-url"),
        ("POST", _SIGNED_URL.split("?")[0]),
        ("PUT", _SESSION_URI),
        ("PATCH", "https://api.mavvrik.dev/tenant-1/metrics/agent/ai/conn-1"),
    ]
    upload_url_request = mavvrik_wire.requests[1]
    assert upload_url_request.url.params["datetime"] == "2026-01-01"
    uploaded_csv = gzip.decompress(mavvrik_wire.requests[3].read()).decode()
    header, row = uploaded_csv.strip().splitlines()
    assert "ChargePeriodStart" in header.split(",")
    assert "gpt-4o" in row
    assert "2026-01-01T00:00:00Z" in row


@pytest.mark.asyncio
async def test_export_window_honours_max_rows_cap(monkeypatch, mavvrik_env) -> None:
    await _seed_mavvrik_wire()
    monkeypatch.setenv("MAVVRIK_FOCUS_MAX_ROWS", "25")
    query_raw = _install_usage_rows(monkeypatch, [])

    await MavvrikFocusLogger().export_usage_data(start_time_utc=_WINDOW_START, end_time_utc=_WINDOW_END)

    assert query_raw.await_args.args[1:] == (_WINDOW_START, _WINDOW_END, 25)

import base64
import gzip
import json
import time
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.rust_bridge._native import NativeClickHouseSpendConfig, NativeClickHouseSpendStorage
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


def _storage(url: str, retention_days: int = 14) -> NativeClickHouseSpendStorage:
    return NativeClickHouseSpendStorage(NativeClickHouseSpendConfig("spend_test", url, retention_days))


@pytest.mark.asyncio
async def test_native_spend_writer_preserves_python_mappings(recording_server: RecordingServer) -> None:
    recording_server.enqueue(ResponseSpec(body=""))
    storage: Final = _storage(recording_server.base_url)
    attributes: Final = MappingProxyType({"label": "雪"})
    row: Final = MappingProxyType(
        {
            "start_time": 1_234,
            "end_time": 2_345,
            "completion_start_time": None,
            "metadata": attributes,
            "request_tags": ("first", "second"),
            "EngineReceivedMs": -1,
        }
    )
    before: Final = time.time_ns() // 1_000_000
    await storage.insert_rows((row,))
    after: Final = time.time_ns() // 1_000_000
    request: Final = recording_server.requests[0]
    stored: Final = json.loads(gzip.decompress(request.raw_body))
    assert before <= stored["EngineReceivedMs"] <= after
    assert stored == {
        "start_time": "1970-01-01T00:00:01.234Z",
        "end_time": "1970-01-01T00:00:02.345Z",
        "completion_start_time": None,
        "metadata": attributes,
        "request_tags": ["first", "second"],
        "EngineReceivedMs": stored["EngineReceivedMs"],
    }
    assert row["EngineReceivedMs"] == -1
    assert parse_qs(urlsplit(request.path).query)["query"] == ["INSERT INTO `spend_test`.spend_logs FORMAT JSONEachRow"]


@pytest.mark.asyncio
async def test_native_spend_writer_rejects_python_objects_before_io(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = 0
    invalid: Final = object()
    with pytest.raises(ValueError, match=type(invalid).__name__):
        await _storage(recording_server.base_url).insert_rows(({"metadata": invalid},))
    assert recording_server.requests == []


@pytest.mark.parametrize("status", (400, 503), ids=("bad-row", "unavailable"))
@pytest.mark.asyncio
async def test_native_spend_writer_maps_insert_failures(recording_server: RecordingServer, status: int) -> None:
    recording_server.enqueue(ResponseSpec(status=status, body="denied"))
    with pytest.raises(RuntimeError, match=f"insert failed with HTTP status {status}"):
        await _storage(recording_server.base_url).insert_rows(({"request_id": "request"},))


@pytest.mark.parametrize("limit", ("0", "invalid"))
@pytest.mark.asyncio
async def test_native_spend_writer_validates_configured_limit(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    recording_server.expected_requests = 0
    monkeypatch.setenv("CLICKHOUSE_TRACE_MAX_INSERT_BYTES", limit)
    with pytest.raises(ValueError, match="CLICKHOUSE_TRACE_MAX_INSERT_BYTES must be a positive integer"):
        await _storage(recording_server.base_url).insert_rows(({"request_id": "request"},))
    assert recording_server.requests == []


@pytest.mark.asyncio
async def test_native_spend_writer_bounds_encoded_bytes(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording_server.expected_requests = 0
    monkeypatch.setenv("CLICKHOUSE_TRACE_MAX_INSERT_BYTES", "64")
    with pytest.raises(OverflowError, match="encoded size limit"):
        await _storage(recording_server.base_url).insert_rows(({"metadata": "雪" * 64},))
    assert recording_server.requests == []


@pytest.mark.asyncio
async def test_native_spend_schema_uses_writer_credentials_and_surfaces_failure(
    recording_server: RecordingServer,
) -> None:
    recording_server.expected_requests = 2
    recording_server.enqueue(ResponseSpec(body=b""))
    recording_server.enqueue(ResponseSpec(status=403, body="denied"))
    writer_url: Final = recording_server.base_url.replace("http://", "http://writer:p%40ss%2Fword%25@")
    with pytest.raises(RuntimeError, match="schema setup failed with HTTP status 403"):
        await _storage(writer_url + "?database=wrong&readonly=1", 7).ensure_schema()
    assert recording_server.requests[0].raw_body.startswith(b"CREATE DATABASE IF NOT EXISTS")
    assert recording_server.requests[1].raw_body.startswith(b"CREATE TABLE IF NOT EXISTS")
    assert "readonly" not in parse_qs(urlsplit(recording_server.requests[0].path).query)
    assert recording_server.requests[0].headers["authorization"] == (
        "Basic " + base64.b64encode(b"writer:p@ss/word%").decode()
    )

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


@pytest.mark.asyncio
async def test_native_spend_rollup_reads_through_bounded_query(recording_server: RecordingServer) -> None:
    rows: Final = [
        {
            "bucket": "2026-10-24T21:00:00",
            "tag": "prod",
            "spend": 0.5,
            "prompt_tokens": "12",
            "completion_tokens": 3,
            "cache_read_input_tokens": "0",
            "cache_creation_input_tokens": "0",
            "api_requests": "2",
            "successful_requests": "1",
            "failed_requests": "1",
            "total_response_time_ms": "340",
            "timed_requests": "2",
            "distinct_api_keys": "1",
        }
    ]
    recording_server.enqueue(ResponseSpec(body={"meta": [], "data": rows, "rows": 1}))
    result: Final = await _storage(recording_server.base_url).fetch_rollup(
        {
            "start_ms": 1_000,
            "end_ms": 61_000,
            "bucket": "1h",
            "group_by": ["tag"],
            "filters": {"tag": ["prod", "it's"]},
            "limit": 10,
        }
    )
    assert result == [
        {
            "bucket": "2026-10-24T21:00:00",
            "tag": "prod",
            "spend": 0.5,
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "api_requests": 2,
            "successful_requests": 1,
            "failed_requests": 1,
            "total_response_time_ms": 340,
            "timed_requests": 2,
            "distinct_api_keys": 1,
        }
    ]
    request: Final = recording_server.requests[0]
    query: Final = parse_qs(urlsplit(request.path).query)
    assert query["readonly"] == ["1"]
    assert query["database"] == ["spend_test"]
    assert query["param_filter_tag"] == ["['prod','it\\'s']"]
    assert query["param_limit"] == ["10"]
    assert b"ARRAY JOIN request_tags AS tag" in request.raw_body
    assert b"GROUP BY bucket, tag" in request.raw_body


@pytest.mark.parametrize(
    ("request_body", "match"),
    (
        ({"start_ms": 5, "end_ms": 5, "group_by": ["tag"]}, "end_ms must be after start_ms"),
        ({"start_ms": 1, "end_ms": 2, "group_by": ["request_id"]}, "Matching variant not found"),
        ({"start_ms": 1, "end_ms": 2, "sql": "DROP TABLE spend_logs"}, "unknown field"),
    ),
    ids=("window", "dimension", "raw-sql"),
)
@pytest.mark.asyncio
async def test_native_spend_rollup_rejects_invalid_requests_before_io(
    recording_server: RecordingServer, request_body: dict[str, object], match: str
) -> None:
    recording_server.expected_requests = 0
    with pytest.raises(ValueError, match=match):
        await _storage(recording_server.base_url).fetch_rollup(request_body)
    assert recording_server.requests == []

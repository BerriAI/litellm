import base64
import gzip
import json
import time
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.rust_bridge._native import NativeClickHouseSpendConfig, NativeClickHouseSpendStorage
from tests._support.provider import SharedProvider
from tests._support.wire import Reply

pytestmark = pytest.mark.requires_rust_extension


def _storage(url: str, retention_days: int = 14) -> NativeClickHouseSpendStorage:
    return NativeClickHouseSpendStorage(NativeClickHouseSpendConfig("spend_test", url, retention_days))


@pytest.mark.asyncio
async def test_native_spend_writer_preserves_python_mappings(recording_server: SharedProvider) -> None:
    recording_server.expect(Reply(body=b""))
    storage: Final = _storage(recording_server.wire.url)
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
    requests: Final = recording_server.received()
    assert len(requests) == 1
    request: Final = requests[0]
    stored: Final = json.loads(gzip.decompress(request.body))
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
    assert parse_qs(urlsplit(request.target).query)["query"] == [
        "INSERT INTO `spend_test`.spend_logs FORMAT JSONEachRow"
    ]


@pytest.mark.asyncio
async def test_native_spend_writer_rejects_python_objects_before_io(recording_server: SharedProvider) -> None:
    invalid: Final = object()
    with pytest.raises(ValueError, match=type(invalid).__name__):
        await _storage(recording_server.wire.url).insert_rows(({"metadata": invalid},))
    requests: Final = recording_server.received()
    assert requests == ()


@pytest.mark.parametrize("status", (400, 503), ids=("bad-row", "unavailable"))
@pytest.mark.asyncio
async def test_native_spend_writer_maps_insert_failures(recording_server: SharedProvider, status: int) -> None:
    recording_server.expect(Reply(status=status, body=b"denied"))
    with pytest.raises(RuntimeError, match=f"insert failed with HTTP status {status}"):
        await _storage(recording_server.wire.url).insert_rows(({"request_id": "request"},))
    requests: Final = recording_server.received()
    assert len(requests) == 1


@pytest.mark.parametrize("limit", ("0", "invalid"))
@pytest.mark.asyncio
async def test_native_spend_writer_validates_configured_limit(
    recording_server: SharedProvider, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    monkeypatch.setenv("CLICKHOUSE_TRACE_MAX_INSERT_BYTES", limit)
    with pytest.raises(ValueError, match="CLICKHOUSE_TRACE_MAX_INSERT_BYTES must be a positive integer"):
        await _storage(recording_server.wire.url).insert_rows(({"request_id": "request"},))
    requests: Final = recording_server.received()
    assert requests == ()


@pytest.mark.asyncio
async def test_native_spend_writer_bounds_encoded_bytes(
    recording_server: SharedProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLICKHOUSE_TRACE_MAX_INSERT_BYTES", "64")
    with pytest.raises(OverflowError, match="encoded size limit"):
        await _storage(recording_server.wire.url).insert_rows(({"metadata": "雪" * 64},))
    requests: Final = recording_server.received()
    assert requests == ()


@pytest.mark.asyncio
async def test_native_spend_schema_uses_writer_credentials_and_surfaces_failure(
    recording_server: SharedProvider,
) -> None:
    recording_server.expect(Reply(body=b""), Reply(status=403, body=b"denied"))
    writer_url: Final = recording_server.wire.url.replace("http://", "http://writer:p%40ss%2Fword%25@")
    with pytest.raises(RuntimeError, match="schema setup failed with HTTP status 403"):
        await _storage(writer_url + "?database=wrong&readonly=1", 7).ensure_schema()
    requests: Final = recording_server.received()
    assert len(requests) == 2
    assert requests[0].body.startswith(b"CREATE DATABASE IF NOT EXISTS")
    assert requests[1].body.startswith(b"CREATE TABLE IF NOT EXISTS")
    assert "readonly" not in parse_qs(urlsplit(requests[0].target).query)
    assert requests[0].headers["authorization"] == ("Basic " + base64.b64encode(b"writer:p@ss/word%").decode())

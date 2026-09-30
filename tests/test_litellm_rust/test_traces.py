import base64
import gzip
import json
import time
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.rust_bridge._native import NativeTraceStorage
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
async def test_trace_reader_projects_connection_and_parameters(recording_server: RecordingServer) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [{"trace_id": "trace-1"}]}))
    reader_url: Final = recording_server.base_url.replace("http://", "http://reader:p%40ss%2Fword%25@")
    storage: Final = NativeTraceStorage("trace_test", recording_server.base_url, reader_url + "?database=wrong")
    response: Final = json.loads(
        await storage.query("trace_spans", {"trace_id": "trace-1", "team_ids": [], "api_key_hash": "", "trace_ref": ""})
    )
    request: Final = recording_server.requests[0]
    parameters: Final = parse_qs(urlsplit(request.path).query)
    assert response["data"] == [{"trace_id": "trace-1"}]
    assert b"o.TraceId = {trace_id:String}" in request.raw_body
    assert b"trace-1" not in request.raw_body
    assert parameters["database"] == ["trace_test"]
    assert parameters["param_trace_id"] == ["trace-1"]
    assert parameters["readonly"] == ["1"]
    assert "user" not in parameters
    assert "password" not in parameters
    assert request.headers["authorization"] == "Basic " + base64.b64encode(b"reader:p@ss/word%").decode()


@pytest.mark.asyncio
async def test_trace_reader_rejects_success_status_with_embedded_error(recording_server: RecordingServer) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [], "exception": "query failed"}))
    storage: Final = NativeTraceStorage("trace_test", recording_server.base_url, recording_server.base_url)
    with pytest.raises(RuntimeError, match="invalid or failed JSON"):
        await storage.query("trace_spans", {"trace_id": "trace-1", "team_ids": [], "api_key_hash": "", "trace_ref": ""})


@pytest.mark.asyncio
async def test_trace_reader_rejects_arbitrary_sql() -> None:
    storage: Final = NativeTraceStorage("trace_test", "http://localhost:8123", "http://localhost:8123")
    with pytest.raises(ValueError, match="unknown ClickHouse read query"):
        await storage.query("SELECT 1", {})


@pytest.mark.asyncio
async def test_schema_binding_rejects_invalid_database() -> None:
    with pytest.raises(ValueError, match=r"database.*retention"):
        NativeTraceStorage("db; DROP DATABASE default", "http://localhost:8123")


@pytest.mark.asyncio
async def test_schema_binding_rejects_non_positive_retention() -> None:
    storage: Final = NativeTraceStorage("traces", "http://localhost:8123")
    with pytest.raises(ValueError, match=r"database.*retention"):
        await storage.ensure_schema(0, 14)


@pytest.mark.asyncio
async def test_schema_setup_uses_writer_credentials_and_rejects_failed_statement(
    recording_server: RecordingServer,
) -> None:
    recording_server.expected_requests = 2
    recording_server.enqueue(ResponseSpec(body=""))
    recording_server.enqueue(ResponseSpec(status=403, body="denied"))
    writer_url: Final = recording_server.base_url.replace("http://", "http://writer:p%40ss%2Fword%25@")
    storage: Final = NativeTraceStorage("trace_test", writer_url + "?database=wrong&readonly=1")
    with pytest.raises(RuntimeError, match="schema setup failed with HTTP status 403"):
        await storage.ensure_schema(7, 14)
    assert len(recording_server.requests) == 2
    assert recording_server.requests[0].raw_body.startswith(b"CREATE DATABASE IF NOT EXISTS")
    assert recording_server.requests[1].raw_body.startswith(b"CREATE TABLE IF NOT EXISTS")
    assert "readonly" not in parse_qs(urlsplit(recording_server.requests[0].path).query)
    assert (
        recording_server.requests[0].headers["authorization"]
        == "Basic " + base64.b64encode(b"writer:p@ss/word%").decode()
    )


@pytest.mark.asyncio
async def test_insert_encodes_and_sends_rows(recording_server: RecordingServer) -> None:
    recording_server.enqueue(ResponseSpec(body=""))
    storage: Final = NativeTraceStorage("trace_test", recording_server.base_url)
    started_ms: Final = time.time_ns() // 1_000_000
    await storage.insert_rows("otel_traces", [{"Timestamp": 1_234_567_890, "Input": "hello"}])
    finished_ms: Final = time.time_ns() // 1_000_000
    request: Final = recording_server.requests[0]
    row: Final = json.loads(gzip.decompress(request.raw_body))
    assert started_ms <= row["EngineReceivedMs"] <= finished_ms
    assert {key: value for key, value in row.items() if key != "EngineReceivedMs"} == {
        "Input": "hello",
        "Timestamp": "1970-01-01T00:00:01.23456789Z",
    }
    assert parse_qs(urlsplit(request.path).query)["query"] == [
        "INSERT INTO `trace_test`.otel_traces FORMAT JSONEachRow"
    ]
    assert request.headers["content-encoding"] == "gzip"

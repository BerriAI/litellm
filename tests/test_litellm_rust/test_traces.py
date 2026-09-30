import base64
import gzip
import json
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
    rows: Final = json.loads(await storage.query("span_detail", {"trace_id": "trace-1", "span_id": "span-1"}))["data"]
    request: Final = recording_server.requests[0]
    parameters: Final = parse_qs(urlsplit(request.path).query)
    assert rows == [{"trace_id": "trace-1"}]
    assert b"WHERE TraceId = {trace_id:String} AND SpanId = {span_id:String}" in request.raw_body
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
        await storage.query("span_detail", {})


@pytest.mark.asyncio
async def test_trace_reader_rejects_raw_sql(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = 0
    storage: Final = NativeTraceStorage("trace_test", recording_server.base_url, recording_server.base_url)
    with pytest.raises(ValueError, match="invalid trace read query"):
        await storage.query("SELECT * FROM otel_traces", {})
    assert recording_server.requests == []


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
    assert recording_server.requests[0].headers["authorization"] == "Basic " + base64.b64encode(
        b"writer:p@ss/word%"
    ).decode()


@pytest.mark.asyncio
async def test_insert_encodes_and_sends_rows(recording_server: RecordingServer) -> None:
    recording_server.enqueue(ResponseSpec(body=""))
    storage: Final = NativeTraceStorage(
        "trace_test", recording_server.base_url + "?input_format_skip_unknown_fields=1"
    )
    await storage.insert_rows("otel_traces", [{"Timestamp": 1_234_567_890, "Input": "hello"}])
    request: Final = recording_server.requests[0]
    assert json.loads(gzip.decompress(request.raw_body)) == {
        "Input": "hello",
        "Timestamp": "1970-01-01T00:00:01.23456789Z",
    }
    assert parse_qs(urlsplit(request.path).query)["query"] == [
        "INSERT INTO `trace_test`.otel_traces FORMAT JSONEachRow"
    ]
    assert parse_qs(urlsplit(request.path).query)["input_format_skip_unknown_fields"] == ["0"]
    assert request.headers["content-encoding"] == "gzip"

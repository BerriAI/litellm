import base64
import gzip
import json
import time
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import JsonValue

from litellm.constants import OTLP_MAX_ATTRIBUTE_VALUE_BYTES
from litellm.rust_bridge._native import NativeTraceConfig, NativeTraceStorage
from litellm.rust_bridge.trace.generated.models import ActivityAvailability, LensAccessParams
from litellm.rust_bridge.trace.generated.types import TraceScope
from litellm.rust_bridge.trace.storage import ClickHouseStorage, TraceStorageConfig, span_rows
from litellm.tracing import Tenant, TraceReceiver, TracingPayloadTooLargeError
from tests._support.recording_server import RecordingServer, ResponseSpec


pytestmark = pytest.mark.requires_rust_extension


def _native_storage(database: str, url: str, retention_days: int = 14) -> NativeTraceStorage:
    return NativeTraceStorage(NativeTraceConfig(database, url, retention_days, OTLP_MAX_ATTRIBUTE_VALUE_BYTES))


@pytest.fixture
def span_row() -> dict[str, JsonValue]:
    return {
        "span_id": "span-1",
        "parent_span_id": "",
        "name": "root",
        "type": "agent",
        "agent": "",
        "framework": "",
        "status": "STATUS_CODE_OK",
        "status_message": "",
        "error_truncated": 0,
        "start_ns": "1000000000",
        "duration_ns": "1000",
        "service": "test",
        "input_preview": "hello",
        "model": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "litellm_request_id": "",
        "team_id": "",
        "api_key_hash": "",
        "user_id": "",
    }


@pytest.fixture
def span_params() -> dict[str, str | int | list[str]]:
    return {"trace_id": "trace-1", "trace_ref": "", "all_teams": 1, "user_id": "", "team_ids": []}


@pytest.mark.asyncio
async def test_trace_reader_projects_connection_and_parameters(
    recording_server: RecordingServer, span_row: dict[str, JsonValue], span_params: dict[str, str | int | list[str]]
) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [span_row]}))
    url: Final = recording_server.base_url.replace("http://", "http://reader:p%40ss%2Fword%25@")
    storage: Final = _native_storage("trace_test", url + "?database=wrong")
    rows: Final = json.loads(await storage.query("trace_spans", span_params))
    request: Final = recording_server.requests[0]
    parameters: Final = parse_qs(urlsplit(request.path).query)
    assert rows == {"data": [span_row]}
    assert b"o.TraceId = {trace_id:String}" in request.raw_body
    assert parameters["database"] == ["trace_test"]
    assert parameters["param_trace_id"] == ["trace-1"]
    assert parameters["readonly"] == ["1"]
    assert "user" not in parameters
    assert "password" not in parameters
    assert request.headers["authorization"] == "Basic " + base64.b64encode(b"reader:p@ss/word%").decode()


@pytest.mark.asyncio
async def test_trace_reader_rejects_success_status_with_embedded_error(
    recording_server: RecordingServer, span_params: dict[str, str | int | list[str]]
) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [], "exception": "query failed"}))
    storage: Final = _native_storage("trace_test", recording_server.base_url)
    with pytest.raises(RuntimeError, match="invalid or failed JSON"):
        await storage.query("trace_spans", span_params)


@pytest.mark.asyncio
async def test_reader_rejects_arbitrary_sql_before_sending(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = 0
    storage: Final = _native_storage("trace_test", recording_server.base_url)
    with pytest.raises(ValueError, match="unknown ClickHouse read query"):
        await storage.query("SELECT 1", {})


@pytest.mark.asyncio
async def test_from_env_reads_with_clickhouse_url(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": []}))
    monkeypatch.setenv("CLICKHOUSE_URL", recording_server.base_url)
    monkeypatch.delenv("CLICKHOUSE_READER_URL", raising=False)
    scope: Final[TraceScope] = {"all_teams": 1, "user_id": "", "team_ids": ()}
    page: Final = await TraceReceiver.from_env().list_traces(scope, 0, 1)
    assert page == {"data": (), "next_cursor": None}
    assert len(recording_server.requests) == 1


@pytest.mark.asyncio
async def test_schema_setup_uses_configured_retention(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = None
    recording_server.default_response = ResponseSpec(body=b"")
    storage: Final = _native_storage("trace_test", recording_server.base_url, 7)
    await storage.ensure_schema()
    ttl_statements: Final = tuple(
        request.raw_body for request in recording_server.requests if b"MODIFY TTL" in request.raw_body
    )
    assert all(b"INTERVAL 7 DAY" in statement for statement in ttl_statements)
    assert tuple(request.raw_body.strip() for request in recording_server.requests[-3:]) == (
        b"ALTER TABLE `trace_test`.otel_traces MODIFY TTL toDateTime(Timestamp) + INTERVAL 7 DAY",
        b"ALTER TABLE `trace_test`.agent_traces_by_key MODIFY TTL toDateTime(StartTs) + INTERVAL 7 DAY",
        b"ALTER TABLE `trace_test`.spend_logs MODIFY TTL toDateTime(start_time) + INTERVAL 7 DAY",
    )


@pytest.mark.asyncio
async def test_schema_setup_uses_writer_credentials_and_rejects_failed_statement(
    recording_server: RecordingServer,
) -> None:
    recording_server.expected_requests = 2
    recording_server.enqueue(ResponseSpec(body=b""))
    recording_server.enqueue(ResponseSpec(status=403, body="denied"))
    writer_url: Final = recording_server.base_url.replace("http://", "http://writer:p%40ss%2Fword%25@")
    storage: Final = _native_storage("trace_test", writer_url + "?database=wrong&readonly=1", 7)
    with pytest.raises(RuntimeError, match="schema setup failed with HTTP status 403"):
        await storage.ensure_schema()
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
    storage: Final = _native_storage("trace_test", recording_server.base_url)
    before: Final = time.time_ns() // 1_000_000
    await storage.insert_rows("otel_traces", [{"Timestamp": 1_234_567_890, "Input": "hello", "EngineReceivedMs": -1}])
    after: Final = time.time_ns() // 1_000_000
    request: Final = recording_server.requests[0]
    row: Final = json.loads(gzip.decompress(request.raw_body))
    assert before <= row["EngineReceivedMs"] <= after
    assert row == {
        "Input": "hello",
        "Timestamp": "1970-01-01T00:00:01.23456789Z",
        "EngineReceivedMs": row["EngineReceivedMs"],
    }
    assert parse_qs(urlsplit(request.path).query)["query"] == [
        "INSERT INTO `trace_test`.otel_traces FORMAT JSONEachRow"
    ]
    assert request.headers["content-encoding"] == "gzip"


def _resource_export(attribute_bytes: int, span_count: int, groups: int = 1) -> bytes:
    span: Final = {
        "traceId": "01" * 16,
        "spanId": "02" * 8,
        "name": "shared-resource",
        "startTimeUnixNano": "1",
        "endTimeUnixNano": "2",
    }
    resource: Final = {
        "resource": {
            "attributes": [
                {"key": "shared", "value": {"stringValue": "x" * attribute_bytes}},
                {"key": "litellm.team_id", "value": {"stringValue": "spoofed"}},
            ]
        },
        "scopeSpans": [
            {
                "scope": {"name": "scope-" * 32, "version": "v" * 128},
                "spans": [{**span, "spanId": f"{index + 1:016x}"} for index in range(span_count)],
            }
        ],
    }
    return json.dumps({"resourceSpans": [resource] * groups}).encode()


@pytest.mark.asyncio
async def test_resource_fanout_reaches_insert_with_identical_values(recording_server: RecordingServer) -> None:
    body: Final = _resource_export(16 * 1024, 1024)
    receiver: Final = TraceReceiver(ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test")))
    tenant: Final = Tenant("team-a", "key-a", "org-a")
    assert await receiver.ingest(body, "application/json", None, tenant) == 1024
    encoded: Final = gzip.decompress(recording_server.requests[0].raw_body)
    actual: Final = tuple(json.loads(line) for line in encoded.splitlines())
    expected: Final = span_rows(body, "application/json", tenant)
    assert len(encoded) < 64 * 1024 * 1024
    assert tuple({key: value for key, value in row.items() if key != "EngineReceivedMs"} for row in actual) == tuple(
        {**row, "Timestamp": "1970-01-01T00:00:00.000000001Z"} for row in expected
    )
    assert len({row["EngineReceivedMs"] for row in actual}) == 1


@pytest.mark.asyncio
async def test_shared_resource_still_hits_insert_limit_before_transport(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = 0
    body: Final = _resource_export(64 * 1024, 1024)
    receiver: Final = TraceReceiver(ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test")))
    with pytest.raises(TracingPayloadTooLargeError, match="encoded size limit"):
        await receiver.ingest(body, "application/json", None, Tenant("team-a", "key-a"))
    assert recording_server.requests == []


@pytest.mark.asyncio
async def test_insert_validates_values_without_pydantic_copy(recording_server: RecordingServer) -> None:
    storage: Final = ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test"))
    invalid: Final = object()
    with pytest.raises(ValueError, match=type(invalid).__name__):
        await storage.insert_rows("otel_traces", [{"ResourceAttributes": invalid}])
    attributes: Final = MappingProxyType({"service.name": "trace-test"})
    await storage.insert_rows(
        "otel_traces",
        (MappingProxyType({"Timestamp": 1, "ResourceAttributes": attributes, "SpanAttributes": attributes}),),
    )
    stored: Final = json.loads(gzip.decompress(recording_server.requests[0].raw_body))
    assert stored["Timestamp"] == "1970-01-01T00:00:00.000000001Z"
    assert stored["ResourceAttributes"] == attributes
    assert stored["SpanAttributes"] == attributes


@pytest.mark.asyncio
async def test_trace_receiver_reads_with_only_one_clickhouse_url(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    span_row: dict[str, JsonValue],
    span_params: dict[str, str | int | list[str]],
) -> None:
    monkeypatch.setenv("CLICKHOUSE_URL", recording_server.base_url)
    monkeypatch.setenv("CLICKHOUSE_DATABASE", "trace_test")
    monkeypatch.delenv("CLICKHOUSE_READER_URL", raising=False)
    recording_server.enqueue(ResponseSpec(body={"data": [span_row]}))
    receiver: Final = TraceReceiver.from_env()
    trace: Final = await receiver.get_trace("trace-1", {"all_teams": 1, "user_id": "", "team_ids": ()}, "ref")
    assert trace is not None
    assert trace["spans"][0]["span_id"] == span_row["span_id"]
    assert trace["spans"][0]["duration_ms"] == int(str(span_row["duration_ns"])) / 1_000_000
    parameters: Final = parse_qs(urlsplit(recording_server.requests[0].path).query)
    assert parameters["database"] == ["trace_test"]
    assert parameters["readonly"] == ["1"]


@pytest.mark.asyncio
async def test_lens_read_uses_the_shared_native_query_and_returns_typed_rows(
    recording_server: RecordingServer,
) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [{"traces": 0, "requests": 1}]}))
    storage: Final = ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test"))
    rows: Final = await storage.lens_availability(LensAccessParams(all_teams=0, team="team-a", key_hash="key-a"))
    assert rows == (ActivityAvailability(traces=False, requests=True),)
    parameters: Final = parse_qs(urlsplit(recording_server.requests[0].path).query)
    assert parameters["param_all_teams"] == ["0"]
    assert parameters["param_team"] == ["team-a"]
    assert parameters["param_key_hash"] == ["key-a"]

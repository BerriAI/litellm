import base64
import gzip
import json
import time
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.rust_bridge._native import NativeTraceStorage, trace_decode_otlp
from litellm.rust_bridge.traces import ClickHouseStorage, NormalizedSpan, normalized_field_definitions
from litellm.tracing import Tenant, TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing.decode import decode_otlp
from litellm.tracing.store import TraceStore
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
async def test_trace_reader_projects_connection_and_parameters(recording_server: RecordingServer) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [{"trace_id": "trace-1"}]}))
    reader_url: Final = recording_server.base_url.replace("http://", "http://reader:p%40ss%2Fword%25@")
    storage: Final = NativeTraceStorage("trace_test", recording_server.base_url, reader_url + "?database=wrong")
    rows: Final = json.loads(await storage.query("trace_spans", {"trace_id": "trace-1"}))
    request: Final = recording_server.requests[0]
    parameters: Final = parse_qs(urlsplit(request.path).query)
    assert rows == {"data": [{"trace_id": "trace-1"}]}
    assert b"o.TraceId = {trace_id:String}" in request.raw_body
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
        await storage.query("trace_spans", {})


@pytest.mark.asyncio
async def test_reader_rejects_arbitrary_sql_before_sending(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = 0
    storage: Final = NativeTraceStorage("trace_test", recording_server.base_url, recording_server.base_url)
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


def test_decode_and_tenant_stamping_share_resources_without_crossing_groups() -> None:
    body: Final = _resource_export(128, 2, 2)
    native: Final = trace_decode_otlp(body, "application/json")
    assert native[0]["scope_name"] is native[1]["scope_name"]
    assert native[0]["scope_version"] is native[1]["scope_version"]
    assert native[0]["resource_attributes"] is native[1]["resource_attributes"]
    assert native[2]["resource_attributes"] is native[3]["resource_attributes"]
    assert native[0]["resource_attributes"] is not native[2]["resource_attributes"]
    rows: Final = decode_otlp(body, "application/json")
    first: Final = Tenant("team-a", "key-a", "org-a").stamp_rows(rows)
    second: Final = Tenant("team-b", "key-b", "org-b").stamp_rows(rows)
    assert first[0]["ResourceAttributes"] is first[1]["ResourceAttributes"]
    assert first[2]["ResourceAttributes"] is first[3]["ResourceAttributes"]
    assert first[0]["ResourceAttributes"] is not first[2]["ResourceAttributes"]
    assert first[0]["ResourceAttributes"] is not second[0]["ResourceAttributes"]
    assert first[0]["ResourceAttributes"] == {
        "shared": "x" * 128,
        "litellm.team_id": "team-a",
        "litellm.api_key_hash": "key-a",
        "litellm.org_id": "org-a",
    }
    assert second[0]["ResourceAttributes"]["litellm.team_id"] == "team-b"
    assert rows[0]["ResourceAttributes"] == {"shared": "x" * 128, "litellm.team_id": "spoofed"}


def test_normalized_field_contract_matches_decoded_rust_span() -> None:
    body: Final = _resource_export(8, 1)
    spans: Final = trace_decode_otlp(body, "application/json")
    fields: Final = normalized_field_definitions()
    assert len(spans) == 1
    assert {field.name for field in fields} == set(spans[0]["normalized"]) == set(NormalizedSpan.model_fields)
    assert len({field.clickhouse_column for field in fields}) == len(fields)


@pytest.mark.asyncio
async def test_resource_fanout_reaches_insert_with_identical_values(recording_server: RecordingServer) -> None:
    body: Final = _resource_export(16 * 1024, 1024)
    receiver: Final = TraceReceiver(TraceStore(ClickHouseStorage("trace_test", recording_server.base_url)))
    tenant: Final = Tenant("team-a", "key-a", "org-a")
    assert await receiver.ingest(body, "application/json", None, tenant) == 1024
    encoded: Final = gzip.decompress(recording_server.requests[0].raw_body)
    actual: Final = tuple(json.loads(line) for line in encoded.splitlines())
    expected: Final = tenant.stamp_rows(decode_otlp(body, "application/json"))
    assert len(encoded) < 64 * 1024 * 1024
    assert tuple({key: value for key, value in row.items() if key != "EngineReceivedMs"} for row in actual) == tuple(
        {**row, "Timestamp": "1970-01-01T00:00:00.000000001Z"} for row in expected
    )
    assert len({row["EngineReceivedMs"] for row in actual}) == 1


@pytest.mark.asyncio
async def test_shared_resource_still_hits_insert_limit_before_transport(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = 0
    body: Final = _resource_export(64 * 1024, 1024)
    receiver: Final = TraceReceiver(TraceStore(ClickHouseStorage("trace_test", recording_server.base_url)))
    with pytest.raises(TracingPayloadTooLargeError, match="encoded size limit"):
        await receiver.ingest(body, "application/json", None, Tenant("team-a", "key-a"))
    assert recording_server.requests == []


@pytest.mark.asyncio
async def test_insert_validates_values_without_pydantic_copy(recording_server: RecordingServer) -> None:
    storage: Final = ClickHouseStorage("trace_test", recording_server.base_url)
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

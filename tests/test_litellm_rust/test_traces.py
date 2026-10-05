import base64
import gzip
import json
import math
import re
import time
from collections.abc import Generator, Iterator
from contextlib import closing
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE, OTLP_MAX_ATTRIBUTE_VALUE_BYTES
from litellm.rust_bridge._native import NativeTraceConfig, NativeTraceStorage
from litellm.rust_bridge.trace.generated.models import TraceQueryHelp
from litellm.rust_bridge.trace.generated.requests import Span as SpanModel
from litellm.rust_bridge.trace.generated.requests import TraceMetadata, TraceSpansPage
from litellm.rust_bridge.trace.generated.types import AllQueryScope, Trace, TracePage
from litellm.rust_bridge.trace.storage import ClickHouseStorage, TraceStorageConfig, span_rows
from litellm.tracing import Tenant, TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing.types import SpendLogRecord
from scripts.seed_tracing_fixtures import (
    TRACE,
    TRACE_FIXTURES,
    Copies,
    FixtureReplay,
    bulk_span_rows,
    copied_trace_id,
    copy_clickhouse,
    fixture_capture,
    fixture_replays,
    long_sessions,
    rebase_spend,
    response_pattern,
    spend_fixtures,
)
from tests.test_litellm_rust.support.clickhouse import clickhouse_service
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension
QUERY_ROWS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_TRACE_PAGE: Final = TypeAdapter(TracePage)


class CapturedSpendRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    spend: float
    prompt_tokens: int
    completion_tokens: int


class CapturedSpendQuery(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: tuple[CapturedSpendRow, ...]


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


@pytest.mark.asyncio
async def test_trace_reader_projects_connection_and_parameters(
    recording_server: RecordingServer,
    span_row: dict[str, JsonValue],
) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [span_row]}))
    url: Final = recording_server.base_url.replace("http://", "http://reader:p%40ss%2Fword%25@")
    storage: Final = _native_storage("trace_test", url + "?database=wrong")
    trace: Final = TypeAdapter(Trace).validate_python(
        await storage.get_trace("trace-1", AllQueryScope(kind="all"), "ref")
    )
    request: Final = recording_server.requests[0]
    parameters: Final = parse_qs(urlsplit(request.path).query)
    assert trace["spans"][0]["span_id"] == span_row["span_id"]
    assert parameters["database"] == ["trace_test"]
    assert parameters["param_trace_id"] == ["trace-1"]
    assert parameters["readonly"] == ["1"]
    assert "user" not in parameters
    assert "password" not in parameters
    assert request.headers["authorization"] == "Basic " + base64.b64encode(b"reader:p@ss/word%").decode()


@pytest.mark.asyncio
async def test_trace_reader_rejects_success_status_with_embedded_error(recording_server: RecordingServer) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [], "exception": "query failed"}))
    storage: Final = _native_storage("trace_test", recording_server.base_url)
    with pytest.raises(RuntimeError, match="query failed"):
        await storage.get_trace("trace-1", AllQueryScope(kind="all"), "ref")


@pytest.mark.asyncio
async def test_schema_binding_rejects_invalid_database() -> None:
    with pytest.raises(ValueError, match=r"database.*retention"):
        NativeTraceConfig("db; DROP DATABASE default", "http://localhost:8123", 14, OTLP_MAX_ATTRIBUTE_VALUE_BYTES)


@pytest.mark.asyncio
async def test_schema_binding_rejects_non_positive_retention() -> None:
    with pytest.raises(ValueError, match=r"database.*retention"):
        NativeTraceConfig("traces", "http://localhost:8123", 0, OTLP_MAX_ATTRIBUTE_VALUE_BYTES)


def test_invalid_url_error_does_not_expose_credentials() -> None:
    with pytest.raises(RuntimeError, match="invalid ClickHouse HTTP URL") as error:
        NativeTraceConfig("traces", "secret://writer:password@example.com", 7, OTLP_MAX_ATTRIBUTE_VALUE_BYTES)
    assert "password" not in str(error.value)


@pytest.mark.asyncio
async def test_from_env_reads_with_clickhouse_url(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": []}))
    monkeypatch.setenv("CLICKHOUSE_URL", recording_server.base_url)
    monkeypatch.delenv("CLICKHOUSE_READER_URL", raising=False)
    scope: Final = AllQueryScope(kind="all")
    page: Final = await TraceReceiver.from_env().list_traces(scope, 0, 1)
    assert page["data"] == ()
    assert page["next_cursor"] is None
    assert (page["window"]["start_ms"], page["window"]["end_ms"]) == (0, 1)
    assert len(recording_server.requests) == 1


@pytest.fixture
def run_row() -> dict[str, JsonValue]:
    return {
        "trace_id": "trace",
        "trace_ref": "ref",
        "team_id": "",
        "api_key_hash": "",
        "user_id": "",
        "name": "run",
        "service": "test",
        "input_preview": "",
        "status": "STATUS_CODE_OK",
        "start_ms": 1000,
        "duration_ns": 1_000_000,
        "span_count": 0,
        "agent_count": 0,
        "agent_invocations": 0,
        "agent_names": [],
        "frameworks": [],
        "llm_calls": 0,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "models": [],
        "error_count": 0,
    }


@pytest.mark.parametrize("params", ({}, {"start_ms": 1}))
def test_trace_pages_keep_the_effective_window_through_the_http_native_boundary(
    recording_server: RecordingServer,
    run_row: dict[str, JsonValue],
    params: dict[str, int],
) -> None:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.tracing_endpoints import provide_receiver, router

    recording_server.expected_requests = None
    recording_server.default_response = ResponseSpec(body={"data": []})
    rows: Final = tuple(
        {**run_row, "trace_id": f"trace-{index:03d}", "trace_ref": f"ref-{index:03d}"}
        for index in range(AGENT_TRACING_LIST_PAGE_SIZE + 1)
    )
    recording_server.enqueue(ResponseSpec(body={"data": rows}))
    storage: Final = ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, token="test"
    )
    app.dependency_overrides[provide_receiver] = lambda: TraceReceiver(storage)
    with TestClient(app) as client:
        first: Final = client.get("/v1/traces", params=params)
        assert first.status_code == 200, first.text
        first_body: Final = _TRACE_PAGE.validate_python(first.json())
        first_parameters: Final = parse_qs(urlsplit(recording_server.requests[0].path).query)
        cursor: Final = first_body["next_cursor"]
        assert cursor
        assert len(first_body["data"]) == AGENT_TRACING_LIST_PAGE_SIZE
        recording_server.enqueue(ResponseSpec(body={"data": [rows[0]]}))
        second: Final = client.get("/v1/traces", params={**params, "cursor": cursor})
        assert second.status_code == 200, second.text
        second_body: Final = _TRACE_PAGE.validate_python(second.json())
        refs: Final = tuple(row["id"] for row in (*first_body["data"], *second_body["data"]))
        assert refs == tuple(row["trace_ref"] for row in reversed(rows))
        assert second_body["next_cursor"] is None
        assert second_body["window"] == first_body["window"]
        assert second_body["data"][0]["root_status"] == "ok"
        assert not second_body["data"][0]["has_error"]
        run_parameters: Final = tuple(
            parse_qs(urlsplit(request.path).query)
            for request in recording_server.requests
            if "param_sort_key" in parse_qs(urlsplit(request.path).query)
        )
        assert len(run_parameters) == 2
        assert (run_parameters[1]["param_start_ms"], run_parameters[1]["param_end_ms"]) == (
            first_parameters["param_start_ms"],
            first_parameters["param_end_ms"],
        )
        assert run_parameters[1]["param_as_of_ms"] == first_parameters["param_as_of_ms"]
        sent: Final = len(recording_server.requests)
        changed: Final = client.get(
            "/v1/traces", params={"cursor": cursor, "end_ms": int(first_parameters["param_end_ms"][0]) + 1}
        )
        assert changed.status_code == 400, changed.text
        assert len(recording_server.requests) == sent
        changed_cutoff: Final = client.get(
            "/v1/traces", params={"cursor": cursor, "as_of_ms": first_body["window"]["as_of_ms"] - 1}
        )
        assert changed_cutoff.status_code == 400, changed_cutoff.text
        assert len(recording_server.requests) == sent


@pytest.mark.asyncio
async def test_schema_setup_uses_configured_retention(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = None
    storage: Final = _native_storage("trace_test", recording_server.base_url, 7)
    await storage.ensure_schema()
    ttl_statements: Final = tuple(
        request.raw_body for request in recording_server.requests if b"MODIFY TTL" in request.raw_body
    )
    assert len(ttl_statements) == 3
    assert all(b"INTERVAL 7 DAY" in statement for statement in ttl_statements)


@pytest.mark.asyncio
async def test_schema_setup_uses_writer_credentials_and_rejects_failed_statement(
    recording_server: RecordingServer,
) -> None:
    recording_server.expected_requests = 2
    recording_server.enqueue(ResponseSpec(body=""))
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


@pytest.mark.parametrize(
    ("role", "user_id", "expected_status"),
    (
        ("proxy_admin", None, 200),
        ("proxy_admin_viewer", None, 200),
        ("internal_user", "user", 200),
        ("internal_user", None, 403),
    ),
)
def test_trace_sql_endpoint_enforces_ownership_and_preserves_clickhouse_envelope(
    recording_server: RecordingServer, role: str, user_id: str | None, expected_status: int
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.authorization_dependencies import get_log_team_lookup
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.tracing_endpoints import provide_receiver, provide_trace_query_secret, router

    envelope: Final = {
        "meta": [{"name": "answer", "type": "UInt8"}],
        "data": [{"answer": 42}],
        "rows": 1,
        "statistics": {"elapsed": 0.01, "rows_read": 1, "bytes_read": 1},
    }
    recording_server.expected_requests = 15 if expected_status == 200 else 0
    if expected_status == 200:
        for _ in range(14):
            recording_server.enqueue(ResponseSpec(body=""))
        recording_server.enqueue(ResponseSpec(body=envelope))
    storage: Final = ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[provide_trace_query_secret] = lambda: "test-master-secret"
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role, user_id=user_id, token="test")
    app.dependency_overrides[provide_receiver] = lambda: TraceReceiver(storage)

    async def permitted_teams(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        return ()

    app.dependency_overrides[get_log_team_lookup] = lambda: permitted_teams
    with TestClient(app) as client:
        result: Final = client.post("/v1/traces/query", json={"sql": "SELECT 42 AS answer"})
        assert result.status_code == expected_status, result.text
        if expected_status == 403:
            assert result.json()["detail"] == "Not allowed to view logs"
            assert result.json()["code"] == "forbidden"
            return
        assert result.json() == envelope
        assert recording_server.requests[-1].raw_body == b"SELECT 42 AS answer"
        assert client.post("/v1/traces/query", json={"sql": "  "}).status_code == 400
        assert client.post("/v1/traces/query", json={}).status_code == 422


@pytest.mark.parametrize("discovery_fails", (False, True))
def test_trace_help_endpoint_runs_native_schema_and_metadata_discovery(
    recording_server: RecordingServer, discovery_fails: bool
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.tracing_endpoints import provide_receiver, provide_trace_query_secret, router

    recording_server.expected_requests = 23
    for _ in range(14):
        recording_server.enqueue(ResponseSpec(body=""))
    for response in (
        {"data": [{"name": "Model", "type": "String"}]},
        {"data": []},
        {"data": []},
        {"data": []},
        {"data": []},
        {"data": []},
    ):
        recording_server.enqueue(ResponseSpec(body=response))
    metadata: Final = (
        ResponseSpec(status=503, body="discovery failed")
        if discovery_fails
        else ResponseSpec(body={"data": [{"metadata": '{"custom": {"label": "hello"}}'}]})
    )
    recording_server.enqueue(metadata)
    recording_server.enqueue(ResponseSpec(body={"data": [{"key": "custom.span"}]}))
    recording_server.enqueue(ResponseSpec(body={"data": [{"key": "custom.resource"}]}))
    storage: Final = ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[provide_trace_query_secret] = lambda: "test-master-secret"
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role="proxy_admin", token="test")
    app.dependency_overrides[provide_receiver] = lambda: TraceReceiver(storage)
    with TestClient(app) as client:
        result: Final = client.get("/v1/traces/query/help")
    assert result.status_code == 200, result.text
    body: Final = result.json()
    assert body["guide"].startswith("Trace SQL query guide")
    assert body["tables"][0]["columns"] == [{"name": "Model", "type": "String"}]
    if discovery_fails:
        assert body["metadata"]["fields"] == []
        assert "503" in body["metadata"]["error"]
    else:
        assert "JSONExtractRaw(metadata, 'custom', 'label')" in body["guide"]
        assert body["metadata"]["fields"][1] == {
            "path": ["custom", "label"],
            "types": ["string"],
            "expression": "JSONExtractRaw(metadata, 'custom', 'label')",
        }
    assert body["attributes"][0]["fields"][0]["expression"] == "span_attributes['custom.span']"
    assert body["attributes"][1]["fields"][0]["expression"] == "resource_attributes['custom.resource']"


@pytest.mark.parametrize(
    ("clickhouse_status", "body", "expected_status", "database_code"),
    (
        (400, b"ClickHouse rejected the query", 400, None),
        (404, b"ClickHouse rejected the query", 400, None),
        (500, b"Code: 62. Invalid syntax", 400, 62),
        (403, b"Code: 497. Access denied", 400, 497),
        (500, b"Code: 241. Memory limit exceeded", 422, 241),
        (500, b"ClickHouse rejected the query", 503, None),
        (503, b"ClickHouse rejected the query", 503, None),
        (200, b'{"data":[]}', 503, None),
    ),
)
def test_trace_sql_endpoint_distinguishes_query_errors_from_reader_failures(
    recording_server: RecordingServer,
    clickhouse_status: int,
    body: bytes,
    expected_status: int,
    database_code: int | None,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.tracing_endpoints import provide_receiver, provide_trace_query_secret, router

    recording_server.expected_requests = 16
    for _ in range(14):
        recording_server.enqueue(ResponseSpec(body=""))
    recording_server.enqueue(ResponseSpec(status=clickhouse_status, body=body))
    envelope: Final = {
        "meta": [{"name": "answer", "type": "UInt8"}],
        "data": [{"answer": 42}],
        "rows": 1,
        "statistics": {"elapsed": 0.01, "rows_read": 1, "bytes_read": 1},
    }
    recording_server.enqueue(ResponseSpec(body=envelope))
    storage: Final = ClickHouseStorage(TraceStorageConfig(recording_server.base_url, "trace_test"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[provide_trace_query_secret] = lambda: "test-master-secret"
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role="proxy_admin", token="test")
    app.dependency_overrides[provide_receiver] = lambda: TraceReceiver(storage)
    with TestClient(app) as client:
        failed: Final = client.post("/v1/traces/query", json={"sql": "SELEC 42"})
        assert failed.status_code == expected_status, failed.text
        assert failed.json().get("database_code") == database_code
        assert (
            failed.json()["code"]
            == {400: "query_rejected", 422: "query_limit_exceeded", 503: "query_unavailable"}[expected_status]
        )
        if database_code is not None:
            assert failed.json()["detail"] == body.decode()
        recovered: Final = client.post("/v1/traces/query", json={"sql": "SELECT 42 AS answer"})
        assert recovered.status_code == 200, recovered.text
        assert recovered.json() == envelope
    assert recording_server.requests[-2].raw_body == b"SELEC 42"


@pytest.mark.asyncio
async def test_trace_receiver_reads_with_only_one_clickhouse_url(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    span_row: dict[str, JsonValue],
) -> None:
    monkeypatch.setenv("CLICKHOUSE_URL", recording_server.base_url)
    monkeypatch.setenv("CLICKHOUSE_DATABASE", "trace_test")
    monkeypatch.delenv("CLICKHOUSE_READER_URL", raising=False)
    recording_server.enqueue(ResponseSpec(body={"data": [span_row]}))
    receiver: Final = TraceReceiver.from_env()
    trace: Final = await receiver.get_trace("trace-1", AllQueryScope(kind="all"), "ref")
    assert trace is not None
    assert trace["spans"][0]["span_id"] == span_row["span_id"]
    assert trace["spans"][0]["duration_ms"] == int(str(span_row["duration_ns"])) / 1_000_000
    parameters: Final = parse_qs(urlsplit(recording_server.requests[0].path).query)
    assert parameters["database"] == ["trace_test"]
    assert parameters["readonly"] == ["1"]


@dataclass(frozen=True, slots=True)
class SeededTraceAPI:
    client: TestClient
    storage: ClickHouseStorage
    spends: tuple[SpendLogRecord, ...]
    help: TraceQueryHelp

    def query_example(self, name: str) -> tuple[dict[str, JsonValue], ...]:
        example: Final = next(example for example in self.help.examples if example.name == name)
        response: Final = self.client.post("/v1/traces/query", json={"sql": example.sql})
        assert response.status_code == 200, response.text
        return QUERY_ROWS.validate_python(response.json()["data"])


@pytest.fixture
def seeded_trace_api(clickhouse_url: str) -> Iterator[SeededTraceAPI]:
    from scripts.seed_tracing_fixtures import (
        TRACE_FIXTURES,
        fixture_replays,
        rebase_spend,
    )

    spends: Final = dict(spend_fixtures())["openai_agents_swarm"]
    pattern: Final = re.compile("|".join(re.escape(row["response_id"]) for row in spends))
    replays: Final = fixture_replays(TRACE_FIXTURES, time.time_ns() // 1_000_000, "query-api", pattern)
    swarm: Final = next(replay for replay in replays if replay.name == "openai_agents_swarm")
    rebased: Final = rebase_spend(spends, swarm.offset_ms, swarm.namespace, pattern)
    stamped: Final[tuple[SpendLogRecord, ...]] = tuple(
        {**row, "team_id": "team-a", "api_key": "fixture-key", "user": "fixture-user"} for row in rebased
    )
    yield from _fixture_trace_api(clickhouse_url, replays, stamped)


def _fixture_trace_api(
    clickhouse_url: str, replays: tuple[FixtureReplay, ...], stamped: tuple[SpendLogRecord, ...]
) -> Generator[SeededTraceAPI]:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.tracing_endpoints import provide_receiver, provide_trace_query_secret, router

    storage: Final = ClickHouseStorage(TraceStorageConfig(clickhouse_url, "trace_test"))
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[provide_trace_query_secret] = lambda: "fixture-secret"
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, team_id="team-a", token="fixture-key", user_id="fixture-user"
    )
    app.dependency_overrides[provide_receiver] = lambda: TraceReceiver(storage)
    with TestClient(app) as client:
        assert client.portal is not None
        client.portal.call(storage.ensure_schema)
        ingested: Final = tuple(client.post("/v1/traces", json=replay.export) for replay in replays)
        for result in ingested:
            assert result.status_code == 200, result.text
        client.portal.call(storage.insert_rows, "spend_logs", stamped)
        response: Final = client.get("/v1/traces/query/help")
        assert response.status_code == 200, response.text
        yield SeededTraceAPI(client, storage, stamped, TraceQueryHelp.model_validate(response.json()))


def test_fixture_backed_help_examples_execute_through_query_api(seeded_trace_api: SeededTraceAPI) -> None:
    api: Final = seeded_trace_api
    assert {"traces", "spans", "calls"} <= {table.name for table in api.help.tables}
    assert api.help.metadata.error is None
    assert api.help.metadata.sampled_rows == len(api.spends)
    assert any(field.path == ("fixture_capture", "name") for field in api.help.metadata.fields)
    for example in api.help.examples:
        api.query_example(example.name)
    records: Final = api.query_example("Recent spend records")
    assert {str(row["request_id"]) for row in records} == {row["request_id"] for row in api.spends}
    total: Final = sum(row["spend"] or 0 for row in api.spends)
    recorded: Final = api.query_example("Recorded spend by trace")
    assert len(recorded) == 1
    assert recorded[0]["trace_id"] == api.spends[0]["trace_id"]
    assert int(str(recorded[0]["requests"])) == len(api.spends)
    assert math.isclose(float(str(recorded[0]["recorded_spend"])), total)
    detail: Final = _trace(api, api.spends[0]["trace_id"])
    assert math.isclose(detail["summary"]["spend"] or 0, total)
    unmatched: Final = api.query_example("LLM spans without a direct spend match")
    assert unmatched
    assert all(row["trace_id"] != api.spends[0]["trace_id"] for row in unmatched)
    unpriced: Final = _trace(api, str(unmatched[0]["trace_id"]))
    assert unpriced["summary"]["spend"] is None


@pytest.mark.parametrize("spend", (None, 0.0, 0.125), ids=("unknown", "free", "paid"))
def test_query_model_totals_deduplicate_and_preserve_unknown_cost(
    seeded_trace_api: SeededTraceAPI, spend: float | None
) -> None:
    api: Final = seeded_trace_api
    original: Final = api.spends[0]
    replacement: Final[SpendLogRecord] = {**original, "end_time": original["end_time"] + 1, "spend": spend}
    assert api.client.portal is not None
    api.client.portal.call(api.storage.insert_rows, "spend_logs", (replacement,))
    totals: Final = api.query_example("Spend and tokens by model")
    row: Final = next(row for row in totals if row["model"] == original["model"])
    model_spends: Final = tuple(row for row in api.spends if row["model"] == original["model"])
    assert int(str(row["requests"])) == len(model_spends)
    assert int(str(row["input_tokens"])) == sum(row["prompt_tokens"] for row in model_spends)
    assert int(str(row["output_tokens"])) == sum(row["completion_tokens"] for row in model_spends)
    assert int(str(row["unknown_cost_requests"])) == int(spend is None)
    if spend is None:
        assert row["spend"] is None
    else:
        assert math.isclose(
            float(str(row["spend"])), sum(row["spend"] or 0 for row in model_spends) - (original["spend"] or 0) + spend
        )


def test_query_correlation_requires_key_or_user_ownership_within_a_team(seeded_trace_api: SeededTraceAPI) -> None:
    api: Final = seeded_trace_api
    original: Final = api.spends[0]
    unrelated: Final[SpendLogRecord] = {
        **original,
        "request_id": "unrelated-request",
        "api_key": "other-key",
        "user": "other-user",
    }
    assert api.client.portal is not None
    api.client.portal.call(api.storage.insert_rows, "spend_logs", (unrelated,))
    matches: Final = api.query_example("Traces correlated with LLM call metadata")
    assert {str(row["request_id"]) for row in matches} == {row["request_id"] for row in api.spends}
    assert all(row["request_id"] != unrelated["request_id"] for row in matches)


def _captured_replays(
    namespace: str,
) -> tuple[tuple[FixtureReplay, ...], tuple[tuple[str, tuple[SpendLogRecord, ...]], ...]]:
    captures: Final = spend_fixtures()
    pattern: Final = response_pattern(tuple(chain.from_iterable(rows for _, rows in captures)))
    replays: Final = fixture_replays(TRACE_FIXTURES, time.time_ns() // 1_000_000, namespace, pattern)
    by_name: Final = MappingProxyType(dict(captures))
    return replays, tuple(
        (
            replay.name,
            tuple(
                _stamp(row) for row in rebase_spend(by_name[replay.name], replay.offset_ms, replay.namespace, pattern)
            ),
        )
        for replay in replays
        if replay.name in by_name
    )


def _stamp(row: SpendLogRecord) -> SpendLogRecord:
    return {**row, "team_id": "team-a", "api_key": "fixture-key", "user": "fixture-user"}


@pytest.fixture(scope="module")
def captured_trace_api() -> Iterator[SeededTraceAPI]:
    replays, paired = _captured_replays("captured-api")
    with clickhouse_service() as url:
        yield from _fixture_trace_api(url, replays, tuple(chain.from_iterable(rows for _, rows in paired)))


@pytest.mark.parametrize("name", tuple(name for name, _ in spend_fixtures()))
def test_captured_sdk_cost_survives_seeding_and_is_queryable(name: str, captured_trace_api: SeededTraceAPI) -> None:
    api: Final = captured_trace_api
    rows: Final = tuple(row for row in api.spends if fixture_capture("", row).name == name)
    assert rows
    capture: Final = fixture_capture(name, rows[0])
    detail: Final = _trace(api, capture.trace_id)
    original: Final = span_rows((TRACE_FIXTURES / f"{name}.json").read_bytes(), "application/json")
    assert detail["summary"]["span_count"] == len(original)
    if capture.spend_linked and capture.spend_complete:
        assert detail["summary"]["spend"] is not None
        assert math.isclose(detail["summary"]["spend"], sum(row["spend"] or 0 for row in rows))
    else:
        assert detail["summary"]["spend"] is None
    query: Final = api.client.post(
        "/v1/traces/query",
        json={
            "sql": "SELECT request_id, spend, prompt_tokens, completion_tokens FROM spend_logs FINAL "
            f"WHERE JSONExtractString(metadata, 'fixture_capture', 'name') = '{name}' LIMIT 100"
        },
    )
    assert query.status_code == 200, query.text
    records: Final = CapturedSpendQuery.model_validate_json(query.content).data
    assert {row.request_id for row in records} == {row["request_id"] for row in rows}
    assert math.isclose(sum(row.spend for row in records), sum(row["spend"] or 0 for row in rows))
    assert sum(row.prompt_tokens for row in records) == sum(row["prompt_tokens"] for row in rows)
    assert sum(row.completion_tokens for row in records) == sum(row["completion_tokens"] for row in rows)


def test_server_side_copies_keep_every_capture_linked_to_its_spend() -> None:
    replays, paired = _captured_replays("copied-api")
    copies: Final = Copies(
        trace_ids=tuple(sorted(frozenset(str(span["TraceId"]) for span in bulk_span_rows(replays, Tenant("", ""))))),
        request_ids=tuple(row["request_id"] for _, rows in paired for row in rows),
        numbers=range(1, 3),
        step_ms=60_000,
        source="seed-copied-api-",
        target="seed-copied-api-c",
    )
    (session,) = long_sessions(replays, paired, "seed-copied-api-", "seed-copied-api-c", (3,))
    session_spend: Final = sum(row["spend"] or 0 for row in dict(paired)["openai_agents_swarm"])
    session_spans: Final = len(
        span_rows((TRACE_FIXTURES / "openai_agents_swarm.json").read_bytes(), "application/json")
    )
    with (
        clickhouse_service() as url,
        closing(_fixture_trace_api(url, replays, tuple(chain.from_iterable(rows for _, rows in paired)))) as seeded,
    ):
        api: Final = next(seeded)
        assert api.client.portal is not None
        for plan in (copies, session):
            api.client.portal.call(_copy_clickhouse, url, plan)
        for name, rows in paired:
            _assert_capture(api, name, rows, fixture_capture(name, rows[0]).trace_id)
            _assert_capture(api, name, rows, copied_trace_id(fixture_capture(name, rows[0]).trace_id, "2"))
        trace: Final = _trace(api, copied_trace_id(session.trace_ids[0], session.session))
        assert trace["summary"]["span_count"] == 1 + 3 * (session_spans - 1)
        (root,) = (span for span in trace["spans"] if not span["parent_span_id"])
        assert {span["parent_span_id"] for span in trace["spans"] if span["parent_span_id"]} <= {
            span["span_id"] for span in trace["spans"]
        }
        assert root["start_offset_ms"] == min(span["start_offset_ms"] for span in trace["spans"])
        assert root["start_offset_ms"] + root["duration_ms"] >= max(
            span["start_offset_ms"] + span["duration_ms"] for span in trace["spans"]
        )
        assert trace["summary"]["spend"] == pytest.approx(3 * session_spend)


def _trace(api: SeededTraceAPI, trace_id: str) -> Trace:
    listed: Final = api.client.get(
        "/v1/traces",
        params={
            "q": f"trace_id:{trace_id}",
            "start_ms": 0,
            "end_ms": time.time_ns() // 1_000_000 + 86_400_000,
            "page_size": 2,
        },
    )
    assert listed.status_code == 200, listed.text
    page: Final = _TRACE_PAGE.validate_json(listed.content)
    (summary,) = page["data"]
    assert summary["trace_id"] == trace_id
    response: Final = api.client.get(f"/v1/traces/{summary['id']}")
    assert response.status_code == 200, response.text
    metadata: Final = TraceMetadata.model_validate_json(response.content)
    first: Final = _span_page(api, summary["id"], None)
    spans: Final = tuple(_trace_spans(api, summary["id"], first))
    return TRACE.validate_python(
        {
            **metadata.model_dump(mode="json"),
            "spans": tuple(span.model_dump(mode="json") for span in spans),
            "next_cursor": None,
        }
    )


def _span_page(api: SeededTraceAPI, id: str, cursor: str | None) -> TraceSpansPage:
    params: Final = {"page_size": 200} if cursor is None else {"page_size": 200, "cursor": cursor}
    response: Final = api.client.get(f"/v1/traces/{id}/spans", params=params)
    assert response.status_code == 200, response.text
    return TraceSpansPage.model_validate_json(response.content)


def _trace_spans(api: SeededTraceAPI, id: str, page: TraceSpansPage) -> Iterator[SpanModel]:
    yield from page.data
    if page.next_cursor is not None:
        yield from _trace_spans(api, id, _span_page(api, id, page.next_cursor))


def _assert_capture(api: SeededTraceAPI, name: str, rows: tuple[SpendLogRecord, ...], trace_id: str) -> None:
    capture: Final = fixture_capture(name, rows[0])
    summary: Final = _trace(api, trace_id)["summary"]
    assert summary["span_count"] == len(span_rows((TRACE_FIXTURES / f"{name}.json").read_bytes(), "application/json"))
    assert summary["spend"] == (
        pytest.approx(sum(row["spend"] or 0 for row in rows))
        if capture.spend_linked and capture.spend_complete
        else None
    )


async def _copy_clickhouse(url: str, copies: Copies) -> None:
    async with httpx.AsyncClient(base_url=url, params={"database": "trace_test"}) as client:
        await copy_clickhouse(client, "trace_test", copies)

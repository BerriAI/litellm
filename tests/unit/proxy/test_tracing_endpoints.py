"""
Tests for the agent tracing endpoints (litellm/proxy/tracing_endpoints.py).
"""

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from types import ModuleType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy import tracing_endpoints
from litellm.proxy._types import LitellmUserRoles, ProxyLifespanState, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.tracing_runtime import manage_tracing, provide_storage
from litellm.rust_bridge import loader
from litellm.rust_bridge.trace_queries import SPAN_DETAIL, SpanDetailParams
from litellm.rust_bridge.trace_query_responses import TraceQueryHelp, TraceSQLResponse
from litellm.rust_bridge.traces import AdminQueryScope, ClickHouseStorage, TraceStorageConfig
from litellm.tracing import TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing.store import TraceStore
from litellm.tracing.types import TraceScope

SQL_ENVELOPE: Final = {
    "meta": [{"name": "value", "type": "UInt64"}],
    "data": [{"value": "9007199254740993"}],
    "rows": 1,
    "statistics": {"elapsed": 0.01, "rows_read": 1, "bytes_read": 8},
    "rows_before_limit_at_least": 1,
}
QUERY_HELP: Final[Mapping[str, object]] = {
    "dialect": "test SQL",
    "access": "authenticated scope",
    "response": "JSON envelope",
    "tables": [{"name": "otel_traces", "columns": [{"name": "value", "type": "String", "comment": "label"}]}],
    "normalized_fields": [],
    "metadata": {
        "table": "spend_logs",
        "column": "metadata",
        "fields": [],
        "sampled_rows": 0,
        "invalid_json_rows": 0,
        "truncated": True,
        "sample_sql": "SELECT metadata FROM traces",
        "scope": "bounded sample",
        "error": "discovery unavailable",
    },
    "attributes": [],
    "relationships": [],
    "examples": [{"name": "recent", "sql": "SELECT * FROM traces LIMIT 1"}],
    "gotchas": ["Keep queries bounded"],
    "guide": "scoped",
}


TEAM_KEY = UserAPIKeyAuth(
    token="hashed-key", team_id="team-research", org_id="org-1", user_role=LitellmUserRoles.INTERNAL_USER
)
TRACE_RESPONSE: Final = {
    "summary": {
        "trace_id": "t1",
        "name": "trace",
        "service": "test",
        "input_preview": "",
        "start_time": "2026-01-01T00:00:00Z",
        "duration_ms": 0,
        "status": "ok",
        "span_count": 0,
        "agent_count": 0,
        "agent_invocations": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "error_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "models": [],
        "spend": None,
    },
    "agents": [],
    "spans": [],
}
SPAN_DETAIL_RESPONSE: Final = {
    "span_id": "s1",
    "input": "",
    "output": "",
    "input_ui": {"kind": "text", "text": ""},
    "output_ui": {"kind": "text", "text": ""},
    "attributes": {},
}


@pytest.mark.parametrize(
    ("auth", "scope", "can_write"),
    (
        pytest.param(
            UserAPIKeyAuth(token="admin-key", team_id="team-a", user_role=LitellmUserRoles.PROXY_ADMIN),
            TraceScope(all_teams=1, user_id="", team_ids=(), api_key_hash=""),
            True,
            id="admin",
        ),
        pytest.param(
            UserAPIKeyAuth(token="view-key", team_id="team-a", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
            TraceScope(all_teams=1, user_id="", team_ids=(), api_key_hash=""),
            False,
            id="view-only-admin",
        ),
        pytest.param(
            TEAM_KEY,
            TraceScope(all_teams=0, user_id="", team_ids=(), api_key_hash="hashed-key"),
            True,
            id="team-key",
        ),
        pytest.param(
            UserAPIKeyAuth(token="hashed-key", user_role=LitellmUserRoles.INTERNAL_USER),
            TraceScope(all_teams=0, user_id="", team_ids=(), api_key_hash="hashed-key"),
            True,
            id="teamless-key",
        ),
    ),
)
def test_trace_read_and_write_permissions(
    client: TestClient, receiver: MagicMock, auth: UserAPIKeyAuth, scope: TraceScope, can_write: bool
) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth

    read: Final = client.get("/v1/traces?start_ms=1&end_ms=2")
    assert read.status_code == 200, read.text
    receiver.list_traces.assert_awaited_once_with(scope=scope, start_ms=1, end_ms=2, cursor=None)

    write: Final = client.post("/v1/traces", json={})
    assert write.status_code == (200 if can_write else 403), write.text
    if not can_write:
        receiver.ingest.assert_not_awaited()
        return
    receiver.ingest.assert_awaited_once()
    tenant: Final = receiver.ingest.await_args.kwargs["tenant"]
    assert (tenant.team_id, tenant.api_key_hash, tenant.org_id) == (
        auth.team_id or "",
        auth.token or "",
        auth.org_id or "",
    )


@pytest.fixture
def receiver(client) -> MagicMock:
    fake = MagicMock()
    fake.ingest = AsyncMock(return_value=1)
    fake.list_traces = AsyncMock(return_value={"data": [], "next_cursor": None})
    fake.get_trace = AsyncMock(return_value=None)
    fake.get_span = AsyncMock(return_value=None)
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: fake
    return fake


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(tracing_endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: TEAM_KEY
    return TestClient(app)


@pytest.mark.parametrize("native_available", [True, False])
def test_501_when_tracing_not_enabled(
    client: TestClient, native_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from google.rpc.status_pb2 import Status

    from litellm.rust_bridge import loader

    if not native_available:
        monkeypatch.setattr(loader, "_cached_bridge", None)
    response: Final = client.post("/v1/traces", content=b"")
    assert response.status_code == 501
    assert response.headers["content-type"] == "application/x-protobuf"
    assert Status.FromString(response.content).message == (
        "Agent tracing is not enabled. Set `tracing:` in general_settings and CLICKHOUSE_URL."
        if native_available
        else ""
    )
    assert client.get("/v1/traces").status_code == 501


def test_post_protobuf_returns_empty_protobuf(client, receiver):
    response = client.post(
        "/v1/traces",
        content=b"\x0a\x00",
        headers={"content-type": "application/x-protobuf", "content-encoding": "gzip"},
    )
    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["content-type"] == "application/x-protobuf"
    kwargs = receiver.ingest.call_args.kwargs
    assert kwargs["body"] is not None
    assert kwargs["content_type"] == "application/x-protobuf"
    assert kwargs["content_encoding"] == "gzip"
    assert kwargs["tenant"].team_id == "team-research"


def test_post_json_returns_empty_json(client, receiver):
    response = client.post("/v1/traces", content=b"{}", headers={"content-type": "application/json"})
    assert response.status_code == 200
    assert response.json() == {}


def test_post_clickhouse_failure_is_503_with_retry_after(client, receiver):
    receiver.ingest.side_effect = RuntimeError("ClickHouse unavailable")
    response = client.post("/v1/traces", content=b"", headers={"content-type": "application/x-protobuf"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == str(tracing_endpoints.OTLP_RETRY_AFTER_SECONDS)


def test_post_too_large_is_413(client, receiver):
    receiver.ingest.side_effect = TracingPayloadTooLargeError("OTLP body exceeds 10 bytes")
    response = client.post("/v1/traces", content=b"x" * 20)
    assert response.status_code == 413
    from google.rpc.status_pb2 import Status

    assert "exceeds" in Status.FromString(response.content).message


def test_list_traces_passes_scope_window_and_cursor(client, receiver):
    response = client.get("/v1/traces", params={"start_ms": 1, "end_ms": 2, "cursor": "abc"})
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}
    receiver.list_traces.assert_awaited_once_with(
        scope={"all_teams": 0, "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"},
        start_ms=1,
        end_ms=2,
        cursor="abc",
    )


def test_list_traces_defaults_to_last_24h(client, receiver):
    client.get("/v1/traces")
    kwargs = receiver.list_traces.call_args.kwargs
    assert kwargs["end_ms"] - kwargs["start_ms"] == tracing_endpoints.MS_PER_DAY
    assert kwargs["cursor"] is None


def test_get_trace_404_and_200(client, receiver):
    assert client.get("/v1/traces/missing").status_code == 404
    receiver.get_trace.return_value = TRACE_RESPONSE
    response = client.get("/v1/traces/t1")
    assert response.status_code == 200
    assert response.json() == TRACE_RESPONSE
    receiver.get_trace.assert_awaited_with(
        "t1", {"all_teams": 0, "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}, ""
    )


def test_get_span_404_and_200(client, receiver):
    assert client.get("/v1/traces/t1/spans/s1").status_code == 404
    receiver.get_span.return_value = SPAN_DETAIL_RESPONSE
    response = client.get("/v1/traces/t1/spans/s1")
    assert response.status_code == 200
    assert response.json()["span_id"] == "s1"
    receiver.get_span.assert_awaited_with(
        "t1", "s1", {"all_teams": 0, "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}, ""
    )


def test_get_span_serves_ui_content_from_stored_payloads(client):
    storage = MagicMock()
    stored_output = '{"role": "ai", "content": "", "tool_calls": [{"name": "lookup", "args": {"id": 7}}]}'
    storage.query = AsyncMock(
        return_value=[{"span_id": "s1", "input": '{"city": "Paris"}', "output": stored_output, "attributes": {}}]
    )
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: TraceReceiver(TraceStore(storage))
    body = client.get("/v1/traces/t1/spans/s1?trace_ref=run-one").json()
    assert body["output"] == stored_output
    assert body["input_ui"] == {"kind": "fields", "fields": [{"key": "city", "value": "Paris"}]}
    assert body["output_ui"] == {
        "kind": "messages",
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup", "arguments": '{"id": 7}'}]}
        ],
    }


def test_trace_detail_passes_scoped_reference(client, receiver):
    receiver.get_trace.return_value = TRACE_RESPONSE
    assert client.get("/v1/traces/t1?trace_ref=run-one").status_code == 200
    receiver.get_trace.assert_awaited_with(
        "t1", {"all_teams": 0, "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}, "run-one"
    )


def test_invalid_export_and_cursor_are_client_errors(client, receiver):
    from litellm.tracing.decode import InvalidOTLPPayloadError

    receiver.ingest.side_effect = InvalidOTLPPayloadError("invalid OTLP trace payload")
    assert client.post("/v1/traces", content=b"broken").status_code == 400
    receiver.list_traces.side_effect = ValueError("Invalid trace cursor")
    assert client.get("/v1/traces?cursor=broken").status_code == 400


def test_teamless_key_without_token_gets_403_on_reads(client, receiver):
    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER
    )
    assert client.get("/v1/traces").status_code == 403
    receiver.list_traces.assert_not_called()


def test_view_only_admin_cannot_ingest_traces(client, receiver):
    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        token="admin-key", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    )
    response = client.post("/v1/traces", content=b"{}")
    assert response.status_code == 403
    receiver.ingest.assert_not_called()


@pytest.mark.parametrize(
    "status_code, field, message",
    [(401, "detail", "Invalid API key"), (403, "message", "Not allowed to ingest agent traces")],
)
def test_auth_failure_precedes_disabled_receiver(
    client: TestClient, status_code: int, field: str, message: str
) -> None:
    def unavailable() -> None:
        return None

    def authenticate() -> UserAPIKeyAuth:
        if status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

    client.app.dependency_overrides[user_api_key_auth] = authenticate
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = unavailable
    response: Final = client.post("/v1/traces", content=b"{}", headers={"content-type": "application/json"})
    assert response.status_code == status_code
    assert response.json() == {field: message}


def test_disabled_receiver_precedes_read_scope_rejection(client: TestClient) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER
    )
    response: Final = client.get("/v1/traces")
    assert response.status_code == 501
    assert response.json() == {
        "detail": "Agent tracing is not enabled. Set `tracing:` in general_settings and CLICKHOUSE_URL."
    }


@pytest.mark.requires_rust_extension
def test_injected_receiver_persists_authenticated_tenant(client: TestClient) -> None:
    storage: Final = MagicMock(spec=ClickHouseStorage)
    storage.insert_rows = AsyncMock()
    tracing: Final = TraceReceiver(TraceStore(storage))
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: tracing
    response: Final = client.post(
        "/v1/traces",
        json={
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "litellm.team_id", "value": {"stringValue": "spoofed-team"}},
                            {"key": "litellm.api_key_hash", "value": {"stringValue": "spoofed-key"}},
                            {"key": "litellm.org_id", "value": {"stringValue": "spoofed-org"}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "01" * 16,
                                    "spanId": "02" * 8,
                                    "name": "dependency-injection",
                                    "startTimeUnixNano": "1000000000",
                                    "endTimeUnixNano": "1000000001",
                                }
                            ]
                        }
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json() == {}
    storage.insert_rows.assert_awaited_once()
    table, rows = storage.insert_rows.await_args.args
    assert table == "otel_traces"
    assert len(rows) == 1
    assert rows[0]["TeamId"] == TEAM_KEY.team_id
    assert rows[0]["ApiKeyHash"] == TEAM_KEY.token
    assert rows[0]["ResourceAttributes"] == {
        "litellm.team_id": TEAM_KEY.team_id,
        "litellm.api_key_hash": TEAM_KEY.token,
        "litellm.org_id": TEAM_KEY.org_id,
        "litellm.user_id": TEAM_KEY.user_id or "",
    }


def test_lifespan_receivers_are_app_local() -> None:
    first_storage: Final = MagicMock(spec=ClickHouseStorage)
    first_storage.query = AsyncMock(
        return_value=[
            {
                "span_id": "first-span",
                "input": "first-input",
                "output": "",
                "attributes": {},
            }
        ]
    )
    second_storage: Final = MagicMock(spec=ClickHouseStorage)
    second_storage.query = AsyncMock(
        return_value=[
            {
                "span_id": "second-span",
                "input": "second-input",
                "output": "",
                "attributes": {},
            }
        ]
    )
    first_receiver: Final = TraceReceiver(TraceStore(first_storage))
    second_receiver: Final = TraceReceiver(TraceStore(second_storage))
    first_storage.ensure_schema = AsyncMock()
    second_storage.ensure_schema = AsyncMock()

    @asynccontextmanager
    async def first_lifespan(app: FastAPI) -> AsyncGenerator[ProxyLifespanState, None]:
        async with manage_tracing(True, lambda: first_receiver) as receiver:
            state: Final[ProxyLifespanState] = {"tracing_receiver": receiver}
            yield state

    @asynccontextmanager
    async def second_lifespan(app: FastAPI) -> AsyncGenerator[ProxyLifespanState, None]:
        async with manage_tracing(True, lambda: second_receiver) as receiver:
            state: Final[ProxyLifespanState] = {"tracing_receiver": receiver}
            yield state

    first_app: Final = FastAPI(lifespan=first_lifespan)
    second_app: Final = FastAPI(lifespan=second_lifespan)
    first_app.include_router(tracing_endpoints.router)
    second_app.include_router(tracing_endpoints.router)
    first_app.dependency_overrides[user_api_key_auth] = lambda: TEAM_KEY
    second_app.dependency_overrides[user_api_key_auth] = lambda: TEAM_KEY

    with TestClient(first_app) as first_client:
        with TestClient(second_app) as second_client:
            second_response: Final = second_client.get("/v1/traces/t1/spans/second-span?trace_ref=second-run")
            simultaneous: Final = first_client.get("/v1/traces/t1/spans/first-span?trace_ref=first-run")
        first_response: Final = first_client.get("/v1/traces/t1/spans/first-span?trace_ref=first-run")
        assert simultaneous.json() == first_response.json()
    first_storage.ensure_schema.assert_awaited_once()
    second_storage.ensure_schema.assert_awaited_once()

    assert first_response.status_code == second_response.status_code == 200
    assert first_response.json() == {
        "span_id": "first-span",
        "input": "first-input",
        "output": "",
        "attributes": {},
        "input_ui": {"kind": "text", "text": "first-input"},
        "output_ui": {"kind": "text", "text": ""},
    }
    assert second_response.json() == {
        "span_id": "second-span",
        "input": "second-input",
        "output": "",
        "attributes": {},
        "input_ui": {"kind": "text", "text": "second-input"},
        "output_ui": {"kind": "text", "text": ""},
    }
    assert first_storage.query.await_count == 2
    first_storage.query.assert_awaited_with(
        SPAN_DETAIL,
        SpanDetailParams(
            all_teams=0,
            user_id="",
            team_ids=(),
            api_key_hash=TEAM_KEY.token,
            trace_id="t1",
            span_id="first-span",
            trace_ref="first-run",
        ),
    )
    second_storage.query.assert_awaited_once_with(
        SPAN_DETAIL,
        SpanDetailParams(
            all_teams=0,
            user_id="",
            team_ids=(),
            api_key_hash=TEAM_KEY.token,
            trace_id="t1",
            span_id="second-span",
            trace_ref="second-run",
        ),
    )


@pytest.mark.parametrize("auth", [TEAM_KEY, UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)])
def test_query_validation_precedes_trace_access_checks(client: TestClient, auth: UserAPIKeyAuth) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth
    response: Final = client.get("/v1/traces", params={"start_ms": "invalid"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "start_ms"]


@pytest.mark.parametrize("enabled", [True, False])
def test_unavailable_lifespan_receiver_returns_501(enabled: bool) -> None:
    storage: Final = MagicMock(spec=ClickHouseStorage)
    storage.ensure_schema = AsyncMock(side_effect=RuntimeError("storage unavailable"))
    tracing: Final = TraceReceiver(TraceStore(storage))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[ProxyLifespanState, None]:
        async with manage_tracing(enabled, lambda: tracing) as receiver:
            state: Final[ProxyLifespanState] = {"tracing_receiver": receiver}
            yield state

    app: Final = FastAPI(lifespan=lifespan)
    app.include_router(tracing_endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: TEAM_KEY
    with TestClient(app) as client:
        response: Final = client.get("/v1/traces")
    assert response.status_code == 501
    assert storage.ensure_schema.await_count == int(enabled)
    storage.query.assert_not_called()


def test_lens_reads_from_the_lifespan_storage() -> None:
    from litellm.proxy.lens.endpoints import router as lens_router

    storage: Final = MagicMock(spec=ClickHouseStorage)
    storage.ensure_schema = AsyncMock()
    storage.lens_sample = AsyncMock(return_value=[])
    tracing: Final = TraceReceiver(TraceStore(storage))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[ProxyLifespanState, None]:
        async with manage_tracing(True, lambda: tracing) as receiver:
            state: Final[ProxyLifespanState] = {"tracing_receiver": receiver}
            yield state

    app: Final = FastAPI(lifespan=lifespan)
    app.include_router(lens_router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    with TestClient(app) as client:
        response: Final = client.post(
            "/lens/preview/sample",
            json={"settings": {"name": "Review", "model": "analysis", "context": "Find failed executions"}},
        )
    assert response.status_code == 200, response.text
    assert response.json()["executions"] == []
    storage.lens_sample.assert_awaited_once()
    assert storage.lens_sample.await_args.args[0].all_teams == 1


def test_lens_reads_from_injected_storage_without_receiver() -> None:
    from litellm.proxy.lens.endpoints import router as lens_router
    from litellm.proxy.lens.sources import Storage

    storage: Final = MagicMock(spec=Storage)
    storage.lens_sample = AsyncMock(return_value=[])
    app: Final = FastAPI()
    app.include_router(lens_router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    app.dependency_overrides[provide_storage] = lambda: storage

    with TestClient(app) as client:
        response: Final = client.post(
            "/lens/preview/sample",
            json={"settings": {"name": "Review", "model": "analysis", "context": "Find failed executions"}},
        )

    assert response.status_code == 200, response.text
    assert response.json()["executions"] == []
    storage.lens_sample.assert_awaited_once()


@pytest.mark.parametrize(
    ("auth", "expected_scope"),
    (
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), {"kind": "admin"}),
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY), {"kind": "admin"}),
        (TEAM_KEY, {"kind": "logs", "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}),
        (
            UserAPIKeyAuth(token="project-key", team_id="team-a", project_id="project-a"),
            {"kind": "logs", "user_id": "", "team_ids": (), "api_key_hash": "project-key"},
        ),
        (UserAPIKeyAuth(token="solo-key"), {"kind": "logs", "user_id": "", "team_ids": (), "api_key_hash": "solo-key"}),
    ),
)
def test_sql_and_help_use_authenticated_scope(
    client: TestClient, receiver: MagicMock, auth: UserAPIKeyAuth, expected_scope: dict[str, str]
) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    receiver.store.storage.query_sql = AsyncMock(return_value=TraceSQLResponse.model_validate(SQL_ENVELOPE))
    receiver.store.storage.query_help = AsyncMock(return_value=TraceQueryHelp.model_validate(QUERY_HELP))
    result: Final = client.post("/v1/traces/query", json={"sql": "SELECT * FROM otel_traces"})
    assert result.status_code == 200, result.text
    assert result.json() == SQL_ENVELOPE
    receiver.store.storage.query_sql.assert_awaited_once_with(
        "SELECT * FROM otel_traces", expected_scope, "test-secret"
    )
    help_result: Final = client.get("/v1/traces/query/help")
    assert help_result.status_code == 200, help_result.text
    assert help_result.json() == QUERY_HELP
    receiver.store.storage.query_help.assert_awaited_once_with(expected_scope, "test-secret")
    forged: Final = client.post("/v1/traces/query", json={"sql": "SELECT 1", "scope": {"kind": "admin"}})
    assert forged.status_code == 422, forged.text
    assert receiver.store.storage.query_sql.await_count == 1


@pytest.mark.parametrize("auth", (UserAPIKeyAuth(), UserAPIKeyAuth(team_id="a", project_id="p")))
def test_sql_rejects_missing_identity_without_querying(
    client: TestClient, receiver: MagicMock, auth: UserAPIKeyAuth
) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    result: Final = client.post("/v1/traces/query", json={"sql": "SELECT * FROM otel_traces"})
    assert result.status_code == 403, result.text
    assert client.get("/v1/traces/query/help").status_code == 403
    receiver.store.storage.query_sql.assert_not_called()
    receiver.store.storage.query_help.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status"), ((ValueError("invalid SQL"), 400), (RuntimeError("reader unavailable"), 503))
)
def test_sql_reports_rejected_queries_and_unavailable_readers(
    client: TestClient, receiver: MagicMock, error: Exception, status: int
) -> None:
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    receiver.store.storage.query_sql = AsyncMock(side_effect=error)
    result: Final = client.post("/v1/traces/query", json={"sql": "SELECT 1"})
    assert result.status_code == status, result.text
    receiver.store.storage.query_sql.assert_awaited_once_with(
        "SELECT 1", {"kind": "logs", "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}, "test-secret"
    )


def test_query_help_does_not_fall_back_when_reader_provisioning_fails(client: TestClient, receiver: MagicMock) -> None:
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    receiver.store.storage.query_help = AsyncMock(side_effect=RuntimeError("reader provisioning failed"))
    result: Final = client.get("/v1/traces/query/help")
    assert result.status_code == 503, result.text
    receiver.store.storage.query_help.assert_awaited_once_with(
        {"kind": "logs", "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}, "test-secret"
    )


@pytest.mark.parametrize("secret", (None, "configured-master-key"))
def test_queries_require_a_proxy_secret(
    client: TestClient, receiver: MagicMock, monkeypatch: pytest.MonkeyPatch, secret: str | None
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", secret)
    receiver.store.storage.query_sql = AsyncMock(return_value=TraceSQLResponse.model_validate(SQL_ENVELOPE))
    result: Final = client.post("/v1/traces/query", json={"sql": "SELECT 1"})
    if secret is None:
        assert result.status_code == 503, result.text
        assert "master key" in result.json()["detail"]
        receiver.store.storage.query_sql.assert_not_awaited()
        return
    assert result.status_code == 200, result.text
    receiver.store.storage.query_sql.assert_awaited_once_with(
        "SELECT 1", {"kind": "logs", "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}, secret
    )


class _NativeConfig:
    def __init__(self, database: str, url: str, retention_days: int) -> None:
        pass


class _NativeReturningHelp(ModuleType):
    def __init__(self, help_payload: Mapping[str, object]) -> None:
        super().__init__("native_traces")

        class Storage:
            def __init__(self, config: _NativeConfig) -> None:
                pass

            async def query_help(self, scope: AdminQueryScope, secret: str) -> Mapping[str, object]:
                return help_payload

        self.NativeTraceConfig: Final = _NativeConfig
        self.NativeTraceStorage: Final = Storage
        self.trace_decode_otlp: Final = list
        self.trace_encode_error: Final = bytes
        self.trace_normalized_field_definitions: Final = list


async def test_storage_validates_the_native_query_help_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "_cached_bridge", _NativeReturningHelp(QUERY_HELP))
    storage: Final = ClickHouseStorage(TraceStorageConfig("http://clickhouse:8123"))
    assert await storage.query_help({"kind": "admin"}, "secret") == TraceQueryHelp.model_validate(QUERY_HELP)


@pytest.mark.parametrize(
    "drift",
    (
        {
            "metadata": {
                "table": "spend_logs",
                "column": "metadata",
                "fields": [{"path": ["a"], "types": ["boolen"], "expression": "a"}],
                "sampled_rows": 1,
                "invalid_json_rows": 0,
                "truncated": False,
                "sample_sql": "SELECT metadata FROM spend_logs",
                "scope": "bounded sample",
            }
        },
        {"tables": [{"name": "traces", "columns": [{"name": "value", "type": "String"}]}]},
        {"unexpected": True},
    ),
)
async def test_storage_rejects_native_query_help_that_drifts_from_the_contract(
    monkeypatch: pytest.MonkeyPatch, drift: Mapping[str, object]
) -> None:
    monkeypatch.setattr(loader, "_cached_bridge", _NativeReturningHelp({**QUERY_HELP, **drift}))
    storage: Final = ClickHouseStorage(TraceStorageConfig("http://clickhouse:8123"))
    with pytest.raises(RuntimeError, match="invalid response"):
        await storage.query_help({"kind": "admin"}, "secret")

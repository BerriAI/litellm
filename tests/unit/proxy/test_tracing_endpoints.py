"""
Tests for the agent tracing endpoints (litellm/proxy/tracing_endpoints.py).
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy import proxy_server, tracing_endpoints
from litellm.proxy._types import LitellmUserRoles, ProxyLifespanState, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.spend_tracking import spend_management_endpoints
from litellm.proxy.spend_tracking.log_visibility import log_visibility
from litellm.proxy.tracing_runtime import manage_tracing, provide_storage
from litellm.rust_bridge.traces import ClickHouseStorage
from litellm.tracing import TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing.store import AmbiguousTraceError, TraceStore
from litellm.tracing.types import TraceScope

TEAM_KEY = UserAPIKeyAuth(
    token="hashed-key", team_id="team-research", org_id="org-1", user_role=LitellmUserRoles.INTERNAL_USER
)


@pytest.mark.asyncio
async def test_read_scope_for_admin_sees_everything():
    for role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        auth = UserAPIKeyAuth(token="k", team_id="team-a", user_role=role)
        assert await tracing_endpoints.read_scope_for(auth) == {
            "all_teams": 1,
            "user_id": "",
            "team_ids": (),
            "api_key_hash": "",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth",
    (
        UserAPIKeyAuth(token="key-a", team_id="team-a", user_id="reader"),
        UserAPIKeyAuth(token="dashboard-session", user_id="reader"),
    ),
)
async def test_read_scope_for_user_covers_all_their_keys(auth, monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(
        spend_management_endpoints,
        "_get_permitted_team_ids_for_spend_logs",
        AsyncMock(return_value=[]),
    )
    scope: Final = await tracing_endpoints.read_scope_for(auth)

    assert scope == {"all_teams": 0, "user_id": "reader", "team_ids": (), "api_key_hash": ""}


@pytest.mark.asyncio
async def test_read_scope_includes_permitted_teams(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking import spend_management_endpoints

    permitted: Final = AsyncMock(return_value=["t1"])
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(spend_management_endpoints, "_get_permitted_team_ids_for_spend_logs", permitted)

    scope: Final = await tracing_endpoints.read_scope_for(UserAPIKeyAuth(token="key-a", user_id="reader"))

    assert scope == {"all_teams": 0, "user_id": "reader", "team_ids": ("t1",), "api_key_hash": ""}
    permitted.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_scope_falls_back_to_user_only_when_team_lookup_fails(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking import spend_management_endpoints

    permitted: Final = AsyncMock(side_effect=RuntimeError("Postgres unavailable"))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(spend_management_endpoints, "_get_permitted_team_ids_for_spend_logs", permitted)

    scope: Final = await tracing_endpoints.read_scope_for(UserAPIKeyAuth(token="key-a", user_id="reader"))

    assert scope == {"all_teams": 0, "user_id": "reader", "team_ids": (), "api_key_hash": ""}
    permitted.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_scope_for_key_without_user_is_key_only():
    scope: Final = await tracing_endpoints.read_scope_for(
        UserAPIKeyAuth(token="hashed-key", team_id="team-a", user_role=LitellmUserRoles.INTERNAL_USER)
    )
    assert scope == {"all_teams": 0, "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}


@pytest.mark.asyncio
async def test_log_visibility_without_user_or_token_is_forbidden():
    with pytest.raises(HTTPException) as e:
        await log_visibility(UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER))
    assert e.value.status_code == 403


def test_tenant_for_comes_from_auth():
    tenant: Final = tracing_endpoints.tenant_for(TEAM_KEY)
    assert (tenant.team_id, tenant.api_key_hash, tenant.org_id, tenant.user_id) == (
        "team-research",
        "hashed-key",
        "org-1",
        "",
    )
    blank: Final = tracing_endpoints.tenant_for(UserAPIKeyAuth())
    assert (blank.team_id, blank.api_key_hash, blank.org_id, blank.user_id) == ("", "", "", "")
    user_tenant: Final = tracing_endpoints.tenant_for(UserAPIKeyAuth(user_id="reader"))
    assert user_tenant.user_id == "reader"


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
    assert (tenant.team_id, tenant.api_key_hash, tenant.org_id, tenant.user_id) == (
        auth.team_id or "",
        auth.token or "",
        auth.org_id or "",
        auth.user_id or "",
    )


@pytest.fixture
def receiver(client) -> MagicMock:
    fake = MagicMock()
    fake.ingest = AsyncMock(return_value=1)
    fake.list_traces = AsyncMock(return_value={"data": [], "next_cursor": None})
    fake.get_trace = AsyncMock(return_value=None)
    fake.get_span = AsyncMock(return_value=None)
    fake.get_span_error = AsyncMock(
        return_value={"span_id": "s1", "message": "error", "total_chars": 5, "next_cursor": None}
    )
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
    trace = {"summary": {"trace_id": "t1"}, "agents": [], "spans": []}
    receiver.get_trace.return_value = trace
    response = client.get("/v1/traces/t1")
    assert response.status_code == 200
    assert response.json() == trace
    receiver.get_trace.assert_awaited_with(
        "t1", {"all_teams": 0, "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}, ""
    )


def test_get_span_404_and_200(client, receiver):
    assert client.get("/v1/traces/t1/spans/s1").status_code == 404
    receiver.get_span.return_value = {"span_id": "s1", "input": "", "output": "", "attributes": {}}
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
        side_effect=[
            [{"trace_ref": "ref-a"}],
            [{"span_id": "s1", "input": '{"city": "Paris"}', "output": stored_output, "attributes": {}}],
        ]
    )
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: TraceReceiver(TraceStore(storage))
    body = client.get("/v1/traces/t1/spans/s1").json()
    assert body["output"] == stored_output
    assert body["input_ui"] == {"kind": "fields", "fields": [{"key": "city", "value": "Paris"}]}
    assert body["output_ui"] == {
        "kind": "messages",
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup", "arguments": '{"id": 7}'}]}
        ],
    }


def test_trace_detail_passes_scoped_reference(client, receiver):
    receiver.get_trace.return_value = {"summary": {"trace_id": "t1"}, "agents": [], "spans": []}
    assert client.get("/v1/traces/t1?trace_ref=run-one").status_code == 200
    receiver.get_trace.assert_awaited_with(
        "t1", {"all_teams": 0, "user_id": "", "team_ids": (), "api_key_hash": "hashed-key"}, "run-one"
    )


def test_ambiguous_trace_and_span_require_reference(client, receiver):
    receiver.get_trace.side_effect = AmbiguousTraceError("Multiple traces have this ID; provide trace_ref")
    receiver.get_span.side_effect = AmbiguousTraceError("Multiple traces have this ID; provide trace_ref")
    assert client.get("/v1/traces/reused").status_code == 409
    assert client.get("/v1/traces/reused/spans/span").status_code == 409


def test_invalid_export_and_cursor_are_client_errors(client, receiver):
    from litellm.tracing.decode import InvalidOTLPPayloadError

    receiver.ingest.side_effect = InvalidOTLPPayloadError("invalid OTLP trace payload")
    assert client.post("/v1/traces", content=b"broken").status_code == 400
    receiver.list_traces.side_effect = ValueError("Invalid trace cursor")
    assert client.get("/v1/traces?cursor=broken").status_code == 400
    receiver.get_span_error.side_effect = ValueError("Invalid diagnostic cursor")
    assert client.get("/v1/traces/t1/spans/s1/error?cursor=broken").status_code == 400


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


def test_ingest_does_not_resolve_read_visibility(client, receiver, monkeypatch: pytest.MonkeyPatch) -> None:
    visibility: Final = AsyncMock()
    monkeypatch.setattr(tracing_endpoints, "log_visibility", visibility)

    response = client.post("/v1/traces", json={})

    assert response.status_code == 200
    visibility.assert_not_awaited()


def test_span_error_uses_request_log_visibility(client, receiver, monkeypatch: pytest.MonkeyPatch) -> None:
    permitted_teams: Final = AsyncMock(return_value=["team-permitted"])
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(
        spend_management_endpoints,
        "_get_permitted_team_ids_for_spend_logs",
        permitted_teams,
    )
    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        token="key-a",
        team_id="team-a",
        user_id="reader",
        user_role=LitellmUserRoles.TEAM,
    )

    response = client.get("/v1/traces/t1/spans/s1/error")

    assert response.status_code == 200
    receiver.get_span_error.assert_awaited_once_with(
        "t1",
        "s1",
        {"all_teams": 0, "user_id": "reader", "team_ids": ("team-permitted",), "api_key_hash": ""},
        "",
        None,
    )
    permitted_teams.assert_awaited_once()


def test_ambiguous_span_error_requires_reference(client, receiver) -> None:
    receiver.get_span_error.side_effect = AmbiguousTraceError("Multiple traces have this ID; provide trace_ref")

    response = client.get("/v1/traces/reused/spans/span/error")

    assert response.status_code == 409


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
    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        token=TEAM_KEY.token,
        team_id=TEAM_KEY.team_id,
        org_id=TEAM_KEY.org_id,
        user_id="reader",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    response: Final = client.post(
        "/v1/traces",
        json={
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "litellm.user_id", "value": {"stringValue": "spoofed-user"}},
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
    assert rows[0]["UserId"] == "reader"
    assert rows[0]["TeamId"] == TEAM_KEY.team_id
    assert rows[0]["ApiKeyHash"] == TEAM_KEY.token
    assert rows[0]["ResourceAttributes"] == {
        "litellm.user_id": "reader",
        "litellm.team_id": TEAM_KEY.team_id,
        "litellm.api_key_hash": TEAM_KEY.token,
        "litellm.org_id": TEAM_KEY.org_id,
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
        "span_detail",
        {
            "all_teams": 0,
            "user_id": "",
            "team_ids": (),
            "api_key_hash": TEAM_KEY.token,
            "trace_id": "t1",
            "span_id": "first-span",
            "trace_ref": "first-run",
        },
    )
    second_storage.query.assert_awaited_once_with(
        "span_detail",
        {
            "all_teams": 0,
            "user_id": "",
            "team_ids": (),
            "api_key_hash": TEAM_KEY.token,
            "trace_id": "t1",
            "span_id": "second-span",
            "trace_ref": "second-run",
        },
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
    assert storage.lens_sample.await_args.args[0]["all_teams"] == 1


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

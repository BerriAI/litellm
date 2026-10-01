"""
Tests for the agent tracing endpoints (litellm/proxy/tracing_endpoints.py).
"""

from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy import tracing_endpoints
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.spend_tracking import spend_management_endpoints
from litellm.proxy.spend_tracking.log_visibility import log_visibility
from litellm.tracing import TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing.store import AmbiguousTraceError, ClickHouseTraceStore

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


# ---------------------------------------------------------------- endpoints


@pytest.fixture
def receiver(monkeypatch) -> MagicMock:
    fake = MagicMock()
    fake.ingest = AsyncMock(return_value=1)
    fake.list_traces = AsyncMock(return_value={"data": [], "next_cursor": None})
    fake.get_trace = AsyncMock(return_value=None)
    fake.get_span = AsyncMock(return_value=None)
    monkeypatch.setattr(tracing_endpoints, "receiver", fake)
    return fake


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(tracing_endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: TEAM_KEY
    return TestClient(app)


def test_501_when_tracing_not_enabled(client, monkeypatch):
    monkeypatch.setattr(tracing_endpoints, "receiver", None)
    assert client.post("/v1/traces", content=b"").status_code == 501
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
    assert kwargs["body"] == b"\x0a\x00"
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
    assert "exceeds" in response.json()["detail"]


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


def test_get_span_serves_ui_content_from_stored_payloads(client, monkeypatch):
    storage = MagicMock()
    stored_output = '{"role": "ai", "content": "", "tool_calls": [{"name": "lookup", "args": {"id": 7}}]}'
    storage.query = AsyncMock(
        side_effect=[
            [{"trace_ref": "ref-a"}],
            [{"span_id": "s1", "input": '{"city": "Paris"}', "output": stored_output, "attributes": {}}],
        ]
    )
    monkeypatch.setattr(tracing_endpoints, "receiver", TraceReceiver(ClickHouseTraceStore(storage)))
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

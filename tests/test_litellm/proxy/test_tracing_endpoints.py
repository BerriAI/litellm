"""
Tests for the agent tracing endpoints (litellm/proxy/tracing_endpoints.py).
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy import tracing_endpoints
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.tracing import TracingPayloadTooLargeError

TEAM_KEY = UserAPIKeyAuth(
    token="hashed-key", team_id="team-research", org_id="org-1", user_role=LitellmUserRoles.INTERNAL_USER
)


# ---------------------------------------------------------------- scope / tenant


def test_scope_for_admin_sees_everything():
    for role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        auth = UserAPIKeyAuth(token="k", team_id="team-a", user_role=role)
        assert tracing_endpoints.scope_for(auth) == {"team_ids": [], "api_key_hash": ""}


def test_scope_for_team_key_sees_its_team():
    assert tracing_endpoints.scope_for(TEAM_KEY) == {"team_ids": ["team-research"], "api_key_hash": ""}


def test_scope_for_teamless_key_sees_only_its_own_traces():
    auth = UserAPIKeyAuth(token="hashed-key", user_role=LitellmUserRoles.INTERNAL_USER)
    assert tracing_endpoints.scope_for(auth) == {"team_ids": [""], "api_key_hash": "hashed-key"}


def test_scope_for_no_team_no_token_is_forbidden():
    with pytest.raises(HTTPException) as e:
        tracing_endpoints.scope_for(UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER))
    assert e.value.status_code == 403


def test_tenant_for_comes_from_auth():
    tenant = tracing_endpoints.tenant_for(TEAM_KEY)
    assert (tenant.team_id, tenant.api_key_hash, tenant.org_id) == ("team-research", "hashed-key", "org-1")
    blank = tracing_endpoints.tenant_for(UserAPIKeyAuth())
    assert (blank.team_id, blank.api_key_hash, blank.org_id) == ("", "", "")


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
    for response in (
        client.post("/v1/traces", content=b""),
        client.get("/v1/traces"),
        client.get("/v1/traces/trace-id"),
        client.get("/v1/traces/trace-id/spans/span-id"),
    ):
        assert response.status_code == 501
        assert "CLICKHOUSE_URL and CLICKHOUSE_READER_URL" in response.json()["detail"]


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
        scope={"team_ids": ["team-research"], "api_key_hash": ""}, start_ms=1, end_ms=2, cursor="abc"
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
    receiver.get_trace.assert_awaited_with("t1", {"team_ids": ["team-research"], "api_key_hash": ""})


def test_get_span_404_and_200(client, receiver):
    assert client.get("/v1/traces/t1/spans/s1").status_code == 404
    receiver.get_span.return_value = {"span_id": "s1", "input": "", "output": "", "attributes": {}}
    response = client.get("/v1/traces/t1/spans/s1")
    assert response.status_code == 200
    assert response.json()["span_id"] == "s1"
    receiver.get_span.assert_awaited_with("t1", "s1", {"team_ids": ["team-research"], "api_key_hash": ""})


def test_teamless_key_without_token_gets_403_on_reads(client, receiver):
    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER
    )
    assert client.get("/v1/traces").status_code == 403
    receiver.list_traces.assert_not_called()

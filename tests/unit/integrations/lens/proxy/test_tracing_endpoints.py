"""
Tests for the agent tracing endpoints (litellm/integrations/lens/proxy/tracing_endpoints.py).
"""

import asyncio
import gzip
import json
from queue import Queue
from collections.abc import AsyncGenerator, Iterable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from itertools import chain
from typing import Final, Literal, NamedTuple
from unittest.mock import AsyncMock, MagicMock, call
from urllib.parse import urlsplit

import httpx
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span
from pydantic import JsonValue, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.constants import (
    AGENT_TRACING_AGENT_LIST_LIMIT,
    DEFAULT_AGENT_TRACING_RETENTION_DAYS,
    TRACE_READ_RETRY_AFTER_SECONDS,
)
from litellm.integrations.lens import TraceReceiver
from litellm.integrations.lens.errors import TraceChanged
from litellm.integrations.lens.generated.models import TraceQueryHelp
from litellm.integrations.lens.generated.responses import TraceSQLResponse
from litellm.integrations.lens.generated.types import TraceScope
from litellm.integrations.lens.proxy import tracing_endpoints
from litellm.integrations.lens.proxy.buffering import FORWARD_BUFFER_BUDGET
from litellm.integrations.lens.proxy.tracing_runtime import manage_tracing
from litellm.integrations.lens.remote import MAX_RESPONSE_BYTES, LensConnection, RemoteTraceStore
from litellm.integrations.lens.storage import LensTraceStorage
from litellm.integrations.lens.types import TraceAgent, TraceAgentList
from litellm.proxy._types import LitellmUserRoles, ProxyLifespanState, UserAPIKeyAuth
from litellm.proxy.auth.authorization import OwnedRows, ReadScope
from litellm.proxy.auth.authorization_dependencies import get_log_team_lookup
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

SQL_ROWS: Final[tuple[Mapping[str, JsonValue], ...]] = (
    {
        "value": "9007199254740993",
        "count": 42,
        "fraction": 2.5,
        "nested": {"values": [True, None, "text"]},
    },
)
SQL_RESPONSE: Final = TraceSQLResponse(data=SQL_ROWS)
QUERY_HELP: Final[Mapping[str, object]] = {
    "dialect": "test SQL",
    "access": "authenticated scope",
    "response": (
        'JSON object {"data": [rows]}; each row maps selected columns to values; 64-bit integers may be strings'
    ),
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
    user_id="user",
    token="hashed-key",
    team_id="team-research",
    org_id="org-1",
    user_role=LitellmUserRoles.INTERNAL_USER,
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
        "priced_calls": 0,
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
SPAN_ERROR_RESPONSE: Final = {
    "span_id": "s1",
    "message": "span error",
    "total_chars": 10,
    "next_cursor": None,
}
CONVERSATION_RESPONSE: Final = {
    "turns": [
        {
            "trace_id": "earlier-trace",
            "trace_ref": "earlier-run",
            "span_id": "earlier-span",
            "start_time": "2026-01-01T00:00:00Z",
            "input": "earlier input",
            "output": "earlier output",
            "input_ui": {"kind": "text", "text": "earlier input"},
            "output_ui": {"kind": "text", "text": "earlier output"},
        }
    ],
    "next_cursor": "more-turns",
}
NOW_MS: Final = 1_800_000_000_000


class RequestValidationError(TypedDict):
    type: ReadOnly[str]
    loc: ReadOnly[list[str | int]]


def _validation_errors(response: Response) -> tuple[RequestValidationError, ...]:
    return tuple(TypeAdapter(list[RequestValidationError]).validate_python(response.json()["detail"]))


def _assert_validation_error(response: Response, error_type: str, location: tuple[str | int, ...]) -> None:
    assert response.status_code == 422, response.text
    assert any(
        error["type"] == error_type and tuple(error["loc"]) == location for error in _validation_errors(response)
    )


@pytest.mark.parametrize(
    ("auth", "scope"),
    (
        pytest.param(
            UserAPIKeyAuth(token="admin-key", team_id="team-a", user_role=LitellmUserRoles.PROXY_ADMIN),
            TraceScope(all_teams=1, user_id="", team_ids=()),
            id="admin",
        ),
        pytest.param(
            UserAPIKeyAuth(token="view-key", team_id="team-a", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
            TraceScope(all_teams=1, user_id="", team_ids=()),
            id="view-only-admin",
        ),
        pytest.param(
            TEAM_KEY,
            TraceScope(all_teams=0, user_id="user", team_ids=()),
            id="team-key",
        ),
        pytest.param(
            UserAPIKeyAuth(user_id="user", token="hashed-key", user_role=LitellmUserRoles.INTERNAL_USER),
            TraceScope(all_teams=0, user_id="user", team_ids=()),
            id="teamless-key",
        ),
        pytest.param(
            UserAPIKeyAuth(token="hashed-key", user_role=LitellmUserRoles.INTERNAL_USER),
            None,
            id="key-without-user-can-only-write",
        ),
    ),
)
def test_trace_read_permissions(
    client: TestClient, receiver: MagicMock, auth: UserAPIKeyAuth, scope: TraceScope | None
) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth

    read: Final = client.get("/v1/traces?start_ms=1&end_ms=2")
    assert read.status_code == (403 if scope is None else 200), read.text
    if scope is None:
        receiver.list_traces.assert_not_awaited()
    else:
        receiver.list_traces.assert_awaited_once_with(scope=scope, start_ms=1, end_ms=2, cursor=None)


@pytest.fixture
def receiver(client) -> MagicMock:
    fake = MagicMock()
    fake.ingest = AsyncMock(return_value=1)
    fake.list_traces = AsyncMock(return_value={"data": [], "next_cursor": None})
    fake.get_trace = AsyncMock(return_value=None)
    fake.get_span = AsyncMock(return_value=None)
    fake.get_conversation = AsyncMock(return_value=None)
    fake.list_agents = AsyncMock(return_value=TraceAgentList(agents=()))
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: fake
    return fake


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(tracing_endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: TEAM_KEY

    async def lookup(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        return ()

    app.dependency_overrides[get_log_team_lookup] = lambda: lookup
    return TestClient(app)


AGENT_SPANS: Final = b"\x0a\x02agent spans exactly as the agent sent them"


def _forward_agent_spans(
    client: TestClient, lens: httpx.MockTransport, headers: Mapping[str, str], path: str = "/v1/traces"
) -> Response:
    tracing: Final = TraceReceiver(
        storage=LensTraceStorage(RemoteTraceStore(httpx.AsyncClient(base_url="http://lens", transport=lens)))
    )
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: tracing
    return client.post(path, content=AGENT_SPANS, headers=headers)


@pytest.mark.parametrize("path", ("/v1/traces", "/v1/logs"))
def test_agent_telemetry_reaches_lens_unchanged_under_the_owner_of_the_key_that_sent_it(
    client: TestClient, path: str
) -> None:
    received: Final = Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        received.put_nowait(request)
        return httpx.Response(200, content=b"stored")

    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="hashed-key", team_id="team-a", org_id="org-a", user_id="user-a"
    )
    response: Final = _forward_agent_spans(
        client,
        httpx.MockTransport(accept),
        {
            "content-type": "application/x-protobuf",
            "content-encoding": "gzip",
            "x-litellm-api-key-hash": "another-key",
            "x-litellm-team-id": "another-team",
        },
        path,
    )

    forwarded: Final = received.get_nowait()
    owner: Final = {name: value for name, value in forwarded.headers.items() if name.startswith("x-litellm-")}
    assert (response.status_code, response.content, response.headers["content-type"]) == (
        200,
        b"stored",
        "application/x-protobuf",
    )
    assert (forwarded.url.path, forwarded.content, forwarded.headers["content-encoding"]) == (
        path,
        AGENT_SPANS,
        "gzip",
    )
    assert owner == {
        "x-litellm-api-key-hash": "hashed-key",
        "x-litellm-team-id": "team-a",
        "x-litellm-org-id": "org-a",
        "x-litellm-user-id": "user-a",
    }


@pytest.mark.parametrize("media_type", ("application/json", "application/x-protobuf"))
def test_agent_spans_are_retried_later_when_lens_cannot_be_reached(client: TestClient, media_type: str) -> None:
    from google.rpc.status_pb2 import Status

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("lens is down", request=request)

    response: Final = _forward_agent_spans(client, httpx.MockTransport(refuse), {"content-type": media_type})

    message: Final = (
        response.json()["message"] if media_type == "application/json" else Status.FromString(response.content).message
    )
    assert (response.status_code, message, response.headers["retry-after"]) == (
        503,
        "Lens is unavailable. Please try again.",
        str(tracing_endpoints.OTLP_RETRY_AFTER_SECONDS),
    )


def test_agent_spans_larger_than_the_forwarding_limit_are_refused_before_lens_sees_them(client: TestClient) -> None:
    received: Final = Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        received.put_nowait(request)
        return httpx.Response(200)

    tracing: Final = TraceReceiver(
        storage=LensTraceStorage(
            RemoteTraceStore(httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)))
        )
    )
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: tracing

    response: Final = client.post(
        "/v1/traces", content=bytes(MAX_RESPONSE_BYTES + 1), headers={"content-type": "application/x-protobuf"}
    )

    assert (response.status_code, received.empty()) == (413, True)
    FORWARD_BUFFER_BUDGET.reserve(FORWARD_BUFFER_BUDGET.capacity)
    FORWARD_BUFFER_BUDGET.release(FORWARD_BUFFER_BUDGET.capacity)


@pytest.mark.parametrize("path", ("/v1/traces", "/v1/logs"))
def test_agent_telemetry_is_refused_on_a_gateway_without_tracing(client: TestClient, path: str) -> None:
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: None

    response: Final = client.post(path, content=AGENT_SPANS, headers={"content-type": "application/json"})

    assert (response.status_code, response.json()["message"]) == (501, "Agent tracing is not enabled on this gateway.")


def test_list_traces_passes_scope_window_and_cursor(client, receiver):
    response = client.get("/v1/traces", params={"start_ms": 1, "end_ms": 2, "cursor": "abc"})
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}
    receiver.list_traces.assert_awaited_once_with(
        scope={"all_teams": 0, "user_id": "user", "team_ids": ()},
        start_ms=1,
        end_ms=2,
        cursor="abc",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN))
@pytest.mark.parametrize(
    ("endpoint", "operation", "result"),
    (
        ("/v1/traces", "list", {"data": [], "next_cursor": None}),
        ("/v1/traces/agents", "query", {"data": []}),
        ("/v1/traces/t1", "trace", TRACE_RESPONSE),
        ("/v1/traces/t1/conversation", "conversation", CONVERSATION_RESPONSE),
        ("/v1/traces/t1/spans/s1", "span", SPAN_DETAIL_RESPONSE),
        ("/v1/traces/t1/spans/s1/error", "span_error", SPAN_ERROR_RESPONSE),
    ),
)
async def test_key_and_team_filters_reach_lens_without_replacing_authenticated_scope(
    client: TestClient, role: LitellmUserRoles, endpoint: str, operation: str, result: Mapping[str, JsonValue]
) -> None:
    requests: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(200, json=result)

    async def lookup(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        return ("permitted-team",)

    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="reader", user_role=role)
    client.app.dependency_overrides[get_log_team_lookup] = lambda: lookup
    client.app.dependency_overrides[tracing_endpoints.current_time_ms] = lambda: NOW_MS
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as worker:
        tracing: Final = TraceReceiver(LensTraceStorage(RemoteTraceStore(worker)))
        client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: tracing
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app), base_url="http://gateway"
        ) as gateway:
            response: Final = await gateway.get(
                endpoint,
                params={"team_id": "requested-team", "api_key_hash": "requested-key", "start_ms": 123, "end_ms": 456},
            )
    assert response.status_code == 200, response.text
    forwarded: Final = json.loads(requests.get_nowait().content)
    assert requests.empty()
    assert forwarded["operation"] == operation
    assert forwarded["parameters" if operation == "query" else "scope"] == {
        "all_teams": 1 if role == LitellmUserRoles.PROXY_ADMIN else 0,
        "user_id": "" if role == LitellmUserRoles.PROXY_ADMIN else "reader",
        "team_ids": [] if role == LitellmUserRoles.PROXY_ADMIN else ["permitted-team"],
        "filter_team_id": "requested-team",
        "filter_api_key_hash": "requested-key",
        **({"start_ms": 123, "end_ms": 456, "limit": AGENT_TRACING_AGENT_LIST_LIMIT} if operation == "query" else {}),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("exists", (True, False))
async def test_conversation_transport_preserves_scoped_reference_and_continuation(
    client: TestClient, exists: bool
) -> None:
    def accept(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/read"
        assert json.loads(request.content) == {
            "operation": "conversation",
            "scope": {
                "all_teams": 0,
                "user_id": "user",
                "team_ids": [],
                "filter_team_id": "selected-team",
                "filter_api_key_hash": "selected-key",
            },
            "trace_id": "t1",
            "trace_ref": "selected-run",
            "cursor": "previous-page",
            "page_size": 7,
        }
        return httpx.Response(200, content=json.dumps(CONVERSATION_RESPONSE if exists else None))

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as worker:
        tracing: Final = TraceReceiver(LensTraceStorage(RemoteTraceStore(worker)))
        client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: tracing
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app), base_url="http://gateway"
        ) as gateway:
            response: Final = await gateway.get(
                "/v1/traces/t1/conversation",
                params={
                    "team_id": "selected-team",
                    "api_key_hash": "selected-key",
                    "trace_ref": "selected-run",
                    "cursor": "previous-page",
                    "page_size": 7,
                },
            )
    assert response.status_code == (200 if exists else 404), response.text
    assert response.json() == (CONVERSATION_RESPONSE if exists else {"detail": "Trace t1 not found"})


@pytest.mark.parametrize("query", ("page_size=0", "page_size=51", "cursor=" + "x" * 513))
def test_conversation_rejects_unbounded_parameters(client: TestClient, receiver: MagicMock, query: str) -> None:
    response: Final = client.get(f"/v1/traces/t1/conversation?{query}")
    assert response.status_code == 422, response.text
    receiver.get_conversation.assert_not_awaited()


def test_list_traces_defaults_to_last_24h(client, receiver):
    client.get("/v1/traces")
    kwargs = receiver.list_traces.call_args.kwargs
    assert kwargs["end_ms"] - kwargs["start_ms"] == tracing_endpoints.MS_PER_DAY
    assert kwargs["cursor"] is None


@pytest.mark.parametrize(
    ("params", "expected_start_ms", "expected_end_ms"),
    (
        ({}, NOW_MS - tracing_endpoints.MS_PER_DAY, NOW_MS),
        ({"start_ms": 123}, 123, NOW_MS),
        ({"end_ms": -7}, NOW_MS - tracing_endpoints.MS_PER_DAY, -7),
    ),
)
def test_list_traces_resolves_default_bounds_from_injected_clock(
    client: TestClient,
    receiver: MagicMock,
    params: Mapping[str, int],
    expected_start_ms: int,
    expected_end_ms: int,
) -> None:
    client.app.dependency_overrides[tracing_endpoints.current_time_ms] = lambda: NOW_MS
    response: Final = client.get("/v1/traces", params=params)
    assert response.status_code == 200, response.text
    receiver.list_traces.assert_awaited_once_with(
        scope={"all_teams": 0, "user_id": "user", "team_ids": ()},
        start_ms=expected_start_ms,
        end_ms=expected_end_ms,
        cursor=None,
    )


@pytest.mark.parametrize(
    ("params", "expected_start_ms", "expected_end_ms"),
    (
        ({}, NOW_MS - DEFAULT_AGENT_TRACING_RETENTION_DAYS * tracing_endpoints.MS_PER_DAY, NOW_MS),
        ({"start_ms": 123, "end_ms": 456}, 123, 456),
    ),
)
def test_list_trace_agents_passes_reader_scope_and_window(
    client: TestClient,
    receiver: MagicMock,
    params: Mapping[str, int],
    expected_start_ms: int,
    expected_end_ms: int,
) -> None:
    client.app.dependency_overrides[tracing_endpoints.current_time_ms] = lambda: NOW_MS
    receiver.list_agents.return_value = TraceAgentList(
        agents=(
            TraceAgent(
                name="moyai",
                runs=3,
                failed_runs=1,
                last_seen=datetime(2026, 10, 7, 20, 31, tzinfo=timezone.utc),
                frameworks=("openai-agents",),
            ),
        )
    )
    response: Final = client.get("/v1/traces/agents", params=params)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "agents": [
            {
                "name": "moyai",
                "runs": 3,
                "failed_runs": 1,
                "last_seen": "2026-10-07T20:31:00Z",
                "frameworks": ["openai-agents"],
            }
        ]
    }
    receiver.list_agents.assert_awaited_once_with(
        scope={"all_teams": 0, "user_id": "user", "team_ids": ()},
        start_ms=expected_start_ms,
        end_ms=expected_end_ms,
    )
    receiver.get_trace.assert_not_awaited()


def test_list_trace_agents_requires_read_access(client: TestClient, receiver: MagicMock) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        token="hashed-key", user_role=LitellmUserRoles.INTERNAL_USER
    )
    response: Final = client.get("/v1/traces/agents")
    assert response.status_code == 403, response.text
    receiver.list_agents.assert_not_awaited()


def test_list_trace_agents_maps_storage_outage_to_503(client: TestClient, receiver: MagicMock) -> None:
    receiver.list_agents.side_effect = RuntimeError("private database details")
    response: Final = client.get("/v1/traces/agents")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "expected_scope"),
    (
        pytest.param(
            LitellmUserRoles.PROXY_ADMIN,
            TraceScope(all_teams=1, user_id="", team_ids=()),
            id="admin",
        ),
        pytest.param(
            LitellmUserRoles.INTERNAL_USER,
            TraceScope(all_teams=0, user_id="agent-owner", team_ids=("managed-team",)),
            id="owner-and-permitted-teams",
        ),
    ),
)
async def test_agent_picker_reads_through_worker_with_authenticated_scope(
    client: TestClient, role: LitellmUserRoles, expected_scope: TraceScope
) -> None:
    requests: Final = asyncio.Queue[httpx.Request]()
    secret: Final = "test-only-lens-service-secret-32-characters"

    def accept(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "agent_name": "research-agent",
                        "runs": "3",
                        "failed_runs": "1",
                        "last_seen_ms": "1791405060000",
                        "frameworks": ["openai-agents"],
                    }
                ]
            },
        )

    async def lookup(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        return ("managed-team",)

    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="agent-owner", token="user-key", team_id="unmanaged-team", user_role=role
    )
    client.app.dependency_overrides[get_log_team_lookup] = lambda: lookup
    async with httpx.AsyncClient(
        base_url="http://lens",
        headers={"Authorization": f"Bearer {secret}"},
        transport=httpx.MockTransport(accept),
    ) as worker:
        tracing: Final = TraceReceiver(storage=LensTraceStorage(RemoteTraceStore(worker)))
        client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: tracing
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app), base_url="http://gateway"
        ) as gateway:
            response: Final = await gateway.get("/v1/traces/agents", params={"start_ms": 123, "end_ms": 456})
    assert response.status_code == 200, response.text
    assert response.json() == {
        "agents": [
            {
                "name": "research-agent",
                "runs": 3,
                "failed_runs": 1,
                "last_seen": "2026-10-07T20:31:00Z",
                "frameworks": ["openai-agents"],
            }
        ]
    }
    request: Final = requests.get_nowait()
    assert requests.empty()
    assert request.method == "POST"
    assert request.url.path == "/internal/read"
    assert request.headers["Authorization"] == f"Bearer {secret}"
    assert json.loads(request.content) == {
        "operation": "query",
        "name": "trace_agents",
        "parameters": {
            **expected_scope,
            "team_ids": list(expected_scope["team_ids"]),
            "filter_team_id": "",
            "filter_api_key_hash": "",
            "start_ms": 123,
            "end_ms": 456,
            "limit": AGENT_TRACING_AGENT_LIST_LIMIT,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "code"), ((503, "unavailable"), (413, "too_large")))
async def test_agent_picker_reports_worker_failures_without_leaking_details(
    client: TestClient, status: int, code: str
) -> None:
    async with httpx.AsyncClient(
        base_url="http://lens",
        transport=httpx.MockTransport(lambda request: httpx.Response(status, text="private storage details")),
    ) as worker:
        tracing: Final = TraceReceiver(storage=LensTraceStorage(RemoteTraceStore(worker)))
        client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: tracing
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app), base_url="http://gateway"
        ) as gateway:
            response: Final = await gateway.get("/v1/traces/agents", params={"start_ms": 123, "end_ms": 456})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "private storage details" not in response.text
    if status == 503:
        assert response.headers["Retry-After"] == str(TRACE_READ_RETRY_AFTER_SECONDS)


def test_list_traces_forwards_large_and_negative_bounds_unchanged(client: TestClient, receiver: MagicMock) -> None:
    response: Final = client.get("/v1/traces", params={"start_ms": 2**63, "end_ms": -1, "cursor": "next"})
    assert response.status_code == 200, response.text
    receiver.list_traces.assert_awaited_once_with(
        scope={"all_teams": 0, "user_id": "user", "team_ids": ()},
        start_ms=2**63,
        end_ms=-1,
        cursor="next",
    )


def test_get_trace_404_and_200(client, receiver):
    assert client.get("/v1/traces/missing").status_code == 404
    receiver.get_trace.return_value = TRACE_RESPONSE
    response = client.get("/v1/traces/t1")
    assert response.status_code == 200
    assert response.json() == TRACE_RESPONSE
    receiver.get_trace.assert_awaited_with("t1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "", None, None)


def test_get_span_404_and_200(client, receiver):
    assert client.get("/v1/traces/t1/spans/s1").status_code == 404
    receiver.get_span.return_value = SPAN_DETAIL_RESPONSE
    response = client.get("/v1/traces/t1/spans/s1")
    assert response.status_code == 200
    assert response.json()["span_id"] == "s1"
    receiver.get_span.assert_awaited_with("t1", "s1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "")


@pytest.mark.parametrize("suffix,cursor,page_size", [("", None, None), ("&cursor=next&page_size=200", "next", 200)])
def test_trace_detail_passes_scoped_reference(client, receiver, suffix, cursor, page_size):
    receiver.get_trace.return_value = TRACE_RESPONSE
    assert client.get(f"/v1/traces/t1?trace_ref=run-one{suffix}").status_code == 200
    receiver.get_trace.assert_awaited_with(
        "t1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one", cursor, page_size
    )


@pytest.mark.parametrize("page_size", (1, 500))
def test_trace_detail_accepts_page_size_bounds(client: TestClient, receiver: MagicMock, page_size: int) -> None:
    receiver.get_trace.return_value = TRACE_RESPONSE
    response: Final = client.get("/v1/traces/t1", params={"trace_ref": "run-one", "page_size": page_size})
    assert response.status_code == 200, response.text
    receiver.get_trace.assert_awaited_once_with(
        "t1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one", None, page_size
    )


@pytest.mark.parametrize(
    ("value", "error_type"),
    (("0", "greater_than_equal"), ("501", "less_than_equal"), ("abc", "int_parsing")),
)
def test_trace_detail_reports_page_size_validation(
    client: TestClient, receiver: MagicMock, value: str, error_type: str
) -> None:
    response: Final = client.get("/v1/traces/t1", params={"page_size": value})
    _assert_validation_error(response, error_type, ("query", "page_size"))
    receiver.get_trace.assert_not_awaited()


def test_trace_read_routes_accept_and_forward_512_character_cursors(client: TestClient, receiver: MagicMock) -> None:
    cursor: Final = "x" * 512
    receiver.get_trace.return_value = TRACE_RESPONSE
    receiver.get_span_error = AsyncMock(return_value=SPAN_ERROR_RESPONSE)
    client.app.dependency_overrides[tracing_endpoints.current_time_ms] = lambda: NOW_MS

    list_response: Final = client.get("/v1/traces", params={"cursor": cursor})
    detail_response: Final = client.get("/v1/traces/t1", params={"trace_ref": "run-one", "cursor": cursor})
    error_response: Final = client.get(
        "/v1/traces/t1/spans/s1/error", params={"trace_ref": "run-one", "cursor": cursor}
    )

    assert list_response.status_code == 200, list_response.text
    assert detail_response.status_code == 200, detail_response.text
    assert error_response.status_code == 200, error_response.text
    receiver.list_traces.assert_awaited_once_with(
        scope={"all_teams": 0, "user_id": "user", "team_ids": ()},
        start_ms=NOW_MS - tracing_endpoints.MS_PER_DAY,
        end_ms=NOW_MS,
        cursor=cursor,
    )
    receiver.get_trace.assert_awaited_once_with(
        "t1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one", cursor, None
    )
    receiver.get_span_error.assert_awaited_once_with(
        "t1", "s1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one", cursor
    )


@pytest.mark.parametrize(
    "path",
    ("/v1/traces", "/v1/traces/t1", "/v1/traces/t1/spans/s1/error"),
)
def test_trace_read_routes_reject_513_character_cursors(client: TestClient, receiver: MagicMock, path: str) -> None:
    response: Final = client.get(path, params={"cursor": "x" * 513})
    _assert_validation_error(response, "string_too_long", ("query", "cursor"))


def test_trace_read_routes_ignore_unknown_query_parameters(client: TestClient, receiver: MagicMock) -> None:
    receiver.get_trace.return_value = TRACE_RESPONSE
    receiver.get_span.return_value = SPAN_DETAIL_RESPONSE
    receiver.get_span_error = AsyncMock(return_value=SPAN_ERROR_RESPONSE)

    list_params: Final = {"start_ms": 1, "end_ms": 2, "cursor": "list-cursor"}
    list_response: Final = client.get("/v1/traces", params=list_params)
    list_unknown_response: Final = client.get("/v1/traces", params={**list_params, "foo": "bar"})
    detail_params: Final = {"trace_ref": "run-one", "cursor": "detail-cursor", "page_size": 10}
    detail_response: Final = client.get("/v1/traces/t1", params=detail_params)
    detail_unknown_response: Final = client.get("/v1/traces/t1", params={**detail_params, "foo": "bar"})
    span_response: Final = client.get("/v1/traces/t1/spans/s1", params={"trace_ref": "run-one"})
    span_unknown_response: Final = client.get("/v1/traces/t1/spans/s1", params={"trace_ref": "run-one", "foo": "bar"})
    error_params: Final = {"trace_ref": "run-one", "cursor": "error-cursor"}
    error_response: Final = client.get("/v1/traces/t1/spans/s1/error", params=error_params)
    error_unknown_response: Final = client.get("/v1/traces/t1/spans/s1/error", params={**error_params, "foo": "bar"})

    assert list_response.status_code == 200, list_response.text
    assert list_unknown_response.status_code == 200, list_unknown_response.text
    assert detail_response.status_code == 200, detail_response.text
    assert detail_unknown_response.status_code == 200, detail_unknown_response.text
    assert span_response.status_code == 200, span_response.text
    assert span_unknown_response.status_code == 200, span_unknown_response.text
    assert error_response.status_code == 200, error_response.text
    assert error_unknown_response.status_code == 200, error_unknown_response.text
    receiver.list_traces.assert_has_awaits(
        (
            call(scope={"all_teams": 0, "user_id": "user", "team_ids": ()}, start_ms=1, end_ms=2, cursor="list-cursor"),
            call(scope={"all_teams": 0, "user_id": "user", "team_ids": ()}, start_ms=1, end_ms=2, cursor="list-cursor"),
        )
    )
    receiver.get_trace.assert_has_awaits(
        (
            call("t1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one", "detail-cursor", 10),
            call("t1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one", "detail-cursor", 10),
        )
    )
    receiver.get_span.assert_has_awaits(
        (
            call("t1", "s1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one"),
            call("t1", "s1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one"),
        )
    )
    receiver.get_span_error.assert_has_awaits(
        (
            call("t1", "s1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one", "error-cursor"),
            call("t1", "s1", {"all_teams": 0, "user_id": "user", "team_ids": ()}, "run-one", "error-cursor"),
        )
    )


@pytest.mark.parametrize(
    "path,method",
    (
        ("/v1/traces", "list_traces"),
        ("/v1/traces/t1", "get_trace"),
        ("/v1/traces/t1/conversation", "get_conversation"),
        ("/v1/traces/t1/spans/s1", "get_span"),
        ("/v1/traces/t1/spans/s1/error", "get_span_error"),
    ),
)
@pytest.mark.parametrize(
    "error,status,code,message",
    (
        (
            RuntimeError("private database details"),
            503,
            "unavailable",
            "Traces are temporarily unavailable. Please try again.",
        ),
        (
            OverflowError("private query details"),
            413,
            "too_large",
            "Trace is too large for this view. Use a filtered trace query.",
        ),
        (
            TraceChanged("Trace changed while paging; refresh the trace to continue"),
            409,
            "trace_changed",
            "Trace changed while paging; refresh the trace to continue",
        ),
        (ValueError("Invalid span cursor"), 400, "invalid_request", "Invalid span cursor"),
    ),
)
def test_read_failures_carry_a_code_per_kind_without_exposing_database_details(
    client: TestClient,
    receiver: MagicMock,
    path: str,
    method: str,
    error: Exception,
    status: int,
    code: str,
    message: str,
) -> None:
    getattr(receiver, method).side_effect = error
    response: Final = client.get(path)
    assert response.status_code == status
    assert response.json() == {"detail": {"code": code, "message": message}}
    retry_after: Final = response.headers.get("Retry-After")
    assert (retry_after == str(TRACE_READ_RETRY_AFTER_SECONDS)) == (status == 503), retry_after


@pytest.mark.parametrize(
    "query",
    ("page_size=0", "page_size=501", "cursor=" + "x" * 513),
    ids=("zero-page-size", "oversized-page-size", "oversized-cursor"),
)
def test_trace_page_rejects_unbounded_parameters(client: TestClient, receiver: MagicMock, query: str) -> None:
    response: Final = client.get(f"/v1/traces/t1?{query}")
    assert response.status_code == 422
    receiver.get_trace.assert_not_awaited()


def test_invalid_cursor_is_a_client_error(client: TestClient, receiver: MagicMock) -> None:
    receiver.list_traces.side_effect = ValueError("Invalid trace cursor")
    assert client.get("/v1/traces?cursor=broken").status_code == 400


@pytest.mark.parametrize(
    "auth",
    (
        UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER),
        UserAPIKeyAuth(token="key"),
        UserAPIKeyAuth(token="key", team_id="unpermitted"),
        UserAPIKeyAuth(user_id="", token="key"),
    ),
)
def test_key_without_user_cannot_read_traces(client: TestClient, auth: UserAPIKeyAuth) -> None:
    storage: Final = MagicMock(spec=LensTraceStorage)
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: TraceReceiver(storage)
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    for path in (
        "/v1/traces",
        "/v1/traces/t1",
        "/v1/traces/t1/conversation",
        "/v1/traces/t1/spans/s1",
        "/v1/traces/t1/spans/s1/error",
        "/v1/traces/query/help",
    ):
        response: Final = client.get(path)
        assert response.status_code == 403, response.text
    query: Final = client.post("/v1/traces/query", json={"sql": "SELECT * FROM otel_traces"})
    assert query.status_code == 403, query.text
    for read in (
        storage.list_traces,
        storage.get_trace,
        storage.get_conversation,
        storage.get_span,
        storage.get_span_error,
    ):
        read.assert_not_called()
    storage.query_sql.assert_not_called()
    storage.query_help.assert_not_called()


def test_disabled_receiver_precedes_read_scope_rejection(client: TestClient) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER
    )
    response: Final = client.get("/v1/traces")
    assert response.status_code == 501
    assert response.json() == {
        "detail": "Agent tracing is not enabled. Configure the Lens service and LITELLM_LENS_URL."
    }


def test_lifespan_receivers_are_app_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LENS_URL", "http://lens.test")
    monkeypatch.setenv("LITELLM_LENS_SERVICE_TOKEN", "test-service-token-with-32-characters")
    first_storage: Final = MagicMock(spec=LensTraceStorage)
    first_storage.get_span = AsyncMock(return_value={**SPAN_DETAIL_RESPONSE, "span_id": "first-span"})
    second_storage: Final = MagicMock(spec=LensTraceStorage)
    second_storage.get_span = AsyncMock(return_value={**SPAN_DETAIL_RESPONSE, "span_id": "second-span"})
    first_receiver: Final = TraceReceiver(first_storage)
    second_receiver: Final = TraceReceiver(second_storage)
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
    first_storage.ensure_schema.assert_not_awaited()
    second_storage.ensure_schema.assert_not_awaited()

    assert first_response.status_code == second_response.status_code == 200
    assert first_response.json()["span_id"] == "first-span"
    assert second_response.json()["span_id"] == "second-span"
    scope: Final = TraceScope(all_teams=0, user_id=TEAM_KEY.user_id or "", team_ids=())
    assert first_storage.get_span.await_count == 2
    first_storage.get_span.assert_awaited_with("t1", "first-span", scope, "first-run")
    second_storage.get_span.assert_awaited_once_with("t1", "second-span", scope, "second-run")


@pytest.mark.parametrize("auth", [TEAM_KEY, UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)])
def test_query_validation_precedes_trace_access_checks(client: TestClient, auth: UserAPIKeyAuth) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth
    response: Final = client.get("/v1/traces", params={"start_ms": "invalid"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "start_ms"]


@pytest.mark.parametrize("enabled", [True, False])
def test_unconfigured_lifespan_receiver_returns_501(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_LENS_URL", raising=False)
    storage: Final = MagicMock(spec=LensTraceStorage)
    storage.ensure_schema = AsyncMock(side_effect=RuntimeError("storage unavailable"))
    tracing: Final = TraceReceiver(storage)

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
    storage.ensure_schema.assert_not_awaited()
    storage.list_traces.assert_not_called()


@pytest.mark.parametrize(
    ("auth", "expected_scope"),
    (
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), {"kind": "all"}),
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY), {"kind": "all"}),
        (TEAM_KEY, {"kind": "owned", "user_id": "user", "team_ids": ()}),
        (
            UserAPIKeyAuth(user_id="user", token="project-key", team_id="team-a", project_id="project-a"),
            {"kind": "owned", "user_id": "user", "team_ids": ()},
        ),
        (
            UserAPIKeyAuth(user_id="user", token="solo-key"),
            {"kind": "owned", "user_id": "user", "team_ids": ()},
        ),
    ),
)
def test_sql_and_help_use_authenticated_scope(
    client: TestClient, receiver: MagicMock, auth: UserAPIKeyAuth, expected_scope: dict[str, str]
) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    receiver.storage.query_sql = AsyncMock(return_value=SQL_RESPONSE)
    receiver.storage.query_help = AsyncMock(return_value=TraceQueryHelp.model_validate(QUERY_HELP))
    result: Final = client.post("/v1/traces/query", json={"sql": "SELECT * FROM otel_traces"})
    assert result.status_code == 200, result.text
    assert result.json() == {"data": list(SQL_ROWS)}
    assert type(result.json()["data"][0]["value"]) is str
    assert type(result.json()["data"][0]["count"]) is int
    assert type(result.json()["data"][0]["fraction"]) is float
    receiver.storage.query_sql.assert_awaited_once_with("SELECT * FROM otel_traces", expected_scope, "test-secret")
    help_result: Final = client.get("/v1/traces/query/help")
    assert help_result.status_code == 200, help_result.text
    assert help_result.json() == QUERY_HELP
    receiver.storage.query_help.assert_awaited_once_with(expected_scope, "test-secret")
    forged: Final = client.post("/v1/traces/query", json={"sql": "SELECT 1", "scope": {"kind": "all"}})
    assert forged.status_code == 422, forged.text
    assert receiver.storage.query_sql.await_count == 1


def test_sql_query_returns_empty_data(client: TestClient, receiver: MagicMock) -> None:
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    receiver.storage.query_sql = AsyncMock(return_value=TraceSQLResponse(data=()))

    result: Final = client.post("/v1/traces/query", json={"sql": "SELECT * FROM otel_traces"})

    assert result.status_code == 200, result.text
    assert result.json() == {"data": []}


def test_sql_query_openapi_declares_a_closed_response_object(client: TestClient) -> None:
    openapi: Final = client.app.openapi()
    response: Final = openapi["paths"]["/v1/traces/query"]["post"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    component_name: Final = response["$ref"].rsplit("/", 1)[-1]
    component: Final = openapi["components"]["schemas"][component_name]

    assert set(component["properties"]) == {"data"}
    assert component["additionalProperties"] is False


@pytest.mark.parametrize(
    ("body", "error_type", "location"),
    (
        (b"{}", "missing", ("body", "sql")),
        (b'{"sql": null}', "string_type", ("body", "sql")),
        (b'{"sql": 1}', "string_type", ("body", "sql")),
        (b'{"sql": "SELECT 1", "extra": true}', "extra_forbidden", ("body", "extra")),
        (b"{", "json_invalid", ("body", 1)),
        (b"[]", "model_attributes_type", ("body",)),
    ),
)
def test_sql_query_rejects_invalid_request_bodies(
    client: TestClient, receiver: MagicMock, body: bytes, error_type: str, location: tuple[str | int, ...]
) -> None:
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    receiver.storage.query_sql = AsyncMock()
    response: Final = client.post("/v1/traces/query", content=body, headers={"content-type": "application/json"})
    _assert_validation_error(response, error_type, location)
    receiver.storage.query_sql.assert_not_awaited()


@pytest.mark.parametrize("auth", (UserAPIKeyAuth(), UserAPIKeyAuth(team_id="a", project_id="p")))
def test_sql_rejects_missing_identity_without_querying(
    client: TestClient, receiver: MagicMock, auth: UserAPIKeyAuth
) -> None:
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    result: Final = client.post("/v1/traces/query", json={"sql": "SELECT * FROM otel_traces"})
    assert result.status_code == 403, result.text
    assert client.get("/v1/traces/query/help").status_code == 403
    receiver.storage.query_sql.assert_not_called()
    receiver.storage.query_help.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status"), ((ValueError("invalid SQL"), 400), (RuntimeError("reader unavailable"), 503))
)
def test_sql_reports_rejected_queries_and_unavailable_readers(
    client: TestClient, receiver: MagicMock, error: Exception, status: int
) -> None:
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    receiver.storage.query_sql = AsyncMock(side_effect=error)
    result: Final = client.post("/v1/traces/query", json={"sql": "SELECT 1"})
    assert result.status_code == status, result.text
    receiver.storage.query_sql.assert_awaited_once_with(
        "SELECT 1", {"kind": "owned", "user_id": "user", "team_ids": ()}, "test-secret"
    )


def test_query_help_does_not_fall_back_when_reader_provisioning_fails(client: TestClient, receiver: MagicMock) -> None:
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"
    receiver.storage.query_help = AsyncMock(side_effect=RuntimeError("reader provisioning failed"))
    result: Final = client.get("/v1/traces/query/help")
    assert result.status_code == 503, result.text
    receiver.storage.query_help.assert_awaited_once_with(
        {"kind": "owned", "user_id": "user", "team_ids": ()}, "test-secret"
    )


@pytest.mark.parametrize("secret", (None, "configured-master-key"))
def test_queries_require_a_proxy_secret(
    client: TestClient, receiver: MagicMock, monkeypatch: pytest.MonkeyPatch, secret: str | None
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", secret)
    receiver.storage.query_sql = AsyncMock(return_value=SQL_RESPONSE)
    result: Final = client.post("/v1/traces/query", json={"sql": "SELECT 1"})
    if secret is None:
        assert result.status_code == 503, result.text
        assert "master key" in result.json()["detail"]
        receiver.storage.query_sql.assert_not_awaited()
        return
    assert result.status_code == 200, result.text
    receiver.storage.query_sql.assert_awaited_once_with(
        "SELECT 1", {"kind": "owned", "user_id": "user", "team_ids": ()}, secret
    )


@pytest.mark.parametrize(
    ("auth", "teams", "expected"),
    (
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), ("team-a",), (1, "", ())),
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY), ("team-a",), (1, "", ())),
        (UserAPIKeyAuth(user_id="user", token="key", team_id="unpermitted"), ("a", "b"), (0, "user", ("a", "b"))),
        (UserAPIKeyAuth(user_id="user", token="key"), (), (0, "user", ())),
        (UserAPIKeyAuth(user_id="user"), ("a",), (0, "user", ("a",))),
    ),
)
def test_shared_trace_permissions_reach_read_and_sql_boundaries(
    client: TestClient,
    auth: UserAPIKeyAuth,
    teams: tuple[str, ...],
    expected: tuple[Literal[0, 1], str, tuple[str, ...]],
) -> None:
    async def lookup(caller: UserAPIKeyAuth) -> tuple[str, ...]:
        assert caller is auth
        return teams

    team_lookup: Final = AsyncMock(side_effect=lookup)
    storage: Final = MagicMock(spec=LensTraceStorage)
    storage.get_span = AsyncMock(return_value=SPAN_DETAIL_RESPONSE)
    storage.query_sql = AsyncMock(return_value=SQL_RESPONSE)
    storage.query_help = AsyncMock(return_value=TraceQueryHelp.model_validate(QUERY_HELP))
    client.app.dependency_overrides[user_api_key_auth] = lambda: auth
    client.app.dependency_overrides[get_log_team_lookup] = lambda: team_lookup
    client.app.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: TraceReceiver(storage)
    client.app.dependency_overrides[tracing_endpoints.provide_trace_query_secret] = lambda: "test-secret"

    response: Final = client.get("/v1/traces/t1/spans/s1?trace_ref=run-one")
    assert response.status_code == 200, response.text
    assert response.json()["span_id"] == "s1"
    storage.get_span.assert_awaited_once_with(
        "t1", "s1", TraceScope(all_teams=expected[0], user_id=expected[1], team_ids=expected[2]), "run-one"
    )
    sql_response: Final = client.post("/v1/traces/query", json={"sql": "SELECT * FROM otel_traces"})
    assert sql_response.status_code == 200, sql_response.text
    assert sql_response.json() == {"data": list(SQL_ROWS)}
    assert client.get("/v1/traces/query/help").json() == QUERY_HELP
    query_scope: Final = (
        {"kind": "all"}
        if expected[0]
        else {
            "kind": "owned",
            "user_id": expected[1],
            "team_ids": expected[2],
        }
    )
    storage.query_sql.assert_awaited_once_with("SELECT * FROM otel_traces", query_scope, "test-secret")
    storage.query_help.assert_awaited_once_with(query_scope, "test-secret")
    assert team_lookup.await_count == (
        3
        if auth.user_id and auth.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        else 0
    )


@pytest.mark.parametrize(
    ("scope", "expected"),
    (
        (OwnedRows(None), ("", ())),
        (OwnedRows("user"), ("user", ())),
        (OwnedRows("user", ("a", "b")), ("user", ("a", "b"))),
    ),
)
def test_trace_storage_permissions_map_owned_rows(
    scope: ReadScope,
    expected: tuple[str, tuple[str, ...]],
) -> None:
    assert tracing_endpoints._trace_scope(scope) == TraceScope(all_teams=0, user_id=expected[0], team_ids=expected[1])
    assert tracing_endpoints.trace_query_scope(scope) == {
        "kind": "owned",
        "user_id": expected[0],
        "team_ids": expected[1],
    }


@pytest.mark.parametrize("cursor,page_size", ((None, None), ("next", 200)))
async def test_storage_preserves_page_cursor_and_normalizes_remote_trace_data(
    cursor: str | None, page_size: int | None
) -> None:
    scope: Final[TraceScope] = {"all_teams": 0, "user_id": "owner", "team_ids": ()}

    def accept(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "operation": "trace",
            "scope": {**scope, "team_ids": []},
            "trace_id": "t1",
            "trace_ref": "run",
            "cursor": cursor,
            "page_size": page_size,
        }
        return httpx.Response(200, json={**TRACE_RESPONSE, "next_cursor": "more"})

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        trace: Final = await LensTraceStorage(RemoteTraceStore(client)).get_trace("t1", scope, "run", cursor, page_size)
    assert trace is not None
    assert trace["next_cursor"] == "more"
    assert trace["spans"] == ()
    assert trace["summary"]["span_count"] == 0


async def test_storage_validates_the_remote_query_help_value() -> None:
    async with httpx.AsyncClient(
        base_url="http://lens", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=QUERY_HELP))
    ) as client:
        assert await LensTraceStorage(RemoteTraceStore(client)).query_help(
            {"kind": "all"}, "secret"
        ) == TraceQueryHelp.model_validate(QUERY_HELP)


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
async def test_storage_rejects_remote_query_help_that_drifts_from_the_contract(drift: Mapping[str, object]) -> None:
    async with httpx.AsyncClient(
        base_url="http://lens",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={**QUERY_HELP, **drift})),
    ) as client:
        with pytest.raises(RuntimeError, match="invalid response"):
            await LensTraceStorage(RemoteTraceStore(client)).query_help({"kind": "all"}, "secret")


_LENS_URL: Final = "http://lens.test"


def _spans(resources: Iterable[ResourceSpans]) -> tuple[Span, ...]:
    scopes: Final = chain.from_iterable(resource.scope_spans for resource in resources)
    return tuple(chain.from_iterable(scope.spans for scope in scopes))


class _Upload(NamedTuple):
    path: str
    authorization: str
    body: bytes


@pytest.fixture
def lens_uploads(monkeypatch: pytest.MonkeyPatch) -> list[_Upload]:
    """Every OTLP export the gateway makes, taken at the HTTP library so nothing leaves the process."""
    uploads: Final[list[_Upload]] = []  # mutable-ok: collects what the exporter thread sends during the test

    def send(_session: requests.Session, request: requests.PreparedRequest, **_options: object) -> requests.Response:
        raw: Final = request.body if isinstance(request.body, bytes) else b""
        body: Final = gzip.decompress(raw) if request.headers.get("content-encoding") == "gzip" else raw
        uploads.append(_Upload(urlsplit(request.url or "").path, request.headers.get("authorization", ""), body))
        response: Final = requests.Response()
        response.status_code = 200
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setenv("LITELLM_LENS_URL", _LENS_URL)
    monkeypatch.setenv("LITELLM_LENS_SERVICE_TOKEN", "test-service-token-with-32-characters")
    return uploads


def _in_process_lens(connection: LensConnection) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=connection.url, transport=httpx.MockTransport(lambda _: httpx.Response(200)))


@pytest.mark.asyncio
async def test_connected_gateway_sends_lens_no_spans_of_its_own_for_a_model_call(
    monkeypatch: pytest.MonkeyPatch, lens_uploads: list[_Upload]
) -> None:
    import litellm
    from litellm.integrations.otel.logger import OpenTelemetryV2
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "open_telemetry_logger", None)

    async with manage_tracing(
        True, lambda: TraceReceiver(MagicMock(spec=LensTraceStorage)), client_factory=_in_process_lens
    ):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "what is the capital of France"}],
            mock_response="Paris",
            metadata={"user_api_key_user_id": "user-1"},
        )
        await asyncio.sleep(0.2)
        assert not any(isinstance(callback, OpenTelemetryV2) for callback in litellm.callbacks)

    assert tuple(path for path, _, _ in lens_uploads if path in ("/v1/traces", "/v1/logs")) == ()

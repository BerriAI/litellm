import asyncio
import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from litellm.rust_bridge.trace.errors import TraceChanged
from litellm.rust_bridge.trace.generated.types import AllQueryScope, TraceScope
from litellm.tracing.remote import LensConnection, RemoteTraceStore, bounded_response


@pytest.mark.parametrize(
    "url", ("", "ftp://lens", "http://", "http://user:secret@lens", "https://lens?q=1", "https://lens/#x")
)
def test_service_url_rejects_unsupported_or_credential_bearing_destinations(url: str) -> None:
    with pytest.raises(ValueError, match="URL"):
        LensConnection.from_env({"LITELLM_LENS_URL": url, "LITELLM_LENS_SERVICE_TOKEN": "x" * 32})


def test_connection_requires_a_strong_secret_and_preserves_the_configured_prefix() -> None:
    with pytest.raises(ValueError, match="secret"):
        LensConnection.from_env({"LITELLM_LENS_URL": "https://lens", "LITELLM_LENS_SERVICE_TOKEN": "short"})
    connection: Final = LensConnection.from_env(
        {"LITELLM_LENS_URL": "https://lens/prefix/", "LITELLM_LENS_SERVICE_TOKEN": "x" * 32}
    )
    assert connection.url == "https://lens/prefix"
    assert "x" * 32 not in repr(connection)


async def _read_case(
    store: RemoteTraceStore, operation: str, scope: TraceScope, query_scope: AllQueryScope
) -> tuple[JsonValue, Mapping[str, object]]:
    match operation:
        case "list":
            return (
                await store.list_traces(scope, 10, 20, "next", 17),
                {
                    "operation": operation,
                    "scope": scope,
                    "start_ms": 10,
                    "end_ms": 20,
                    "cursor": "next",
                    "limit": 17,
                },
            )
        case "trace":
            return (
                await store.get_trace("trace", scope, "ref", "next", 17),
                {
                    "operation": operation,
                    "scope": scope,
                    "trace_id": "trace",
                    "trace_ref": "ref",
                    "cursor": "next",
                    "page_size": 17,
                },
            )
        case "span":
            return (
                await store.get_span("trace", "span", scope, "ref"),
                {
                    "operation": operation,
                    "scope": scope,
                    "trace_id": "trace",
                    "trace_ref": "ref",
                    "span_id": "span",
                },
            )
        case "span_error":
            return (
                await store.get_span_error("trace", "span", scope, "ref", "next"),
                {
                    "operation": operation,
                    "scope": scope,
                    "trace_id": "trace",
                    "trace_ref": "ref",
                    "span_id": "span",
                    "cursor": "next",
                },
            )
        case "sql":
            return (
                json.loads(await store.query_sql("SELECT 1", query_scope, "unused-local-secret")),
                {"operation": operation, "scope": query_scope, "sql": "SELECT 1"},
            )
        case "help":
            return (
                await store.query_help(query_scope, "unused-local-secret"),
                {"operation": operation, "scope": query_scope},
            )
        case _:
            return (
                json.loads(await store.query("lens_sample", {"source": "traces"})),
                {"operation": operation, "name": "lens_sample", "parameters": {"source": "traces"}},
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("list", "trace", "span", "span_error", "sql", "help", "query"))
async def test_remote_reads_preserve_scope_and_pagination(operation: str) -> None:
    scope: Final = TraceScope(all_teams=0, user_id="owner", team_ids=("team",))
    query_scope: Final = AllQueryScope(kind="all")
    requests: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(200, json={"data": [{"value": "safe"}]})

    async with httpx.AsyncClient(base_url="http://lens/prefix/", transport=httpx.MockTransport(accept)) as client:
        store: Final = RemoteTraceStore(client)
        result, expected = await _read_case(store, operation, scope, query_scope)
        request: Final = requests.get_nowait()
        assert request.url.path == "/prefix/internal/read"
        assert json.loads(request.content) == json.loads(json.dumps(expected))
        assert result == {"data": [{"value": "safe"}]}
        assert b"unused-local-secret" not in request.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error",
    ((400, ValueError), (409, TraceChanged), (413, OverflowError), (503, RuntimeError), (302, RuntimeError)),
)
async def test_remote_failures_preserve_public_error_categories_without_leaking_storage_details(
    status: int, error: type[Exception]
) -> None:
    async with httpx.AsyncClient(
        base_url="http://lens",
        transport=httpx.MockTransport(lambda request: httpx.Response(status, text="private storage credentials")),
    ) as client:
        with pytest.raises(error) as failure:
            await RemoteTraceStore(client).get_trace("trace", TraceScope(all_teams=1, user_id="", team_ids=()), "ref")
        assert "private storage credentials" not in str(failure.value)


@pytest.mark.asyncio
async def test_network_failure_is_retryable_without_exposing_the_remote_url() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private storage credentials", request=request)

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(RuntimeError, match="Lens trace storage is unavailable"):
            await RemoteTraceStore(client).query_help(AllQueryScope(kind="all"), "secret")


@pytest.mark.asyncio
async def test_reads_reject_oversized_responses_and_invalid_json() -> None:
    with pytest.raises(RuntimeError, match="size limit"):
        await bounded_response(httpx.Response(200, content=b"abcd"), 3)
    assert await bounded_response(httpx.Response(200, content=b"abcd"), 4) == b"abcd"
    async with httpx.AsyncClient(
        base_url="http://lens", transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"{"))
    ) as client:
        with pytest.raises(ValueError, match="Invalid Lens response"):
            await RemoteTraceStore(client).query_help(AllQueryScope(kind="all"), "secret")


@pytest.mark.asyncio
async def test_gateway_cannot_relay_otlp_or_write_arbitrary_tables() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No network access is allowed for schema setup or refused uploads")

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(fail)) as client:
        store: Final = RemoteTraceStore(client)
        await store.ensure_schema()
        with pytest.raises(RuntimeError, match="directly"):
            await store.ingest(b"{}", "application/json", {})
        with pytest.raises(ValueError, match="request records and feedback"):
            await store.insert_rows("otel_traces", ())


@pytest.mark.asyncio
async def test_request_records_use_the_internal_service_endpoint() -> None:
    requests: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        await RemoteTraceStore(client).insert_rows("spend_logs", ({"request_id": "r"},))
    request: Final = requests.get_nowait()
    assert request.url.path == "/internal/spend"
    assert json.loads(request.content) == [{"request_id": "r"}]


@pytest.mark.asyncio
async def test_feedback_rows_use_the_internal_feedback_endpoint() -> None:
    requests: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        await RemoteTraceStore(client).insert_rows("lens_feedback", ({"TraceId": "t", "Score": 2},))
    request: Final = requests.get_nowait()
    assert request.url.path == "/internal/feedback"
    assert json.loads(request.content) == [{"TraceId": "t", "Score": 2}]

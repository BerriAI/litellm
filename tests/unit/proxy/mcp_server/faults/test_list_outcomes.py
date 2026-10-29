"""Classification and rendering matrix for per-server tools/list outcomes: every failure mode maps
to exactly one category, wire values never carry upstream prose, and single-upstream HTTP statuses
stay truthful to who failed."""

import sys
from typing import Final

if sys.version_info < (3, 11):  # BaseExceptionGroup is a builtin only from 3.11
    from exceptiongroup import BaseExceptionGroup

import httpx
import httpx2
import pytest
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED, ErrorData

from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.mcp_server.exceptions import (
    MCPServerListError,
    MCPUpstreamAuthError,
)
from litellm.proxy.mcp_server.faults.list_outcomes import (
    ServerListFault,
    ServerListOk,
    classify_list_exception,
    list_fault_http_status,
    outcome_wire_value,
)


def test_carried_fault_passes_through():
    fault = ServerListFault(tag="timeout")
    assert classify_list_exception(MCPServerListError(fault, "srv")) is fault


def test_upstream_auth_error_maps_to_auth_required_and_forbidden():
    assert classify_list_exception(MCPUpstreamAuthError(401, None, "srv")).tag == "auth_required"
    assert classify_list_exception(MCPUpstreamAuthError(403, None, "srv")).tag == "forbidden"


def test_proxy_rate_limit_error_maps_to_rate_limited() -> None:
    fault: Final = classify_list_exception(ProxyRateLimitError(detail="server RPM exceeded"))
    assert fault == ServerListFault(tag="rate_limited", status_code=429)


def test_timeout_and_connection_errors_classify_without_status():
    assert classify_list_exception(TimeoutError()).tag == "timeout"
    assert classify_list_exception(ConnectionError()).tag == "unreachable"


def test_upstream_json_rpc_error_code_is_never_read_as_an_http_status():
    """JSON-RPC error codes and HTTP status codes are different namespaces, so an upstream is free
    to answer with application code 408. Classifying that number as a gateway timeout would report
    a 504 the gateway never caused. A client timeout reaches here already expressed as a
    ``TimeoutError``, so this taxonomy never has to read the code to tell them apart."""
    upstream_error = MCPError(code=int(httpx.codes.REQUEST_TIMEOUT), message="re-authenticate and retry")
    assert classify_list_exception(upstream_error).tag != "timeout"
    assert list_fault_http_status(classify_list_exception(upstream_error)) != 504


def test_embedded_upstream_response_status_wins():
    response = httpx.Response(503, request=httpx.Request("POST", "https://mcp.example.com/mcp"))
    exc = httpx.HTTPStatusError("boom", request=response.request, response=response)
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = exc
    fault = classify_list_exception(wrapped)
    assert fault.tag == "upstream_error"
    assert fault.status_code == 503


def test_embedded_401_classifies_auth_required():
    response = httpx.Response(401, request=httpx.Request("POST", "https://mcp.example.com/mcp"))
    exc = httpx.HTTPStatusError("no", request=response.request, response=response)
    assert classify_list_exception(exc).tag == "auth_required"


def test_context_response_does_not_shadow_the_causal_chain_response():
    real_response = httpx.Response(401, request=httpx.Request("POST", "https://mcp.example.com/mcp"))
    real = httpx.HTTPStatusError("upstream rejected", request=real_response.request, response=real_response)
    incidental_response = httpx.Response(500, request=httpx.Request("POST", "https://hooks.example.com/log"))
    incidental = httpx.HTTPStatusError(
        "logging hook failed", request=incidental_response.request, response=incidental_response
    )
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = real
    wrapper.__context__ = incidental
    fault = classify_list_exception(wrapper)
    assert fault.tag == "auth_required"
    assert fault.status_code == 401


def test_exception_group_members_are_searched_in_raise_order():
    first_response = httpx.Response(502, request=httpx.Request("POST", "https://mcp.example.com/mcp"))
    first = httpx.HTTPStatusError("first", request=first_response.request, response=first_response)
    second_response = httpx.Response(503, request=httpx.Request("POST", "https://mcp.example.com/mcp"))
    second = httpx.HTTPStatusError("second", request=second_response.request, response=second_response)
    fault = classify_list_exception(BaseExceptionGroup("task group", [first, second]))
    assert fault.tag == "upstream_error"
    assert fault.status_code == 502


def test_unknown_exception_is_internal():
    assert classify_list_exception(ValueError("who knows")).tag == "internal"


@pytest.mark.parametrize("sdk_wrapped", (False, True))
@pytest.mark.parametrize("nested", (False, True))
def test_sse_event_limit_is_a_safe_upstream_fault(sdk_wrapped: bool, nested: bool) -> None:
    limit: Final = 1024
    decoder_message: Final = f"Server-sent event exceeded the {limit} byte limit."
    failure: Final = (
        MCPError(code=CONNECTION_CLOSED, message=f"SSE stream failed: {decoder_message}")
        if sdk_wrapped else httpx2.SSEError(decoder_message)
    )
    wrapped: Final = BaseExceptionGroup("private-upstream-secret", [failure]) if nested else failure
    fault: Final = classify_list_exception(wrapped)
    message: Final = (
        f"Upstream MCP SSE event exceeded the {limit} byte per-event limit. "
        "Paginate tools/list responses so each event stays below the limit."
    )
    assert fault == ServerListFault(tag="upstream_error", sse_event_limit_bytes=limit)
    assert fault.public_message == message
    assert list_fault_http_status(fault) == 502
    assert outcome_wire_value(fault) == {
        "status": "upstream_error", "reason": "sse_event_too_large", "limit_bytes": limit, "message": message,
    }


@pytest.mark.parametrize(
    "failure",
    (
        MCPError(code=CONNECTION_CLOSED, message="Connection closed"),
        MCPError(code=-32603, message="SSE stream failed: Server-sent event exceeded the 1024 byte limit."),
        MCPError(code=CONNECTION_CLOSED, message="SSE stream failed: Server-sent event exceeded the 0 byte limit."),
        MCPError(code=CONNECTION_CLOSED, message="SSE stream failed: Server-sent event exceeded the -1 byte limit."),
        MCPError(code=CONNECTION_CLOSED, message=(
            "SSE stream failed: Server-sent event exceeded the 10000000000000000000 byte limit."
        )),
        MCPError(code=CONNECTION_CLOSED, message=(
            "SSE stream failed: Server-sent event exceeded the 1024 byte limit. private-secret"
        )),
        httpx2.SSEError("private-secret Server-sent event exceeded the 1024 byte limit."),
        httpx2.SSEError("Expected response with content type 'text/event-stream', got 'private-secret'."),
        RuntimeError("SSE stream failed: Server-sent event exceeded the 1024 byte limit."),
    ),
)
def test_unrelated_errors_do_not_claim_an_sse_event_limit(failure: Exception) -> None:
    fault: Final = classify_list_exception(failure)
    assert fault.sse_event_limit_bytes is None
    assert fault.public_message is None
    assert "private-secret" not in str(outcome_wire_value(fault))


@pytest.mark.parametrize("status_code", (401, 403))
def test_upstream_auth_takes_precedence_over_sse_event_limit(status_code: int) -> None:
    request: Final = httpx2.Request("POST", "https://upstream.example/mcp")
    response: Final = httpx2.Response(status_code, request=request)
    failure: Final = BaseExceptionGroup("upstream errors", [
        httpx2.SSEError("Server-sent event exceeded the 1024 byte limit."),
        httpx2.HTTPStatusError("private-secret", request=request, response=response),
    ])
    fault: Final = classify_list_exception(failure)
    assert fault.tag == ("auth_required" if status_code == 401 else "forbidden")
    assert list_fault_http_status(fault) == status_code
    assert fault.sse_event_limit_bytes is None


def test_wire_value_carries_no_prose():
    fault = ServerListFault(tag="upstream_error", status_code=500)
    assert outcome_wire_value(fault) == {"status": "upstream_error", "http_status": 500}
    assert outcome_wire_value(ServerListOk(tool_count=7)) == {"status": "ok", "tool_count": 7}
    assert outcome_wire_value(ServerListFault(tag="timeout")) == {"status": "timeout"}
    assert outcome_wire_value(ServerListFault(tag="rate_limited", status_code=429)) == {
        "status": "rate_limited",
        "http_status": 429,
    }


@pytest.mark.parametrize(
    "tag,status_code,expected",
    [
        ("auth_required", 401, 401),
        ("auth_required", None, 401),
        ("forbidden", 403, 403),
        ("rate_limited", 429, 429),
        ("timeout", None, 504),
        ("unreachable", None, 502),
        ("upstream_error", 500, 502),
        ("internal", None, 500),
    ],
)
def test_single_upstream_http_status_is_truthful(tag, status_code, expected):
    assert list_fault_http_status(ServerListFault(tag=tag, status_code=status_code)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_message", [None, "client disconnected"])
async def test_cancelled_fetch_propagates_instead_of_masquerading_as_a_healthy_empty_server(cancel_message: str | None):
    """A cancelled per-server fetch must not masquerade as ok(tool_count=0), and it is not an upstream
    fault either: the caller cancelled it (a sibling server already answered, or the request went
    away), so the cancellation propagates instead of being logged as a listing failure."""
    import asyncio
    from unittest.mock import MagicMock

    from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager

    manager = MCPServerManager()
    client = MagicMock()
    started = asyncio.Event()

    async def list_tools(raise_on_error: bool = False) -> list[object]:
        started.set()
        await asyncio.Event().wait()
        return []

    client.list_tools = list_tools
    listing = asyncio.create_task(manager._fetch_tools_with_timeout(client, "cancelled_srv"))
    await started.wait()
    listing.cancel(cancel_message)

    with pytest.raises(asyncio.CancelledError):
        await listing


@pytest.mark.asyncio
async def test_client_side_cancelled_error_is_a_classified_fault_not_a_cancellation():
    """The MCP client raises its own CancelledError when a session's teardown outlives its cleanup
    budget while the listing task keeps running. That is one server's fault, so it classifies as an
    internal listing failure instead of cancelling the aggregate listing around it."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from litellm.mcp_client.client import SESSION_CLEANUP_TIMEOUT_MESSAGE
    from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager

    manager = MCPServerManager()
    client = MagicMock()
    client.list_tools = AsyncMock(side_effect=asyncio.CancelledError(SESSION_CLEANUP_TIMEOUT_MESSAGE))

    with pytest.raises(MCPServerListError) as exc_info:
        await manager._fetch_tools_with_timeout(client, "slow_teardown_srv")

    assert exc_info.value.fault.tag == "internal"


def test_auth_challenge_and_status_come_from_the_causal_response():
    """An incidental 403 raised while handling the causal 401 (context chain) must not shadow it:
    the carrier channel and the challenge both derive from the response on the explicit causal
    chain, so the caller is challenged to authenticate rather than told it is forbidden."""
    from litellm.proxy.mcp_server.faults.list_outcomes import upstream_auth_challenge

    causal = httpx.HTTPStatusError(
        "auth",
        request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        response=httpx.Response(
            401,
            headers={"www-authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known"'},
            request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        ),
    )
    incidental = httpx.HTTPStatusError(
        "hook",
        request=httpx.Request("POST", "https://hook.example.com/log"),
        response=httpx.Response(403, request=httpx.Request("POST", "https://hook.example.com/log")),
    )
    wrapper = RuntimeError("fetch failed")
    wrapper.__cause__ = causal
    wrapper.__context__ = incidental

    result = upstream_auth_challenge(wrapper)
    assert result is not None
    status_code, challenge = result
    assert status_code == 401
    assert challenge == 'Bearer resource_metadata="https://mcp.example.com/.well-known"'


def test_raise_classified_list_failure_routes_auth_to_upstream_auth_error():
    """The single choice-point sends 401/403 through MCPUpstreamAuthError with the upstream's own
    challenge and everything else through MCPServerListError, so fetch sites cannot drift."""
    from litellm.proxy.mcp_server.faults.list_outcomes import raise_classified_list_failure

    auth_exc = httpx.HTTPStatusError(
        "auth",
        request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        response=httpx.Response(
            401,
            headers={"www-authenticate": "Bearer realm=x"},
            request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        ),
    )
    with pytest.raises(MCPUpstreamAuthError) as auth_info:
        raise_classified_list_failure(auth_exc, "srv")
    assert auth_info.value.status_code == 401
    assert auth_info.value.www_authenticate == "Bearer realm=x"

    with pytest.raises(MCPServerListError) as fault_info:
        raise_classified_list_failure(RuntimeError("boom"), "srv")
    assert fault_info.value.fault.tag == "internal"


def test_causal_auth_behind_unrelated_response_is_still_found():
    """The auth scan must not end at the first response of any status: a causal 401 sitting deeper
    in the tree than an unrelated 5xx (retry attempts, multi-stream task groups) must still surface
    with its challenge, or the client is told upstream_error and never re-authenticates."""
    from litellm.proxy.mcp_server.faults.list_outcomes import upstream_auth_challenge

    deep_auth = httpx.HTTPStatusError(
        "auth",
        request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        response=httpx.Response(
            401,
            headers={"www-authenticate": "Bearer realm=upstream"},
            request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        ),
    )
    earlier_5xx = httpx.HTTPStatusError(
        "flaky attempt",
        request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        response=httpx.Response(500, request=httpx.Request("POST", "https://mcp.example.com/mcp")),
    )
    earlier_5xx.__cause__ = deep_auth
    wrapper = RuntimeError("fetch failed")
    wrapper.__cause__ = earlier_5xx

    result = upstream_auth_challenge(wrapper)
    assert result is not None
    assert result == (401, "Bearer realm=upstream")


def test_classification_agrees_with_auth_scan_on_nested_auth():
    """classify_list_exception derives its auth arm from the same scan as the carrier choice-point,
    so a nested 401 behind a 5xx classifies auth_required, never upstream_error(500)."""
    deep_auth = httpx.HTTPStatusError(
        "auth",
        request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        response=httpx.Response(401, request=httpx.Request("POST", "https://mcp.example.com/mcp")),
    )
    earlier_5xx = httpx.HTTPStatusError(
        "flaky attempt",
        request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        response=httpx.Response(500, request=httpx.Request("POST", "https://mcp.example.com/mcp")),
    )
    earlier_5xx.__cause__ = deep_auth
    wrapper = RuntimeError("fetch failed")
    wrapper.__cause__ = earlier_5xx

    fault = classify_list_exception(wrapper)
    assert fault.tag == "auth_required"
    assert fault.status_code == 401


def test_pure_non_auth_response_still_classifies_upstream_error():
    """With no auth response anywhere in the tree, the first response in deliberate order still
    drives the generic upstream_error classification."""
    exc = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("POST", "https://mcp.example.com/mcp"),
        response=httpx.Response(502, request=httpx.Request("POST", "https://mcp.example.com/mcp")),
    )
    fault = classify_list_exception(exc)
    assert fault.tag == "upstream_error"
    assert fault.status_code == 502


@pytest.mark.parametrize("status, challenge, expected", [
    (403, 'Bearer error="insufficient_scope", scope="tools.write"', "tools.write"),
    (403, 'bEaReR error=insufficient_scope, scope="tools.read tools.write"', "tools.read tools.write"),
    (403, 'Bearer error="insufficient_scope"', ""),
        (403, 'Bearer error="insufficient_scope", scope=', None),
    (403, 'Bearer error="insufficient_scope", scope', None),
    (401, 'Bearer error="insufficient_scope", scope="tools.write"', None),
    (403, 'Bearer error="invalid_token", scope="tools.write"', None),
    (403, '', None),
    (403, 'Basic error="insufficient_scope", scope="tools.write"', None),
    (403, 'Bearer error_description="error=insufficient_scope", scope="tools.write"', None),
    (403, 'Bearer error="insufficient_scope", error="invalid_token"', None),
    (403, 'Bearer error="insufficient_scope", scope="tools.write\r\ninjected"', None),
    (403, 'Bearer error="insufficient_scope", scope="tools.write", Basic realm="other"', None),
])
@pytest.mark.parametrize("client", ["httpx", "httpx2"])
def test_managed_scope_step_up_only_accepts_unambiguous_bearer_challenges(status, challenge, expected, client):
    import httpx2

    from litellm.proxy.mcp_server.faults.list_outcomes import upstream_insufficient_scope

    transport = httpx if client == "httpx" else httpx2
    response = transport.Response(status, headers={"www-authenticate": challenge}, request=transport.Request("POST", "https://peer.test/mcp"))
    cause = transport.HTTPStatusError("upstream rejected", request=response.request, response=response)
    wrapped = RuntimeError("transport failure")
    wrapped.__cause__ = cause
    assert upstream_insufficient_scope(wrapped) == expected


def _rate_limited_error(retry_after: str | None = None) -> httpx.HTTPStatusError:
    request: Final = httpx.Request("POST", "https://mcp.example.com/mcp")
    headers: Final = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx.HTTPStatusError(
        "rate limited",
        request=request,
        response=httpx.Response(429, headers=headers, request=request),
    )


@pytest.mark.asyncio
async def test_retry_upstream_rate_limit_backs_off_then_reraises() -> None:
    from litellm.proxy.mcp_server.faults.list_outcomes import retry_upstream_rate_limit

    sleeps: Final[list[float]] = []
    calls: Final[list[httpx.HTTPStatusError]] = []
    failure: Final = _rate_limited_error()

    async def record(delay: float) -> None:
        sleeps.append(delay)

    async def always_limited() -> None:
        calls.append(failure)
        raise failure

    expected_error: Final = pytest.raises(httpx.HTTPStatusError)
    with expected_error as exc_info:
        await retry_upstream_rate_limit(
            always_limited,
            "srv",
            max_retries=3,
            base_delay=0.5,
            max_delay=5.0,
            sleep=record,
            jitter=lambda: 0.0,
        )

    assert exc_info.value is failure
    assert sleeps == [0.5, 1.0, 2.0]
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_retry_upstream_rate_limit_honors_retry_after_up_to_cap() -> None:
    from litellm.proxy.mcp_server.faults.list_outcomes import retry_upstream_rate_limit

    sleeps: Final[list[float]] = []

    async def record(delay: float) -> None:
        sleeps.append(delay)

    async def always_limited() -> None:
        raise _rate_limited_error("120")

    with pytest.raises(httpx.HTTPStatusError):
        await retry_upstream_rate_limit(
            always_limited,
            "srv",
            max_retries=3,
            base_delay=0.5,
            max_delay=5.0,
            sleep=record,
            jitter=lambda: 0.0,
        )

    assert sleeps == [5.0, 5.0, 5.0]


@pytest.mark.asyncio
async def test_retry_upstream_rate_limit_parses_http_date_retry_after() -> None:
    import email.utils
    import time

    from litellm.proxy.mcp_server.faults.list_outcomes import retry_upstream_rate_limit

    sleeps: Final[list[float]] = []
    attempts: Final[list[None]] = []

    async def record(delay: float) -> None:
        sleeps.append(delay)

    async def limited() -> str:
        attempts.append(None)
        if len(attempts) == 1:
            raise _rate_limited_error(email.utils.formatdate(time.time() + 3, usegmt=True))
        raise _rate_limited_error("abc")

    with pytest.raises(httpx.HTTPStatusError):
        await retry_upstream_rate_limit(
            limited,
            "srv",
            max_retries=3,
            base_delay=0.5,
            max_delay=5.0,
            sleep=record,
            jitter=lambda: 0.0,
        )

    assert 0 < sleeps[0] <= 3.5
    assert sleeps[1:] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_retry_upstream_rate_limit_passes_through_non_429() -> None:
    from litellm.proxy.mcp_server.faults.list_outcomes import retry_upstream_rate_limit

    sleeps: Final[list[float]] = []
    calls = 0
    failure: Final = ValueError("broken")

    async def record(delay: float) -> None:
        sleeps.append(delay)

    async def broken() -> None:
        nonlocal calls
        calls += 1
        raise failure

    expected_error: Final = pytest.raises(ValueError, match="broken")
    with expected_error as exc_info:
        await retry_upstream_rate_limit(
            broken, "srv", max_retries=3, base_delay=0.5, max_delay=5.0, sleep=record, jitter=lambda: 0.0
        )

    assert exc_info.value is failure
    assert calls == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_retry_upstream_rate_limit_negative_max_retries_is_a_single_attempt() -> None:
    from litellm.proxy.mcp_server.faults.list_outcomes import retry_upstream_rate_limit

    sleeps: Final[list[float]] = []
    calls = 0
    failure: Final = _rate_limited_error("0")

    async def record(delay: float) -> None:
        sleeps.append(delay)

    async def always_limited() -> None:
        nonlocal calls
        calls += 1
        raise failure

    expected_error: Final = pytest.raises(httpx.HTTPStatusError)
    with expected_error as exc_info:
        await retry_upstream_rate_limit(
            always_limited,
            "srv",
            max_retries=-1,
            base_delay=0.5,
            max_delay=5.0,
            sleep=record,
            jitter=lambda: 0.0,
        )

    assert exc_info.value is failure
    assert calls == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_retry_deadline_skips_sleep_that_would_pass_deadline() -> None:
    from litellm.proxy.mcp_server.faults.list_outcomes import retry_upstream_rate_limit

    now: Final[list[float]] = [100.0]
    sleeps: Final[list[float]] = []
    calls = 0
    failure: Final = _rate_limited_error()

    async def record(delay: float) -> None:
        sleeps.append(delay)

    async def always_limited() -> None:
        nonlocal calls
        calls += 1
        raise failure

    expected_error: Final = pytest.raises(httpx.HTTPStatusError)
    with expected_error as exc_info:
        await retry_upstream_rate_limit(
            always_limited,
            "srv",
            max_retries=3,
            base_delay=2.0,
            max_delay=5.0,
            sleep=record,
            jitter=lambda: 0.0,
            deadline_seconds=1.0,
            clock=lambda: now[0],
        )

    assert exc_info.value is failure
    assert calls == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_retry_attempt_is_interrupted_at_deadline() -> None:
    import anyio

    from litellm.proxy.mcp_server.faults.list_outcomes import retry_upstream_rate_limit

    calls = 0
    failure: Final = _rate_limited_error("0")

    async def fetch() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        await anyio.sleep(5)

    expected_error: Final = pytest.raises(httpx.HTTPStatusError)
    with anyio.fail_after(5):
        with expected_error as exc_info:
            await retry_upstream_rate_limit(
                fetch, "srv", max_retries=3, base_delay=0.0, max_delay=5.0, deadline_seconds=0.2
            )

    assert exc_info.value is failure


@pytest.mark.asyncio
async def test_retry_deadline_does_not_bound_first_attempt() -> None:
    import anyio

    from litellm.proxy.mcp_server.faults.list_outcomes import retry_upstream_rate_limit

    async def slow() -> str:
        await anyio.sleep(0.15)
        return "tools"

    result: Final = await retry_upstream_rate_limit(
        slow, "srv", max_retries=3, base_delay=0.5, max_delay=5.0, deadline_seconds=0.05
    )

    assert result == "tools"


@pytest.mark.asyncio
async def test_retry_upstream_rate_limit_never_retries_cancellation() -> None:
    import asyncio

    from litellm.proxy.mcp_server.faults.list_outcomes import retry_upstream_rate_limit

    sleeps: Final[list[float]] = []
    calls = 0

    async def record(delay: float) -> None:
        sleeps.append(delay)

    async def cancelled() -> None:
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry_upstream_rate_limit(
            cancelled, "srv", max_retries=3, base_delay=0.5, max_delay=5.0, sleep=record, jitter=lambda: 0.0
        )

    assert calls == 1
    assert sleeps == []


def test_listing_failed_outright_only_when_every_server_faulted_non_auth() -> None:
    from litellm.proxy.mcp_server.faults.list_outcomes import (
        ServerListOk,
        listing_failed_outright,
    )

    assert not listing_failed_outright({})
    assert listing_failed_outright({"a": ServerListFault(tag="upstream_error", status_code=429)})
    assert listing_failed_outright(
        {"a": ServerListFault(tag="timeout"), "b": ServerListFault(tag="rate_limited", status_code=429)}
    )
    assert not listing_failed_outright(
        {"a": ServerListFault(tag="upstream_error", status_code=500), "b": ServerListOk(tool_count=2)}
    )
    assert not listing_failed_outright({"a": ServerListFault(tag="auth_required", status_code=401)})
    assert not listing_failed_outright({"a": ServerListFault(tag="forbidden", status_code=403)})

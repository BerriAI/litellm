"""Classification and rendering matrix for per-server tools/list outcomes: every failure mode maps
to exactly one category, wire values never carry upstream prose, and single-upstream HTTP statuses
stay truthful to who failed."""

import httpx
import pytest

from litellm.proxy._experimental.mcp_server.exceptions import (
    MCPServerListError,
    MCPUpstreamAuthError,
)
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
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


def test_timeout_and_connection_errors_classify_without_status():
    assert classify_list_exception(TimeoutError()).tag == "timeout"
    assert classify_list_exception(ConnectionError()).tag == "unreachable"


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


def test_wire_value_carries_no_prose():
    fault = ServerListFault(tag="upstream_error", status_code=500)
    assert outcome_wire_value(fault) == {"status": "upstream_error", "http_status": 500}
    assert outcome_wire_value(ServerListOk(tool_count=7)) == {"status": "ok", "tool_count": 7}
    assert outcome_wire_value(ServerListFault(tag="timeout")) == {"status": "timeout"}


@pytest.mark.parametrize(
    "tag,status_code,expected",
    [
        ("auth_required", 401, 401),
        ("auth_required", None, 401),
        ("forbidden", 403, 403),
        ("timeout", None, 504),
        ("unreachable", None, 502),
        ("upstream_error", 500, 502),
        ("internal", None, 500),
    ],
)
def test_single_upstream_http_status_is_truthful(tag, status_code, expected):
    assert list_fault_http_status(ServerListFault(tag=tag, status_code=status_code)) == expected


@pytest.mark.asyncio
async def test_cancelled_fetch_is_a_classified_fault_not_a_healthy_empty_server():
    """A cancelled per-server fetch must not masquerade as ok(tool_count=0): cancellation was already
    suppressed before the outcome plumbing existed, so it stays suppressed, but as an internal fault
    the outcome reporting can see."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    manager = MCPServerManager()
    client = MagicMock()
    client.list_tools = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(MCPServerListError) as exc_info:
        await manager._fetch_tools_with_timeout(client, "cancelled_srv")

    assert exc_info.value.fault.tag == "internal"


def test_auth_challenge_and_status_come_from_the_causal_response():
    """An incidental 403 raised while handling the causal 401 (context chain) must not shadow it:
    the carrier channel and the challenge both derive from the response on the explicit causal
    chain, so the caller is challenged to authenticate rather than told it is forbidden."""
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import upstream_auth_challenge

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
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import raise_classified_list_failure

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
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import upstream_auth_challenge

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

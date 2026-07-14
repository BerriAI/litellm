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

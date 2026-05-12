"""
Tests for the dynamic_mcp_route handler in proxy_server.py.

Covers the resolution order:
  1. Registered MCP server alias  → forwards to /mcp/{name}
  2. Comma-separated list          → short-circuits before any DB call;
                                     forwarded to /mcp/{segment}
  3. Toolset name (cached)         → sets toolset scope, forwards to /mcp
  4. MCP access group tag (cached) → forwards to /mcp/{name} when the group
                                     resolves to at least one server
  5. Unknown name                  → 404

Patch targets are at the source modules because dynamic_mcp_route
uses lazy local imports inside the function body.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_MCP_MANAGER = "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager"
_HANDLE_HTTP = (
    "litellm.proxy._experimental.mcp_server.server.handle_streamable_http_mcp"
)
_STREAM_ASGI = "litellm.proxy.proxy_server._stream_mcp_asgi_response"
_PRISMA = "litellm.proxy.proxy_server.prisma_client"
_IS_ACCESS_GROUP = "litellm.proxy.proxy_server._is_mcp_access_group_cached"
_FORWARD = "litellm.proxy.proxy_server._mcp_forward_as_path"


def _make_request(path: str = "/test/mcp"):
    """Minimal fake Starlette Request."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [],
        "query_string": b"",
        "server": ("localhost", 4000),
        "scheme": "http",
    }

    async def receive():
        return {"type": "http.request", "body": b"{}"}

    return Request(scope=scope, receive=receive)


def _fake_server(name: str = "my_server", server_id: str = "server-id-1"):
    s = MagicMock()
    s.name = name
    s.server_id = server_id
    return s


def _fake_toolset(name: str = "my_toolset", toolset_id: str = "ts-1"):
    t = MagicMock()
    t.toolset_id = toolset_id
    t.name = name
    return t


async def _ok_mcp_handle(scope, receive, send):
    """Stub MCP handler that returns HTTP 200."""
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"{}"})


# ---------------------------------------------------------------------------
# 1. Registered MCP server alias
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_mcp_route_resolves_registered_server():
    """When the segment matches a known server alias the request is forwarded
    to /mcp/{name} and the handler returns 200."""
    from starlette.responses import Response

    from litellm.proxy.proxy_server import dynamic_mcp_route

    request = _make_request("/my_server/mcp")
    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=_fake_server("my_server"))

    fake_forward = AsyncMock(return_value=Response(content=b"{}", status_code=200))

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_FORWARD, new=fake_forward),
    ):
        response = await dynamic_mcp_route("my_server", request)

    assert response.status_code == 200
    fake_forward.assert_awaited_once_with("my_server", request)


# ---------------------------------------------------------------------------
# 2. Comma-separated list (short-circuits before toolset DB call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_mcp_route_comma_list_forwarded():
    """A comma-separated segment is forwarded directly without hitting
    the toolset or access-group DB lookups."""
    from starlette.responses import Response

    from litellm.proxy.proxy_server import dynamic_mcp_route

    segment = "github_mcp,zapier"
    request = _make_request(f"/{segment}/mcp")

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=None)

    fake_forward = AsyncMock(return_value=Response(content=b"{}", status_code=200))

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_FORWARD, new=fake_forward),
    ):
        response = await dynamic_mcp_route(segment, request)

    assert response.status_code == 200
    fake_forward.assert_awaited_once_with(segment, request)
    # Toolset lookup must NOT be called for comma names
    fake_mgr.get_toolset_by_name_cached.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Toolset name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_mcp_route_resolves_toolset():
    """When the segment is a toolset name the toolset context var is set
    and the request is forwarded to /mcp (not /mcp/{name})."""
    from litellm.proxy.proxy_server import dynamic_mcp_route

    request = _make_request("/my_toolset/mcp")
    fake_toolset = _fake_toolset("my_toolset", "ts-42")

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=None)
    fake_mgr.get_toolset_by_name_cached = AsyncMock(return_value=fake_toolset)

    captured_toolset_id = None
    captured_scope = {}

    async def fake_stream(fn, scope, receive):
        nonlocal captured_toolset_id
        from litellm.proxy._experimental.mcp_server.server import (
            _mcp_active_toolset_id,
        )

        captured_toolset_id = _mcp_active_toolset_id.get()
        captured_scope.update(scope)

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_PRISMA, new=MagicMock()),
        patch(_STREAM_ASGI, new=AsyncMock(side_effect=fake_stream)),
    ):
        await dynamic_mcp_route("my_toolset", request)

    assert captured_toolset_id == "ts-42"
    assert captured_scope.get("path") == "/mcp"


# ---------------------------------------------------------------------------
# 4. MCP access group tag (cached)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_mcp_route_resolves_access_group():
    """When the segment is an MCP access group the request is forwarded (not 404)."""
    from starlette.responses import Response

    from litellm.proxy.proxy_server import dynamic_mcp_route

    request = _make_request("/dev_group/mcp")

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=None)
    fake_mgr.get_toolset_by_name_cached = AsyncMock(return_value=None)

    fake_forward = AsyncMock(return_value=Response(content=b"{}", status_code=200))

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_PRISMA, new=MagicMock()),
        patch(_IS_ACCESS_GROUP, new=AsyncMock(return_value=True)),
        patch(_FORWARD, new=fake_forward),
    ):
        response = await dynamic_mcp_route("dev_group", request)

    assert response.status_code == 200
    fake_forward.assert_awaited_once_with("dev_group", request)


@pytest.mark.asyncio
async def test_dynamic_mcp_route_access_group_called_with_correct_name():
    """The access group lookup receives exactly the segment from the URL."""
    from starlette.responses import Response

    from litellm.proxy.proxy_server import dynamic_mcp_route

    request = _make_request("/qa_tools/mcp")

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=None)
    fake_mgr.get_toolset_by_name_cached = AsyncMock(return_value=None)

    is_group = AsyncMock(return_value=True)

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_PRISMA, new=MagicMock()),
        patch(_IS_ACCESS_GROUP, new=is_group),
        patch(
            _FORWARD,
            new=AsyncMock(return_value=Response(content=b"{}", status_code=200)),
        ),
    ):
        await dynamic_mcp_route("qa_tools", request)

    is_group.assert_awaited_once_with("qa_tools")


# ---------------------------------------------------------------------------
# 5. Unknown name → 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_mcp_route_unknown_name_returns_404():
    """A segment that is not a server, toolset, or access group → 404."""
    from litellm.proxy.proxy_server import dynamic_mcp_route

    request = _make_request("/does_not_exist/mcp")

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=None)
    fake_mgr.get_toolset_by_name_cached = AsyncMock(return_value=None)

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_PRISMA, new=MagicMock()),
        patch(_IS_ACCESS_GROUP, new=AsyncMock(return_value=False)),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await dynamic_mcp_route("does_not_exist", request)

    assert exc_info.value.status_code == 404
    assert "does_not_exist" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_dynamic_mcp_route_empty_access_group_returns_404():
    """An access group tag that resolves to zero servers still returns 404."""
    from litellm.proxy.proxy_server import dynamic_mcp_route

    request = _make_request("/empty_group/mcp")

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=None)
    fake_mgr.get_toolset_by_name_cached = AsyncMock(return_value=None)

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_PRISMA, new=MagicMock()),
        patch(_IS_ACCESS_GROUP, new=AsyncMock(return_value=False)),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await dynamic_mcp_route("empty_group", request)

    assert exc_info.value.status_code == 404

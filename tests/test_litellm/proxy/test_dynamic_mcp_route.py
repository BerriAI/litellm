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

from unittest.mock import ANY, AsyncMock, MagicMock, patch

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
_USER_API_KEY_CACHE = "litellm.proxy.proxy_server.user_api_key_cache"
_GET_ACCESS_GROUP_SERVERS = (
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp."
    "MCPRequestHandler._get_mcp_servers_from_access_groups"
)
_FORWARD = "litellm.proxy.proxy_server._mcp_forward_as_path"
_RESOLVE_CSV = "litellm.proxy.proxy_server._resolve_mcp_csv_tokens"


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
async def test_dynamic_mcp_route_comma_list_forwarded_when_tokens_resolve():
    """A comma-separated segment is forwarded after every token is resolved as
    a known server / access group. Forwarding uses the deduped, validated
    token list (so unknown / duplicate tokens cannot leak through). The
    toolset DB lookup is bypassed entirely for comma names."""
    from starlette.responses import Response

    from litellm.proxy.proxy_server import dynamic_mcp_route

    segment = "github_mcp,zapier"
    request = _make_request(f"/{segment}/mcp")

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=None)

    fake_forward = AsyncMock(return_value=Response(content=b"{}", status_code=200))
    fake_resolve = AsyncMock(return_value=["github_mcp", "zapier"])

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_RESOLVE_CSV, new=fake_resolve),
        patch(_FORWARD, new=fake_forward),
    ):
        response = await dynamic_mcp_route(segment, request)

    assert response.status_code == 200
    fake_forward.assert_awaited_once_with("github_mcp,zapier", request)
    fake_mgr.get_toolset_by_name_cached.assert_not_called()


@pytest.mark.asyncio
async def test_dynamic_mcp_route_comma_list_returns_404_when_no_tokens_resolve():
    """A comma-separated segment with zero resolved tokens must 404 instead of
    forwarding (downstream filter falls back to full allowed_mcp_servers when
    no token matches, which would silently broaden scope)."""
    from litellm.proxy.proxy_server import dynamic_mcp_route

    segment = "ghost1,ghost2"
    request = _make_request(f"/{segment}/mcp")

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=None)

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_RESOLVE_CSV, new=AsyncMock(return_value=[])),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await dynamic_mcp_route(segment, request)

    assert exc_info.value.status_code == 404
    assert segment in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_dynamic_mcp_route_comma_list_forwards_only_resolved_subset():
    """If only a subset of CSV tokens resolve, the request is forwarded with
    just that subset (so unknown tokens cannot ride along into the downstream
    server filter)."""
    from starlette.responses import Response

    from litellm.proxy.proxy_server import dynamic_mcp_route

    segment = "github_mcp,ghost,zapier"
    request = _make_request(f"/{segment}/mcp")

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=None)

    fake_forward = AsyncMock(return_value=Response(content=b"{}", status_code=200))
    fake_resolve = AsyncMock(return_value=["github_mcp", "zapier"])

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_RESOLVE_CSV, new=fake_resolve),
        patch(_FORWARD, new=fake_forward),
    ):
        await dynamic_mcp_route(segment, request)

    fake_forward.assert_awaited_once_with("github_mcp,zapier", request)


@pytest.mark.asyncio
async def test_resolve_mcp_csv_tokens_dedupes_and_caps():
    """_resolve_mcp_csv_tokens dedupes tokens exact-match (so distinct casings
    are preserved — downstream resolution may be case-sensitive), drops empty
    fragments, and stops looking up after DEFAULT_MCP_NAMESPACE_CSV_MAX_TOKENS
    unique tokens to bound DB / cache fan-out."""
    from litellm.constants import DEFAULT_MCP_NAMESPACE_CSV_MAX_TOKENS
    from litellm.proxy.proxy_server import _resolve_mcp_csv_tokens

    fake_mgr = MagicMock()
    fake_mgr.get_mcp_server_by_name = MagicMock(return_value=_fake_server())

    # "github_mcp" appears twice (once with surrounding whitespace) — must be
    # collapsed to a single entry. "GITHUB_MCP" is a distinct exact token and
    # is kept (downstream resolution may be case-sensitive).
    csv = ",,github_mcp, github_mcp ,GITHUB_MCP," + ",".join(
        f"srv_{i}" for i in range(DEFAULT_MCP_NAMESPACE_CSV_MAX_TOKENS + 5)
    )

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_IS_ACCESS_GROUP, new=AsyncMock(return_value=False)),
    ):
        resolved = await _resolve_mcp_csv_tokens(csv, client_ip=None)

    assert len(resolved) == DEFAULT_MCP_NAMESPACE_CSV_MAX_TOKENS
    assert resolved[0] == "github_mcp"
    assert "GITHUB_MCP" in resolved
    assert resolved.count("github_mcp") == 1


@pytest.mark.asyncio
async def test_resolve_mcp_csv_tokens_drops_unknown_and_resolves_access_groups():
    """Unknown tokens are dropped; access-group tokens are accepted via the
    cached existence helper (no per-call uncached DB hit)."""
    from litellm.proxy.proxy_server import _resolve_mcp_csv_tokens

    fake_mgr = MagicMock()
    # Only "registered_srv" is a known server alias.
    fake_mgr.get_mcp_server_by_name = MagicMock(
        side_effect=lambda name, client_ip=None: (
            _fake_server(name) if name == "registered_srv" else None
        )
    )

    # "dev_group" is a real access group; "ghost" is not.
    is_group = AsyncMock(side_effect=lambda name: name == "dev_group")

    with (
        patch(_MCP_MANAGER, fake_mgr),
        patch(_IS_ACCESS_GROUP, new=is_group),
    ):
        resolved = await _resolve_mcp_csv_tokens(
            "registered_srv,dev_group,ghost", client_ip=None
        )

    assert resolved == ["registered_srv", "dev_group"]
    # Access-group lookup must NOT be called for "registered_srv" (already
    # matched as a server alias) but MUST be called for "dev_group" and
    # "ghost" (the only tokens that fall through to the access-group check).
    assert {call.args[0] for call in is_group.await_args_list} == {
        "dev_group",
        "ghost",
    }


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


@pytest.mark.asyncio
async def test_is_mcp_access_group_cached_caches_positive_result():
    """Known access groups are cached after resolving to one or more servers."""
    from litellm.proxy.proxy_server import _is_mcp_access_group_cached

    fake_cache = MagicMock()
    fake_cache.async_get_cache = AsyncMock(return_value=None)
    fake_cache.async_set_cache = AsyncMock()
    get_access_group_servers = AsyncMock(return_value=["server-id"])

    with (
        patch(_USER_API_KEY_CACHE, new=fake_cache),
        patch(_GET_ACCESS_GROUP_SERVERS, new=get_access_group_servers),
    ):
        result = await _is_mcp_access_group_cached("dev_group")

    assert result is True
    get_access_group_servers.assert_awaited_once_with(["dev_group"])
    fake_cache.async_set_cache.assert_awaited_once_with(
        key="mcp_access_group_exists:dev_group",
        value=True,
        ttl=ANY,
    )


@pytest.mark.asyncio
async def test_is_mcp_access_group_cached_caches_negative_result_briefly():
    """Empty access-group lookups are cached with a short TTL so unauthenticated
    callers cannot force a fresh DB lookup per request for unknown names."""
    from litellm.constants import DEFAULT_MCP_ACCESS_GROUP_NEGATIVE_CACHE_TTL
    from litellm.proxy.proxy_server import _is_mcp_access_group_cached

    fake_cache = MagicMock()
    fake_cache.async_get_cache = AsyncMock(return_value=None)
    fake_cache.async_set_cache = AsyncMock()
    get_access_group_servers = AsyncMock(return_value=[])

    with (
        patch(_USER_API_KEY_CACHE, new=fake_cache),
        patch(_GET_ACCESS_GROUP_SERVERS, new=get_access_group_servers),
    ):
        result = await _is_mcp_access_group_cached("dev_group")

    assert result is False
    get_access_group_servers.assert_awaited_once_with(["dev_group"])
    fake_cache.async_set_cache.assert_awaited_once_with(
        key="mcp_access_group_exists:dev_group",
        value=False,
        ttl=DEFAULT_MCP_ACCESS_GROUP_NEGATIVE_CACHE_TTL,
    )


@pytest.mark.asyncio
async def test_is_mcp_access_group_cached_returns_cached_negative_without_db():
    """A cached False entry short-circuits the DB lookup on subsequent calls."""
    from litellm.proxy.proxy_server import _is_mcp_access_group_cached

    fake_cache = MagicMock()
    fake_cache.async_get_cache = AsyncMock(return_value=False)
    fake_cache.async_set_cache = AsyncMock()
    get_access_group_servers = AsyncMock(return_value=["server-id"])

    with (
        patch(_USER_API_KEY_CACHE, new=fake_cache),
        patch(_GET_ACCESS_GROUP_SERVERS, new=get_access_group_servers),
    ):
        result = await _is_mcp_access_group_cached("never_existed")

    assert result is False
    get_access_group_servers.assert_not_awaited()
    fake_cache.async_set_cache.assert_not_awaited()


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

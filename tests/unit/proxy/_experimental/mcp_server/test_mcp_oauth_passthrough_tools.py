from litellm.proxy._experimental.mcp_server import operations as mcp_operations
"""Unit tests for MCP OAuth passthrough tool-fetch behavior."""

import asyncio
import logging
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


from litellm.proxy._experimental.mcp_server.exceptions import MCPServerListError, MCPUpstreamAuthError
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _extract_upstream_auth_failure,
)
from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def test_extract_upstream_auth_failure_finds_401_in_http_status_error():
    response = httpx.Response(
        status_code=401,
        headers={"www-authenticate": 'Bearer resource_metadata="https://x"'},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    exc = httpx.HTTPStatusError("401", request=response.request, response=response)

    result = _extract_upstream_auth_failure(exc)
    assert result == (401, 'Bearer resource_metadata="https://x"')


def test_extract_upstream_auth_failure_walks_exception_group():
    response = httpx.Response(
        status_code=401,
        headers={"www-authenticate": "Bearer"},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    inner = httpx.HTTPStatusError("401", request=response.request, response=response)

    try:
        raise ExceptionGroup("wrapped", [inner])
    except Exception as group:
        result = _extract_upstream_auth_failure(group)

    assert result == (401, "Bearer")


def test_extract_upstream_auth_failure_returns_none_for_non_auth():
    assert _extract_upstream_auth_failure(RuntimeError("boom")) is None


def _auth_status_error(status_code: int, www_authenticate: str) -> httpx.HTTPStatusError:
    response = httpx.Response(
        status_code=status_code,
        headers={"www-authenticate": www_authenticate},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    return httpx.HTTPStatusError(str(status_code), request=response.request, response=response)


def test_extract_upstream_auth_failure_finds_401_behind_cause_chain():
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = _auth_status_error(401, "Bearer")
    assert _extract_upstream_auth_failure(wrapper) == (401, "Bearer")


def test_extract_upstream_auth_failure_finds_401_behind_context_chain():
    wrapper = RuntimeError("wrapped")
    wrapper.__context__ = _auth_status_error(401, "Bearer")
    assert _extract_upstream_auth_failure(wrapper) == (401, "Bearer")


def test_extract_upstream_auth_failure_prefers_causal_chain_over_context():
    """A 403 raised incidentally while handling the real 401 (surviving only as ``__context__``)
    must not shadow the 401 on the explicit ``raise ... from`` chain."""
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = _auth_status_error(401, "Bearer realm=real")
    wrapper.__context__ = _auth_status_error(403, "Bearer realm=incidental")
    assert _extract_upstream_auth_failure(wrapper) == (401, "Bearer realm=real")


@pytest.mark.asyncio
async def test_fetch_tools_from_passthrough_raises_on_upstream_401():
    manager = MCPServerManager()
    passthrough_server = MCPServer(
        server_id="p1",
        name="sample_docs",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )

    response = httpx.Response(
        status_code=401,
        headers={"www-authenticate": 'Bearer resource_metadata="https://upstream"'},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    upstream_error = httpx.HTTPStatusError(
        "401", request=response.request, response=response
    )

    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(side_effect=upstream_error)

    with pytest.raises(MCPUpstreamAuthError) as exc_info:
        await manager._fetch_tools_with_timeout(mock_client, passthrough_server.name)

    assert exc_info.value.status_code == 401
    assert exc_info.value.www_authenticate == (
        'Bearer resource_metadata="https://upstream"'
    )
    assert exc_info.value.server_name == "sample_docs"
    mock_client.list_tools.assert_awaited_with(raise_on_error=True)


@pytest.mark.asyncio
async def test_fetch_tools_from_delegated_oauth2_raises_on_upstream_401():
    manager = MCPServerManager()
    delegated_server = MCPServer(
        server_id="oauth1",
        name="delegated_docs",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        delegate_auth_to_upstream=True,
    )

    response = httpx.Response(
        status_code=401,
        headers={"www-authenticate": 'Bearer resource_metadata="https://upstream"'},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    upstream_error = httpx.HTTPStatusError(
        "401", request=response.request, response=response
    )

    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(side_effect=upstream_error)

    with pytest.raises(MCPUpstreamAuthError) as exc_info:
        await manager._fetch_tools_with_timeout(mock_client, delegated_server.name)

    assert exc_info.value.status_code == 401
    assert exc_info.value.www_authenticate == (
        'Bearer resource_metadata="https://upstream"'
    )
    assert exc_info.value.server_name == "delegated_docs"
    mock_client.list_tools.assert_awaited_with(raise_on_error=True)


@pytest.mark.asyncio
async def test_fetch_tools_from_client_credentials_oauth2_surfaces_upstream_401():
    """The auth_type carve-out was removed: a client_credentials (M2M) server now
    surfaces an upstream 401 as MCPUpstreamAuthError too, instead of swallowing it
    to an empty list, so single-server routes can return a 401 challenge."""
    manager = MCPServerManager()
    m2m_server = MCPServer(
        server_id="oauth-m2m",
        name="m2m_docs",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        delegate_auth_to_upstream=True,
        oauth2_flow="client_credentials",
    )

    response = httpx.Response(
        status_code=401,
        headers={"www-authenticate": 'Bearer resource_metadata="https://upstream"'},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    upstream_error = httpx.HTTPStatusError(
        "401", request=response.request, response=response
    )

    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(side_effect=upstream_error)

    with pytest.raises(MCPUpstreamAuthError) as exc_info:
        await manager._fetch_tools_with_timeout(mock_client, m2m_server.name)

    assert exc_info.value.status_code == 401
    assert exc_info.value.server_name == "m2m_docs"
    mock_client.list_tools.assert_awaited_with(raise_on_error=True)


@pytest.mark.asyncio
async def test_fetch_tools_from_passthrough_returns_tools_on_success():
    manager = MCPServerManager()
    passthrough_server = MCPServer(
        server_id="p1",
        name="sample_docs",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization"],
        oauth_passthrough=True,
    )

    tool = MagicMock()
    tool.name = "list_documents"
    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(return_value=[tool])

    tools = await manager._fetch_tools_with_timeout(mock_client, passthrough_server.name)
    assert tools == [tool]


def test_to_http_exception_preserves_upstream_www_authenticate():
    err = MCPUpstreamAuthError(
        status_code=401,
        www_authenticate='Bearer resource_metadata="https://upstream/.well-known/oauth-protected-resource"',
        server_name="sample_docs",
    )

    http_exc = err.to_http_exception()
    assert http_exc.status_code == 401
    assert http_exc.headers == {
        "www-authenticate": 'Bearer resource_metadata="https://upstream/.well-known/oauth-protected-resource"'
    }


def test_to_http_exception_skips_fabrication_when_base_url_missing():
    """Without ``base_url`` we cannot build an RFC 9728 §3.2-compliant absolute
    URI, so we omit the fabricated ``WWW-Authenticate`` challenge entirely
    instead of emitting a relative URI strict clients reject."""
    err = MCPUpstreamAuthError(
        status_code=401,
        www_authenticate=None,
        server_name="sample_docs",
    )

    http_exc = err.to_http_exception()
    assert http_exc.status_code == 401
    assert http_exc.headers is None


def test_to_http_exception_fabricates_absolute_resource_metadata_with_base_url():
    err = MCPUpstreamAuthError(
        status_code=401,
        www_authenticate=None,
        server_name="sample_docs",
    )

    http_exc = err.to_http_exception(base_url="https://gateway.example.com/")
    assert http_exc.status_code == 401
    assert http_exc.headers == {
        "www-authenticate": 'Bearer resource_metadata="https://gateway.example.com/.well-known/oauth-protected-resource/mcp/sample_docs"'
    }


def test_to_http_exception_skips_challenge_for_non_401_status():
    err = MCPUpstreamAuthError(
        status_code=403,
        www_authenticate=None,
        server_name="sample_docs",
    )

    http_exc = err.to_http_exception()
    assert http_exc.status_code == 403
    assert http_exc.headers is None


@pytest.mark.asyncio
async def test_fetch_tools_from_gateway_managed_surfaces_upstream_401():
    """An oauth2 server that is neither pass-through nor delegate now surfaces an
    upstream 401 as MCPUpstreamAuthError as well; the auth_type carve-out that
    swallowed it to [] was removed. A missing upstream WWW-Authenticate is carried
    through as None (the single-server route fabricates one from the gateway URL)."""
    manager = MCPServerManager()
    oauth2_server = MCPServer(
        server_id="o1",
        name="keycloak_whoami",
        url="https://upstream/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
    )

    response = httpx.Response(
        status_code=401,
        headers={},
        request=httpx.Request("GET", "https://upstream/mcp"),
    )
    upstream_error = httpx.HTTPStatusError(
        "401", request=response.request, response=response
    )
    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(side_effect=upstream_error)

    with pytest.raises(MCPUpstreamAuthError) as exc_info:
        await manager._fetch_tools_with_timeout(mock_client, oauth2_server.name)

    assert exc_info.value.status_code == 401
    assert exc_info.value.www_authenticate is None
    assert exc_info.value.server_name == "keycloak_whoami"
    mock_client.list_tools.assert_awaited_with(raise_on_error=True)


def _http_server(server_id: str, name: str, **kwargs) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=name,
        alias=name,
        url=f"https://{name}/mcp",
        transport=MCPTransport.http,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_aggregate_list_tools_absorbs_one_unauthenticated_server():
    """Regression: across the aggregate (/mcp), a delegate/passthrough server that raises
    MCPUpstreamAuthError must not empty every other server's tools. Re-raising it on the
    aggregate path (introduced with the passthrough feature) zeroed the whole list because the
    fan-out gather propagated it. The failed server now contributes an "auth_required" outcome
    instead of vanishing, so it stays distinguishable from a healthy server with no tools."""
    from unittest.mock import patch

    from mcp.types import Tool as MCPTool

    from litellm.proxy._experimental.mcp_server import server as mcp_server
    from litellm.proxy._types import UserAPIKeyAuth

    delegate = _http_server(
        "s1", "delegate_docs", auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True
    )
    working = _http_server("s2", "working_docs", auth_type=MCPAuth.none)
    good_tool = MCPTool(name="working_docs-read", description="d", inputSchema={"type": "object"})

    async def fake_get_tools(server, **kwargs):
        if server.server_id == delegate.server_id:
            raise MCPUpstreamAuthError(status_code=401, www_authenticate=None, server_name=server.name)
        return [good_tool]

    with patch.object(mcp_operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[delegate, working])), patch.object(
        mcp_operations, "_prefetch_oauth_creds_for_user", AsyncMock(return_value={})
    ), patch.object(mcp_operations, "_prepare_mcp_server_headers", MagicMock(return_value=(None, None))), patch.object(
        mcp_operations, "_get_user_oauth_extra_headers_from_db", AsyncMock(return_value=None)
    ), patch.object(
        mcp_operations, "filter_tools_by_key_team_permissions", AsyncMock(side_effect=lambda tools, **k: tools)
    ), patch.object(
        mcp_operations.global_mcp_server_manager, "get_tools_from_server", AsyncMock(side_effect=fake_get_tools)
    ):
        listing = await mcp_operations._get_tools_from_mcp_servers(
            user_api_key_auth=UserAPIKeyAuth(token="h", user_id="u1"),
            mcp_auth_header=None,
            mcp_servers=None,
        )

    assert [t.name for t in listing.tools] == ["working_docs-read"]
    assert listing.outcomes["delegate_docs"].tag == "auth_required"
    assert listing.outcomes["working_docs"].tag == "ok"


@pytest.mark.asyncio
async def test_single_server_route_also_absorbs_upstream_auth_error():
    """A single-server route (/<server>/mcp) absorbs an upstream-auth error just like the aggregate:
    the failing server contributes no tools and an "auth_required" outcome rather than re-raising.
    Surfacing it to the client as a 401 + WWW-Authenticate challenge cannot be done from this list
    handler — the MCP session manager serializes a raise into a JSON-RPC error, not an HTTP 401 — so
    re-auth surfacing is handled by a request-scope preemptive check, tracked separately."""
    from unittest.mock import patch

    from litellm.proxy._experimental.mcp_server import server as mcp_server
    from litellm.proxy._experimental.mcp_server.mcp_context import mcp_gateway_server_name
    from litellm.proxy._types import UserAPIKeyAuth

    delegate = _http_server(
        "s1", "delegate_docs", auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True
    )

    async def fake_get_tools(server, **kwargs):
        raise MCPUpstreamAuthError(status_code=401, www_authenticate=None, server_name=server.name)

    # /<server>/mcp sets the path-derived single-server scope; absorption must hold even then.
    token = mcp_gateway_server_name.set("delegate_docs")
    try:
        with patch.object(mcp_operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[delegate])), patch.object(
            mcp_operations, "_prefetch_oauth_creds_for_user", AsyncMock(return_value={})
        ), patch.object(mcp_operations, "_prepare_mcp_server_headers", MagicMock(return_value=(None, None))), patch.object(
            mcp_operations, "_get_user_oauth_extra_headers_from_db", AsyncMock(return_value=None)
        ), patch.object(
            mcp_operations.global_mcp_server_manager, "get_tools_from_server", AsyncMock(side_effect=fake_get_tools)
        ):
            listing = await mcp_operations._get_tools_from_mcp_servers(
                user_api_key_auth=UserAPIKeyAuth(token="h", user_id="u1"),
                mcp_auth_header=None,
                mcp_servers=["delegate_docs"],
            )
        assert listing.tools == []
        assert listing.outcomes["delegate_docs"].tag == "auth_required"
    finally:
        mcp_gateway_server_name.reset(token)


@pytest.mark.asyncio
async def test_aggregate_with_single_accessible_server_still_absorbs():
    """Regression for the route-misclassification: an aggregate request (/mcp, mcp_servers=None)
    from a key that can access exactly one server must still absorb that server's
    MCPUpstreamAuthError, not surface it. Keying the surface decision off the allowed count rather
    than the request filter would re-raise here and leave the aggregate broken for one-server
    permission sets."""
    from unittest.mock import patch

    from litellm.proxy._experimental.mcp_server import server as mcp_server
    from litellm.proxy._types import UserAPIKeyAuth

    delegate = _http_server(
        "s1", "delegate_docs", auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True
    )

    async def fake_get_tools(server, **kwargs):
        raise MCPUpstreamAuthError(status_code=401, www_authenticate=None, server_name=server.name)

    with patch.object(mcp_operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[delegate])), patch.object(
        mcp_operations, "_prefetch_oauth_creds_for_user", AsyncMock(return_value={})
    ), patch.object(mcp_operations, "_prepare_mcp_server_headers", MagicMock(return_value=(None, None))), patch.object(
        mcp_operations, "_get_user_oauth_extra_headers_from_db", AsyncMock(return_value=None)
    ), patch.object(
        mcp_operations.global_mcp_server_manager, "get_tools_from_server", AsyncMock(side_effect=fake_get_tools)
    ):
        # Aggregate route: no explicit server filter, even though only one server is accessible.
        listing = await mcp_operations._get_tools_from_mcp_servers(
            user_api_key_auth=UserAPIKeyAuth(token="h", user_id="u1"),
            mcp_auth_header=None,
            mcp_servers=None,
        )

    assert listing.tools == []
    assert listing.outcomes["delegate_docs"].tag == "auth_required"


@pytest.mark.asyncio
async def test_fetch_tools_logs_upstream_request_details_on_500(caplog):
    manager = MCPServerManager()
    request = httpx.Request(
        "POST",
        "https://upstream/apis/mcp",
        headers={"Authorization": "Bearer upstream-token-0123456789"},
        content=b'{"method":"initialize","jsonrpc":"2.0","id":0}',
    )
    response = httpx.Response(500, request=request)
    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(
        side_effect=httpx.HTTPStatusError("500", request=request, response=response)
    )

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        with pytest.raises(MCPServerListError):
            await manager._fetch_tools_with_timeout(mock_client, "sample_docs")

    assert "POST https://upstream/ -> HTTP 500" in caplog.text
    assert '"method":"initialize"' in caplog.text
    assert "upstream-token-0123456789" not in caplog.text



@pytest.mark.asyncio
async def test_client_creation_failure_logs_sanitized_exchange(monkeypatch, caplog):
    manager = MCPServerManager()
    server = MCPServer(server_id="sample", name="sample", url="https://upstream/mcp", transport=MCPTransport.http, auth_type=MCPAuth.none)
    request = httpx.Request("POST", "https://upstream/mcp?credential=query-secret")
    response = httpx.Response(500, request=request, json={"error":"missing_scope"})
    error = httpx.HTTPStatusError("query-secret", request=request, response=response)
    monkeypatch.setattr(manager, "create_mcp_client", AsyncMock(side_effect=error))
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        with pytest.raises(MCPServerListError):
            await manager._get_tools_from_server(server)
    assert "POST https://upstream/ -> HTTP 500" in caplog.text
    assert "missing_scope" in caplog.text and "query-secret" not in caplog.text


@pytest.mark.parametrize(
    "oauth_headers,server_headers,authorized",
    [
        ({"Authorization": "Bearer upstream"}, None, True),
        ({"AUTHORIZATION": "Bearer upstream"}, None, True),
        ({"x-unrelated": "present"}, None, False),
        (None, {"catalog": {"Authorization": "Bearer scoped"}}, True),
        (None, {"other-server": {"Authorization": "Bearer unrelated"}}, False),
        (None, {"catalog": {"x-unrelated": "present"}}, False),
        (None, {"catalog": "Bearer legacy"}, True),
        (None, {"catalog": "   "}, False),
    ],
)
def test_passthrough_admission_recognizes_only_matching_authorization(oauth_headers, server_headers, authorized):
    from litellm.proxy._experimental.mcp_server.operations import _client_has_passthrough_authorization
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(server_id="catalog", name="catalog", alias="catalog", transport=MCPTransport.http)
    assert _client_has_passthrough_authorization(server, oauth_headers, server_headers) is authorized


@pytest.mark.parametrize("base_url, request_path, scope, expected", [
    ("https://gateway.test/proxy", "/alias/mcp", "tools.write", 'Bearer resource_metadata="https://gateway.test/proxy/.well-known/oauth-protected-resource/alias/mcp", error="insufficient_scope", scope="tools.write"'),
    ("https://gateway.test/proxy", "/mcp/alias", "", 'Bearer resource_metadata="https://gateway.test/proxy/.well-known/oauth-protected-resource/mcp/alias", error="insufficient_scope"'),
    (None, None, "tools.write", 'Bearer error="insufficient_scope", scope="tools.write"'),
])
def test_managed_step_up_challenge_uses_only_the_gateway_metadata(base_url, request_path, scope, expected):
    error = MCPUpstreamAuthError(status_code=403, www_authenticate=None, server_name="alias", required_scope=scope)
    response = error.to_http_exception(base_url=base_url, request_path=request_path)
    assert response.status_code == 403
    assert response.headers == {"www-authenticate": expected}


@pytest.mark.asyncio
async def test_scope_challenge_replaces_pending_stream_headers_once():
    from litellm.proxy._experimental.mcp_server.scope_challenge import OAuthScopeResponse

    send = AsyncMock()
    response = OAuthScopeResponse(send, "https://gateway.example", "/docs/mcp")
    await response({"type": "http.response.start", "status": 200, "headers": []})
    body = asyncio.create_task(response({"type": "http.response.body", "body": b"event: message\n"}))
    await asyncio.sleep(0)
    send.assert_not_awaited()
    response.deny(MCPUpstreamAuthError(403, None, "docs", required_scope="tools.write"))
    await asyncio.wait_for(body, 1)
    await response({"type": "http.response.body", "body": b"private tool error"})
    assert send.await_count == 2
    start, result = [call.args[0] for call in send.await_args_list]
    assert start["status"] == 403
    challenge = dict(start["headers"])[b"www-authenticate"].decode()
    assert challenge == ('Bearer resource_metadata="https://gateway.example/.well-known/oauth-protected-resource/docs/mcp", '
                         'error="insufficient_scope", scope="tools.write"')
    assert result["body"] == b'{"detail":"Forbidden"}'


@pytest.mark.asyncio
async def test_scope_authorization_releases_progress_without_waiting_for_tool_result():
    import httpx2
    from mcp.server.context import ServerRequestContext
    from starlette.requests import Request

    from litellm.proxy._experimental.mcp_server.mcp_context import active_mcp_request_ctx_var
    from litellm.proxy._experimental.mcp_server.scope_challenge import (
        SCOPE_RESPONSE_KEY,
        OAuthScopeResponse,
    )

    from litellm.proxy._experimental.mcp_server.legacy_callbacks import record_upstream_tool_authorization

    send = AsyncMock()
    gate = OAuthScopeResponse(send, "https://gateway.example", "/mcp/docs")
    request = Request({"type": "http", SCOPE_RESPONSE_KEY: gate})
    context = ServerRequestContext(session=MagicMock(), lifespan_context={}, protocol_version="2025-11-25", method="tools/call", request=request)
    token = active_mcp_request_ctx_var.set(context)
    try:
        await gate({"type": "http.response.start", "status": 200, "headers": []})
        progress = asyncio.create_task(gate({"type": "http.response.body", "body": b"progress", "more_body": True}))
        await asyncio.sleep(0)
        for method, status in [("initialize", 200), ("tools/list", 200), ("tools/call", 403)]:
            await record_upstream_tool_authorization(httpx2.Response(status, request=httpx2.Request("POST", "https://upstream.example/mcp", json={"method": method})))
            assert gate.pending
        unread = httpx2.Request("POST", "https://upstream.example/mcp", content=iter([b'{"method":"tools/call"}']))
        await record_upstream_tool_authorization(httpx2.Response(200, request=unread))
        assert gate.pending
        # Use the actual HTTP response hook contract, before reading the tool response body.
        from litellm.experimental_mcp_client.client import MCPClient

        upstream = MCPClient(server_url="https://upstream.example/mcp")
        async with upstream._create_httpx_client_factory(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, content=b"tool result")))() as client:
            async with client.stream("POST", "https://upstream.example/mcp", json={"method": "tools/call"}):
                await asyncio.wait_for(progress, 1)
                assert send.await_args_list[-1].args[0]["body"] == b"progress"
        gate.deny(MCPUpstreamAuthError(403, None, "docs", required_scope="tools.write"))
        await gate({"type": "http.response.body", "body": b"result"})
        assert send.await_args_list[0].args[0]["status"] == 200
        assert send.await_args_list[-1].args[0]["body"] == b"result"
    finally:
        active_mcp_request_ctx_var.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_scope_middleware_releases_waiting_response_on_validation_or_cancellation(failure):
    from mcp.server.context import ServerRequestContext
    from starlette.requests import Request

    from litellm.proxy._experimental.mcp_server.scope_challenge import SCOPE_RESPONSE_KEY, OAuthScopeResponse, finish_scope_response

    send = AsyncMock()
    gate = OAuthScopeResponse(send, "https://gateway.example", "/mcp/docs")
    context = ServerRequestContext(session=MagicMock(), lifespan_context={}, protocol_version="2025-11-25", method="tools/call", request=Request({"type": "http", SCOPE_RESPONSE_KEY: gate}))
    with pytest.raises(failure):
        await finish_scope_response(context, AsyncMock(side_effect=failure))
    await gate({"type": "http.response.start", "status": 200, "headers": []})
    await gate({"type": "http.response.body", "body": b"error"})
    assert not gate.pending
    assert send.await_args_list[-1].args[0]["body"] == b"error"


@pytest.mark.asyncio
async def test_scope_response_preserves_transport_validation_errors():
    from litellm.proxy._experimental.mcp_server.scope_challenge import OAuthScopeResponse

    send = AsyncMock()
    gate = OAuthScopeResponse(send, "https://gateway.example", "/mcp/docs")
    await gate({"type": "http.response.start", "status": 400, "headers": []})
    await gate({"type": "http.response.body", "body": b"invalid request"})
    assert send.await_args_list[0].args[0]["status"] == 400
    assert send.await_args_list[1].args[0]["body"] == b"invalid request"


@pytest.mark.asyncio
@pytest.mark.parametrize("denied", [False, True])
@pytest.mark.parametrize("stateless", [False, True])
@pytest.mark.parametrize("padding_size", [0, 131072])
async def test_streamable_sdk_relays_scope_challenge_before_committing_http_response(denied, stateless, padding_size):
    import httpx2
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.types import CallToolResult, TextContent

    from litellm.proxy._experimental.mcp_server.legacy_callbacks import get_scope_response
    from litellm.proxy._experimental.mcp_server.mcp_context import active_mcp_request_ctx_var
    from litellm.proxy._experimental.mcp_server.scope_challenge import SCOPE_RESPONSE_KEY, OAuthScopeResponse, finish_scope_response

    async def call_tool(context, params):
        token = active_mcp_request_ctx_var.set(context)
        try:
            gate = get_scope_response()
            assert gate is not None
            if denied:
                gate.deny(MCPUpstreamAuthError(403, None, "docs", required_scope="tools.write"))
            return CallToolResult(content=[TextContent(type="text", text="done")])
        finally:
            active_mcp_request_ctx_var.reset(token)

    server = Server("scope-test", on_call_tool=call_tool)
    server.middleware.append(finish_scope_response)
    manager = StreamableHTTPSessionManager(server, stateless=stateless)

    async def app(scope, receive, send):
        response = OAuthScopeResponse(send, "https://gateway.example", "/mcp/docs")
        scope[SCOPE_RESPONSE_KEY] = response
        await manager.handle_request(scope, receive, response)

    async with manager.run():
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)) as client:
            headers = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-11-25"}
            if not stateless:
                initialized = await asyncio.wait_for(client.post("https://gateway.example/mcp/docs", headers=headers, json={
                    "jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
                        "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "scope-test" + "x" * padding_size, "version": "1"},
                    },
                }), 3)
                assert initialized.status_code == 200
                headers["MCP-Session-Id"] = initialized.headers["mcp-session-id"]
                notified = await client.post("https://gateway.example/mcp/docs", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
                assert notified.status_code == 202
            response = await asyncio.wait_for(client.post("https://gateway.example/mcp/docs", headers=headers,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "write", "arguments": {"padding": "x" * padding_size}}}), 3)
    assert response.status_code == (403 if denied else 200)
    if denied:
        assert 'error="insufficient_scope"' in response.headers["www-authenticate"]
        assert response.json() == {"detail": "Forbidden"}
    else:
        assert '"text":"done"' in response.text
        assert "www-authenticate" not in response.headers


@pytest.mark.asyncio
async def test_scope_challenges_are_isolated_between_concurrent_requests():
    import httpx2
    from mcp.server.context import ServerRequestContext
    from starlette.requests import Request

    from litellm.proxy._experimental.mcp_server.legacy_callbacks import get_scope_response, record_upstream_tool_authorization
    from litellm.proxy._experimental.mcp_server.mcp_context import active_mcp_request_ctx_var
    from litellm.proxy._experimental.mcp_server.scope_challenge import SCOPE_RESPONSE_KEY, OAuthScopeResponse

    async def request_scope(denied):
        send = AsyncMock()
        gate = OAuthScopeResponse(send, "https://gateway.example", "/mcp/docs")
        context = ServerRequestContext(session=MagicMock(), lifespan_context={}, protocol_version="2025-11-25", method="tools/call", request=Request({"type": "http", SCOPE_RESPONSE_KEY: gate}))
        token = active_mcp_request_ctx_var.set(context)
        try:
            await gate({"type": "http.response.start", "status": 200, "headers": []})
            await asyncio.sleep(0)
            assert get_scope_response() is gate
            if denied:
                gate.deny(MCPUpstreamAuthError(403, None, "docs", required_scope="tools.write"))
            else:
                await record_upstream_tool_authorization(httpx2.Response(200, request=httpx2.Request("POST", "https://upstream.example/mcp", json={"method": "tools/call"})))
            await gate({"type": "http.response.body", "body": b"done"})
            return send.await_args_list[0].args[0]["status"]
        finally:
            active_mcp_request_ctx_var.reset(token)

    assert await asyncio.gather(request_scope(True), request_scope(False)) == [403, 200]
    assert get_scope_response() is None


@pytest.mark.parametrize("body", [b"", b"{", b"[]", b'{"method":"initialize"}'])
def test_non_tool_payloads_do_not_hold_response_headers(body):
    from litellm.proxy._experimental.mcp_server.scope_challenge import is_tool_call_request, scope_response_from_request

    assert not is_tool_call_request(body)
    assert scope_response_from_request(None) is None


@pytest.mark.asyncio
async def test_scope_middleware_preserves_non_http_requests():
    from mcp.server.context import ServerRequestContext
    from mcp.types import CallToolResult, TextContent

    from litellm.proxy._experimental.mcp_server.scope_challenge import finish_scope_response

    context = ServerRequestContext(session=MagicMock(), lifespan_context={}, protocol_version="2025-11-25", method="tools/call")
    async def call_next(request):
        return CallToolResult(content=[TextContent(type="text", text=request.method)])

    returned = await finish_scope_response(context, call_next)
    assert returned.content[0].text == "tools/call"

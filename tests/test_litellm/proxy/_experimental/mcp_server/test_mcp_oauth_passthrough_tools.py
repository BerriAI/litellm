from litellm.proxy._experimental.mcp_server import operations as mcp_operations
"""Unit tests for MCP OAuth passthrough tool-fetch behavior."""

import logging
import sys
from typing import Final
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
        mcp_operations.global_mcp_server_manager, "_get_tools_from_server", AsyncMock(side_effect=fake_get_tools)
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
    from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_gateway_server_name
    from litellm.proxy._types import UserAPIKeyAuth

    delegate = _http_server(
        "s1", "delegate_docs", auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True
    )

    async def fake_get_tools(server, **kwargs):
        raise MCPUpstreamAuthError(status_code=401, www_authenticate=None, server_name=server.name)

    # /<server>/mcp sets the path-derived single-server scope; absorption must hold even then.
    token = _mcp_gateway_server_name.set("delegate_docs")
    try:
        with patch.object(mcp_operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[delegate])), patch.object(
            mcp_operations, "_prefetch_oauth_creds_for_user", AsyncMock(return_value={})
        ), patch.object(mcp_operations, "_prepare_mcp_server_headers", MagicMock(return_value=(None, None))), patch.object(
            mcp_operations, "_get_user_oauth_extra_headers_from_db", AsyncMock(return_value=None)
        ), patch.object(
            mcp_operations.global_mcp_server_manager, "_get_tools_from_server", AsyncMock(side_effect=fake_get_tools)
        ):
            listing = await mcp_operations._get_tools_from_mcp_servers(
                user_api_key_auth=UserAPIKeyAuth(token="h", user_id="u1"),
                mcp_auth_header=None,
                mcp_servers=["delegate_docs"],
            )
        assert listing.tools == []
        assert listing.outcomes["delegate_docs"].tag == "auth_required"
    finally:
        _mcp_gateway_server_name.reset(token)


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
        mcp_operations.global_mcp_server_manager, "_get_tools_from_server", AsyncMock(side_effect=fake_get_tools)
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
    monkeypatch.setattr(manager, "_create_mcp_client", AsyncMock(side_effect=error))
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


@pytest.mark.asyncio
async def test_listing_transport_preserves_auth_challenge_before_sse_success() -> None:
    from starlette.exceptions import HTTPException
    from litellm.proxy._experimental.mcp_server.server import MCPAuthResponse

    send: Final = AsyncMock()
    response: Final = MCPAuthResponse(send)
    await response.send(
        {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
    )
    await response.send({"type": "http.response.body", "body": b": ping\r\n\r\n", "more_body": True})
    response.challenge = HTTPException(
        401, "Unauthorized", headers={"WWW-Authenticate": 'Bearer resource_metadata="http://localhost/mcp-metadata"'}
    )
    await response.send(
        {
            "type": "http.response.body",
            "body": b'event: message\r\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\r\n\r\n',
            "more_body": True,
        }
    )
    await response.send({"type": "http.response.body", "body": b"", "more_body": False})
    sent: Final = tuple(call.args[0] for call in send.await_args_list)
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [401]
    assert dict(sent[0]["headers"])[b"www-authenticate"] == b'Bearer resource_metadata="http://localhost/mcp-metadata"'
    assert b'"tools"' not in b"".join(m.get("body", b"") for m in sent)
    assert sent[-1].get("more_body", False) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 400, 403])
async def test_listing_transport_preserves_non_auth_responses(status: int) -> None:
    from litellm.proxy._experimental.mcp_server.server import MCPAuthResponse

    send: Final = AsyncMock()
    response: Final = MCPAuthResponse(send)
    start: Final = {
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json")],
    }
    body: Final = {
        "type": "http.response.body",
        "body": b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}',
        "more_body": False,
    }
    await response.send(start)
    await response.send(body)
    assert tuple(call.args[0] for call in send.await_args_list) == (start, body)


@pytest.mark.asyncio
async def test_protocol_listing_does_not_report_success_when_every_server_requires_auth() -> None:
    from unittest.mock import patch
    from mcp.types import ListToolsRequest
    from litellm.proxy._experimental.mcp_server import operations
    from litellm.proxy._experimental.mcp_server.contracts import OperationContext
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing, ServerListFault
    from litellm.proxy._types import UserAPIKeyAuth

    context: Final = OperationContext(_caller=UserAPIKeyAuth(user_id="reader"))
    listing: Final = AggregateToolListing([], {"github": ServerListFault(tag="auth_required", status_code=401)})
    with patch.object(operations, "_list_mcp_tools", AsyncMock(return_value=listing)):
        with pytest.raises(MCPUpstreamAuthError) as caught:
            await operations.GatewayOperations().execute(ListToolsRequest(), context)
    assert caught.value.status_code == 401
    assert caught.value.server_name == "github"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("tools/list", "prompts/list", "resources/list", "resources/templates/list"))
@pytest.mark.parametrize("protocol", ("2025-06-18", "2025-11-25"))
@pytest.mark.parametrize("json_response", (False, True))
@pytest.mark.parametrize("stateful", (False, True))
@pytest.mark.parametrize("path", ("/mcp", "/github/mcp"))
async def test_streamable_http_listing_returns_late_oauth_challenge(
    monkeypatch: pytest.MonkeyPatch, stateful: bool, path: str, json_response: bool, protocol: str, method: str
) -> None:
    from litellm.proxy._experimental.mcp_server import server
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing, ServerListFault
    from litellm.proxy._types import UserAPIKeyAuth

    auth: Final = UserAPIKeyAuth(api_key="test-owner", user_id="test-user")
    monkeypatch.setattr(
        server, "extract_mcp_auth_context", AsyncMock(return_value=(auth, None, None, None, None, None))
    )
    monkeypatch.setattr(server, "_raise_preemptive_401_for_unauthenticated_servers", AsyncMock())
    monkeypatch.setattr(server, "_check_passthrough_upstream_auth", AsyncMock())
    listing: Final = AsyncMock(
        return_value=AggregateToolListing(
            [],
            {
                "github": ServerListFault(
                    tag="auth_required",
                    status_code=401,
                    www_authenticate='Bearer resource_metadata="http://gateway/.well-known/oauth-protected-resource/mcp/github"',
                )
            },
        )
    )
    if method != "tools/list":
        listing.side_effect = MCPUpstreamAuthError(
            401, 'Bearer resource_metadata="http://gateway/.well-known/oauth-protected-resource/mcp/github"', "github"
        )
    list_function: Final = {
        "tools/list": "_list_mcp_tools",
        "prompts/list": "_list_mcp_prompts",
        "resources/list": "_list_mcp_resources",
        "resources/templates/list": "_list_mcp_resource_templates",
    }[method]
    monkeypatch.setattr(server.operations, list_function, listing)
    monkeypatch.setattr(server.operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[]))
    monkeypatch.setattr(server.operations, "_raise_if_initialize_grants_no_mcp_servers", AsyncMock())
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    monkeypatch.setattr(
        server,
        "session_manager_stateless",
        StreamableHTTPSessionManager(app=server.server, stateless=True, json_response=json_response),
    )
    monkeypatch.setattr(
        server,
        "session_manager_stateful",
        StreamableHTTPSessionManager(app=server.server, stateless=False, json_response=json_response),
    )
    await server.initialize_session_managers()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://gateway"
        ) as client:
            headers: Final = {"accept": "application/json, text/event-stream", "mcp-protocol-version": protocol}
            if stateful:
                initialized: Final = await client.post(
                    path,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 0,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": protocol,
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1"},
                        },
                    },
                )
                assert initialized.status_code == 200, initialized.text
                client.headers["mcp-session-id"] = initialized.headers["mcp-session-id"]
                notification: Final = await client.post(
                    path, headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"}
                )
                assert notification.status_code == 202
            malformed: Final = await client.post(
                path, headers={**headers, "content-type": "application/json"}, content=b"{"
            )
            assert malformed.status_code == 400
            listing.assert_not_awaited()
            response: Final = await client.post(
                path, headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": method}
            )
        assert response.status_code == 401, response.text
        assert (
            response.headers["www-authenticate"]
            == 'Bearer resource_metadata="http://gateway/.well-known/oauth-protected-resource/mcp/github"'
        )
        listing.assert_awaited_once()
    finally:
        await server.shutdown_session_managers()


@pytest.mark.asyncio
async def test_late_auth_failure_does_not_rewrite_committed_stream() -> None:
    from fastapi import HTTPException
    from litellm.proxy._experimental.mcp_server.server import MCPAuthResponse

    send: Final = AsyncMock()
    response: Final = MCPAuthResponse(send)
    start: Final = {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
    progress: Final = {
        "type": "http.response.body",
        "body": b'data: {"method":"notifications/progress"}\n\n',
        "more_body": True,
    }
    error: Final = {
        "type": "http.response.body",
        "body": b'data: {"error":{"code":-32600,"message":"Upstream authorization failed (HTTP 401)"}}\n\n',
        "more_body": False,
    }
    await response.send(start)
    await response.send(progress)
    response.challenge = HTTPException(401, headers={"WWW-Authenticate": "Bearer"})
    await response.send(error)
    assert tuple(call.args[0] for call in send.await_args_list) == (start, progress, error)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("prompts", "resources", "resource_templates"))
async def test_optional_listing_propagates_auth_without_discarding_healthy_servers(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy._experimental.mcp_server import operations

    blocked: Final = _http_server("blocked", "blocked")
    healthy: Final = _http_server("healthy", "healthy")
    fetch: Final = AsyncMock(side_effect=MCPUpstreamAuthError(401, "Bearer", "blocked"))
    manager: Final = MagicMock()
    setattr(manager, f"get_{kind}_from_server", fetch)
    monkeypatch.setattr(operations, "global_mcp_server_manager", manager)
    monkeypatch.setattr(operations, "_prepare_mcp_server_headers", MagicMock(return_value=(None, None)))
    monkeypatch.setattr(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[blocked]))
    listing: Final = getattr(operations, f"_list_mcp_{kind}")
    with pytest.raises(MCPUpstreamAuthError) as failure:
        await listing()
    assert failure.value.www_authenticate == "Bearer"
    fetch.assert_awaited_once()
    fetch.reset_mock(side_effect=True)
    fetch.side_effect = [MCPUpstreamAuthError(401, "Bearer", "blocked"), []]
    monkeypatch.setattr(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[blocked, healthy]))
    assert await listing() == []
    assert fetch.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    (
        "get_prompts_from_server",
        "get_resources_from_server",
        "get_resource_templates_from_server",
        "get_prompt_from_server",
        "read_resource_from_server",
    ),
)
@pytest.mark.parametrize("status", (401, 403))
async def test_manager_preserves_auth_failures_for_prompts_and_resources(
    operation: str, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException
    from pydantic import AnyUrl

    manager: Final = MCPServerManager()
    upstream: Final = _http_server("upstream", "upstream")
    create: Final = AsyncMock(side_effect=HTTPException(status, headers={"WWW-Authenticate": "Bearer"}))
    monkeypatch.setattr(manager, "_create_mcp_client", create)
    kwargs: Final = (
        {"prompt_name": "example"}
        if operation == "get_prompt_from_server"
        else {"url": AnyUrl("https://example.com/resource")}
        if operation == "read_resource_from_server"
        else {}
    )
    with pytest.raises(MCPUpstreamAuthError) as failure:
        await getattr(manager, operation)(server=upstream, user_api_key_auth=None, **kwargs)
    assert failure.value.status_code == status
    assert failure.value.www_authenticate == "Bearer"
    assert failure.value.server_name == "upstream"
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_optional_listing_preserves_cancellation() -> None:
    import asyncio
    from litellm.proxy._experimental.mcp_server.operations import _collect_mcp_listing

    fetch: Final = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _collect_mcp_listing((_http_server("upstream", "upstream"),), fetch)
    fetch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("get_prompt_from_server", "read_resource_from_server"))
async def test_prompt_and_resource_calls_preserve_static_headers_and_non_auth_failures(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import AnyUrl

    manager: Final = MCPServerManager()
    upstream: Final = _http_server("upstream", "upstream", static_headers={"x-upstream": "configured"})
    failure: Final = RuntimeError("Upstream unavailable")
    create: Final = AsyncMock(side_effect=failure)
    monkeypatch.setattr(manager, "_create_mcp_client", create)
    kwargs: Final = (
        {"prompt_name": "example"}
        if operation == "get_prompt_from_server"
        else {"url": AnyUrl("https://example.com/resource")}
    )
    with pytest.raises(RuntimeError) as caught:
        await getattr(manager, operation)(server=upstream, user_api_key_auth=None, **kwargs)
    assert caught.value is failure
    assert create.await_args.kwargs["extra_headers"] == {"x-upstream": "configured"}
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_transport_preserves_sse_priming_event_on_success() -> None:
    from litellm.proxy._experimental.mcp_server.server import MCPAuthResponse

    send: Final = AsyncMock()
    response: Final = MCPAuthResponse(send)
    start: Final = {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
    priming: Final = {"type": "http.response.body", "body": b"id: resume-token\r\ndata: \r\n\r\n", "more_body": True}
    tools: Final = {
        "type": "http.response.body",
        "body": b'data: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n\n',
        "more_body": True,
    }
    await response.send(start)
    await response.send(priming)
    send.assert_not_awaited()
    await response.send(tools)
    assert tuple(call.args[0] for call in send.await_args_list) == (start, priming, tools)


@pytest.mark.asyncio
async def test_tool_call_preserves_resolver_http_challenge(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException
    from mcp.types import CallToolRequest, CallToolRequestParams
    from litellm.proxy._experimental.mcp_server import operations
    from litellm.proxy._experimental.mcp_server.contracts import OperationContext

    challenge: Final = HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})
    call: Final = AsyncMock(side_effect=challenge)
    monkeypatch.setattr(operations, "call_mcp_tool", call)
    with pytest.raises(HTTPException) as failure:
        await operations.GatewayOperations().execute(
            CallToolRequest(params=CallToolRequestParams(name="upstream-tool", arguments={})),
            OperationContext(_caller=None),
        )
    assert failure.value is challenge
    call.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_handler_preserves_unrelated_protocol_errors(
    _mcp_request_ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp import MCPError
    from mcp.types import CallToolRequestParams
    from litellm.proxy._experimental.mcp_server import server

    failure: Final = MCPError(code=-32602, message="Invalid tool parameters")
    execute: Final = AsyncMock(side_effect=failure)
    gateway: Final = MagicMock()
    gateway.execute = execute
    monkeypatch.setattr(server.operations, "GatewayOperations", MagicMock(return_value=gateway))
    monkeypatch.setattr(
        server, "get_or_extract_auth_context", AsyncMock(return_value=(None, None, None, None, None, None, None))
    )
    with pytest.raises(MCPError) as caught:
        await server.mcp_server_tool_call(_mcp_request_ctx(), CallToolRequestParams(name="example", arguments={}))
    assert caught.value is failure
    execute.assert_awaited_once()

import asyncio
import contextvars
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from mcp import ReadResourceResult, Resource
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    Prompt,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
)

from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPTransport,
    UserAPIKeyAuth,
)
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.fixture(autouse=True)
def cleanup_mcp_global_state():
    """Clean up MCP global state before and after each test.

    This fixture ensures test isolation when running with pytest-xdist
    parallel execution. Without this, global_mcp_server_manager state
    can leak between tests causing mock assertion failures.
    """
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        # Clear before test
        global_mcp_server_manager.registry.clear()
        global_mcp_server_manager.tool_name_to_mcp_server_name_mapping.clear()
        yield
        # Clear after test
        global_mcp_server_manager.registry.clear()
        global_mcp_server_manager.tool_name_to_mcp_server_name_mapping.clear()
    except ImportError:
        # MCP not available, skip cleanup
        yield


@pytest.mark.asyncio
async def test_mcp_server_tool_call_body_contains_request_data():
    """Test that proxy_server_request body contains name and arguments"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            mcp_server_tool_call,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Setup test data
    tool_name = "test_tool"
    tool_arguments = {"param1": "value1", "param2": 123}

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # Mock the add_litellm_data_to_request function to capture the data
    captured_data = {}

    async def mock_add_litellm_data_to_request(data, request, user_api_key_dict, proxy_config):
        captured_data.update(data)
        # Simulate the proxy_server_request creation
        captured_data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": {},
            "body": data.copy(),  # This is what we want to test
        }
        return captured_data

    # Mock the call_mcp_tool function to avoid actual tool execution
    async def mock_call_mcp_tool(*args, **kwargs):
        return [{"type": "text", "text": "mocked response"}]

    with patch(
        "litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request",
        mock_add_litellm_data_to_request,
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.call_mcp_tool",
            mock_call_mcp_tool,
        ):
            with patch(
                "litellm.proxy.proxy_server.proxy_config",
                MagicMock(),
            ):
                # Call the function
                await mcp_server_tool_call(tool_name, tool_arguments)

    # Verify the body contains the expected data
    assert "proxy_server_request" in captured_data
    assert "body" in captured_data["proxy_server_request"]

    body = captured_data["proxy_server_request"]["body"]
    assert body["name"] == tool_name
    assert body["arguments"] == tool_arguments


def test_prepare_mcp_server_headers_case_insensitive_extra_headers():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _prepare_mcp_server_headers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    server = MCPServer(
        server_id="server-case",
        name="server",
        transport=MCPTransport.http,
        extra_headers=["Authorization"],
    )

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers={
            "x-litellm-api-key": "Bearer sk-litellm-key",
            "authorization": "Bearer token",
        },
    )

    assert server_auth_header is None
    assert extra_headers == {"Authorization": "Bearer token"}


def test_prepare_mcp_server_headers_passthrough_strips_authorization_without_admission_header():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _prepare_mcp_server_headers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    server = MCPServer(
        server_id="server-passthrough-no-admission",
        name="server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization", "x-request-id"],
        oauth_passthrough=True,
    )

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers={
            "authorization": "Bearer sk-litellm-key",
            "x-request-id": "req-789",
        },
    )

    assert server_auth_header is None
    assert extra_headers == {"x-request-id": "req-789"}


def test_prepare_mcp_server_headers_passthrough_forwards_authorization_for_anonymous_admission():
    """Cold-start return per RFC 9728: client admits anonymously through
    the pass-through fallback in :meth:`MCPRequestHandler.process_mcp_request`
    (``user_api_key_auth.api_key is None``) and the ``Authorization`` bearer
    is the upstream OAuth token — it must be forwarded, not stripped."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _prepare_mcp_server_headers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    from litellm.proxy._types import UserAPIKeyAuth

    server = MCPServer(
        server_id="server-passthrough-anon-admission",
        name="server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization", "x-request-id"],
        oauth_passthrough=True,
    )

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers={
            "authorization": "Bearer upstream-oauth-token",
            "x-request-id": "req-790",
        },
        user_api_key_auth=UserAPIKeyAuth(),
    )

    assert server_auth_header is None
    assert extra_headers == {
        "Authorization": "Bearer upstream-oauth-token",
        "x-request-id": "req-790",
    }


def test_prepare_mcp_server_headers_passthrough_strips_authorization_for_authenticated_admission():
    """When admission validated ``Authorization`` as a LiteLLM key
    (``user_api_key_auth.api_key`` is set, no explicit ``x-litellm-api-key``),
    the bearer must still be stripped to avoid leaking the gateway key
    upstream."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _prepare_mcp_server_headers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    from litellm.proxy._types import UserAPIKeyAuth

    server = MCPServer(
        server_id="server-passthrough-authenticated",
        name="server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization", "x-request-id"],
        oauth_passthrough=True,
    )

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers={
            "authorization": "Bearer sk-litellm-key",
            "x-request-id": "req-791",
        },
        user_api_key_auth=UserAPIKeyAuth(api_key="sk-litellm-key"),
    )

    assert server_auth_header is None
    assert extra_headers == {"x-request-id": "req-791"}


def test_prepare_mcp_server_headers_oauth2_m2m_omits_litellm_caller_authorization():
    """M2M OAuth must not put caller Bearer (LiteLLM API key) into extra_headers (#23652)."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _prepare_mcp_server_headers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    server = MCPServer(
        server_id="m2m-server",
        name="m2m",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow="client_credentials",
        token_url="https://auth.example.com/token",
    )
    caller_key = {"Authorization": "Bearer sk-litellm-caller"}

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=caller_key,
        raw_headers=None,
    )

    assert server_auth_header is None
    assert extra_headers is None


def test_prepare_mcp_server_headers_oauth2_interactive_drops_caller_authorization():
    """A v2-migrated interactive OAuth (authorization_code) server must NOT forward the
    caller's Authorization: the resolver injects the stored per-user token, so a
    caller-supplied bearer must not override another user's stored credential. Non-auth
    headers are still carried; only the credential is dropped."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _prepare_mcp_server_headers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    caller_oauth = {"Authorization": "Bearer caller-supplied-token"}

    server = MCPServer(
        server_id="3lo-server",
        name="3lo",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow=None,
    )

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=caller_oauth,
        raw_headers=None,
    )

    assert server_auth_header is None
    # Caller's Authorization is dropped (only key present) -> extra_headers is None.
    assert extra_headers is None


def test_prepare_mcp_server_headers_m2m_skips_authorization_from_raw_extra_headers():
    """M2M must not merge caller Authorization from raw_headers when extra_headers lists it."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _prepare_mcp_server_headers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    server = MCPServer(
        server_id="m2m-raw",
        name="m2m",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow="client_credentials",
        token_url="https://auth.example.com/token",
        extra_headers=["Authorization", "X-Custom"],
    )

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers={"Authorization": "Bearer sk-1234"},
        raw_headers={
            "authorization": "Bearer sk-1234",
            "x-custom": "trace",
        },
    )

    assert server_auth_header is None
    assert extra_headers is not None
    assert "Authorization" not in extra_headers
    assert extra_headers.get("X-Custom") == "trace"


def _client_forwarded_mode_server(server_id: str, auth_type) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=server_id,
        transport=MCPTransport.http,
        auth_type=auth_type,
    )


def _prepare_headers_in_scope(server: MCPServer, scope_servers):
    from litellm.proxy._experimental.mcp_server.server import (
        _prepare_mcp_server_headers,
    )

    return _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers={"Authorization": "Bearer upstream-token"},
        raw_headers={
            "x-litellm-api-key": "Bearer sk-litellm-key",
            "authorization": "Bearer upstream-token",
        },
        user_api_key_auth=UserAPIKeyAuth(api_key="sk-litellm-key"),
        scope_servers=scope_servers,
    )


def test_prepare_mcp_server_headers_withholds_global_authorization_when_scope_fans_out():
    """One caller bearer must not be replayed against multiple upstreams (RFC 9700
    cross-resource replay): in a fan-out scope with a second Authorization-consuming
    server, the client-forwarded modes get no global Authorization."""
    delegate = _client_forwarded_mode_server("od-fanout", MCPAuth.oauth_delegate)
    second_consumer = _client_forwarded_mode_server("tp-fanout", MCPAuth.true_passthrough)

    _, extra_headers = _prepare_headers_in_scope(delegate, [delegate, second_consumer])

    assert not extra_headers or "authorization" not in {k.lower() for k in extra_headers}


def test_prepare_mcp_server_headers_forwards_global_authorization_to_sole_consumer():
    """Non-consuming servers (static api_key) in scope do not make the forward ambiguous."""
    delegate = _client_forwarded_mode_server("od-sole", MCPAuth.oauth_delegate)
    static_server = MCPServer(
        server_id="static-api-key",
        name="static-api-key",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
        authentication_token="static-key",
    )

    _, extra_headers = _prepare_headers_in_scope(delegate, [delegate, static_server])

    assert extra_headers == {"Authorization": "Bearer upstream-token"}


def test_prepare_mcp_server_headers_scope_counts_legacy_delegate_as_consumer():
    """Legacy upstream-delegated oauth2 servers still receive the caller's Authorization on the
    v1 path, so their presence in scope must suppress the new modes' forward too."""
    delegate = _client_forwarded_mode_server("od-vs-legacy", MCPAuth.oauth_delegate)
    legacy_delegate = MCPServer(
        server_id="legacy-delegate",
        name="legacy-delegate",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        delegate_auth_to_upstream=True,
    )

    _, extra_headers = _prepare_headers_in_scope(delegate, [delegate, legacy_delegate])

    assert not extra_headers or "authorization" not in {k.lower() for k in extra_headers}


def test_prepare_mcp_server_headers_fanout_withhold_survives_extra_headers_loop():
    """Regression: when fan-out withholds the request-wide Authorization from a client-forwarded
    server, the later server.extra_headers copy loop must not re-add it from raw_headers even if
    the server lists Authorization in extra_headers. Otherwise one bearer is replayed across every
    consuming upstream in the scope (the exact cross-resource replay the withholding prevents)."""
    delegate = MCPServer(
        server_id="od-extra-hdr",
        name="od-extra-hdr",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth_delegate,
        extra_headers=["Authorization"],
    )
    second_consumer = _client_forwarded_mode_server("tp-peer", MCPAuth.true_passthrough)

    _, extra_headers = _prepare_headers_in_scope(delegate, [delegate, second_consumer])

    assert not extra_headers or "authorization" not in {k.lower() for k in extra_headers}


def test_prepare_mcp_server_headers_sole_consumer_still_forwards_via_extra_headers():
    """Guard the fix does not over-withhold: with no second consumer in scope, a client-forwarded
    server that lists Authorization in extra_headers still forwards the caller's bearer."""
    delegate = MCPServer(
        server_id="od-extra-sole",
        name="od-extra-sole",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth_delegate,
        extra_headers=["Authorization"],
    )
    static_server = MCPServer(
        server_id="static-peer",
        name="static-peer",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
        authentication_token="static-key",
    )

    _, extra_headers = _prepare_headers_in_scope(delegate, [delegate, static_server])

    assert extra_headers is not None
    assert extra_headers.get("Authorization") == "Bearer upstream-token"


@pytest.mark.asyncio
async def test_call_tool_m2m_skips_authorization_headers():
    """M2M call_tool must not forward caller Authorization in oauth2/raw headers."""
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    manager = MCPServerManager()
    server = MCPServer(
        server_id="m2m-call-tool",
        name="m2m-call-tool",
        server_name="m2m-call-tool",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow="client_credentials",
        token_url="https://auth.example.com/token",
        client_id="cid",
        client_secret="csecret",
        extra_headers=["Authorization", "X-Custom"],
    )

    mock_client = MagicMock()
    mock_client.call_tool = AsyncMock(return_value=MagicMock())

    with patch.object(manager, "_create_mcp_client", new=AsyncMock(return_value=mock_client)) as create_client_mock:
        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="echo",
            arguments={"message": "hello"},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers={"Authorization": "Bearer sk-1234"},
            raw_headers={"authorization": "Bearer sk-1234", "x-custom": "trace"},
            proxy_logging_obj=None,
        )

    create_kwargs = create_client_mock.await_args.kwargs
    extra_headers = create_kwargs["extra_headers"] or {}
    assert "Authorization" not in extra_headers
    assert extra_headers.get("X-Custom") == "trace"


@pytest.mark.asyncio
async def test_get_prompts_from_mcp_servers_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_prompts_from_mcp_servers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")

    server_a = MagicMock(name="server_a_obj")
    server_a.name = "server_a"
    server_a.alias = "server_a"
    server_a.server_name = "server_a"
    server_a.server_id = "a"
    server_a.auth_type = None
    server_a.extra_headers = None

    server_b = MagicMock(name="server_b_obj")
    server_b.name = "server_b"
    server_b.alias = "server_b"
    server_b.server_name = "server_b"
    server_b.server_id = "b"
    server_b.auth_type = None
    server_b.extra_headers = None

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            AsyncMock(return_value=[server_a, server_b]),
        ) as mock_allowed,
        patch(
            "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
            return_value=(None, None),
        ) as mock_headers,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
    ):
        mock_manager.get_prompts_from_server = AsyncMock(
            side_effect=[
                [Prompt(name="hello", description="hi")],
                [Prompt(name="howdy", description="hey")],
            ]
        )

        prompts = await _get_prompts_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    mock_allowed.assert_awaited_once()
    assert mock_headers.call_count == 2
    assert mock_manager.get_prompts_from_server.await_count == 2
    assert {prompt.name for prompt in prompts} == {"hello", "howdy"}


@pytest.mark.asyncio
async def test_get_resources_from_mcp_servers_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_resources_from_mcp_servers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="user")

    server_a = MagicMock(name="server_a_obj")
    server_a.name = "server_a"
    server_a.alias = "server_a"
    server_a.server_name = "server_a"
    server_a.server_id = "a"
    server_a.auth_type = None
    server_a.extra_headers = None

    server_b = MagicMock(name="server_b_obj")
    server_b.name = "server_b"
    server_b.alias = "server_b"
    server_b.server_name = "server_b"
    server_b.server_id = "b"
    server_b.auth_type = None
    server_b.extra_headers = None

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            AsyncMock(return_value=[server_a, server_b]),
        ) as mock_allowed,
        patch(
            "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
            return_value=(None, None),
        ) as mock_headers,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
    ):
        mock_manager.get_resources_from_server = AsyncMock(
            side_effect=[
                [
                    Resource(
                        name="resource_a",
                        uri="https://example.com/a",
                    )
                ],
                [
                    Resource(
                        name="resource_b",
                        uri="https://example.com/b",
                    )
                ],
            ]
        )

        resources = await _get_resources_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    mock_allowed.assert_awaited_once()
    assert mock_headers.call_count == 2
    assert mock_manager.get_resources_from_server.await_count == 2
    assert {resource.name for resource in resources} == {
        "resource_a",
        "resource_b",
    }


@pytest.mark.asyncio
async def test_get_resource_templates_from_mcp_servers_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_resource_templates_from_mcp_servers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="user")

    server = MagicMock(name="server_obj")
    server.name = "server"
    server.alias = "server"
    server.server_name = "server"
    server.server_id = "server-id"
    server.auth_type = None
    server.extra_headers = None

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            AsyncMock(return_value=[server]),
        ) as mock_allowed,
        patch(
            "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
            return_value=(None, None),
        ) as mock_headers,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
    ):
        mock_manager.get_resource_templates_from_server = AsyncMock(
            return_value=[
                ResourceTemplate(
                    name="template",
                    description="desc",
                    uriTemplate="https://example.com/resource/{id}",
                )
            ]
        )

        templates = await _get_resource_templates_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    mock_allowed.assert_awaited_once()
    mock_headers.assert_called_once()
    mock_manager.get_resource_templates_from_server.assert_awaited_once()
    assert [template.name for template in templates] == ["template"]


@pytest.mark.asyncio
async def test_mcp_get_prompt_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import mcp_get_prompt
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")

    server = MagicMock()
    server.name = "server_a"

    prompt_result = MagicMock(name="prompt_result")

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            AsyncMock(return_value=[server]),
        ) as mock_allowed,
        patch(
            "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
            return_value=({"Authorization": "token"}, {"X-Test": "1"}),
        ) as mock_headers,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
    ):
        mock_manager.get_prompt_from_server = AsyncMock(return_value=prompt_result)

        result = await mcp_get_prompt(
            name="server_a-hello",  # prefixed name since server prefixes are always added
            arguments={"foo": "bar"},
            user_api_key_auth=user_api_key_auth,
        )

    mock_allowed.assert_awaited_once()
    mock_headers.assert_called_once_with(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers=None,
        user_api_key_auth=user_api_key_auth,
    )
    mock_manager.get_prompt_from_server.assert_awaited_once_with(
        server=server,
        prompt_name="hello",
        arguments={"foo": "bar"},
        mcp_auth_header={"Authorization": "token"},
        extra_headers={"X-Test": "1"},
        raw_headers=None,
    )
    assert result is prompt_result


@pytest.mark.asyncio
async def test_mcp_read_resource_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import mcp_read_resource
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="key", user_id="user")

    server = MagicMock()
    server.name = "server"

    read_result = ReadResourceResult(
        contents=[
            TextResourceContents(
                uri="https://example.com/resource",
                text="hello world",
                mimeType="text/plain",
            )
        ]
    )

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            AsyncMock(return_value=[server]),
        ) as mock_allowed,
        patch(
            "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
            return_value=({"Authorization": "token"}, {"X-Test": "1"}),
        ) as mock_headers,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
    ):
        mock_manager.read_resource_from_server = AsyncMock(return_value=read_result)

        result = await mcp_read_resource(
            url="https://example.com/resource",
            user_api_key_auth=user_api_key_auth,
        )

    mock_allowed.assert_awaited_once()
    mock_headers.assert_called_once_with(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers=None,
        user_api_key_auth=user_api_key_auth,
    )
    mock_manager.read_resource_from_server.assert_awaited_once_with(
        server=server,
        url="https://example.com/resource",
        mcp_auth_header={"Authorization": "token"},
        extra_headers={"X-Test": "1"},
        raw_headers=None,
    )
    assert result is read_result


def test_normalize_resource_contents_passes_metadata():
    """Test that _normalize_resource_contents preserves meta from ResourceContents (MCP 1.26.0+)."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _normalize_resource_contents,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    meta = {"version": "1.0", "source": "test"}
    contents = [
        TextResourceContents(
            uri="https://example.com/resource",
            text="hello world",
            mimeType="text/plain",
            meta=meta,
        )
    ]

    result = _normalize_resource_contents(contents)

    assert len(result) == 1
    assert result[0].content == "hello world"
    assert result[0].mime_type == "text/plain"
    assert result[0].meta == meta


def test_normalize_resource_contents_blob_with_metadata():
    """Test that _normalize_resource_contents preserves meta for BlobResourceContents."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _normalize_resource_contents,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    meta = {"encoding": "base64"}
    contents = [
        BlobResourceContents(
            uri="https://example.com/image.png",
            blob="aGVsbG8=",
            mimeType="image/png",
            meta=meta,
        )
    ]

    result = _normalize_resource_contents(contents)

    assert len(result) == 1
    assert result[0].content == "aGVsbG8="
    assert result[0].mime_type == "image/png"
    assert result[0].meta == meta


def test_normalize_resource_contents_preserves_empty_metadata():
    """Test that empty dict meta is preserved (truthiness bug fix)."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _normalize_resource_contents,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    empty_meta: dict = {}
    contents = [
        TextResourceContents(
            uri="https://example.com/resource",
            text="hi",
            mimeType="text/plain",
            meta=empty_meta,
        )
    ]

    result = _normalize_resource_contents(contents)

    assert len(result) == 1
    assert result[0].meta == empty_meta
    assert result[0].meta is not None
    assert result[0].meta == {}


def test_normalize_resource_contents_without_metadata():
    """Test that _normalize_resource_contents works when meta is absent (backward compat)."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _normalize_resource_contents,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    contents = [
        TextResourceContents(
            uri="https://example.com/resource",
            text="hello",
            mimeType="text/plain",
        )
    ]

    result = _normalize_resource_contents(contents)

    assert len(result) == 1
    assert result[0].content == "hello"
    assert result[0].meta is None


@pytest.mark.asyncio
async def test_mcp_read_resource_multiple_servers_error():
    try:
        from litellm.proxy._experimental.mcp_server.server import mcp_read_resource
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="key", user_id="user")

    server_a = MagicMock()
    server_b = MagicMock()
    server_a.name = "server_a"
    server_b.name = "server_b"

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        AsyncMock(return_value=[server_a, server_b]),
    ) as mock_allowed:
        with pytest.raises(HTTPException) as exc_info:
            await mcp_read_resource(
                url="https://example.com/resource",
                user_api_key_auth=user_api_key_auth,
            )

    mock_allowed.assert_awaited_once()
    assert exc_info.value.status_code == 400
    assert "Multiple MCP servers" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_continues_when_one_server_fails():
    """Test that _get_tools_from_mcp_servers continues when one server fails"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # Mock servers
    working_server = MagicMock()
    working_server.name = "working_server"
    working_server.alias = "working"
    working_server.allowed_tools = None
    working_server.disallowed_tools = None
    working_server.server_id = "working_server"
    working_server.server_name = "working_server"
    working_server.auth_type = None
    working_server.extra_headers = None

    failing_server = MagicMock()
    failing_server.name = "failing_server"
    failing_server.alias = "failing"
    failing_server.allowed_tools = None
    failing_server.disallowed_tools = None
    failing_server.server_id = "failing_server"
    failing_server.server_name = "failing_server"
    failing_server.auth_type = None
    failing_server.extra_headers = None

    # Mock global_mcp_server_manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["working_server", "failing_server"])
    mock_manager.get_mcp_server_by_id = lambda server_id: (
        working_server if server_id == "working_server" else failing_server
    )
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip_with_info = lambda server_ids, client_ip: (
        server_ids,
        0,
    )

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=True,
        raw_headers=None,
        **kwargs,
    ):
        if server.name == "working_server":
            # Working server returns tools
            tool1 = MagicMock()
            tool1.name = "working_tool_1"
            tool1.description = "Working tool 1"
            tool1.inputSchema = {}
            return [tool1]
        else:
            # Failing server raises an exception
            raise Exception("Server connection failed")

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.verbose_logger",
        ) as mock_logger:
            # Test with server-specific auth headers
            mcp_server_auth_headers = {
                "working": "Bearer working-token",
                "failing": "Bearer failing-token",
            }

            result = await _get_tools_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=None,
                mcp_servers=["working_server", "failing_server"],
                mcp_server_auth_headers=mcp_server_auth_headers,
            )

            # Verify that tools from the working server are returned
            assert len(result) == 1
            assert result[0].name == "working_tool_1"

            # Verify failure logging
            mock_logger.exception.assert_any_call(
                "Error getting tools from server failing_server: Server connection failed"
            )

            # Verify success logging
            mock_logger.info.assert_any_call("Successfully fetched 1 tools total from all MCP servers")


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_handles_all_servers_failing():
    """Test that _get_tools_from_mcp_servers handles all servers failing gracefully"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # Mock servers
    failing_server1 = MagicMock()
    failing_server1.name = "failing_server1"
    failing_server1.alias = "failing1"
    failing_server1.allowed_tools = None
    failing_server1.disallowed_tools = None
    failing_server1.server_id = "failing_server1"
    failing_server1.server_name = "failing_server1"
    failing_server1.auth_type = None
    failing_server1.extra_headers = None

    failing_server2 = MagicMock()
    failing_server2.name = "failing_server2"
    failing_server2.alias = "failing2"
    failing_server2.allowed_tools = None
    failing_server2.disallowed_tools = None
    failing_server2.server_id = "failing_server2"
    failing_server2.server_name = "failing_server2"
    failing_server2.auth_type = None
    failing_server2.extra_headers = None

    # Mock global_mcp_server_manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["failing_server1", "failing_server2"])
    mock_manager.get_mcp_server_by_id = lambda server_id: (
        failing_server1 if server_id == "failing_server1" else failing_server2
    )
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip_with_info = lambda server_ids, client_ip: (
        server_ids,
        0,
    )

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=True,
        raw_headers=None,
        **kwargs,
    ):
        # All servers fail
        raise Exception(f"Server {server.name} connection failed")

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.verbose_logger",
        ) as mock_logger:
            # Test with server-specific auth headers
            mcp_server_auth_headers = {
                "failing1": "Bearer failing1-token",
                "failing2": "Bearer failing2-token",
            }

            result = await _get_tools_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=None,
                mcp_servers=None,
                mcp_server_auth_headers=mcp_server_auth_headers,
            )

            # Verify that empty list is returned
            assert len(result) == 0

            # Verify failure logging for both servers
            mock_logger.exception.assert_any_call(
                "Error getting tools from server failing_server1: Server failing_server1 connection failed"
            )
            mock_logger.exception.assert_any_call(
                "Error getting tools from server failing_server2: Server failing_server2 connection failed"
            )

            # Verify total logging
            mock_logger.info.assert_any_call("Successfully fetched 0 tools total from all MCP servers")


@pytest.mark.asyncio
async def test_mcp_server_tool_call_body_with_none_arguments():
    """Test that proxy_server_request body handles None arguments correctly"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            mcp_server_tool_call,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Setup test data
    tool_name = "test_tool_no_args"
    tool_arguments = None

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # Mock the add_litellm_data_to_request function to capture the data
    captured_data = {}

    async def mock_add_litellm_data_to_request(data, request, user_api_key_dict, proxy_config):
        captured_data.update(data)
        captured_data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": {},
            "body": data.copy(),
        }
        return captured_data

    # Mock the call_mcp_tool function
    async def mock_call_mcp_tool(*args, **kwargs):
        return [{"type": "text", "text": "mocked response"}]

    with patch(
        "litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request",
        mock_add_litellm_data_to_request,
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.call_mcp_tool",
            mock_call_mcp_tool,
        ):
            with patch(
                "litellm.proxy.proxy_server.proxy_config",
                MagicMock(),
            ):
                # Call the function
                await mcp_server_tool_call(tool_name, tool_arguments)

    # Verify the body contains the expected data
    assert "proxy_server_request" in captured_data
    assert "body" in captured_data["proxy_server_request"]

    body = captured_data["proxy_server_request"]["body"]
    assert body["name"] == tool_name
    assert body["arguments"] == tool_arguments  # Should be None


@pytest.mark.asyncio
async def test_concurrent_initialize_session_managers():
    """Test that concurrent calls to initialize_session_managers don't cause race conditions."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            initialize_session_managers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Import the module to reset state
    import litellm.proxy._experimental.mcp_server.server as mcp_server

    # Reset state before test
    original_initialized = mcp_server._SESSION_MANAGERS_INITIALIZED
    original_session_cm = mcp_server._session_manager_cm
    original_stateful_cm = mcp_server._session_manager_stateful_cm
    original_sse_cm = mcp_server._sse_session_manager_cm
    original_cleanup_task = mcp_server._stateful_auth_context_cleanup_task

    try:
        mcp_server._SESSION_MANAGERS_INITIALIZED = False
        mcp_server._session_manager_cm = None
        mcp_server._session_manager_stateful_cm = None
        mcp_server._sse_session_manager_cm = None

        # Create mock context managers for all three session managers
        mock_cm_stateless = AsyncMock()
        mock_cm_stateless.__aenter__ = AsyncMock()
        mock_cm_stateless.__aexit__ = AsyncMock()

        mock_cm_stateful = AsyncMock()
        mock_cm_stateful.__aenter__ = AsyncMock()
        mock_cm_stateful.__aexit__ = AsyncMock()

        mock_cm_sse = AsyncMock()
        mock_cm_sse.__aenter__ = AsyncMock()
        mock_cm_sse.__aexit__ = AsyncMock()

        with (
            patch.object(
                mcp_server.session_manager_stateless,
                "run",
                return_value=mock_cm_stateless,
            ) as mock_stateless_run,
            patch.object(
                mcp_server.session_manager_stateful,
                "run",
                return_value=mock_cm_stateful,
            ) as mock_stateful_run,
            patch.object(
                mcp_server.sse_session_manager,
                "run",
                return_value=mock_cm_sse,
            ) as mock_sse_run,
            patch("litellm.proxy._experimental.mcp_server.server.verbose_logger"),
        ):
            # Create multiple concurrent tasks that call initialize_session_managers
            async def init_task():
                await initialize_session_managers()
                return "success"

            # Run 10 concurrent initialization attempts
            tasks = [init_task() for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All tasks should complete successfully (no exceptions)
            assert all(result == "success" for result in results), f"Some tasks failed: {results}"

            # Each session manager.run() should only be called once due to the lock
            assert mock_stateless_run.call_count == 1, (
                f"Expected 1 call to session_manager_stateless.run(), got {mock_stateless_run.call_count}"
            )
            assert mock_stateful_run.call_count == 1, (
                f"Expected 1 call to session_manager_stateful.run(), got {mock_stateful_run.call_count}"
            )
            assert mock_sse_run.call_count == 1, (
                f"Expected 1 call to sse_session_manager.run(), got {mock_sse_run.call_count}"
            )

            # The context managers should only be entered once each
            assert mock_cm_stateless.__aenter__.call_count == 1, (
                f"Expected 1 call to stateless __aenter__, got {mock_cm_stateless.__aenter__.call_count}"
            )
            assert mock_cm_stateful.__aenter__.call_count == 1, (
                f"Expected 1 call to stateful __aenter__, got {mock_cm_stateful.__aenter__.call_count}"
            )
            assert mock_cm_sse.__aenter__.call_count == 1, (
                f"Expected 1 call to sse __aenter__, got {mock_cm_sse.__aenter__.call_count}"
            )

            # State should be properly set
            assert mcp_server._SESSION_MANAGERS_INITIALIZED is True

    finally:
        # Cancel the background cleanup task that initialize_session_managers()
        # spawned. Otherwise it keeps running against module-level dicts for the
        # rest of the test session (asyncio_default_fixture_loop_scope=session).
        leaked_task = mcp_server._stateful_auth_context_cleanup_task
        if leaked_task is not None and leaked_task is not original_cleanup_task:
            leaked_task.cancel()

        # Restore original state
        mcp_server._SESSION_MANAGERS_INITIALIZED = original_initialized
        mcp_server._session_manager_cm = original_session_cm
        mcp_server._session_manager_stateful_cm = original_stateful_cm
        mcp_server._sse_session_manager_cm = original_sse_cm
        mcp_server._stateful_auth_context_cleanup_task = original_cleanup_task


@pytest.mark.asyncio
async def test_streamable_http_session_manager_is_stateless():
    """
    Test that the StreamableHTTPSessionManager is initialized with both stateless and stateful managers.

    Regression test for GitHub issue #20242 / PR #19809.
    When stateless=False, the mcp library rejects non-initialize requests
    that lack an mcp-session-id header, breaking clients like MCP Inspector,
    curl, and any HTTP client without automatic session management.

    Now we support both:
    - stateless manager for clients without session IDs (curl, Inspector)
    - stateful manager for clients with session IDs (Claude Code, Cursor, VSCode)
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            session_manager_stateful,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # The stateless session manager must be stateless to avoid requiring mcp-session-id
    # on every request. This was regressed by PR #19809 (stateless=True -> False).
    assert session_manager_stateless.stateless is True, (
        "session_manager_stateless must be initialized with stateless=True. "
        "stateless=False breaks MCP clients that don't manage session IDs. "
        "See: https://github.com/BerriAI/litellm/issues/20242"
    )

    # The stateful session manager must be stateful to support progress notifications
    assert session_manager_stateful.stateless is False, (
        "session_manager_stateful must be initialized with stateless=False. "
        "stateless=True breaks progress notifications for clients that manage session IDs."
    )


@pytest.mark.asyncio
async def test_mcp_routing_initialize_to_stateful_no_session_to_stateless():
    """
    Test that routing correctly sends:
    - initialize (no mcp-session-id) → stateful manager (so client gets mcp-session-id)
    - tools/list (no mcp-session-id) → stateless manager (curl, Inspector)
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    async def make_request(method_body: bytes, path: str = "/mcp/progress_test"):
        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer test-key"),
            ],
        }
        receive = AsyncMock(
            return_value={
                "type": "http.request",
                "body": method_body,
                "more_body": False,
            }
        )
        send = AsyncMock()

        stateless_called = []
        stateful_called = []

        async def stateless_handle(s, r, se):
            stateless_called.append(1)

        async def stateful_handle(s, r, se):
            stateful_called.append(1)

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new_callable=AsyncMock,
                return_value=(MagicMock(), None, ["progress_test"], None, None, None),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.set_auth_context",
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch.object(
                session_manager_stateless,
                "handle_request",
                side_effect=stateless_handle,
            ),
            patch.object(
                session_manager_stateful,
                "handle_request",
                side_effect=stateful_handle,
            ),
            patch.object(
                session_manager_stateless,
                "_server_instances",
                {},
            ),
            patch.object(
                session_manager_stateful,
                "_server_instances",
                {},
            ),
        ):
            await handle_streamable_http_mcp(scope, receive, send)

        return bool(stateless_called), bool(stateful_called)

    # initialize → stateful
    init_body = b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}'
    stateless_called, stateful_called = await make_request(init_body)
    assert stateful_called and not stateless_called, "initialize (no session) should route to stateful, not stateless"

    # tools/list → stateless
    tools_body = b'{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
    stateless_called, stateful_called = await make_request(tools_body)
    assert stateless_called and not stateful_called, "tools/list (no session) should route to stateless, not stateful"


@pytest.mark.asyncio
async def test_mcp_routing_chunked_initialize_to_stateful():
    """
    Test that chunked initialize requests route to the stateful manager.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/progress_test",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer test-key"),
        ],
    }
    messages = [
        {
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,',
            "more_body": True,
        },
        {
            "type": "http.request",
            "body": b'"method":"initialize","params":{}}',
            "more_body": False,
        },
    ]
    receive = AsyncMock(side_effect=messages)
    send = AsyncMock()
    stateless_called = []
    stateful_called = []

    async def stateless_handle(s, r, se):
        stateless_called.append(1)

    async def stateful_handle(s, r, se):
        stateful_called.append(1)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(MagicMock(), None, ["progress_test"], None, None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch.object(
            session_manager_stateless,
            "handle_request",
            side_effect=stateless_handle,
        ),
        patch.object(
            session_manager_stateful,
            "handle_request",
            side_effect=stateful_handle,
        ),
        patch.object(
            session_manager_stateless,
            "_server_instances",
            {},
        ),
        patch.object(
            session_manager_stateful,
            "_server_instances",
            {},
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    assert stateful_called and not stateless_called, (
        "chunked initialize (no session) should route to stateful, not stateless"
    )


@pytest.mark.asyncio
async def test_mcp_routing_caps_body_peek_for_oversized_chunked_body():
    """
    A no-session-id POST with a very large chunked body should not force
    the proxy to buffer the entire body just to decide routing — the peek
    should stop once ``_MCP_ROUTING_PEEK_MAX_BYTES`` worth of body has been
    consumed, and the remaining chunks should stream through the original
    receive into the downstream handler.
    """
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    peek_cap = mcp_server._MCP_ROUTING_PEEK_MAX_BYTES
    # First chunk fills the peek budget; subsequent chunks are oversized payload.
    first_chunk = b"x" * peek_cap
    oversized_tail = [b"y" * 65536 for _ in range(4)]

    messages = [
        {"type": "http.request", "body": first_chunk, "more_body": True},
        *[{"type": "http.request", "body": chunk, "more_body": True} for chunk in oversized_tail],
        {"type": "http.request", "body": b"", "more_body": False},
    ]
    receive_calls = {"count": 0}

    async def receive():
        idx = receive_calls["count"]
        receive_calls["count"] += 1
        return messages[idx]

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/progress_test",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer test-key"),
        ],
    }
    send = AsyncMock()

    stateless_received_chunks = []
    receive_count_at_dispatch = {"value": -1}

    async def stateless_handle(s, r, se):
        # Snapshot how many wire reads happened BEFORE dispatch — the cap
        # check is meaningful only against pre-dispatch consumption.
        receive_count_at_dispatch["value"] = receive_calls["count"]
        # Drain the wrapped receive the same way the SDK would.
        while True:
            msg = await r()
            if msg.get("type") != "http.request":
                break
            stateless_received_chunks.append(msg.get("body", b"") or b"")
            if not msg.get("more_body", False):
                break

    async def stateful_handle(s, r, se):
        raise AssertionError("non-initialize POST should not reach stateful manager")

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(MagicMock(), None, ["progress_test"], None, None, None),
        ),
        patch("litellm.proxy._experimental.mcp_server.server.set_auth_context"),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch.object(session_manager_stateless, "handle_request", side_effect=stateless_handle),
        patch.object(session_manager_stateful, "handle_request", side_effect=stateful_handle),
        patch.object(session_manager_stateless, "_server_instances", {}),
        patch.object(session_manager_stateful, "_server_instances", {}),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # The routing peek must stop pulling from the wire once the cap is reached.
    # Without the cap fix, every chunk would have been pulled before dispatch,
    # so this assertion guards against unbounded pre-dispatch buffering.
    assert receive_count_at_dispatch["value"] == 1, (
        "routing should stop reading after the peek cap is filled, "
        f"but consumed {receive_count_at_dispatch['value']} chunks before dispatching"
    )
    # All chunks must still reach the downstream handler via replay+stream.
    total_streamed = sum(len(b) for b in stateless_received_chunks)
    assert total_streamed == len(first_chunk) + sum(len(b) for b in oversized_tail)


@pytest.mark.asyncio
async def test_enforce_stateful_session_cap_evicts_oldest_idle_then_rejects():
    """
    A caller at the per-owner session cap should have its own oldest *idle*
    session evicted to make room for a new one, but be rejected outright when
    every one of its sessions is in flight (nothing safe to evict).
    """
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    terminated = []

    class FakeTransport:
        def __init__(self, session_id):
            self.session_id = session_id

        async def terminate(self):
            terminated.append(self.session_id)

    instances = {f"s{i}": FakeTransport(f"s{i}") for i in range(3)}
    owners = {f"s{i}": "owner-A" for i in range(3)}
    last_seen = {"s0": 1.0, "s1": 2.0, "s2": 3.0}
    contexts = {f"s{i}": MagicMock() for i in range(3)}

    with (
        patch.object(session_manager_stateful, "_server_instances", instances),
        patch.object(mcp_server, "_MAX_STATEFUL_SESSIONS_PER_OWNER", 3),
        patch.dict(mcp_server._stateful_session_owners, owners, clear=True),
        patch.dict(mcp_server._stateful_session_auth_context_last_seen, last_seen, clear=True),
        patch.dict(mcp_server._stateful_session_auth_contexts, contexts, clear=True),
        patch.dict(mcp_server._stateful_session_active_request_counts, {}, clear=True),
    ):
        # All idle -> oldest (s0) is evicted, request may proceed.
        allowed = await mcp_server._enforce_stateful_session_cap_for_owner("owner-A")
        assert allowed is True
        assert terminated == ["s0"]
        assert "s0" not in instances
        assert "s0" not in mcp_server._stateful_session_owners

        # A different owner at the cap is unaffected by owner-A's sessions.
        terminated.clear()
        allowed_other = await mcp_server._enforce_stateful_session_cap_for_owner("owner-B")
        assert allowed_other is True
        assert terminated == []

    # Now every session is in flight -> nothing evictable -> reject.
    terminated.clear()
    instances = {f"s{i}": FakeTransport(f"s{i}") for i in range(3)}
    owners = {f"s{i}": "owner-A" for i in range(3)}
    active = {f"s{i}": 1 for i in range(3)}

    with (
        patch.object(session_manager_stateful, "_server_instances", instances),
        patch.object(mcp_server, "_MAX_STATEFUL_SESSIONS_PER_OWNER", 3),
        patch.dict(mcp_server._stateful_session_owners, owners, clear=True),
        patch.dict(
            mcp_server._stateful_session_auth_context_last_seen,
            {f"s{i}": float(i) for i in range(3)},
            clear=True,
        ),
        patch.dict(mcp_server._stateful_session_active_request_counts, active, clear=True),
    ):
        rejected = await mcp_server._enforce_stateful_session_cap_for_owner("owner-A")
        assert rejected is False
        assert terminated == []
        assert len(instances) == 3


@pytest.mark.asyncio
async def test_mcp_routing_initialize_rejected_when_owner_at_session_cap():
    """
    A new ``initialize`` (no session id) must be rejected with 429 when the
    caller already holds the maximum number of in-flight stateful sessions,
    and must not reach the stateful session manager.
    """
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    cap = 2

    class FakeTransport:
        async def terminate(self):
            pass

    instances = {f"s{i}": FakeTransport() for i in range(cap)}
    owners = {f"s{i}": "owner-X" for i in range(cap)}
    active = {f"s{i}": 1 for i in range(cap)}  # all in flight -> cannot evict
    contexts = {f"s{i}": MagicMock() for i in range(cap)}

    init_body = b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}'
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/progress_test",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer test-key"),
        ],
    }
    receive = AsyncMock(return_value={"type": "http.request", "body": init_body, "more_body": False})
    send = AsyncMock()

    stateful_called = []

    async def stateful_handle(s, r, se):
        stateful_called.append(1)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(MagicMock(), None, ["progress_test"], None, None, None),
        ),
        patch("litellm.proxy._experimental.mcp_server.server.set_auth_context"),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch.object(mcp_server, "_owner_fingerprint_for", return_value="owner-X"),
        patch.object(mcp_server, "_MAX_STATEFUL_SESSIONS_PER_OWNER", cap),
        patch.object(session_manager_stateful, "handle_request", side_effect=stateful_handle),
        patch.object(session_manager_stateful, "_server_instances", instances),
        patch.object(session_manager_stateless, "_server_instances", {}),
        patch.dict(mcp_server._stateful_session_owners, owners, clear=True),
        patch.dict(
            mcp_server._stateful_session_auth_context_last_seen,
            {f"s{i}": float(i) for i in range(cap)},
            clear=True,
        ),
        patch.dict(mcp_server._stateful_session_active_request_counts, active, clear=True),
        patch.dict(mcp_server._stateful_session_auth_contexts, contexts, clear=True),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    assert not stateful_called, "initialize at session cap must not reach the manager"
    start_messages = [
        call.args[0] for call in send.call_args_list if call.args and call.args[0].get("type") == "http.response.start"
    ]
    assert start_messages, "a response should have been sent"
    assert start_messages[0]["status"] == 429


@pytest.mark.asyncio
async def test_stateful_mcp_requests_refresh_session_auth_context():
    """
    Stateful MCP sessions run callbacks in the initialize task's context; the
    stored auth object must be refreshed for each mcp-session-id request.
    """
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            get_auth_context,
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "stateful-session-1"
    initialize_auth = UserAPIKeyAuth(api_key="initialize-key", user_id="user-a")
    current_auth = UserAPIKeyAuth(api_key="current-key", user_id="user-b")
    callback_context = contextvars.copy_context()
    callback_context.run(
        mcp_server.set_auth_context,
        initialize_auth,
        None,
        ["old-server"],
        None,
        None,
        None,
        "1.1.1.1",
    )
    mcp_server._stateful_session_auth_contexts[session_id] = callback_context.run(mcp_server.auth_context_var.get)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/current-server",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer current-key"),
            (b"mcp-session-id", session_id.encode()),
        ],
    }
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()

    captured_context = None

    async def stateful_handle(s, r, se):
        nonlocal captured_context
        captured_context = callback_context.run(get_auth_context)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(
                current_auth,
                "current-mcp-auth",
                ["current-server"],
                {"current-server": {"Authorization": "Bearer server-key"}},
                {"Authorization": "Bearer oauth-key"},
                {"mcp-session-id": session_id},
            ),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch.object(
            session_manager_stateful,
            "handle_request",
            side_effect=stateful_handle,
        ),
        patch.object(
            session_manager_stateful,
            "_server_instances",
            {session_id: MagicMock()},
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    assert captured_context == (
        current_auth,
        "current-mcp-auth",
        ["current-server"],
        {"current-server": {"Authorization": "Bearer server-key"}},
        {"Authorization": "Bearer oauth-key"},
        {"mcp-session-id": session_id},
        "",
    )
    mcp_server._remove_stateful_session_tracking(session_id)


@pytest.mark.asyncio
async def test_initialize_response_capture_accepts_str_headers_and_sets_auth_context():
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "initialize-session-1"
    auth_user = mcp_server.MCPAuthenticatedUser(
        user_api_key_auth=UserAPIKeyAuth(api_key="initialize-key", user_id="user-a")
    )
    previous_auth_user = mcp_server.MCPAuthenticatedUser(
        user_api_key_auth=UserAPIKeyAuth(api_key="previous-key", user_id="user-b")
    )
    sent_messages = []

    async def send(message):
        sent_messages.append(message)

    wrapped_send = mcp_server._wrap_send_with_stateful_session_auth_context(
        send,
        auth_user,
        "owner-fingerprint",
    )
    token = mcp_server.auth_context_var.set(previous_auth_user)
    try:
        await wrapped_send(
            {
                "type": "http.response.start",
                "headers": [("mcp-session-id", session_id)],
            }
        )

        assert mcp_server.auth_context_var.get() is auth_user
        assert mcp_server._stateful_session_auth_contexts[session_id] is auth_user
        assert mcp_server._stateful_session_owners[session_id] == "owner-fingerprint"
        assert session_id in mcp_server._stateful_session_auth_context_last_seen
        assert sent_messages == [
            {
                "type": "http.response.start",
                "headers": [("mcp-session-id", session_id)],
            }
        ]
    finally:
        mcp_server.auth_context_var.reset(token)
        mcp_server._remove_stateful_session_tracking(session_id)


@pytest.mark.asyncio
async def test_initialize_request_tracks_active_session_after_response_header():
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "initialize-active-session-1"
    owner_auth = UserAPIKeyAuth(api_key="initialize-key", user_id="user-a")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer initialize-key"),
        ],
    }
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            "more_body": False,
        }
    )

    async def stateful_handle(s, r, se):
        await se(
            {
                "type": "http.response.start",
                "headers": [(b"mcp-session-id", session_id.encode())],
            }
        )
        assert mcp_server._stateful_session_active_request_counts[session_id] == 1
        now = (
            mcp_server._stateful_session_auth_context_last_seen[session_id]
            + mcp_server._STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS
        )
        await mcp_server._purge_expired_stateful_session_auth_contexts(now=now)
        assert session_id in mcp_server._stateful_session_auth_contexts

    async def stateless_handle(s, r, se):
        raise AssertionError("initialize request should use stateful manager")

    try:
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new_callable=AsyncMock,
                return_value=(owner_auth, None, None, None, None, None),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch.object(
                session_manager_stateful,
                "handle_request",
                side_effect=stateful_handle,
            ),
            patch.object(
                session_manager_stateless,
                "handle_request",
                side_effect=stateless_handle,
            ),
            patch.object(session_manager_stateful, "_server_instances", {}),
        ):
            await handle_streamable_http_mcp(scope, receive, AsyncMock())

        assert session_id not in mcp_server._stateful_session_active_request_counts
        assert session_id in mcp_server._stateful_session_auth_contexts
    finally:
        mcp_server._remove_stateful_session_tracking(session_id)


@pytest.mark.asyncio
async def test_initialize_request_with_existing_session_tracks_new_session():
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    existing_session_id = "existing-initialize-session"
    new_session_id = "reinitialized-session"
    owner_auth = UserAPIKeyAuth(api_key="initialize-key", user_id="user-a")
    owner_fingerprint = mcp_server._owner_fingerprint_for(owner_auth)
    existing_auth_user = mcp_server.MCPAuthenticatedUser(
        user_api_key_auth=owner_auth,
        mcp_auth_header="old-mcp-auth",
        mcp_servers=["old-server"],
        mcp_server_auth_headers={"old-server": {"Authorization": "Bearer old-key"}},
        oauth2_headers={"Authorization": "Bearer old-oauth"},
        raw_headers={"x-old-header": "old"},
        client_ip="old-client-ip",
    )
    initialize_body = b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer initialize-key"),
            (b"mcp-session-id", existing_session_id.encode()),
        ],
    }
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": initialize_body,
            "more_body": False,
        }
    )
    stateful_called = []

    async def stateful_handle(s, r, se):
        stateful_called.append(1)
        message = await r()
        assert message["body"] == initialize_body
        await se(
            {
                "type": "http.response.start",
                "headers": [(b"mcp-session-id", new_session_id.encode())],
            }
        )
        assert mcp_server._stateful_session_auth_contexts[new_session_id]
        assert mcp_server._stateful_session_owners[new_session_id] == owner_fingerprint
        assert mcp_server._stateful_session_active_request_counts[new_session_id] == 1
        now = (
            mcp_server._stateful_session_auth_context_last_seen[new_session_id]
            + mcp_server._STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS
        )
        await mcp_server._purge_expired_stateful_session_auth_contexts(now=now)
        assert new_session_id in mcp_server._stateful_session_auth_contexts
        assert mcp_server._stateful_session_auth_contexts[new_session_id] is not existing_auth_user
        assert mcp_server._stateful_session_auth_contexts[new_session_id].mcp_auth_header == "new-mcp-auth"

    async def stateless_handle(s, r, se):
        raise AssertionError("initialize request with session should use stateful manager")

    try:
        mcp_server._stateful_session_auth_contexts[existing_session_id] = existing_auth_user
        mcp_server._stateful_session_auth_context_last_seen[existing_session_id] = 1.0
        mcp_server._stateful_session_owners[existing_session_id] = owner_fingerprint

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new_callable=AsyncMock,
                return_value=(
                    owner_auth,
                    "new-mcp-auth",
                    ["new-server"],
                    {"new-server": {"Authorization": "Bearer new-key"}},
                    {"Authorization": "Bearer new-oauth"},
                    {"x-new-header": "new"},
                ),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch.object(
                session_manager_stateful,
                "handle_request",
                side_effect=stateful_handle,
            ),
            patch.object(
                session_manager_stateless,
                "handle_request",
                side_effect=stateless_handle,
            ),
            patch.object(
                session_manager_stateful,
                "_server_instances",
                {existing_session_id: MagicMock()},
            ),
        ):
            await handle_streamable_http_mcp(scope, receive, AsyncMock())

        assert stateful_called
        assert new_session_id not in mcp_server._stateful_session_active_request_counts
        assert new_session_id in mcp_server._stateful_session_auth_contexts
        assert mcp_server._stateful_session_auth_contexts[existing_session_id] is existing_auth_user
        assert existing_auth_user.mcp_auth_header == "old-mcp-auth"
        assert existing_auth_user.mcp_servers == ["old-server"]
    finally:
        mcp_server._remove_stateful_session_tracking(existing_session_id)
        mcp_server._remove_stateful_session_tracking(new_session_id)


@pytest.mark.asyncio
async def test_stateful_mcp_auth_contexts_expire_with_idle_sessions():
    """Expired session auth contexts should not remain in memory indefinitely."""
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "expired-stateful-session"
    auth_user = UserAPIKeyAuth(api_key="expired-key", user_id="expired-user")
    transport = MagicMock()
    transport.terminate = AsyncMock()
    now = 1000.0

    mcp_server._stateful_session_auth_contexts[session_id] = auth_user
    mcp_server._stateful_session_auth_context_last_seen[session_id] = (
        now - mcp_server._STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS
    )

    with patch.object(
        mcp_server.session_manager_stateful,
        "_server_instances",
        {session_id: transport},
    ):
        await mcp_server._purge_expired_stateful_session_auth_contexts(now=now)

    assert session_id not in mcp_server._stateful_session_auth_contexts
    assert session_id not in mcp_server._stateful_session_auth_context_last_seen
    transport.terminate.assert_awaited_once()


@pytest.mark.asyncio
async def test_stateful_mcp_auth_contexts_do_not_expire_active_sessions():
    """Active stateful sessions should not be terminated by idle cleanup."""
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "active-stateful-session"
    auth_user = UserAPIKeyAuth(api_key="active-key", user_id="active-user")
    transport = MagicMock()
    transport.terminate = AsyncMock()
    now = 1000.0

    mcp_server._stateful_session_auth_contexts[session_id] = auth_user
    mcp_server._stateful_session_auth_context_last_seen[session_id] = (
        now - mcp_server._STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS
    )
    mcp_server._stateful_session_active_request_counts[session_id] = 1

    try:
        with patch.object(
            mcp_server.session_manager_stateful,
            "_server_instances",
            {session_id: transport},
        ):
            await mcp_server._purge_expired_stateful_session_auth_contexts(now=now)

        assert session_id in mcp_server._stateful_session_auth_contexts
        assert session_id in mcp_server._stateful_session_auth_context_last_seen
        transport.terminate.assert_not_awaited()
    finally:
        mcp_server._stateful_session_auth_contexts.pop(session_id, None)
        mcp_server._stateful_session_auth_context_last_seen.pop(session_id, None)
        mcp_server._stateful_session_active_request_counts.pop(session_id, None)


@pytest.mark.asyncio
async def test_stateful_mcp_auth_context_cleanup_respects_zero_now():
    """Explicit now=0 should be used as-is instead of falling back to monotonic."""
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "zero-now-stateful-session"
    auth_user = UserAPIKeyAuth(api_key="zero-now-key", user_id="zero-now-user")
    transport = MagicMock()
    transport.terminate = AsyncMock()

    mcp_server._stateful_session_auth_contexts[session_id] = auth_user
    mcp_server._stateful_session_auth_context_last_seen[session_id] = 0.0

    try:
        with (
            patch.object(
                mcp_server.session_manager_stateful,
                "_server_instances",
                {session_id: transport},
            ),
            patch.object(
                mcp_server.time,
                "monotonic",
                return_value=mcp_server._STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS + 1,
            ),
        ):
            await mcp_server._purge_expired_stateful_session_auth_contexts(now=0.0)

        assert session_id in mcp_server._stateful_session_auth_contexts
        assert session_id in mcp_server._stateful_session_auth_context_last_seen
        transport.terminate.assert_not_awaited()
    finally:
        mcp_server._stateful_session_auth_contexts.pop(session_id, None)
        mcp_server._stateful_session_auth_context_last_seen.pop(session_id, None)


@pytest.mark.asyncio
async def test_stateful_mcp_cleanup_loop_survives_purge_errors():
    """Cleanup loop should keep running after one purge attempt fails."""
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
    except ImportError:
        pytest.skip("MCP server not available")

    purge = AsyncMock(side_effect=[RuntimeError("terminate failed"), asyncio.CancelledError()])

    with (
        patch.object(mcp_server.asyncio, "sleep", AsyncMock(return_value=None)),
        patch.object(mcp_server, "_purge_expired_stateful_session_auth_contexts", purge),
    ):
        with pytest.raises(asyncio.CancelledError):
            await mcp_server._cleanup_expired_stateful_session_auth_contexts()

    assert purge.await_count == 2


@pytest.mark.asyncio
async def test_owner_fingerprint_distinguishes_oauth_callers():
    """
    OAuth2 passthrough callers all share `UserAPIKeyAuth()` with no api_key
    or user_id. Without folding the upstream bearer into the fingerprint
    they would all collapse to a single 'anonymous' owner and one OAuth
    user could hijack another's mcp-session-id.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _owner_fingerprint_for,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    anon_auth = UserAPIKeyAuth()
    fp_a = _owner_fingerprint_for(anon_auth, {"Authorization": "Bearer token-A"})
    fp_b = _owner_fingerprint_for(anon_auth, {"Authorization": "Bearer token-B"})
    fp_a_again = _owner_fingerprint_for(anon_auth, {"authorization": "Bearer token-A"})
    fp_no_oauth = _owner_fingerprint_for(anon_auth, None)

    assert fp_a != fp_b
    assert fp_a == fp_a_again
    assert fp_a.startswith("oauth:")
    assert fp_no_oauth == "anonymous"
    assert "Bearer token-A" not in fp_a

    # When no API key, user_id, or OAuth bearer is available, fall back to
    # client IP so two unrelated unauthenticated callers from different
    # sources don't collapse to a single 'anonymous' owner and end up able
    # to drive each other's stateful sessions.
    fp_ip_a = _owner_fingerprint_for(anon_auth, None, "10.0.0.1")
    fp_ip_b = _owner_fingerprint_for(anon_auth, None, "10.0.0.2")
    fp_ip_a_again = _owner_fingerprint_for(anon_auth, None, "10.0.0.1")

    assert fp_ip_a != fp_ip_b
    assert fp_ip_a == fp_ip_a_again
    assert fp_ip_a.startswith("ip:")
    assert "10.0.0.1" not in fp_ip_a


@pytest.mark.asyncio
async def test_owner_fingerprint_hashes_custom_api_keys():
    """Custom API key formats should not appear in owner fingerprints."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _owner_fingerprint_for,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    auth = UserAPIKeyAuth(api_key="custom-master-key")
    fp = _owner_fingerprint_for(auth)
    fp_again = _owner_fingerprint_for(auth)

    assert fp == fp_again
    assert fp.startswith("key:")
    assert "custom-master-key" not in fp
    assert fp != "key:custom-master-key"


@pytest.mark.asyncio
async def test_stateful_mcp_session_owner_mismatch_returns_403():
    """
    A stateful mcp-session-id is bound to its creator. A different
    authenticated caller presenting the same session_id must be rejected
    with 403, and the stateful manager must never be invoked.
    """
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "owned-session-1"
    owner_auth = UserAPIKeyAuth(api_key="owner-key", user_id="owner")
    intruder_auth = UserAPIKeyAuth(api_key="intruder-key", user_id="intruder")

    mcp_server._stateful_session_auth_contexts[session_id] = MagicMock()
    mcp_server._stateful_session_owners[session_id] = mcp_server._owner_fingerprint_for(owner_auth)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer intruder-key"),
            (b"mcp-session-id", session_id.encode()),
        ],
    }
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
            "more_body": False,
        }
    )
    sent_messages: list = []

    async def capture_send(message):
        sent_messages.append(message)

    handle_request_mock = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(intruder_auth, None, None, None, None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch.object(
            session_manager_stateful,
            "handle_request",
            side_effect=handle_request_mock,
        ),
        patch.object(
            session_manager_stateful,
            "_server_instances",
            {session_id: MagicMock()},
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, capture_send)

    handle_request_mock.assert_not_awaited()
    statuses = [m["status"] for m in sent_messages if m.get("type") == "http.response.start"]
    assert statuses == [403]

    mcp_server._stateful_session_auth_contexts.pop(session_id, None)
    mcp_server._stateful_session_owners.pop(session_id, None)


@pytest.mark.asyncio
async def test_stateful_mcp_session_serializes_concurrent_requests():
    """
    Concurrent requests on the same stateful mcp-session-id must be
    serialized so they cannot observe each other's mutation of the shared
    MCPAuthenticatedUser while in-flight callbacks are still running.
    """
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "serialized-session-1"
    owner_auth = UserAPIKeyAuth(api_key="owner-key", user_id="owner")
    mcp_server._stateful_session_auth_contexts[session_id] = mcp_server.MCPAuthenticatedUser(
        user_api_key_auth=owner_auth
    )
    mcp_server._stateful_session_owners[session_id] = mcp_server._owner_fingerprint_for(owner_auth)

    inside = 0
    max_inside = 0
    gate = asyncio.Event()

    async def slow_handle(s, r, se):
        nonlocal inside, max_inside
        inside += 1
        max_inside = max(max_inside, inside)
        await gate.wait()
        inside -= 1

    async def make_request():
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(b"mcp-session-id", session_id.encode())],
        }
        receive = AsyncMock(
            return_value={
                "type": "http.request",
                "body": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
                "more_body": False,
            }
        )
        send = AsyncMock()
        await handle_streamable_http_mcp(scope, receive, send)

    try:
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new_callable=AsyncMock,
                return_value=(owner_auth, None, None, None, None, None),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch.object(session_manager_stateful, "handle_request", side_effect=slow_handle),
            patch.object(
                session_manager_stateful,
                "_server_instances",
                {session_id: MagicMock()},
            ),
        ):
            tasks = [asyncio.create_task(make_request()) for _ in range(3)]
            await asyncio.sleep(0.05)
            gate.set()
            await asyncio.gather(*tasks)
    finally:
        mcp_server._stateful_session_auth_contexts.pop(session_id, None)
        mcp_server._stateful_session_owners.pop(session_id, None)
        mcp_server._stateful_session_locks.pop(session_id, None)

    assert max_inside == 1, "concurrent requests on same stateful session must be serialized"


@pytest.mark.asyncio
async def test_stateful_mcp_lock_does_not_leak_when_auth_context_missing():
    """
    If a per-session lock is created for a session_id that is not tracked in
    ``_stateful_session_auth_contexts`` (e.g., a defensive path), the request
    finalizer must drop the lock so it isn't orphaned. The periodic cleanup
    loop only iterates ``_stateful_session_auth_context_last_seen``, so a
    leaked lock would otherwise live forever.
    """
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "untracked-session-1"
    owner_auth = UserAPIKeyAuth(api_key="owner-key", user_id="owner")

    async def handle(s, r, se):
        return None

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"mcp-session-id", session_id.encode())],
    }
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
            "more_body": False,
        }
    )

    try:
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new_callable=AsyncMock,
                return_value=(owner_auth, None, None, None, None, None),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch.object(session_manager_stateful, "handle_request", side_effect=handle),
            patch.object(
                session_manager_stateful,
                "_server_instances",
                {session_id: MagicMock()},
            ),
        ):
            assert session_id not in mcp_server._stateful_session_auth_contexts
            await handle_streamable_http_mcp(scope, receive, AsyncMock())

        assert session_id not in mcp_server._stateful_session_locks, (
            "lock entry must be cleaned up for untracked stateful session"
        )
    finally:
        mcp_server._stateful_session_auth_contexts.pop(session_id, None)
        mcp_server._stateful_session_owners.pop(session_id, None)
        mcp_server._stateful_session_locks.pop(session_id, None)
        mcp_server._stateful_session_active_request_counts.pop(session_id, None)


@pytest.mark.asyncio
async def test_stateful_mcp_get_stream_does_not_block_post():
    """
    A long-lived GET (server-to-client SSE stream) on a stateful session
    must NOT hold the per-session lock — otherwise subsequent POSTs on the
    same mcp-session-id hang for the lifetime of the stream.
    """
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "stream-session-1"
    owner_auth = UserAPIKeyAuth(api_key="owner-key", user_id="owner")
    mcp_server._stateful_session_auth_contexts[session_id] = mcp_server.MCPAuthenticatedUser(
        user_api_key_auth=owner_auth
    )
    mcp_server._stateful_session_owners[session_id] = mcp_server._owner_fingerprint_for(owner_auth)

    stream_release = asyncio.Event()
    post_finished = asyncio.Event()

    async def handle(s, r, se):
        if s.get("method") == "GET":
            await stream_release.wait()
        else:
            post_finished.set()

    async def call(method: str, body: bytes = b""):
        scope = {
            "type": "http",
            "method": method,
            "path": "/mcp",
            "headers": [(b"mcp-session-id", session_id.encode())],
        }
        receive = AsyncMock(
            return_value={
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        )
        await handle_streamable_http_mcp(scope, receive, AsyncMock())

    try:
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new_callable=AsyncMock,
                return_value=(owner_auth, None, None, None, None, None),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch.object(session_manager_stateful, "handle_request", side_effect=handle),
            patch.object(
                session_manager_stateful,
                "_server_instances",
                {session_id: MagicMock()},
            ),
        ):
            stream_task = asyncio.create_task(call("GET"))
            await asyncio.sleep(0.05)
            assert not stream_task.done(), "GET stream should still be open"

            post_task = asyncio.create_task(
                call(
                    "POST",
                    body=b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
                )
            )
            await asyncio.wait_for(post_finished.wait(), timeout=1.0)
            await post_task

            stream_release.set()
            await stream_task
    finally:
        mcp_server._stateful_session_auth_contexts.pop(session_id, None)
        mcp_server._stateful_session_owners.pop(session_id, None)
        mcp_server._stateful_session_locks.pop(session_id, None)


def test_jsonrpc_text_has_top_level_method_ignores_nested_method():
    """The top-level-key scan must not be fooled by a ``method`` field nested
    inside a JSON-RPC response's ``result`` payload — a flat substring search
    would, and that misread is what deadlocks the session lock."""
    from litellm.proxy._experimental.mcp_server.server import (
        _jsonrpc_text_has_top_level_method,
    )

    request = '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{}}'
    assert _jsonrpc_text_has_top_level_method(request) is True

    # method key out of order (after params) is still top-level
    reordered = '{"jsonrpc":"2.0","params":{"x":1},"method":"foo"}'
    assert _jsonrpc_text_has_top_level_method(reordered) is True

    # response whose result nests a "method" key (and arrays of them)
    response = '{"jsonrpc":"2.0","id":1,"result":{"toolResult":{"method":"GET"},"steps":[{"method":"x"}]}}'
    assert _jsonrpc_text_has_top_level_method(response) is False

    # truncated response: result value never closes, no top-level method seen
    truncated = '{"jsonrpc":"2.0","id":1,"result":{"text":"' + "q" * 5000
    assert _jsonrpc_text_has_top_level_method(truncated) is False


@pytest.mark.asyncio
async def test_truncated_jsonrpc_response_with_nested_method_skips_lock():
    """Regression: a large JSON-RPC *response* POST whose ``result`` payload
    nests a ``method`` key must skip the per-session lock so it does not
    deadlock behind the in-flight request POST that is holding the lock while
    it awaits this very response (e.g. sampling/createMessage)."""
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "nested-method-response-session"
    owner_auth = UserAPIKeyAuth(api_key="owner-key", user_id="owner")
    mcp_server._stateful_session_auth_contexts[session_id] = mcp_server.MCPAuthenticatedUser(
        user_api_key_auth=owner_auth
    )
    mcp_server._stateful_session_owners[session_id] = mcp_server._owner_fingerprint_for(owner_auth)

    gate = asyncio.Event()
    request_in_handle = asyncio.Event()
    response_handled = asyncio.Event()

    async def handle(s, r, se):
        msg = await r()
        body = msg.get("body", b"") or b""
        if b'"result"' in body:
            response_handled.set()
        else:
            request_in_handle.set()
            await gate.wait()

    async def call(body: bytes):
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(b"mcp-session-id", session_id.encode())],
        }
        receive = AsyncMock(
            return_value={
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        )
        await handle_streamable_http_mcp(scope, receive, AsyncMock())

    # The in-flight request POST holds the session lock while blocked.
    request_body = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{}}'
    # A JSON-RPC response larger than the routing peek cap so it can't be fully
    # parsed, with a nested "method" key in the first bytes to trip a flat
    # substring heuristic.
    response_body = (
        '{"jsonrpc":"2.0","id":99,"result":{"toolResult":{"method":"GET","payload":"' + ("x" * 5000) + '"}}}'
    ).encode()

    try:
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new_callable=AsyncMock,
                return_value=(owner_auth, None, None, None, None, None),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch.object(session_manager_stateful, "handle_request", side_effect=handle),
            patch.object(
                session_manager_stateful,
                "_server_instances",
                {session_id: MagicMock()},
            ),
        ):
            req_task = asyncio.create_task(call(request_body))
            await asyncio.wait_for(request_in_handle.wait(), timeout=1.0)

            resp_task = asyncio.create_task(call(response_body))
            # Under a flat substring heuristic the response would acquire the
            # lock held by req_task and this wait would time out (deadlock).
            await asyncio.wait_for(response_handled.wait(), timeout=1.0)

            gate.set()
            await asyncio.gather(req_task, resp_task)
    finally:
        gate.set()
        mcp_server._stateful_session_auth_contexts.pop(session_id, None)
        mcp_server._stateful_session_owners.pop(session_id, None)
        mcp_server._stateful_session_locks.pop(session_id, None)
        mcp_server._stateful_session_active_request_counts.pop(session_id, None)


@pytest.mark.asyncio
@pytest.mark.no_parallel
async def test_mcp_routing_with_conflicting_alias_and_group_name():
    """
    Tests (GH #14536) where an MCP server alias (e.g., "group/id")
    conflicts with an access group name (e.g., "group").
    """
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._experimental.mcp_server.server import (
            _get_mcp_servers_in_path,
            _get_tools_from_mcp_servers,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    global_mcp_server_manager.registry.clear()

    # Create two in-memory servers
    specific_server = MCPServer(
        server_id="specific_server_id",
        name="custom_solutions/user_123",
        alias="custom_solutions/user_123",
        transport=MCPTransport.http,
    )
    other_server = MCPServer(
        server_id="other_server_in_group_id",
        name="custom_solutions/another_user_456",
        alias="custom_solutions/another_user_456",
        transport=MCPTransport.http,
    )
    global_mcp_server_manager.registry[specific_server.server_id] = specific_server
    global_mcp_server_manager.registry[other_server.server_id] = other_server

    user_key = UserAPIKeyAuth(api_key="sk-test", team_id="team_custom_solutions")

    # Define the request path that triggers the bug
    test_path = "/mcp/custom_solutions/user_123/chat/completions"

    # This mock will be our "spy" to see which servers are ultimately contacted
    mock_get_tools_spy = AsyncMock(return_value=[])

    # Mock the function that checks DB for an access group named "custom_solutions"
    mock_db_lookup = AsyncMock(return_value=[specific_server.server_id, other_server.server_id])

    mock_get_allowed = AsyncMock(return_value=[specific_server.server_id, other_server.server_id])

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_allowed_mcp_servers",
            mock_get_allowed,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler._get_mcp_servers_from_access_groups",
            mock_db_lookup,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager._get_tools_from_server",
            mock_get_tools_spy,
        ),
    ):
        mcp_servers_from_path = _get_mcp_servers_in_path(test_path)

        await _get_tools_from_mcp_servers(
            user_api_key_auth=user_key,
            mcp_servers=mcp_servers_from_path,
            mcp_auth_header=None,
        )

    # Get the list of actual server objects that the orchestrator tried to contact
    called_servers = [call.kwargs["server"] for call in mock_get_tools_spy.call_args_list]

    assert len(called_servers) == 1, "Should have resolved to exactly one server."
    assert called_servers[0].server_id == specific_server.server_id, (
        "Should have contacted the specific server alias, not the group."
    )


@pytest.mark.asyncio
@pytest.mark.no_parallel
async def test_oauth2_caller_headers_not_forwarded_for_migrated_server():
    """A v2-migrated authorization_code server (like github_mcp) must NOT forward the
    caller's oauth2 Authorization to the MCP client — the resolver injects the stored
    per-user token, so a caller-supplied bearer cannot override another user's credential."""
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    # Clear the registry to avoid conflicts with other tests
    global_mcp_server_manager.registry.clear()

    # Create an OAuth2 MCP server similar to github_mcp configuration
    oauth2_server = MCPServer(
        server_id="github_mcp_server_id",
        name="github_mcp",
        alias="github_mcp",
        transport=MCPTransport.http,
        url="https://api.githubcopilot.com/mcp",
        auth_type=MCPAuth.oauth2,
        client_id="test_github_client_id",
        client_secret="test_github_client_secret",
        scopes=["public_repo", "user:email"],
        authorization_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")

    # Set up OAuth2 headers that would come from the client
    oauth2_headers = {"Authorization": "Bearer github_oauth_token_12345"}

    # Set auth context with OAuth2 headers
    set_auth_context(user_api_key_auth=user_api_key_auth, oauth2_headers=oauth2_headers)

    # This will capture the arguments passed to _create_mcp_client
    captured_client_args = {}

    async def mock_create_mcp_client(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        stdio_env=None,
        **kwargs,
    ):
        # Capture the arguments for verification
        captured_client_args.update(
            {
                "server": server,
                "mcp_auth_header": mcp_auth_header,
                "extra_headers": extra_headers,
                "stdio_env": stdio_env,
                "kwargs": kwargs,
            }
        )
        # Return a mock client that doesn't actually connect
        mock_client = MagicMock()
        return mock_client

    # Mock _fetch_tools_with_timeout to avoid actual network calls
    async def mock_fetch_tools_with_timeout(client, server_name):
        return []  # Return empty list of tools

    with (
        patch.object(
            global_mcp_server_manager,
            "_create_mcp_client",
            side_effect=mock_create_mcp_client,
        ) as mock_create_client,
        patch.object(
            global_mcp_server_manager,
            "_fetch_tools_with_timeout",
            side_effect=mock_fetch_tools_with_timeout,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            AsyncMock(return_value=[oauth2_server]),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._prefetch_oauth_creds_for_user",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_user_oauth_extra_headers_from_db",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        # Call _get_tools_from_mcp_servers which should eventually call _create_mcp_client
        await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,  # Will use all allowed servers
            oauth2_headers=oauth2_headers,
        )

    # Verify that _create_mcp_client was called
    assert mock_create_client.call_count == 1, "Expected _create_mcp_client to be called once"

    # Verify the server passed to _create_mcp_client is the OAuth2 server
    assert captured_client_args["server"].server_id == oauth2_server.server_id
    assert captured_client_args["server"].auth_type == MCPAuth.oauth2

    # Security: a v2-migrated authorization_code server must NOT forward the caller's
    # oauth2 Authorization upstream. The v2 resolver injects the stored per-user token,
    # so a caller-supplied bearer cannot override another user's stored credential.
    extra_headers = captured_client_args["extra_headers"]
    assert extra_headers is None or "Authorization" not in {k.lower() for k in extra_headers}, (
        f"Caller Authorization must not be forwarded, got {extra_headers}"
    )


@pytest.mark.asyncio
async def test_list_tools_single_server_unprefixed_names():
    """When only one MCP server is allowed, list tools should return prefixed names (server prefix is always added)."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # One allowed server
    server = MagicMock()
    server.server_id = "server1"
    server.name = "Zapier MCP"
    server.alias = "zapier"
    server.allowed_tools = None
    server.disallowed_tools = None
    server.server_name = "server1"
    server.auth_type = None
    server.extra_headers = None

    # Mock manager: allow just one server and return a tool based on add_prefix flag
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server1"])
    mock_manager.get_mcp_server_by_id = MagicMock(return_value=server)
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip_with_info = lambda server_ids, client_ip: (
        server_ids,
        0,
    )

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=False,
        raw_headers=None,
        **kwargs,
    ):
        tool = MagicMock()
        tool.name = f"{server.alias}-toolA" if add_prefix else "toolA"
        tool.description = "desc"
        tool.inputSchema = {}
        return [tool]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Server prefix is always added regardless of number of allowed servers
    assert len(tools) == 1
    assert tools[0].name == "zapier-toolA"


@pytest.mark.asyncio
async def test_list_tools_multiple_servers_prefixed_names():
    """When multiple MCP servers are allowed, list tools should return prefixed names."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # Two allowed servers
    server1 = MagicMock()
    server1.server_id = "server1"
    server1.name = "Zapier MCP"
    server1.alias = "zapier"
    server1.allowed_tools = None
    server1.disallowed_tools = None
    server1.server_name = "server1"
    server1.auth_type = None
    server1.extra_headers = None

    server2 = MagicMock()
    server2.server_id = "server2"
    server2.name = "Jira MCP"
    server2.alias = "jira"
    server2.allowed_tools = None
    server2.disallowed_tools = None
    server2.server_name = "server2"
    server2.auth_type = None
    server2.extra_headers = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server1", "server2"])
    mock_manager.get_mcp_server_by_id = lambda server_id: server1 if server_id == "server1" else server2
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip_with_info = lambda server_ids, client_ip: (
        server_ids,
        0,
    )

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=True,
        raw_headers=None,
        **kwargs,
    ):
        tool = MagicMock()
        # When multiple servers, add_prefix should be True -> prefixed names
        tool.name = f"{server.alias}-toolA" if add_prefix else "toolA"
        tool.description = "desc"
        tool.inputSchema = {}
        return [tool]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Should be prefixed since multiple servers are allowed
    names = sorted([t.name for t in tools])
    assert names == ["jira-toolA", "zapier-toolA"]


@pytest.mark.asyncio
async def test_mcp_manager_allows_public_servers_without_permissions():
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    manager = MCPServerManager()
    public_server = MCPServer(
        server_id="public",
        name="public",
        transport=MCPTransport.http,
        allow_all_keys=True,
    )
    manager.registry = {public_server.server_id: public_server}

    with (
        patch(
            "litellm.proxy.management_endpoints.common_utils._user_has_admin_view",
            return_value=False,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPRequestHandler.get_allowed_mcp_servers",
            AsyncMock(return_value=[]),
        ),
    ):
        allowed = await manager.get_allowed_mcp_servers(UserAPIKeyAuth())

    assert allowed == ["public"]


@pytest.mark.asyncio
async def test_mcp_manager_returns_public_when_permission_lookup_fails():
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    manager = MCPServerManager()
    public_server = MCPServer(
        server_id="public",
        name="public",
        transport=MCPTransport.http,
        allow_all_keys=True,
    )
    manager.registry = {public_server.server_id: public_server}

    with (
        patch(
            "litellm.proxy.management_endpoints.common_utils._user_has_admin_view",
            return_value=False,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPRequestHandler.get_allowed_mcp_servers",
            AsyncMock(side_effect=Exception("boom")),
        ),
    ):
        allowed = await manager.get_allowed_mcp_servers(UserAPIKeyAuth())

    assert allowed == ["public"]


@pytest.mark.asyncio
async def test_mcp_manager_merges_public_and_restricted_servers():
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    manager = MCPServerManager()
    public_server = MCPServer(
        server_id="public",
        name="public",
        transport=MCPTransport.http,
        allow_all_keys=True,
    )
    scoped_server = MCPServer(
        server_id="restricted",
        name="restricted",
        transport=MCPTransport.http,
    )
    manager.registry = {
        public_server.server_id: public_server,
        scoped_server.server_id: scoped_server,
    }

    with (
        patch(
            "litellm.proxy.management_endpoints.common_utils._user_has_admin_view",
            return_value=False,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPRequestHandler.get_allowed_mcp_servers",
            AsyncMock(return_value=["restricted"]),
        ),
    ):
        allowed = await manager.get_allowed_mcp_servers(UserAPIKeyAuth())

    assert set(allowed) == {"public", "restricted"}


@pytest.mark.asyncio
async def test_call_mcp_tool_user_unauthorized_access():
    """Test that a user cannot call a tool from a server they don't have access to"""
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.server import call_mcp_tool
    from litellm.proxy._types import UserAPIKeyAuth

    # Create a mock user without access to the server
    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="team-basic",
        object_permission_id="key-permission-123",
    )

    # Mock global_mcp_server_manager.get_mcp_server_by_id to return servers
    # only for allowed servers, not "restricted_server" (the server the user is trying to access)
    allowed_server_obj = MagicMock()
    allowed_server_obj.name = "allowed_server"
    allowed_server_obj.server_name = "allowed_server"
    allowed_server_obj.server_id = "allowed_server"
    allowed_server_obj.alias = "allowed_server"
    allowed_server_obj.allowed_tools = None
    allowed_server_obj.disallowed_tools = None
    allowed_server_obj.auth_type = None
    allowed_server_obj.extra_headers = None

    another_server_obj = MagicMock()
    another_server_obj.name = "another_server"
    another_server_obj.server_name = "another_server"
    another_server_obj.server_id = "another_server"
    another_server_obj.alias = "another_server"
    another_server_obj.allowed_tools = None
    another_server_obj.disallowed_tools = None
    another_server_obj.auth_type = None
    another_server_obj.extra_headers = None

    def mock_get_server_by_id(server_id):
        if server_id == "allowed_server":
            return allowed_server_obj
        elif server_id == "another_server":
            return another_server_obj
        return None

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.get_allowed_mcp_servers",
            AsyncMock(return_value=["allowed_server", "another_server"]),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_id",
            side_effect=mock_get_server_by_id,
        ),
    ):
        # Try to call a tool from "restricted_server" - should raise HTTPException with 403 status
        with pytest.raises(HTTPException) as exc_info:
            await call_mcp_tool(
                name="restricted_server-send_email",
                arguments={
                    "to": "test@example.com",
                    "subject": "Test",
                    "body": "Test",
                },
                user_api_key_auth=mock_user_auth,
                mcp_auth_header="Bearer test_token",
            )

        # Verify the exception details
        assert exc_info.value.status_code == 403
        assert "User not allowed to call this tool" in exc_info.value.detail


@pytest.mark.asyncio
async def test_list_tools_filters_by_key_team_permissions():
    """Test that list_tools filters tools based on key/team mcp_tool_permissions"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
    except ImportError:
        pytest.skip("MCP server not available")

    # Create object permission with tool-level restrictions
    object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm_123",
        mcp_tool_permissions={
            "server1": ["tool1", "tool2"],  # Only allow tool1 and tool2
        },
    )

    user_api_key_auth = UserAPIKeyAuth(
        api_key="test_key",
        user_id="test_user",
        object_permission=object_permission,
    )
    set_auth_context(user_api_key_auth)

    # Mock server
    server = MagicMock()
    server.server_id = "server1"
    server.name = "Test Server"
    server.alias = "test"
    server.short_prefix = None
    server.allowed_tools = None
    server.disallowed_tools = None
    server.server_name = "server1"
    server.auth_type = None
    server.extra_headers = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server1"])
    mock_manager.get_mcp_server_by_id = lambda server_id: server
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip_with_info = lambda server_ids, client_ip: (
        server_ids,
        0,
    )

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=False,
        raw_headers=None,
        **kwargs,
    ):
        # Return 4 tools, but only 2 should be allowed
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "Tool 1"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.description = "Tool 2"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "tool3"
        tool3.description = "Tool 3 - not allowed"
        tool3.inputSchema = {}

        tool4 = MagicMock()
        tool4.name = "tool4"
        tool4.description = "Tool 4 - not allowed"
        tool4.inputSchema = {}

        return [tool1, tool2, tool3, tool4]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Should only return tool1 and tool2
    assert len(tools) == 2
    tool_names = sorted([t.name for t in tools])
    assert tool_names == ["tool1", "tool2"]


@pytest.mark.asyncio
async def test_list_tools_with_team_tool_permissions_inheritance():
    """Test that list_tools correctly applies key/team tool permissions inheritance logic"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import (
            LiteLLM_ObjectPermissionTable,
            UserAPIKeyAuth,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Team allows tool1, tool2, tool3
    team_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="team_perm_123",
        mcp_tool_permissions={
            "server1": ["tool1", "tool2", "tool3"],
        },
    )

    # Key allows tool2, tool3, tool4 - intersection should be tool2, tool3
    key_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="key_perm_456",
        mcp_tool_permissions={
            "server1": ["tool2", "tool3", "tool4"],
        },
    )

    user_api_key_auth = UserAPIKeyAuth(
        api_key="test_key",
        user_id="test_user",
        team_id="team_123",
        object_permission=key_object_permission,
    )
    set_auth_context(user_api_key_auth)

    # Mock server
    server = MagicMock()
    server.server_id = "server1"
    server.name = "Test Server"
    server.alias = "test"
    server.short_prefix = None
    server.allowed_tools = None
    server.disallowed_tools = None
    server.server_name = "server1"
    server.auth_type = None
    server.extra_headers = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server1"])
    mock_manager.get_mcp_server_by_id = lambda server_id: server
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip_with_info = lambda server_ids, client_ip: (
        server_ids,
        0,
    )

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=False,
        raw_headers=None,
        **kwargs,
    ):
        # Return 4 tools
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "Tool 1"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.description = "Tool 2"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "tool3"
        tool3.description = "Tool 3"
        tool3.inputSchema = {}

        tool4 = MagicMock()
        tool4.name = "tool4"
        tool4.description = "Tool 4"
        tool4.inputSchema = {}

        return [tool1, tool2, tool3, tool4]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        # Mock the team object permission retrieval
        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_team_object_permission",
            AsyncMock(return_value=team_object_permission),
        ):
            tools = await _get_tools_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=None,
                mcp_servers=None,
                mcp_server_auth_headers=None,
            )

    # Should only return tool2 and tool3 (intersection of key and team permissions)
    assert len(tools) == 2
    tool_names = sorted([t.name for t in tools])
    assert tool_names == ["tool2", "tool3"]


@pytest.mark.asyncio
async def test_list_tools_with_no_tool_permissions_shows_all():
    """Test that list_tools shows all tools when no mcp_tool_permissions are set"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import UserAPIKeyAuth
    except ImportError:
        pytest.skip("MCP server not available")

    # No tool-level restrictions
    user_api_key_auth = UserAPIKeyAuth(
        api_key="test_key",
        user_id="test_user",
        object_permission=None,
    )
    set_auth_context(user_api_key_auth)

    # Mock server
    server = MagicMock()
    server.server_id = "server1"
    server.name = "Test Server"
    server.alias = "test"
    server.short_prefix = None
    server.allowed_tools = None
    server.disallowed_tools = None
    server.server_name = "server1"
    server.auth_type = None
    server.extra_headers = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server1"])
    mock_manager.get_mcp_server_by_id = lambda server_id: server
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip_with_info = lambda server_ids, client_ip: (
        server_ids,
        0,
    )

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=False,
        raw_headers=None,
        **kwargs,
    ):
        # Return 3 tools
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "Tool 1"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.description = "Tool 2"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "tool3"
        tool3.description = "Tool 3"
        tool3.inputSchema = {}

        return [tool1, tool2, tool3]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Should return all tools when no restrictions
    assert len(tools) == 3
    tool_names = sorted([t.name for t in tools])
    assert tool_names == ["tool1", "tool2", "tool3"]


@pytest.mark.asyncio
async def test_list_tools_strips_prefix_when_matching_permissions():
    """
    Test that tool permission filtering correctly strips prefixes from tool names.

    Tools from MCP servers are prefixed (e.g., "GITMCP-fetch_litellm_documentation"),
    but allowed tools in DB are stored without prefix (e.g., "fetch_litellm_documentation").
    The filtering should strip the prefix before comparing.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
    except ImportError:
        pytest.skip("MCP server not available")

    # Create object permission with tool-level restrictions (WITHOUT prefix)
    object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm_123",
        mcp_tool_permissions={
            "gitmcp_server": [
                "fetch_litellm_documentation",  # No prefix in DB
                "search_litellm_code",  # No prefix in DB
            ],
        },
    )

    user_api_key_auth = UserAPIKeyAuth(
        api_key="test_key",
        user_id="test_user",
        object_permission=object_permission,
    )
    set_auth_context(user_api_key_auth)

    # Mock server
    server = MagicMock()
    server.server_id = "gitmcp_server"
    server.name = "GITMCP"
    server.alias = "GITMCP"
    server.short_prefix = None
    server.server_name = "GITMCP"
    server.allowed_tools = None
    server.disallowed_tools = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["gitmcp_server"])
    mock_manager.get_mcp_server_by_id = MagicMock(return_value=server)
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip_with_info = lambda server_ids, client_ip: (
        server_ids,
        0,
    )

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=True,
        raw_headers=None,
        **kwargs,
    ):
        # Return tools WITH prefix (as they come from MCP server)
        tool1 = MagicMock()
        tool1.name = "GITMCP-fetch_litellm_documentation"  # Prefixed
        tool1.description = "Fetch docs"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "GITMCP-search_litellm_documentation"  # Prefixed, not in allowed list
        tool2.description = "Search docs"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "GITMCP-search_litellm_code"  # Prefixed
        tool3.description = "Search code"
        tool3.inputSchema = {}

        tool4 = MagicMock()
        tool4.name = "GITMCP-fetch_generic_url_content"  # Prefixed, not in allowed list
        tool4.description = "Fetch URL"
        tool4.inputSchema = {}

        return [tool1, tool2, tool3, tool4]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Should only return the 2 tools that match (after stripping prefix)
    assert len(tools) == 2
    tool_names = sorted([t.name for t in tools])
    # Tools still have prefixes in the output, but were filtered correctly
    assert tool_names == [
        "GITMCP-fetch_litellm_documentation",
        "GITMCP-search_litellm_code",
    ]


def test_filter_tools_by_allowed_tools():
    """Test that filter_tools_by_allowed_tools filters tools correctly"""
    from mcp.types import Tool

    from litellm.proxy._experimental.mcp_server.server import (
        filter_tools_by_allowed_tools,
    )
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mcp_server = MCPServer(
        server_id="my_api_mcp",
        name="my_api_mcp",
        alias="my_api_mcp",
        transport=MCPTransport.http,
        allowed_tools=["getpetbyid", "my_api_mcp-findpetsbystatus"],
        disallowed_tools=None,
    )
    tools_to_return = [
        Tool(
            name="my_api_mcp-getpetbyid",
            title=None,
            description="Find pet by ID",
            inputSchema={
                "type": "object",
                "properties": {"petId": {"type": "integer", "description": ""}},
                "required": ["petId"],
            },
            outputSchema=None,
            annotations=None,
        ),
        Tool(
            name="my_api_mcp-findpetsbystatus",
            title=None,
            description="Finds Pets by status",
            inputSchema={
                "type": "object",
                "properties": {"status": {"type": "string", "description": ""}},
                "required": ["status"],
            },
            outputSchema=None,
            annotations=None,
        ),
        Tool(
            name="my_api_mcp-addpet",
            title=None,
            description="Add a new pet to the store",
            inputSchema={
                "type": "object",
                "properties": {
                    "body": {
                        "type": "object",
                        "description": "Request body",
                        "properties": {
                            "name": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    }
                },
                "required": ["body"],
            },
            outputSchema=None,
            annotations=None,
        ),
    ]

    filtered_tools = filter_tools_by_allowed_tools(tools_to_return, mcp_server)

    assert len(filtered_tools) == 2
    assert filtered_tools[0].name == "my_api_mcp-getpetbyid"
    assert filtered_tools[1].name == "my_api_mcp-findpetsbystatus"


def test_apply_tool_overrides():
    """Test that apply_tool_overrides applies custom display names and descriptions."""
    from mcp.types import Tool

    from litellm.proxy._experimental.mcp_server.server import apply_tool_overrides
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mcp_server = MCPServer(
        server_id="my_api_mcp",
        name="my_api_mcp",
        transport=MCPTransport.http,
        tool_name_to_display_name={"getpetbyid": "Get Pet"},
        tool_name_to_description={"getpetbyid": "Custom description for get pet"},
    )
    tools = [
        Tool(
            name="my_api_mcp-getpetbyid",
            title=None,
            description="Original description",
            inputSchema={"type": "object", "properties": {}},
            outputSchema=None,
            annotations=None,
        ),
        Tool(
            name="my_api_mcp-findpetsbystatus",
            title=None,
            description="Finds Pets by status",
            inputSchema={"type": "object", "properties": {}},
            outputSchema=None,
            annotations=None,
        ),
    ]

    result = apply_tool_overrides(tools, mcp_server)

    # First tool should have overridden name and description
    assert result[0].name == "Get Pet"
    assert result[0].description == "Custom description for get pet"
    # Second tool should be unchanged
    assert result[1].name == "my_api_mcp-findpetsbystatus"
    assert result[1].description == "Finds Pets by status"


def test_apply_tool_overrides_no_overrides():
    """Test that apply_tool_overrides returns tools unchanged when no overrides are set."""
    from mcp.types import Tool

    from litellm.proxy._experimental.mcp_server.server import apply_tool_overrides
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mcp_server = MCPServer(
        server_id="my_api_mcp",
        name="my_api_mcp",
        transport=MCPTransport.http,
    )
    tools = [
        Tool(
            name="my_api_mcp-getpetbyid",
            title=None,
            description="Original description",
            inputSchema={"type": "object", "properties": {}},
            outputSchema=None,
            annotations=None,
        ),
    ]

    result = apply_tool_overrides(tools, mcp_server)
    assert result[0].name == "my_api_mcp-getpetbyid"
    assert result[0].description == "Original description"


def _make_db_mcp_server(server_id: str, updated_at: datetime) -> LiteLLM_MCPServerTable:
    return LiteLLM_MCPServerTable(
        server_id=server_id,
        server_name="server",
        alias="server",
        url="https://example.com",
        transport=MCPTransport.http,
        created_at=updated_at,
        updated_at=updated_at,
        mcp_info={},
    )


class TestMCPServerManagerReload:
    @pytest.mark.asyncio
    async def test_reuses_existing_server_when_updated_at_matches(self):
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                MCPServerManager,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        manager = MCPServerManager()
        timestamp = datetime.utcnow()
        existing_server = MCPServer(
            server_id="server-1",
            name="server",
            transport=MCPTransport.http,
            updated_at=timestamp,
        )
        manager.registry = {existing_server.server_id: existing_server}

        db_row = _make_db_mcp_server("server-1", timestamp)

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[db_row])
        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma,
            ),
            patch.object(manager, "build_mcp_server_from_table", AsyncMock()) as mock_build,
        ):
            await manager.reload_servers_from_database()

        mock_build.assert_not_awaited()
        assert manager.registry["server-1"] is existing_server

    @pytest.mark.asyncio
    async def test_rebuilds_server_when_updated_at_changes(self):
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                MCPServerManager,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        manager = MCPServerManager()
        timestamp = datetime.utcnow()
        existing_server = MCPServer(
            server_id="server-1",
            name="server",
            transport=MCPTransport.http,
            updated_at=timestamp,
        )
        manager.registry = {existing_server.server_id: existing_server}

        new_timestamp = timestamp + timedelta(minutes=5)
        db_row = _make_db_mcp_server("server-1", new_timestamp)
        rebuilt_server = MCPServer(
            server_id="server-1",
            name="server",
            transport=MCPTransport.http,
            updated_at=new_timestamp,
        )

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[db_row])
        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma,
            ),
            patch.object(
                manager,
                "build_mcp_server_from_table",
                AsyncMock(return_value=rebuilt_server),
            ) as mock_build,
        ):
            await manager.reload_servers_from_database()

        mock_build.assert_awaited_once_with(db_row, env_vars_are_encrypted=True)
        assert manager.registry["server-1"] is rebuilt_server

    @pytest.mark.asyncio
    async def test_skips_server_when_build_from_database_fails(self, caplog):
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                MCPServerManager,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        manager = MCPServerManager()
        timestamp = datetime.utcnow()
        healthy_row = _make_db_mcp_server("healthy-server", timestamp)
        bad_row = _make_db_mcp_server("bad-server", timestamp)
        another_healthy_row = _make_db_mcp_server("another-healthy-server", timestamp)

        healthy_server = MCPServer(
            server_id="healthy-server",
            name="healthy",
            transport=MCPTransport.http,
            updated_at=timestamp,
        )
        another_healthy_server = MCPServer(
            server_id="another-healthy-server",
            name="another-healthy",
            transport=MCPTransport.http,
            updated_at=timestamp,
        )

        async def build_server(db_row, **kwargs):
            if db_row.server_id == "bad-server":
                raise RuntimeError("transient build failure")
            if db_row.server_id == "healthy-server":
                return healthy_server
            return another_healthy_server

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_many = AsyncMock(
            return_value=[healthy_row, bad_row, another_healthy_row]
        )
        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma,
            ),
            patch.object(
                manager,
                "build_mcp_server_from_table",
                AsyncMock(side_effect=build_server),
            ),
            patch.object(manager, "_maybe_register_openapi_tools", AsyncMock()),
            caplog.at_level("ERROR", logger="LiteLLM"),
        ):
            await manager.reload_servers_from_database()

        assert set(manager.registry) == {"healthy-server", "another-healthy-server"}
        assert manager.registry["healthy-server"] is healthy_server
        assert manager.registry["another-healthy-server"] is another_healthy_server
        assert "Skipping MCP server bad-server" in caplog.text

    @pytest.mark.asyncio
    async def test_skips_server_when_openapi_registration_fails(self, caplog):
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                MCPServerManager,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        manager = MCPServerManager()
        timestamp = datetime.utcnow()
        healthy_row = _make_db_mcp_server("healthy-server", timestamp)
        bad_openapi_row = _make_db_mcp_server("bad-openapi-server", timestamp)
        existing_server = MCPServer(
            server_id="existing-server",
            name="existing",
            transport=MCPTransport.http,
            updated_at=timestamp,
        )
        manager.registry = {existing_server.server_id: existing_server}

        healthy_server = MCPServer(
            server_id="healthy-server",
            name="healthy",
            transport=MCPTransport.http,
            updated_at=timestamp,
        )
        bad_openapi_server = MCPServer(
            server_id="bad-openapi-server",
            name="bad-openapi",
            transport=MCPTransport.http,
            spec_path="https://example.invalid/openapi.json",
            updated_at=timestamp,
        )

        async def build_server(db_row, **kwargs):
            if db_row.server_id == "healthy-server":
                return healthy_server
            return bad_openapi_server

        observed_registries = []

        async def register_openapi_tools(server, **kwargs):
            observed_registries.append(set(manager.registry))
            assert kwargs == {"initialize_mapping": False}
            if server.server_id == "bad-openapi-server":
                raise RuntimeError("blocked address")

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[healthy_row, bad_openapi_row])
        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma,
            ),
            patch.object(
                manager,
                "build_mcp_server_from_table",
                AsyncMock(side_effect=build_server),
            ),
            patch.object(
                manager,
                "_maybe_register_openapi_tools",
                AsyncMock(side_effect=register_openapi_tools),
            ),
            caplog.at_level("ERROR", logger="LiteLLM"),
        ):
            await manager.reload_servers_from_database()

        assert set(manager.registry) == {"healthy-server"}
        assert manager.registry["healthy-server"] is healthy_server
        assert observed_registries == [
            {"existing-server"},
            {"existing-server"},
        ]
        assert "Skipping MCP server bad-openapi-server" in caplog.text


@pytest.mark.asyncio
async def test_call_mcp_tool_logs_failure_via_post_call_failure_hook():
    """
    Regression test for 6267f168...:
    Ensure proxy-side `call_mcp_tool` logs failures via `proxy_logging_obj.post_call_failure_hook`.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            call_mcp_tool,
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport, UserAPIKeyAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    mock_server = MCPServer(
        server_id="server-123",
        name="test_server",
        alias="test_server",
        server_name="test_server",
        url="https://test-server.com/mcp",
        transport=MCPTransport.http,
        mcp_info={"server_name": "test_server"},
    )

    proxy_logging_mock = MagicMock()
    proxy_logging_mock.post_call_failure_hook = AsyncMock()

    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

    with (
        patch.object(
            global_mcp_server_manager,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
            return_value=[mock_server.server_id],
        ),
        patch.object(
            global_mcp_server_manager,
            "get_mcp_server_by_id",
            return_value=mock_server,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers_from_mcp_server_names",
            new_callable=AsyncMock,
            return_value=[mock_server],
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.execute_mcp_tool",
            new_callable=AsyncMock,
            side_effect=Exception("boom"),
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
            proxy_logging_mock,
        ),
    ):
        with pytest.raises(Exception):
            await call_mcp_tool(
                name="test_server-any_tool",
                arguments={"x": 1},
                user_api_key_auth=user_auth,
                litellm_call_id="cid",
            )

    proxy_logging_mock.post_call_failure_hook.assert_awaited_once()
    assert proxy_logging_mock.post_call_failure_hook.await_args.kwargs.get("route") == "/mcp/call_tool"


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_logs_list_tools_to_spendlogs_when_enabled():
    """
    Regression test for 872e5b98...:
    Ensure list-tools logging path calls `async_success_handler` when enabled.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
        )
        from litellm.proxy._types import UserAPIKeyAuth
        from mcp.types import Tool as MCPTool
    except ImportError:
        pytest.skip("MCP server not available")

    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

    server_a = MagicMock(name="server_a_obj")
    server_a.name = "server_a"
    server_a.alias = "server_a"
    server_a.server_name = "server_a"
    server_a.server_id = "a"
    server_a.auth_type = None
    server_a.extra_headers = None

    tool_1 = MCPTool(
        name="server_a-tool_1",
        description="test tool",
        inputSchema={"type": "object"},
    )

    dummy_logging_obj = MagicMock()
    dummy_logging_obj.model_call_details = {"metadata": {"spend_logs_metadata": {}}}
    dummy_logging_obj.async_success_handler = AsyncMock()
    function_setup_kwargs = {}

    def _capture_function_setup(*_args, **kwargs):
        function_setup_kwargs.update(kwargs)
        return dummy_logging_obj, None

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            new=AsyncMock(return_value=[server_a]),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
            return_value=(None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_allowed_tools",
            side_effect=lambda tools, _server: tools,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_key_team_permissions",
            new=AsyncMock(side_effect=lambda tools, **_: tools),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.function_setup",
            side_effect=_capture_function_setup,
        ),
    ):
        mock_manager._get_tools_from_server = AsyncMock(return_value=[tool_1])

        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_auth,
            mcp_auth_header=None,
            mcp_servers=["server_a"],
            mcp_server_auth_headers=None,
            log_list_tools_to_spendlogs=True,
            list_tools_log_source="mcp_protocol",
            request_tags=["team-a"],
        )

    assert tools == [tool_1]
    dummy_logging_obj.async_success_handler.assert_awaited_once()
    assert dummy_logging_obj.async_success_handler.await_args.kwargs["result"] == [tool_1.model_dump(mode="json")]
    assert function_setup_kwargs["metadata"]["tags"] == ["team-a"]

    spend_meta = dummy_logging_obj.model_call_details["metadata"]["spend_logs_metadata"]
    assert spend_meta["tool_count_total"] == 1
    assert spend_meta["allowed_server_count"] == 1
    assert spend_meta["per_server_tool_counts"]["server_a"] == 1


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_returns_tools_when_success_logging_fails():
    """
    Regression test: list_tools should still return fetched tools even if
    async_success_handler raises (e.g. serialization errors in logging path).
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
        )
        from litellm.proxy._types import UserAPIKeyAuth
    except ImportError:
        pytest.skip("MCP server not available")

    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

    server_a = MagicMock(name="server_a_obj")
    server_a.name = "server_a"
    server_a.alias = "server_a"
    server_a.server_name = "server_a"
    server_a.server_id = "a"
    server_a.auth_type = None
    server_a.extra_headers = None

    tool_1 = MagicMock()
    tool_1.name = "server_a-tool_1"

    dummy_logging_obj = MagicMock()
    dummy_logging_obj.model_call_details = {"metadata": {"spend_logs_metadata": {}}}
    dummy_logging_obj.async_success_handler = AsyncMock(
        side_effect=TypeError("Object of type Tool is not JSON serializable")
    )

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            new=AsyncMock(return_value=[server_a]),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
            return_value=(None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_allowed_tools",
            side_effect=lambda tools, _server: tools,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_key_team_permissions",
            new=AsyncMock(side_effect=lambda tools, **_: tools),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.function_setup",
            return_value=(dummy_logging_obj, None),
        ),
    ):
        mock_manager._get_tools_from_server = AsyncMock(return_value=[tool_1])

        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_auth,
            mcp_auth_header=None,
            mcp_servers=["server_a"],
            mcp_server_auth_headers=None,
            log_list_tools_to_spendlogs=True,
            list_tools_log_source="mcp_protocol",
        )

    assert tools == [tool_1]
    dummy_logging_obj.async_success_handler.assert_awaited_once()


def test_tool_name_matches_case_insensitive():
    """Test that _tool_name_matches performs case-insensitive comparison.

    This is critical for OpenAPI-based MCP servers where:
    1. operationIds are often in camelCase (e.g., 'addPet', 'updatePet')
    2. Tool names are lowercased during registration (e.g., 'addpet', 'updatepet')
    3. allowed_tools configuration may use the original camelCase names

    Without case-insensitive matching, all tools would be filtered out.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import _tool_name_matches
    except ImportError:
        pytest.skip("MCP server not available")

    # Test case 1: Unprefixed tool name with camelCase in filter list
    assert _tool_name_matches("addpet", ["addPet", "updatePet"]) is True
    assert _tool_name_matches("updatepet", ["addPet", "updatePet"]) is True
    assert _tool_name_matches("deletepet", ["addPet", "updatePet"]) is False

    # Test case 2: Prefixed tool name with camelCase in filter list
    assert _tool_name_matches("per_store-addpet", ["addPet", "updatePet"]) is True
    assert _tool_name_matches("per_store-updatepet", ["addPet", "updatePet"]) is True
    assert _tool_name_matches("per_store-deletepet", ["addPet", "updatePet"]) is False

    # Test case 3: Mixed case variations
    assert _tool_name_matches("findPetsByStatus", ["findpetsbystatus"]) is True
    assert _tool_name_matches("findpetsbystatus", ["findPetsByStatus"]) is True
    assert _tool_name_matches("FINDPETSBYSTATUS", ["findPetsByStatus"]) is True

    # Test case 4: Full prefixed name in filter list (case-insensitive)
    assert _tool_name_matches("server-addPet", ["server-addpet"]) is True
    assert _tool_name_matches("server-addpet", ["server-addPet"]) is True

    # Test case 5: Ensure non-matching names still don't match
    assert _tool_name_matches("addpet", ["deletePet", "updatePet"]) is False
    assert _tool_name_matches("server-addpet", ["deletePet", "updatePet"]) is False


def test_filter_tools_by_allowed_tools_case_insensitive():
    """Test that filter_tools_by_allowed_tools handles case-insensitive matching.

    Ensures that OpenAPI tools with lowercase names can be filtered using
    camelCase allowed_tools configuration from the OpenAPI spec.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            filter_tools_by_allowed_tools,
        )
        from litellm.types.mcp_server.tool_registry import MCPTool
    except ImportError:
        pytest.skip("MCP server not available")

    # Mock handler function
    def mock_handler(**kwargs):
        return kwargs

    # Create mock tools with lowercase names (as registered from OpenAPI)
    tools = [
        MCPTool(
            name="per_store-addpet",
            description="Add a pet",
            input_schema={"type": "object"},
            handler=mock_handler,
        ),
        MCPTool(
            name="per_store-updatepet",
            description="Update a pet",
            input_schema={"type": "object"},
            handler=mock_handler,
        ),
        MCPTool(
            name="per_store-deletepet",
            description="Delete a pet",
            input_schema={"type": "object"},
            handler=mock_handler,
        ),
        MCPTool(
            name="per_store-findpetsbystatus",
            description="Find pets by status",
            input_schema={"type": "object"},
            handler=mock_handler,
        ),
    ]

    # Create mock server with camelCase allowed_tools (as from OpenAPI spec)
    server = MCPServer(
        server_id="test-server",
        name="per_store",
        transport=MCPTransport.http,
        allowed_tools=["addPet", "updatePet", "findPetsByStatus"],
    )

    # Filter tools
    filtered_tools = filter_tools_by_allowed_tools(tools, server)

    # Should return 3 tools (case-insensitive match)
    assert len(filtered_tools) == 3
    assert any(t.name == "per_store-addpet" for t in filtered_tools)
    assert any(t.name == "per_store-updatepet" for t in filtered_tools)
    assert any(t.name == "per_store-findpetsbystatus" for t in filtered_tools)
    assert not any(t.name == "per_store-deletepet" for t in filtered_tools)


def test_filter_tools_by_allowed_tools_no_filter():
    """Test that filter_tools_by_allowed_tools returns all tools when no filter is set."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            filter_tools_by_allowed_tools,
        )
        from litellm.types.mcp_server.tool_registry import MCPTool
    except ImportError:
        pytest.skip("MCP server not available")

    # Mock handler function
    def mock_handler(**kwargs):
        return kwargs

    tools = [
        MCPTool(
            name="fusion_litellm_mcp-model_list",
            description="List models",
            input_schema={"type": "object"},
            handler=mock_handler,
        ),
        MCPTool(
            name="fusion_litellm_mcp-chat_completion",
            description="Chat completion",
            input_schema={"type": "object"},
            handler=mock_handler,
        ),
    ]

    # Server with no allowed_tools filter
    server = MCPServer(
        server_id="test-server",
        name="fusion_litellm_mcp",
        transport=MCPTransport.http,
        allowed_tools=None,
    )

    filtered_tools = filter_tools_by_allowed_tools(tools, server)

    # Should return all tools when no filter is configured
    assert len(filtered_tools) == 2


def test_filter_tools_enforced_empty_allowlist_blocks_all():
    from mcp.types import Tool

    from litellm.proxy._experimental.mcp_server.server import (
        filter_tools_by_allowed_tools,
    )
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    tools = [
        Tool(
            name="read_wiki_structure",
            title=None,
            description="",
            inputSchema={"type": "object"},
            outputSchema=None,
            annotations=None,
        ),
    ]
    server = MCPServer(
        server_id="deepwiki",
        name="deepwiki",
        transport=MCPTransport.http,
        allowed_tools=[],
        mcp_info={"tool_allowlist_enforced": True},
    )

    assert filter_tools_by_allowed_tools(tools, server) == []


def test_filter_tools_legacy_empty_allowlist_allows_all():
    from mcp.types import Tool

    from litellm.proxy._experimental.mcp_server.server import (
        filter_tools_by_allowed_tools,
    )
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    tools = [
        Tool(
            name="read_wiki_structure",
            title=None,
            description="",
            inputSchema={"type": "object"},
            outputSchema=None,
            annotations=None,
        ),
    ]
    server = MCPServer(
        server_id="legacy",
        name="legacy",
        transport=MCPTransport.http,
        allowed_tools=[],
        mcp_info=None,
    )

    assert len(filter_tools_by_allowed_tools(tools, server)) == 1


def test_check_allowed_or_banned_tools_enforced_empty_denies_calls():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager.__new__(MCPServerManager)
    server = MCPServer(
        server_id="deepwiki",
        name="deepwiki",
        transport=MCPTransport.http,
        allowed_tools=[],
        mcp_info={"tool_allowlist_enforced": True},
    )

    assert manager.check_allowed_or_banned_tools("read_wiki_structure", server) is False


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_injects_stored_oauth2_token():
    """
    When _get_tools_from_mcp_servers is called for an OAuth2 MCP server and no
    oauth2_headers are provided in the request (e.g. a /responses API call from a
    chat UI), the per-user stored token must be fetched from the DB and passed as
    extra_headers to _get_tools_from_server.

    The implementation pre-fetches all user credentials in a single bulk query
    (_prefetch_oauth_creds_for_user) to avoid N+1 queries in the gather loop.

    This covers the bug where OAuth2 MCP tools were always empty in the /responses
    API because the stored credential was never injected.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
        )
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.mcp import MCPAuth
    except ImportError:
        pytest.skip("MCP server not available")

    STORED_TOKEN = "atlassian-oauth-access-token-xyz"
    SERVER_ID = "srv-oauth2-id"
    USER_ID = "user-123"

    user_auth = UserAPIKeyAuth(api_key="test-key", user_id=USER_ID)

    oauth2_server = MagicMock(name="atlassian_server")
    oauth2_server.name = "atlassian_test"
    oauth2_server.alias = "atlassian_test"
    oauth2_server.server_name = "atlassian_test"
    oauth2_server.server_id = SERVER_ID
    oauth2_server.auth_type = MCPAuth.oauth2
    oauth2_server.extra_headers = None

    # Simulate the DB returning a valid credential for this user+server
    prefetched_creds = {SERVER_ID: {"access_token": STORED_TOKEN, "server_id": SERVER_ID}}

    tool_1 = MagicMock()
    tool_1.name = "atlassian_test-search"

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
            new=AsyncMock(return_value=[oauth2_server]),
        ),
        patch(
            # Patch the bulk prefetch so no real DB connection is needed
            "litellm.proxy._experimental.mcp_server.server._prefetch_oauth_creds_for_user",
            new=AsyncMock(return_value=prefetched_creds),
        ) as mock_prefetch,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_allowed_tools",
            side_effect=lambda tools, _server: tools,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_key_team_permissions",
            new=AsyncMock(side_effect=lambda tools, **_: tools),
        ),
    ):
        mock_manager._get_tools_from_server = AsyncMock(return_value=[tool_1])

        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_auth,
            mcp_auth_header=None,
            mcp_servers=["atlassian_test"],
            mcp_server_auth_headers=None,
            oauth2_headers=None,  # No token from request — must fall back to DB
        )

    # Bulk credential prefetch was called once (not once per server)
    mock_prefetch.assert_awaited_once_with(user_auth)

    # The stored token was forwarded to the MCP transport layer as extra_headers
    mock_manager._get_tools_from_server.assert_awaited_once()
    call_kwargs = mock_manager._get_tools_from_server.await_args.kwargs
    assert call_kwargs["extra_headers"] == {"Authorization": f"Bearer {STORED_TOKEN}"}

    assert tools == [tool_1]


# ---------------------------------------------------------------------------
# _merge_gateway_initialize_instructions + ContextVar / InitializationOptions
# ---------------------------------------------------------------------------


def _make_instruction_server(
    server_id="s1",
    name="s1",
    *,
    alias=None,
    server_name=None,
    instructions=None,
    spec_path=None,
    url="https://example.com",
):
    return MCPServer(
        server_id=server_id,
        name=name,
        alias=alias,
        server_name=server_name,
        url=url,
        transport=MCPTransport.http,
        instructions=instructions,
        spec_path=spec_path,
    )


class TestMergeGatewayInitializeInstructions:
    """Tests for _merge_gateway_initialize_instructions."""

    def _merge(self, servers):
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _merge_gateway_initialize_instructions,
            )
        except ImportError:
            pytest.skip("MCP server not available")
        return _merge_gateway_initialize_instructions(servers)

    def test_empty_server_list_returns_none(self):
        """No servers yields no instructions."""
        assert self._merge([]) is None

    def test_single_server_yaml_instructions(self):
        """A single server with YAML instructions returns them verbatim."""
        s = _make_instruction_server(instructions="Use add() for sums.")
        assert self._merge([s]) == "Use add() for sums."

    def test_yaml_instructions_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        s = _make_instruction_server(instructions="  padded  \n")
        assert self._merge([s]) == "padded"

    def test_yaml_override_beats_upstream_cache(self):
        """YAML/DB instructions take precedence over upstream cache."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        global_mcp_server_manager._upstream_initialize_instructions_by_server_id["s1"] = "upstream"
        try:
            s = _make_instruction_server(instructions="yaml wins")
            assert self._merge([s]) == "yaml wins"
        finally:
            global_mcp_server_manager._upstream_initialize_instructions_by_server_id.pop("s1", None)

    def test_upstream_cache_used_when_no_yaml(self):
        """Upstream cached instructions are used when no YAML override is set."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        global_mcp_server_manager._upstream_initialize_instructions_by_server_id["s1"] = "from upstream"
        try:
            s = _make_instruction_server(instructions=None)
            assert self._merge([s]) == "from upstream"
        finally:
            global_mcp_server_manager._upstream_initialize_instructions_by_server_id.pop("s1", None)

    def test_spec_path_servers_skipped(self):
        """OpenAPI (spec_path) servers do not contribute instructions."""
        s = _make_instruction_server(spec_path="/openapi.json", url=None)
        assert self._merge([s]) is None

    def test_no_instructions_no_cache_returns_none(self):
        """Server with no instructions and no cache yields None."""
        s = _make_instruction_server()
        assert self._merge([s]) is None

    def test_multiple_servers_merged_with_labels(self):
        """Multiple servers get label-prefixed and separator-joined."""
        s1 = _make_instruction_server(server_id="a", name="a", alias="Alpha", instructions="instr A")
        s2 = _make_instruction_server(server_id="b", name="b", alias="Beta", instructions="instr B")
        result = self._merge([s1, s2])
        assert result is not None
        assert "[Alpha]" in result and "[Beta]" in result
        assert "instr A" in result and "instr B" in result
        assert "---" in result

    def test_single_server_no_label_wrapping(self):
        """A single server's instructions are not wrapped with a label."""
        s = _make_instruction_server(alias="MyServer", instructions="single")
        result = self._merge([s])
        assert result == "single"
        assert "[MyServer]" not in result

    def test_mixed_yaml_cache_specpath(self):
        """YAML, upstream-cache, and spec_path servers are handled correctly together."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        global_mcp_server_manager._upstream_initialize_instructions_by_server_id["c"] = "cached C"
        try:
            s_yaml = _make_instruction_server(server_id="a", name="a", alias="A", instructions="yaml A")
            s_spec = _make_instruction_server(server_id="b", name="b", alias="B", spec_path="/spec.json", url=None)
            s_cached = _make_instruction_server(server_id="c", name="c", alias="C")
            result = self._merge([s_yaml, s_spec, s_cached])
            assert "yaml A" in result
            assert "cached C" in result
            assert "[B]" not in result
        finally:
            global_mcp_server_manager._upstream_initialize_instructions_by_server_id.pop("c", None)


class TestEnsureUpstreamInitializeInstructionsCached:
    @pytest.mark.asyncio
    async def test_skips_when_yaml_instructions_set(self):
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        server = _make_instruction_server(server_id="yaml-only", instructions="from yaml")
        with patch.object(global_mcp_server_manager, "_create_mcp_client", AsyncMock()) as mock_create:
            await global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(server)
        mock_create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_already_cached(self):
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        server = _make_instruction_server(server_id="cached-only", instructions=None)
        global_mcp_server_manager._upstream_initialize_instructions_by_server_id["cached-only"] = "warm"
        try:
            with patch.object(global_mcp_server_manager, "_create_mcp_client", AsyncMock()) as mock_create:
                await global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(server)
            mock_create.assert_not_awaited()
        finally:
            global_mcp_server_manager._upstream_initialize_instructions_by_server_id.pop("cached-only", None)

    @pytest.mark.asyncio
    async def test_skips_when_spec_path_set(self):
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        server = _make_instruction_server(server_id="openapi-spec", spec_path="/openapi.json", url=None)
        with patch.object(global_mcp_server_manager, "_create_mcp_client", AsyncMock()) as mock_create:
            await global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(server)
        mock_create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_runs_upstream_session_and_caches(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        server = _make_instruction_server(server_id="cold-server", instructions=None)
        fake_client = MagicMock()
        fake_client.run_with_session = AsyncMock(return_value="ok")
        fake_client._last_initialize_instructions = "  upstream says hi  "

        with patch.object(
            global_mcp_server_manager,
            "_create_mcp_client",
            AsyncMock(return_value=fake_client),
        ):
            try:
                await global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(server)
                assert (
                    global_mcp_server_manager._upstream_initialize_instructions_by_server_id["cold-server"]
                    == "upstream says hi"
                )
            finally:
                global_mcp_server_manager._upstream_initialize_instructions_by_server_id.pop("cold-server", None)
                global_mcp_server_manager._upstream_initialize_instructions_probed_at.pop("cold-server", None)

    @pytest.mark.asyncio
    async def test_cooldown_after_empty_upstream_response(self):
        """Upstream returns no instructions → next call within cooldown must not reconnect."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        server = _make_instruction_server(server_id="empty-server", instructions=None)
        fake_client = MagicMock()
        fake_client.run_with_session = AsyncMock(return_value="ok")
        fake_client._last_initialize_instructions = None  # upstream sent nothing

        create = AsyncMock(return_value=fake_client)
        with patch.object(global_mcp_server_manager, "_create_mcp_client", create):
            try:
                await global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(server)
                await global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(server)
                assert create.await_count == 1, "Second probe within cooldown must not reconnect to upstream"
                assert "empty-server" not in global_mcp_server_manager._upstream_initialize_instructions_by_server_id
                assert "empty-server" in global_mcp_server_manager._upstream_initialize_instructions_probed_at
            finally:
                global_mcp_server_manager._upstream_initialize_instructions_probed_at.pop("empty-server", None)

    @pytest.mark.asyncio
    async def test_cooldown_after_upstream_failure(self):
        """run_with_session raises → cooldown applies, no immediate retry."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        server = _make_instruction_server(server_id="boom-server", instructions=None)
        fake_client = MagicMock()
        fake_client.run_with_session = AsyncMock(side_effect=RuntimeError("upstream down"))
        fake_client._last_initialize_instructions = None

        create = AsyncMock(return_value=fake_client)
        with patch.object(global_mcp_server_manager, "_create_mcp_client", create):
            try:
                await global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(server)
                await global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(server)
                assert create.await_count == 1, "Second probe within cooldown must not reconnect after failure"
                assert "boom-server" not in global_mcp_server_manager._upstream_initialize_instructions_by_server_id
                assert "boom-server" in global_mcp_server_manager._upstream_initialize_instructions_probed_at
            finally:
                global_mcp_server_manager._upstream_initialize_instructions_probed_at.pop("boom-server", None)

    @pytest.mark.asyncio
    async def test_reload_resets_probe_cooldown(self):
        """load_servers_from_config clears the negative-cache map so reloads re-probe."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        global_mcp_server_manager._upstream_initialize_instructions_probed_at["reload-target"] = 1.0
        try:
            await global_mcp_server_manager.load_servers_from_config({})
            assert "reload-target" not in global_mcp_server_manager._upstream_initialize_instructions_probed_at
        finally:
            global_mcp_server_manager._upstream_initialize_instructions_probed_at.pop("reload-target", None)


class TestGatewayCreateInitializationOptions:
    """Tests for the patched server.create_initialization_options via ContextVar."""

    def test_no_contextvar_returns_default_options(self):
        """When ContextVar is None, instructions are absent."""
        try:
            from litellm.proxy._experimental.mcp_server.mcp_context import (
                _mcp_gateway_initialize_instructions,
                _mcp_gateway_server_name,
            )
            from litellm.proxy._experimental.mcp_server.server import server
        except ImportError:
            pytest.skip("MCP server not available")

        instructions_token = _mcp_gateway_initialize_instructions.set(None)
        server_name_token = _mcp_gateway_server_name.set(None)
        try:
            opts = server.create_initialization_options()
            assert getattr(opts, "instructions", None) is None
            assert opts.server_name == "litellm-mcp-server"
        finally:
            _mcp_gateway_initialize_instructions.reset(instructions_token)
            _mcp_gateway_server_name.reset(server_name_token)

    @pytest.mark.asyncio
    async def test_scoped_request_uses_configured_server_alias(self):
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _gateway_initialize_instructions_request_scope,
                global_mcp_server_manager,
                server,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scoped_server = MCPServer(
            server_id="server-123",
            name="upstream-server",
            alias="grafana",
            transport=MCPTransport.http,
            url="https://example.com/mcp",
        )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
                new_callable=AsyncMock,
                return_value=[scoped_server],
            ),
            patch.object(
                global_mcp_server_manager,
                "_ensure_upstream_initialize_instructions_cached",
                new_callable=AsyncMock,
            ),
        ):
            async with _gateway_initialize_instructions_request_scope(
                user_api_key_auth=None,
                mcp_servers=["grafana"],
                client_ip=None,
                scoped_server_endpoint=True,
            ):
                assert server.create_initialization_options().server_name == "grafana"

        assert server.create_initialization_options().server_name == "litellm-mcp-server"

    @pytest.mark.asyncio
    async def test_sse_handler_scopes_server_name_from_single_server_path(self):
        try:
            from litellm.proxy._experimental.mcp_server import server as mcp_server
            from litellm.proxy._experimental.mcp_server.server import (
                global_mcp_server_manager,
                handle_sse_mcp,
                server,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scoped_server = MCPServer(
            server_id="server-123",
            name="upstream-server",
            alias="grafana",
            transport=MCPTransport.http,
            url="https://example.com/mcp",
        )
        captured = {}

        async def record_request(scope, receive, send):
            captured["server_name"] = server.create_initialization_options().server_name

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/grafana",
            "headers": [],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new_callable=AsyncMock,
                return_value=(
                    UserAPIKeyAuth(api_key="sk-test"),
                    None,
                    ["grafana"],
                    None,
                    None,
                    None,
                ),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
                new_callable=AsyncMock,
                return_value=[scoped_server],
            ),
            patch.object(
                global_mcp_server_manager,
                "_ensure_upstream_initialize_instructions_cached",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._raise_preemptive_401_for_unauthenticated_servers",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._check_passthrough_upstream_auth",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch.object(
                mcp_server.sse_session_manager,
                "handle_request",
                side_effect=record_request,
            ),
        ):
            await handle_sse_mcp(scope, AsyncMock(), AsyncMock())

        assert captured["server_name"] == "grafana"
        assert server.create_initialization_options().server_name == "litellm-mcp-server"

    def test_contextvar_set_injects_instructions(self):
        """When ContextVar has a value, it appears in InitializationOptions."""
        try:
            from litellm.proxy._experimental.mcp_server.mcp_context import (
                _mcp_gateway_initialize_instructions,
            )
            from litellm.proxy._experimental.mcp_server.server import server
        except ImportError:
            pytest.skip("MCP server not available")

        tok = _mcp_gateway_initialize_instructions.set("hello from merge")
        try:
            opts = server.create_initialization_options()
            assert opts.instructions == "hello from merge"
        finally:
            _mcp_gateway_initialize_instructions.reset(tok)

    def test_contextvar_reset_removes_instructions(self):
        """After resetting the ContextVar, instructions disappear."""
        try:
            from litellm.proxy._experimental.mcp_server.mcp_context import (
                _mcp_gateway_initialize_instructions,
            )
            from litellm.proxy._experimental.mcp_server.server import server
        except ImportError:
            pytest.skip("MCP server not available")

        tok = _mcp_gateway_initialize_instructions.set("temporary")
        _mcp_gateway_initialize_instructions.reset(tok)
        opts = server.create_initialization_options()
        assert getattr(opts, "instructions", None) is None


@pytest.mark.asyncio
async def test_list_tools_with_legacy_db_m2m_server_resolves_oauth2_flow():
    """
    P1 Regression: list_tools path must apply _resolve_oauth2_flow to legacy DB
    rows where oauth2_flow is NULL but M2M credentials are present.

    Without this fix, has_client_credentials returns False and the caller's
    Authorization header is forwarded upstream instead of being blocked.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
        )
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.mcp import MCPAuth
    except ImportError:
        pytest.skip("MCP server not available")

    user_auth = UserAPIKeyAuth(api_key="sk-1234", user_id="test-user")

    # Simulate a legacy DB row: OAuth2 with M2M credentials but oauth2_flow=None
    legacy_server = MagicMock(name="legacy_m2m_server")
    legacy_server.name = "legacy_m2m"
    legacy_server.alias = "legacy_m2m"
    legacy_server.server_name = "legacy_m2m"
    legacy_server.server_id = "legacy-m2m-id"
    legacy_server.auth_type = MCPAuth.oauth2
    legacy_server.oauth2_flow = None  # Legacy: field not set in DB
    legacy_server.token_exchange_endpoint = None
    legacy_server.audience = None
    legacy_server.subject_token_type = None
    legacy_server.token_exchange_profile = None
    legacy_server.token_url = "https://oauth.example.com/token"
    legacy_server.authorization_url = None
    legacy_server.client_id = "client-id"
    legacy_server.client_secret = "client-secret"
    legacy_server.extra_headers = None
    legacy_server.has_client_credentials = False  # This is the bug: should be True
    legacy_server.model_copy = MagicMock(
        side_effect=lambda update: MCPServer(
            server_id=legacy_server.server_id,
            name=legacy_server.name,
            # Carry alias/server_name forward so get_server_prefix resolves to
            # "legacy_m2m" (not the server_id) when the request scope filter
            # matches by alias. Without these, the filter relied on the now-
            # removed silent fail-open fallback.
            alias=legacy_server.alias,
            server_name=legacy_server.server_name,
            transport=MCPTransport.http,
            auth_type=legacy_server.auth_type,
            oauth2_flow=update.get("oauth2_flow", legacy_server.oauth2_flow),
            token_url=legacy_server.token_url,
            authorization_url=legacy_server.authorization_url,
            client_id=legacy_server.client_id,
            client_secret=legacy_server.client_secret,
        )
    )

    tool_1 = MagicMock()
    tool_1.name = "legacy_m2m-tool"

    captured_extra_headers = None

    async def capture_extra_headers(*args, **kwargs):
        nonlocal captured_extra_headers
        captured_extra_headers = kwargs.get("extra_headers")
        return [tool_1]

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_allowed_tools",
            side_effect=lambda tools, _server: tools,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.filter_tools_by_key_team_permissions",
            new=AsyncMock(side_effect=lambda tools, **_: tools),
        ),
    ):
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["legacy-m2m-id"])
        mock_manager.get_mcp_server_by_id = MagicMock(return_value=legacy_server)
        mock_manager.filter_server_ids_by_ip_with_info = MagicMock(return_value=(["legacy-m2m-id"], 0))
        mock_manager._get_tools_from_server = AsyncMock(side_effect=capture_extra_headers)

        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_auth,
            mcp_auth_header=None,
            mcp_servers=["legacy_m2m"],
            mcp_server_auth_headers=None,
            oauth2_headers={"Authorization": "Bearer sk-1234"},  # Caller's token
        )

    # With P1 fix: _get_allowed_mcp_servers applies _resolve_oauth2_flow,
    # so has_client_credentials becomes True and extra_headers should be None
    # (caller's Authorization blocked)
    assert captured_extra_headers is None, (
        "P1 security issue: caller's Authorization header was forwarded to M2M server. "
        "Expected None, got: " + str(captured_extra_headers)
    )
    assert tools == [tool_1]


@pytest.mark.asyncio
async def test_call_tool_empty_extra_headers_returns_none():
    """
    P2 Regression: When all configured extra_headers are filtered out (e.g.
    Authorization for M2M), the resulting extra_headers should be None, not {}.

    Downstream code that checks `if extra_headers is None` will behave
    differently if an empty dict is passed instead.
    """
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.types.mcp import MCPAuth
    except ImportError:
        pytest.skip("MCP server not available")

    manager = MCPServerManager()

    # M2M server with only Authorization in extra_headers
    m2m_server = MCPServer(
        server_id="m2m-srv",
        name="m2m_test",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow="client_credentials",
        token_url="https://oauth.example.com/token",
        client_id="client-id",
        client_secret="client-secret",
        extra_headers=["Authorization"],  # Will be filtered out for M2M
    )

    raw_headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
    }

    captured_extra_headers = None

    async def capture_create_mcp_client(*args, **kwargs):
        nonlocal captured_extra_headers
        captured_extra_headers = kwargs.get("extra_headers")
        # Return a mock client
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=MagicMock(content=[]))
        return mock_client

    with (
        patch.object(
            manager,
            "_create_mcp_client",
            side_effect=capture_create_mcp_client,
        ),
        patch.object(
            manager,
            "get_mcp_server_by_id",
            return_value=m2m_server,
        ),
    ):
        try:
            await manager._call_regular_mcp_tool(
                mcp_server=m2m_server,
                original_tool_name="test_tool",
                arguments={},
                mcp_auth_header=None,
                oauth2_headers=None,
                raw_headers=raw_headers,
            )
        except Exception:
            pass  # We only care about the captured headers

    # With P2 fix: extra_headers should be None (not {}) when all headers filtered
    assert captured_extra_headers is None, (
        "P2 API consistency issue: expected None for empty extra_headers, got: " + str(captured_extra_headers)
    )


# ---------------------------------------------------------------------------
# Pre-flight upstream auth check tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_upstream_auth_returns_upstream_status():
    """_probe_upstream_auth forwards the status code from the upstream server."""
    from litellm.proxy._experimental.mcp_server.server import _probe_upstream_auth

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {"www-authenticate": 'Bearer realm="test"'}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(
        "litellm.proxy._experimental.mcp_server.server.get_async_httpx_client",
        return_value=mock_client,
    ):
        status, www_auth = await _probe_upstream_auth("http://upstream/mcp", "Bearer some-token")

    assert status == 401
    assert www_auth == 'Bearer realm="test"'
    mock_client.post.assert_awaited_once()
    _, kwargs = mock_client.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer some-token"
    assert kwargs["json"]["method"] == "initialize"


@pytest.mark.asyncio
async def test_probe_upstream_auth_surfaces_httpx_status_error():
    """Probe extracts status + WWW-Authenticate from httpx.HTTPStatusError.

    AsyncHTTPHandler.post() calls raise_for_status() internally, so when the
    upstream returns 401/403 the call raises httpx.HTTPStatusError rather than
    returning the response. The probe must catch that specifically (before the
    fail-open `except Exception`) so the auth check is not silently defeated.
    """
    import httpx

    from litellm.proxy._experimental.mcp_server.server import _probe_upstream_auth

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {"www-authenticate": 'Bearer realm="test"'}
    request = httpx.Request("POST", "http://upstream/mcp")
    error = httpx.HTTPStatusError(message="401 Unauthorized", request=request, response=mock_response)

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=error)

    with patch(
        "litellm.proxy._experimental.mcp_server.server.get_async_httpx_client",
        return_value=mock_client,
    ):
        status, www_auth = await _probe_upstream_auth("http://upstream/mcp", "Bearer some-token")

    assert status == 401
    assert www_auth == 'Bearer realm="test"'


@pytest.mark.asyncio
async def test_probe_upstream_auth_fails_open_on_network_error():
    """_probe_upstream_auth returns (200, None) when the network call fails."""
    from litellm.proxy._experimental.mcp_server.server import _probe_upstream_auth

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=Exception("connection refused"))

    with patch(
        "litellm.proxy._experimental.mcp_server.server.get_async_httpx_client",
        return_value=mock_client,
    ):
        status, www_auth = await _probe_upstream_auth("http://upstream/mcp", "Bearer some-token")

    assert status == 200
    assert www_auth is None


def test_get_forwarded_auth_from_scope_extracts_header():
    """Returns Authorization value when x-litellm-api-key is also present."""
    from litellm.proxy._experimental.mcp_server.server import (
        _get_forwarded_auth_from_scope,
    )

    scope = {
        "headers": [
            (b"content-type", b"application/json"),
            (b"x-litellm-api-key", b"sk-litellm-proxy-key"),
            (b"authorization", b"Bearer my-token"),
        ]
    }
    assert _get_forwarded_auth_from_scope(scope) == "Bearer my-token"


def test_get_forwarded_auth_from_scope_returns_none_when_missing():
    from litellm.proxy._experimental.mcp_server.server import (
        _get_forwarded_auth_from_scope,
    )

    assert _get_forwarded_auth_from_scope({"headers": []}) is None


def test_get_forwarded_auth_from_scope_skips_when_no_litellm_key_header():
    """Skip when ``x-litellm-api-key`` is absent.

    Without ``x-litellm-api-key``, the ``Authorization`` header may itself be
    the LiteLLM proxy API key (backward-compat). Forwarding it upstream would
    leak the proxy key, so the helper must return None and the probe must
    not fire.
    """
    from litellm.proxy._experimental.mcp_server.server import (
        _get_forwarded_auth_from_scope,
    )

    scope = {
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer ambiguous-token"),
        ]
    }
    assert _get_forwarded_auth_from_scope(scope) is None


@pytest.mark.asyncio
async def test_create_mcp_client_sampling_disabled_by_default():
    """Sampling callback must be None when allow_sampling is not set (default False)."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()
    server = MCPServer(
        server_id="no-sampling",
        name="no-sampling",
        url="https://example.com/mcp",
        transport=MCPTransport.http,
    )

    client = await manager._create_mcp_client(server=server)
    assert client._sampling_callback is None


@pytest.mark.asyncio
async def test_create_mcp_client_sampling_enabled():
    """Sampling callback must be set when allow_sampling=True."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()
    server = MCPServer(
        server_id="with-sampling",
        name="with-sampling",
        url="https://example.com/mcp",
        transport=MCPTransport.http,
        allow_sampling=True,
    )

    client = await manager._create_mcp_client(server=server)
    assert client._sampling_callback is not None


@pytest.mark.asyncio
async def test_execute_mcp_tool_rest_server_id_authoritative_for_unprefixed_tool():
    """REST server_id + unprefixed tool name must not use global tool-name mapping."""
    from mcp.types import TextContent

    from litellm.proxy._experimental.mcp_server import server as mcp_module

    api_key_server = MCPServer(
        server_id="api-key-server-id",
        name="echo_api_key",
        server_name="echo_api_key",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
        authentication_token="abc123",
    )
    oauth_server = MCPServer(
        server_id="oauth-server-id",
        name="echo_oauth_m2m",
        server_name="echo_oauth_m2m",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        token_url="http://127.0.0.1:8080/token",
        client_id="client",
        client_secret="secret",
    )

    captured: dict = {}

    async def fake_handle_managed_mcp_tool(**kwargs):
        captured.update(kwargs)
        return mcp_module.CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )

    with (
        patch.dict(
            mcp_module.global_mcp_server_manager.tool_name_to_mcp_server_name_mapping,
            {"echo": oauth_server.name},
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "get_registry",
            return_value={
                api_key_server.server_id: api_key_server,
                oauth_server.server_id: oauth_server,
            },
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=oauth_server,
        ),
        patch.object(
            mcp_module,
            "_handle_managed_mcp_tool",
            new=fake_handle_managed_mcp_tool,
        ),
        patch.object(
            mcp_module.MCPRequestHandler,
            "is_tool_allowed",
            return_value=True,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=None,
        ),
    ):
        await mcp_module.execute_mcp_tool(
            name="echo",
            arguments={"message": "hello"},
            allowed_mcp_servers=[api_key_server, oauth_server],
            start_time=datetime.now(),
            requested_server_id=api_key_server.server_id,
        )

    assert captured["server_name"] == "echo_api_key"
    assert captured["name"] == "echo"


@pytest.mark.asyncio
async def test_execute_mcp_tool_rest_server_id_injects_requested_server_credentials():
    """REST server_id must inject the requested server's auth, not a URL-collision peer's."""
    from mcp.types import TextContent

    from litellm.proxy._experimental.mcp_server import server as mcp_module

    requested_server = MCPServer(
        server_id="requested-server-id",
        name="echo_requested",
        server_name="echo_requested",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
        authentication_token="requested-secret",
    )
    collision_server = MCPServer(
        server_id="collision-server-id",
        name="echo_collision",
        server_name="echo_collision",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.bearer_token,
        authentication_token="collision-secret",
    )

    fake_client = MagicMock()
    fake_client._last_initialize_instructions = None
    fake_client.call_tool = AsyncMock(
        return_value=mcp_module.CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
    )

    injected: dict = {}

    async def fake_create_mcp_client(server, **kwargs):
        injected["server"] = server
        return fake_client

    with (
        patch.dict(
            mcp_module.global_mcp_server_manager.tool_name_to_mcp_server_name_mapping,
            {"echo": collision_server.name},
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "get_registry",
            return_value={
                requested_server.server_id: requested_server,
                collision_server.server_id: collision_server,
            },
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_create_mcp_client",
            new=fake_create_mcp_client,
        ),
        patch.object(
            mcp_module.MCPRequestHandler,
            "is_tool_allowed",
            return_value=True,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=None,
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", None),
    ):
        await mcp_module.execute_mcp_tool(
            name="echo",
            arguments={"message": "hello"},
            allowed_mcp_servers=[requested_server, collision_server],
            start_time=datetime.now(),
            requested_server_id=requested_server.server_id,
        )

    routed = injected["server"]
    assert routed.server_id == requested_server.server_id
    assert routed.auth_type == MCPAuth.api_key
    assert routed.authentication_token == "requested-secret"
    assert routed.authentication_token != collision_server.authentication_token


@pytest.mark.asyncio
async def test_execute_mcp_tool_rest_prefixed_tool_still_validates_server_id():
    """Prefixed REST tool names must still match the requested server_id."""
    from litellm.proxy._experimental.mcp_server import server as mcp_module

    api_key_server = MCPServer(
        server_id="api-key-server-id",
        name="echo_api_key",
        server_name="echo_api_key",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
        authentication_token="abc123",
    )
    oauth_server = MCPServer(
        server_id="oauth-server-id",
        name="echo_oauth_m2m",
        server_name="echo_oauth_m2m",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        token_url="http://127.0.0.1:8080/token",
        client_id="client",
        client_secret="secret",
    )

    with (
        patch.object(
            mcp_module.global_mcp_server_manager,
            "get_registry",
            return_value={
                api_key_server.server_id: api_key_server,
                oauth_server.server_id: oauth_server,
            },
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=oauth_server,
        ),
        patch.object(
            mcp_module.MCPRequestHandler,
            "is_tool_allowed",
            return_value=True,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=None,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await mcp_module.execute_mcp_tool(
            name="echo_oauth_m2m-echo",
            arguments={"message": "hello"},
            allowed_mcp_servers=[api_key_server, oauth_server],
            start_time=datetime.now(),
            requested_server_id=api_key_server.server_id,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "tool_server_mismatch"


@pytest.mark.asyncio
async def test_execute_mcp_tool_rest_unauthorized_prefix_still_mismatches():
    """Prefixed name for a registry server the caller cannot access must 403."""
    from litellm.proxy._experimental.mcp_server import server as mcp_module

    api_key_server = MCPServer(
        server_id="api-key-server-id",
        name="echo_api_key",
        server_name="echo_api_key",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
        authentication_token="abc123",
    )
    restricted_server = MCPServer(
        server_id="restricted-server-id",
        name="restricted_server",
        server_name="restricted_server",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.bearer_token,
        authentication_token="secret",
    )

    with (
        patch.object(
            mcp_module.global_mcp_server_manager,
            "get_registry",
            return_value={
                api_key_server.server_id: api_key_server,
                restricted_server.server_id: restricted_server,
            },
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=restricted_server,
        ),
        patch.object(
            mcp_module.MCPRequestHandler,
            "is_tool_allowed",
            return_value=True,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=None,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await mcp_module.execute_mcp_tool(
            name="restricted_server-echo",
            arguments={"message": "hello"},
            allowed_mcp_servers=[api_key_server],
            start_time=datetime.now(),
            requested_server_id=api_key_server.server_id,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "tool_server_mismatch"


@pytest.mark.asyncio
async def test_execute_mcp_tool_rest_hyphenated_upstream_tool_name_routes_to_requested_server():
    """REST server_id + hyphenated upstream tool name (no registry prefix) must route, not 400."""
    from mcp.types import TextContent

    from litellm.proxy._experimental.mcp_server import server as mcp_module

    api_key_server = MCPServer(
        server_id="api-key-server-id",
        name="echo_api_key",
        server_name="echo_api_key",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
        authentication_token="abc123",
    )

    captured: dict = {}

    async def fake_handle_managed_mcp_tool(**kwargs):
        captured.update(kwargs)
        return mcp_module.CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )

    with (
        patch.object(
            mcp_module.global_mcp_server_manager,
            "get_registry",
            return_value={api_key_server.server_id: api_key_server},
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=None,
        ),
        patch.object(
            mcp_module,
            "_handle_managed_mcp_tool",
            new=fake_handle_managed_mcp_tool,
        ),
        patch.object(
            mcp_module.MCPRequestHandler,
            "is_tool_allowed",
            return_value=True,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=None,
        ),
    ):
        await mcp_module.execute_mcp_tool(
            name="text-to-speech",
            arguments={"message": "hello"},
            allowed_mcp_servers=[api_key_server],
            start_time=datetime.now(),
            requested_server_id=api_key_server.server_id,
        )

    assert captured["server_name"] == "echo_api_key"
    assert captured["name"] == "text-to-speech"


@pytest.mark.asyncio
async def test_execute_mcp_tool_sets_model_in_model_call_details():
    """Regression test: MCP tools/call spend logs persisted with model="".

    execute_mcp_tool set logging_obj.model only; the spend-log writer reads
    model_call_details["model"], which stays None when function_setup builds
    the logging object without a "model" kwarg.
    """
    import uuid
    from datetime import timezone

    from litellm.proxy._experimental.mcp_server import server as mcp_module
    from litellm.proxy._types import LitellmUserRoles
    from litellm.utils import Rules, function_setup

    user = UserAPIKeyAuth(
        api_key="sk-user",
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    fake_server = MagicMock()
    fake_server.name = "openapi-petstore"
    fake_server.is_byok = False
    fake_server.auth_type = None
    fake_server.mcp_info = None
    fake_server.server_id = "srv-1"
    fake_server.server_name = "openapi-petstore"

    fake_tool = MagicMock()
    fake_tool.name = "list_pets"

    start_time = datetime.now(timezone.utc)
    litellm_logging_obj, _ = function_setup(
        original_function="call_mcp_tool",
        rules_obj=Rules(),
        start_time=start_time,
        litellm_call_id=str(uuid.uuid4()),
        name="list_pets",
        arguments={"limit": 10},
    )
    assert litellm_logging_obj.model_call_details.get("model") is None

    with (
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=fake_server,
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "pre_call_tool_check",
            new=AsyncMock(return_value={}),
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=fake_tool,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_local_mcp_tool",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.is_tool_allowed",
            return_value=True,
        ),
    ):
        await mcp_module.execute_mcp_tool(
            name="list_pets",
            arguments={"limit": 10},
            allowed_mcp_servers=[fake_server],
            start_time=start_time,
            user_api_key_auth=user,
            litellm_logging_obj=litellm_logging_obj,
        )

    assert litellm_logging_obj.model_call_details["model"] == "MCP: list_pets"
    assert litellm_logging_obj.model == "MCP: list_pets"


@pytest.mark.asyncio
async def test_execute_mcp_tool_rest_unresolved_prefixed_name_routes_to_requested_server():
    """A prefixed REST name that resolves to no tool must still dispatch to the server_id."""
    from mcp.types import TextContent

    from litellm.proxy._experimental.mcp_server import server as mcp_module

    requested_server = MCPServer(
        server_id="rest-target-id",
        name="rest_target",
        server_name="rest_target",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
        authentication_token="abc123",
    )
    prefix_owner = MCPServer(
        server_id="prefix-owner-id",
        name="known_prefix",
        server_name="known_prefix",
        url="http://127.0.0.1:5116/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.bearer_token,
        authentication_token="def456",
    )

    captured: dict = {}

    async def fake_handle_managed_mcp_tool(**kwargs):
        captured.update(kwargs)
        return mcp_module.CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )

    with (
        patch.object(
            mcp_module.global_mcp_server_manager,
            "get_registry",
            return_value={
                requested_server.server_id: requested_server,
                prefix_owner.server_id: prefix_owner,
            },
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=None,
        ),
        patch.object(
            mcp_module,
            "_handle_managed_mcp_tool",
            new=fake_handle_managed_mcp_tool,
        ),
        patch.object(
            mcp_module.MCPRequestHandler,
            "is_tool_allowed",
            return_value=True,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=None,
        ),
    ):
        await mcp_module.execute_mcp_tool(
            name="known_prefix-list_things",
            arguments={"message": "hello"},
            allowed_mcp_servers=[requested_server, prefix_owner],
            start_time=datetime.now(),
            requested_server_id=requested_server.server_id,
        )

    assert captured["server_name"] == "rest_target"
    assert captured["name"] == "list_things"

    routed_server = {
        requested_server.name: requested_server,
        prefix_owner.name: prefix_owner,
    }[captured["server_name"]]
    assert routed_server.server_id == requested_server.server_id
    assert routed_server.auth_type == MCPAuth.api_key
    assert routed_server.authentication_token == "abc123"
    assert routed_server.authentication_token != prefix_owner.authentication_token


@pytest.mark.asyncio
async def test_execute_mcp_tool_rest_prefix_retry_resolution_still_enforces_server_id():
    """A managed tool resolved via the requested server's prefix must still honor the server_id guard."""
    from litellm.proxy._experimental.mcp_server import server as mcp_module

    requested_server = MCPServer(
        server_id="api-key-server-id",
        name="echo_api_key",
        server_name="echo_api_key",
        url="http://127.0.0.1:5115/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
        authentication_token="abc123",
    )
    prefix_owner = MCPServer(
        server_id="prefix-owner-id",
        name="known_prefix",
        server_name="known_prefix",
        url="http://127.0.0.1:5116/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.bearer_token,
        authentication_token="secret",
    )

    def resolve_only_when_requested_prefix_added(tool_name):
        if tool_name == "known_prefix-echo":
            return None
        return prefix_owner

    with (
        patch.object(
            mcp_module.global_mcp_server_manager,
            "get_registry",
            return_value={
                requested_server.server_id: requested_server,
                prefix_owner.server_id: prefix_owner,
            },
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            side_effect=resolve_only_when_requested_prefix_added,
        ),
        patch.object(
            mcp_module.MCPRequestHandler,
            "is_tool_allowed",
            return_value=True,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=None,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await mcp_module.execute_mcp_tool(
            name="known_prefix-echo",
            arguments={"message": "hello"},
            allowed_mcp_servers=[requested_server, prefix_owner],
            start_time=datetime.now(),
            requested_server_id=requested_server.server_id,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "tool_server_mismatch"


# ---------------------------------------------------------------------------
# Regression tests for _get_allowed_mcp_servers_from_mcp_server_names
#
# Prior to the fail-closed fix, an unresolved scope filter (path- or
# header-derived) silently returned the caller's full allowed-server set,
# which made URL/header namespacing appear to work when it did not.
# ---------------------------------------------------------------------------


def _make_mcp_server_for_scope_filter(server_id: str, alias: str) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=alias,
        alias=alias,
        server_name=alias,
        url=f"https://{alias}.test/mcp",
        transport=MCPTransport.http,
        mcp_info={"server_name": alias},
    )


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_from_mcp_server_names_unknown_name_fails_closed():
    """
    Bug fix: requesting an unknown server name (e.g. ``/mcp/<typo>/``) must
    NOT silently fall back to the caller's full allowed-server set.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_allowed_mcp_servers_from_mcp_server_names,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    allowed = [
        _make_mcp_server_for_scope_filter("id-a", "alpha"),
        _make_mcp_server_for_scope_filter("id-b", "beta"),
    ]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp."
        "MCPRequestHandler._get_mcp_servers_from_access_groups",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await _get_allowed_mcp_servers_from_mcp_server_names(
            mcp_servers=["does-not-exist"],
            allowed_mcp_servers=allowed,
        )

    assert result == []


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_from_mcp_server_names_none_returns_all():
    """
    Regression: ``mcp_servers=None`` (no scope filter requested) must still
    return the full allowed-server set. This is the legitimate "no scoping"
    path that the fail-closed fix must not break.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_allowed_mcp_servers_from_mcp_server_names,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    allowed = [
        _make_mcp_server_for_scope_filter("id-a", "alpha"),
        _make_mcp_server_for_scope_filter("id-b", "beta"),
    ]

    result = await _get_allowed_mcp_servers_from_mcp_server_names(
        mcp_servers=None,
        allowed_mcp_servers=allowed,
    )

    assert {s.server_id for s in result} == {"id-a", "id-b"}


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_from_mcp_server_names_known_alias_returns_match():
    """
    Regression: a known server alias must still resolve to exactly that
    server. Guards against the fix accidentally narrowing the happy path.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_allowed_mcp_servers_from_mcp_server_names,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    allowed = [
        _make_mcp_server_for_scope_filter("id-a", "alpha"),
        _make_mcp_server_for_scope_filter("id-b", "beta"),
    ]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp."
        "MCPRequestHandler._get_mcp_servers_from_access_groups",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await _get_allowed_mcp_servers_from_mcp_server_names(
            mcp_servers=["alpha"],
            allowed_mcp_servers=allowed,
        )

    assert [s.server_id for s in result] == ["id-a"]


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_from_mcp_server_names_mixed_known_and_unknown():
    """
    Mixed scope (one valid + one unknown) returns only the resolved server,
    not the full allowed set. Confirms the fail-closed branch only fires
    when NOTHING resolves.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_allowed_mcp_servers_from_mcp_server_names,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    allowed = [
        _make_mcp_server_for_scope_filter("id-a", "alpha"),
        _make_mcp_server_for_scope_filter("id-b", "beta"),
    ]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp."
        "MCPRequestHandler._get_mcp_servers_from_access_groups",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await _get_allowed_mcp_servers_from_mcp_server_names(
            mcp_servers=["alpha", "does-not-exist"],
            allowed_mcp_servers=allowed,
        )

    assert [s.server_id for s in result] == ["id-a"]


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_from_mcp_server_names_access_group_resolves():
    """
    Regression: when a requested name is not a server alias but IS an access
    group, it must still resolve to the underlying servers (not be treated
    as unresolved).
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_allowed_mcp_servers_from_mcp_server_names,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    allowed = [
        _make_mcp_server_for_scope_filter("id-a", "alpha"),
        _make_mcp_server_for_scope_filter("id-b", "beta"),
    ]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp."
        "MCPRequestHandler._get_mcp_servers_from_access_groups",
        new_callable=AsyncMock,
        return_value=["id-b"],
    ):
        result = await _get_allowed_mcp_servers_from_mcp_server_names(
            mcp_servers=["group-name"],
            allowed_mcp_servers=allowed,
        )

    assert [s.server_id for s in result] == ["id-b"]


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_from_mcp_server_names_empty_list_fails_closed():
    """
    Edge case: ``mcp_servers=[]`` (explicit empty scope) is still an
    explicit filter request. Fail closed rather than returning everything.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_allowed_mcp_servers_from_mcp_server_names,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    allowed = [
        _make_mcp_server_for_scope_filter("id-a", "alpha"),
        _make_mcp_server_for_scope_filter("id-b", "beta"),
    ]

    result = await _get_allowed_mcp_servers_from_mcp_server_names(
        mcp_servers=[],
        allowed_mcp_servers=allowed,
    )

    assert result == []


class TestProxyExceptionToHttpException:
    """Auth failures reach the MCP ASGI handlers as ProxyException, not
    HTTPException. The handlers must map them back to their real status and
    headers; otherwise they fall through to the generic 500 handler, dropping
    the 401 + WWW-Authenticate challenge an OAuth client needs to re-authenticate
    and surfacing the tool call as a cancelled/terminated session.
    """

    def test_preserves_401_status_and_www_authenticate_header(self):
        from litellm.proxy._experimental.mcp_server.server import (
            _proxy_exception_to_http_exception,
        )
        from litellm.proxy._types import ProxyException

        exc = ProxyException(
            message="Authentication Error, invalid token",
            type="auth_error",
            param="key",
            code=401,
            headers={"WWW-Authenticate": 'Bearer resource_metadata="/x"'},
        )

        http_exc = _proxy_exception_to_http_exception(exc)

        assert http_exc.status_code == 401
        assert http_exc.detail == "Authentication Error, invalid token"
        assert http_exc.headers["WWW-Authenticate"] == 'Bearer resource_metadata="/x"'

    def test_preserves_403_status(self):
        from litellm.proxy._experimental.mcp_server.server import (
            _proxy_exception_to_http_exception,
        )
        from litellm.proxy._types import ProxyException

        http_exc = _proxy_exception_to_http_exception(
            ProxyException(message="Forbidden", type="auth_error", param="key", code=403)
        )

        assert http_exc.status_code == 403

    def test_non_numeric_code_falls_back_to_500(self):
        from litellm.proxy._experimental.mcp_server.server import (
            _proxy_exception_to_http_exception,
        )
        from litellm.proxy._types import ProxyException

        # ProxyException normalises code to the string "None" when unset.
        http_exc = _proxy_exception_to_http_exception(
            ProxyException(message="boom", type="server_error", param=None, code=None)
        )

        assert http_exc.status_code == 500


class TestStreamableHttpAuthErrorMapping:
    """End-to-end guard for the handler wiring: a ProxyException from auth must
    propagate as the real HTTPException (401 + WWW-Authenticate), not be
    flattened to a generic 500 by the catch-all handler.
    """

    @pytest.mark.asyncio
    async def test_streamable_http_propagates_proxy_exception_as_401(self):
        from litellm.proxy._experimental.mcp_server import server as mcp_module
        from litellm.proxy._types import ProxyException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/some_server",
            "headers": [(b"x-litellm-api-key", b"sk-bad")],
        }

        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        sent = []

        async def send(message):
            sent.append(message)

        auth_failure = ProxyException(
            message="Authentication Error, invalid token",
            type="auth_error",
            param="key",
            code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

        with patch.object(
            mcp_module,
            "extract_mcp_auth_context",
            new=AsyncMock(side_effect=auth_failure),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await mcp_module.handle_streamable_http_mcp(scope, receive, send)

        assert exc_info.value.status_code == 401
        assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"
        # Must not have emitted a 500 body via the generic catch-all.
        assert not any(m.get("type") == "http.response.start" and m.get("status") == 500 for m in sent)

    @pytest.mark.asyncio
    async def test_sse_propagates_proxy_exception_as_401(self):
        from litellm.proxy._experimental.mcp_server import server as mcp_module
        from litellm.proxy._types import ProxyException

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/mcp/some_server",
            "headers": [(b"x-litellm-api-key", b"sk-bad")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        sent = []

        async def send(message):
            sent.append(message)

        auth_failure = ProxyException(
            message="Authentication Error, invalid token",
            type="auth_error",
            param="key",
            code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

        with patch.object(
            mcp_module,
            "extract_mcp_auth_context",
            new=AsyncMock(side_effect=auth_failure),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await mcp_module.handle_sse_mcp(scope, receive, send)

        assert exc_info.value.status_code == 401
        assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"
        assert not any(m.get("type") == "http.response.start" and m.get("status") == 500 for m in sent)


class TestMCPMetaTraceCarrier:
    """`_mcp_meta_trace_carrier` extracts the W3C trace context the MCP client
    propagated in the request's params._meta (SEP-414) so the otel_v2 MCP span can
    parent to the client's span. Exercises the real MCP SDK `RequestParams.Meta`
    shape (extra='allow' preserves the unprefixed keys), not just an injected
    carrier."""

    def test_extracts_trace_context_and_excludes_baggage_and_other_meta(self):
        """Only traceparent/tracestate are carried. The client's W3C ``baggage`` is
        deliberately dropped even though it rides in params._meta: it is
        caller-controlled, and the otel baggage processor stamps allowlisted baggage
        keys onto the span, so honoring it would let a client spoof a span's identity
        (e.g. ``litellm.team.id``). Dropping it at the source is the regression guard."""
        from types import SimpleNamespace

        from mcp.types import RequestParams

        from litellm.proxy._experimental.mcp_server.server import (
            _mcp_meta_trace_carrier,
        )

        meta = RequestParams.Meta.model_validate(
            {
                "traceparent": "00-11111111111111111111111111111111-2222222222222222-01",
                "tracestate": "rojo=1",
                "baggage": "litellm.team.id=spoofed-team,litellm.metadata.user_api_key_user_id=attacker",
                "progressToken": "p1",
            }
        )
        carrier = _mcp_meta_trace_carrier(SimpleNamespace(meta=meta))
        assert carrier == {
            "traceparent": "00-11111111111111111111111111111111-2222222222222222-01",
            "tracestate": "rojo=1",
        }
        assert "baggage" not in carrier

    def test_none_when_no_trace_context(self):
        from types import SimpleNamespace

        from mcp.types import RequestParams

        from litellm.proxy._experimental.mcp_server.server import (
            _mcp_meta_trace_carrier,
        )

        assert _mcp_meta_trace_carrier(None) is None
        assert _mcp_meta_trace_carrier(SimpleNamespace(meta=None)) is None
        only_progress = RequestParams.Meta.model_validate({"progressToken": "p1"})
        assert _mcp_meta_trace_carrier(SimpleNamespace(meta=only_progress)) is None


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_includes_active_servers_submitted_by_user():
    """BYOM submitters can see approved servers they submitted without allow_all_keys."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    submitted_server = _make_mcp_server_for_scope_filter("submitted-1", "user_mcp")
    submitter = UserAPIKeyAuth(
        user_id="submitter-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key="sk-submitter",
    )
    other_user = UserAPIKeyAuth(
        user_id="other-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key="sk-other",
    )

    async def _submitted_ids(prisma_client, user_id):
        return ["submitted-1"] if user_id == "submitter-user" else []

    with (
        patch.object(
            global_mcp_server_manager,
            "get_registry",
            return_value={"submitted-1": submitted_server},
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp."
            "MCPRequestHandler.get_allowed_mcp_servers",
            AsyncMock(return_value=[]),
        ),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_active_submitted_mcp_server_ids_for_user",
            side_effect=_submitted_ids,
        ),
    ):
        submitter_allowed = await global_mcp_server_manager.get_allowed_mcp_servers(submitter)
        other_allowed = await global_mcp_server_manager.get_allowed_mcp_servers(other_user)

    assert "submitted-1" in submitter_allowed
    assert "submitted-1" not in other_allowed


@pytest.mark.asyncio
async def test_get_active_submitted_mcp_server_ids_for_user_queries_active_rows():
    from litellm.proxy._experimental.mcp_server.db import (
        get_active_submitted_mcp_server_ids_for_user,
    )
    from litellm.proxy._types import MCPApprovalStatus

    row = MagicMock()
    row.server_id = "submitted-1"
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[row])

    result = await get_active_submitted_mcp_server_ids_for_user(prisma_client, "submitter-user")

    assert result == ["submitted-1"]
    prisma_client.db.litellm_mcpservertable.find_many.assert_awaited_once_with(
        where={
            "submitted_by": "submitter-user",
            "approval_status": MCPApprovalStatus.active,
        },
    )


@pytest.mark.asyncio
async def test_get_active_submitted_mcp_server_ids_for_user_empty_user_id_skips_db():
    from litellm.proxy._experimental.mcp_server.db import (
        get_active_submitted_mcp_server_ids_for_user,
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_many = AsyncMock()

    assert await get_active_submitted_mcp_server_ids_for_user(prisma_client, "") == []
    prisma_client.db.litellm_mcpservertable.find_many.assert_not_awaited()


# --------------------------------------------------------------------------- #
#  MCP tool-call isError failure logging
# --------------------------------------------------------------------------- #


def _call_tool_result(is_error: bool, text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=is_error)


def _mock_mcp_logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.async_post_mcp_tool_call_hook = AsyncMock()
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.async_failure_handler = AsyncMock()
    return logging_obj


def test_extract_mcp_tool_result_error_message():
    from litellm.proxy._experimental.mcp_server.utils import (
        extract_mcp_tool_result_error_message,
    )

    assert extract_mcp_tool_result_error_message(_call_tool_result(True, "boom")) == "boom"
    assert extract_mcp_tool_result_error_message(_call_tool_result(False, "ok")) is None
    assert (
        extract_mcp_tool_result_error_message(CallToolResult(content=[], isError=True))
        == "MCP tool call returned isError=true"
    )
    assert (
        extract_mcp_tool_result_error_message({"isError": True, "content": [{"type": "text", "text": "denied"}]})
        == "denied"
    )
    assert extract_mcp_tool_result_error_message({"isError": False, "content": []}) is None
    assert extract_mcp_tool_result_error_message({}) is None


@pytest.mark.asyncio
async def test_fire_mcp_tool_call_logging_iserror_logs_failure():
    """Regression test: a CallToolResult with isError=True must go
    down the failure logging path (async_failure_handler + post_call_failure_hook),
    never async_success_handler."""
    from litellm.proxy._experimental.mcp_server.server import (
        _fire_mcp_tool_call_logging,
    )
    from litellm.proxy._experimental.mcp_server.exceptions import MCPToolResultError

    logging_obj = _mock_mcp_logging_obj()
    proxy_logging_mock = MagicMock()
    proxy_logging_mock.post_call_failure_hook = AsyncMock()
    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

    with patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_mock):
        await _fire_mcp_tool_call_logging(
            logging_obj=logging_obj,
            result=_call_tool_result(True, "upstream exploded"),
            start_time=datetime.now(),
            end_time=datetime.now(),
            user_api_key_auth=user_auth,
            request_data={"litellm_call_id": "cid"},
        )

    logging_obj.async_success_handler.assert_not_awaited()
    logging_obj.failure_handler.assert_called_once()
    logging_obj.async_failure_handler.assert_awaited_once()
    tool_error = logging_obj.async_failure_handler.await_args.args[0]
    assert isinstance(tool_error, MCPToolResultError)
    assert str(tool_error) == "upstream exploded"
    logging_obj.has_run_logging.assert_any_call(event_type="sync_success")
    logging_obj.has_run_logging.assert_any_call(event_type="async_success")
    proxy_logging_mock.post_call_failure_hook.assert_awaited_once()
    hook_kwargs = proxy_logging_mock.post_call_failure_hook.await_args.kwargs
    assert hook_kwargs["route"] == "/mcp/call_tool"
    assert hook_kwargs["original_exception"] is tool_error
    assert hook_kwargs["user_api_key_dict"] is user_auth
    logging_obj.async_post_mcp_tool_call_hook.assert_awaited_once()


@pytest.mark.asyncio
async def test_fire_mcp_tool_call_logging_success_path_unchanged():
    """isError=False must keep today's behavior: success handler fires, no
    failure logging, no post_call_failure_hook."""
    from litellm.proxy._experimental.mcp_server.server import (
        _fire_mcp_tool_call_logging,
    )

    logging_obj = _mock_mcp_logging_obj()
    proxy_logging_mock = MagicMock()
    proxy_logging_mock.post_call_failure_hook = AsyncMock()
    result = _call_tool_result(False, "all good")

    with patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_mock):
        await _fire_mcp_tool_call_logging(
            logging_obj=logging_obj,
            result=result,
            start_time=datetime.now(),
            end_time=datetime.now(),
            user_api_key_auth=UserAPIKeyAuth(api_key="test-key", user_id="test-user"),
            request_data={},
        )

    logging_obj.async_success_handler.assert_awaited_once()
    assert logging_obj.async_success_handler.await_args.kwargs["result"] is result
    logging_obj.async_failure_handler.assert_not_awaited()
    logging_obj.failure_handler.assert_not_called()
    proxy_logging_mock.post_call_failure_hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_fire_mcp_tool_call_logging_iserror_without_auth_skips_failure_hook():
    """Without a UserAPIKeyAuth the failure handlers still fire but the proxy
    post_call_failure_hook (which requires one) is skipped."""
    from litellm.proxy._experimental.mcp_server.server import (
        _fire_mcp_tool_call_logging,
    )

    logging_obj = _mock_mcp_logging_obj()
    proxy_logging_mock = MagicMock()
    proxy_logging_mock.post_call_failure_hook = AsyncMock()

    with patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_mock):
        await _fire_mcp_tool_call_logging(
            logging_obj=logging_obj,
            result={"isError": True, "content": [{"type": "text", "text": "denied"}]},
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    logging_obj.async_success_handler.assert_not_awaited()
    logging_obj.async_failure_handler.assert_awaited_once()
    assert str(logging_obj.async_failure_handler.await_args.args[0]) == "denied"
    proxy_logging_mock.post_call_failure_hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_fire_mcp_tool_call_logging_strips_credentials_from_failure_hook():
    """Credential-bearing request_data fields (raw request headers, upstream MCP
    auth headers, OAuth tokens) must never reach post_call_failure_hook
    callbacks; non-credential fields must survive untouched."""
    from litellm.proxy._experimental.mcp_server.server import (
        _fire_mcp_tool_call_logging,
    )

    logging_obj = _mock_mcp_logging_obj()
    proxy_logging_mock = MagicMock()
    proxy_logging_mock.post_call_failure_hook = AsyncMock()
    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
    request_data = {
        "name": "explode",
        "litellm_call_id": "cid",
        "raw_headers": {"authorization": "Bearer sk-caller-secret"},
        "mcp_auth_header": "upstream-secret",
        "mcp_server_auth_headers": {"srv": {"authorization": "Bearer srv-secret"}},
        "oauth2_headers": {"authorization": "Bearer oauth-secret"},
        "user_api_key_auth": user_auth,
    }

    with patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_mock):
        await _fire_mcp_tool_call_logging(
            logging_obj=logging_obj,
            result=_call_tool_result(True, "boom"),
            start_time=datetime.now(),
            end_time=datetime.now(),
            user_api_key_auth=user_auth,
            request_data=request_data,
        )

    proxy_logging_mock.post_call_failure_hook.assert_awaited_once()
    hook_request_data = proxy_logging_mock.post_call_failure_hook.await_args.kwargs["request_data"]
    assert hook_request_data == {"name": "explode", "litellm_call_id": "cid"}
    assert "secret" not in str(hook_request_data)


def _real_mcp_logging_obj(call_id: str):
    from litellm.litellm_core_utils.litellm_logging import Logging

    start_time = datetime.now()
    logging_obj = Logging(
        model="MCP: weather/get_forecast",
        messages=[{"role": "user", "content": "tool call"}],
        stream=False,
        call_type="call_mcp_tool",
        start_time=start_time,
        litellm_call_id=call_id,
        function_id="test-fn",
    )
    logging_obj.update_environment_variables(
        model="MCP: weather/get_forecast",
        user="",
        optional_params={},
        litellm_params={"api_base": ""},
    )
    logging_obj.model_call_details["mcp_tool_call_metadata"] = {
        "name": "get_forecast",
        "arguments": {"city": "Paris"},
        "mcp_server_name": "weather",
    }
    return logging_obj, start_time


@pytest.mark.asyncio
async def test_fire_mcp_tool_call_logging_iserror_builds_failure_payload(monkeypatch):
    """The standard logging payload for an isError=True result must carry
    status='failure' with the tool's error text, so OTel (whose _parse_error
    keys off status) marks the MCP span ERROR."""
    import litellm
    from litellm.proxy._experimental.mcp_server.server import (
        _fire_mcp_tool_call_logging,
    )

    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])

    logging_obj, start_time = _real_mcp_logging_obj("test-mcp-iserror-payload")

    await _fire_mcp_tool_call_logging(
        logging_obj=logging_obj,
        result=_call_tool_result(True, "upstream exploded"),
        start_time=start_time,
        end_time=datetime.now(),
    )

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["status"] == "failure"
    assert payload["error_str"] == "upstream exploded"
    assert payload["error_information"]["error_class"] == "MCPToolResultError"
    assert payload["metadata"]["mcp_tool_call_metadata"]["name"] == "get_forecast"


@pytest.mark.asyncio
async def test_fire_mcp_tool_call_logging_success_builds_success_payload(monkeypatch):
    """isError=False still produces a status='success' payload."""
    import litellm
    from litellm.proxy._experimental.mcp_server.server import (
        _fire_mcp_tool_call_logging,
    )

    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])

    logging_obj, start_time = _real_mcp_logging_obj("test-mcp-success-payload")

    await _fire_mcp_tool_call_logging(
        logging_obj=logging_obj,
        result=_call_tool_result(False, "all good"),
        start_time=start_time,
        end_time=datetime.now(),
    )

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["status"] == "success"


@pytest.mark.asyncio
async def test_fire_mcp_tool_call_logging_iserror_emits_otel_error_span(monkeypatch):
    """End-to-end regression for the OTel symptom: an isError=True tool
    result must reach OTel as an MCP span with StatusCode.ERROR and the tool's
    error message, while isError=False stays non-error."""
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace.status import StatusCode

    import litellm
    from litellm.integrations.otel import OpenTelemetryV2Config
    from litellm.integrations.otel.logger import OpenTelemetryV2
    from litellm.integrations.otel.plumbing import providers
    from litellm.proxy._experimental.mcp_server.server import (
        _fire_mcp_tool_call_logging,
    )

    cfg = OpenTelemetryV2Config(exporter="in_memory", legacy_compat=False)
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    otel_logger = OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider)

    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [otel_logger])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [otel_logger])

    logging_obj, start_time = _real_mcp_logging_obj("test-mcp-iserror-otel")

    await _fire_mcp_tool_call_logging(
        logging_obj=logging_obj,
        result=_call_tool_result(True, "upstream exploded"),
        start_time=start_time,
        end_time=datetime.now(),
    )

    (span,) = exporter.get_finished_spans()
    assert span.name == "tools/call get_forecast"
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "MCPToolResultError"
    assert "upstream exploded" in (span.status.description or "")


@pytest.mark.asyncio
async def test_call_tool_with_legacy_db_m2m_server_resolves_oauth2_flow():
    """
    Finding 3 regression: the call_mcp_tool path must apply the same request-time
    oauth2_flow backstop the listing path does. A legacy DB row with oauth2_flow=NULL
    but the M2M credential shape must reach execute_mcp_tool resolved to
    client_credentials, or the caller's Authorization would be forwarded to an M2M
    upstream on tool execution during a backfill gap (the list path was covered, the
    call path was not).
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import call_mcp_tool
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.mcp import MCPAuth
    except ImportError:
        pytest.skip("MCP server not available")

    user_auth = UserAPIKeyAuth(api_key="sk-1234", user_id="test-user")

    legacy_server = MCPServer(
        server_id="legacy-m2m-id",
        name="legacy_m2m",
        alias="legacy_m2m",
        server_name="legacy_m2m",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow=None,  # legacy: unstamped
        token_url="https://oauth.example.com/token",
        client_id="client-id",
        client_secret="client-secret",
    )
    assert legacy_server.has_client_credentials is False

    captured_servers = {}

    async def capture_execute(*args, **kwargs):
        captured_servers["allowed"] = kwargs.get("allowed_mcp_servers")
        return MagicMock(name="call_tool_result")

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        ) as mock_manager,
        patch(
            "litellm.proxy._experimental.mcp_server.server.execute_mcp_tool",
            side_effect=capture_execute,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers_from_mcp_server_names",
            new=AsyncMock(side_effect=lambda mcp_servers, allowed_mcp_servers: allowed_mcp_servers),
        ),
    ):
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["legacy-m2m-id"])
        mock_manager.get_mcp_server_by_id = MagicMock(return_value=legacy_server)

        await call_mcp_tool(name="legacy_m2m-tool", arguments={}, user_api_key_auth=user_auth)

    resolved = captured_servers["allowed"]
    assert resolved and resolved[0].oauth2_flow == "client_credentials"
    assert resolved[0].has_client_credentials is True


@pytest.mark.parametrize(
    "url, expected",
    [
        # only the origin may be logged: userinfo, query, fragment, and the PATH are all stripped,
        # because hosted MCP servers routinely embed the credential in the path (e.g. /mcp/s/<token>)
        ("https://user:s3cr3t@mcp.example.com/mcp?token=abcd1234&x=1", "https://mcp.example.com"),
        ("https://mcp.example.com/mcp#frag", "https://mcp.example.com"),
        ("https://host:8443/a/b?q=1", "https://host:8443"),
        ("https://mcp.zapier.com/api/mcp/s/NDgzcret-token/mcp", "https://mcp.zapier.com"),
        ("https://mcp.notion.com/mcp", "https://mcp.notion.com"),
        (None, None),
        ("", None),
        ("not a url", None),
        ("http://[::1", None),
    ],
)
def test_redact_mcp_resource_url_strips_credentials(url, expected):
    """The MCP tool-call log records the upstream resource, so the URL must be redacted to
    scheme+host+path: userinfo, query string, and fragment (which can carry embedded tokens or
    secret parameters) must never reach spend-log metadata or logging callbacks."""
    from litellm.proxy._experimental.mcp_server.server import _redact_mcp_resource_url

    assert _redact_mcp_resource_url(url) == expected

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import GetPromptRequest, GetPromptRequestParams, GetPromptResult

from litellm.proxy._experimental.mcp_server.operations import GatewayOperations, prepare_context
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.mark.asyncio
async def test_oauth_prefetch_failure_does_not_log_caller_or_exception_text(caplog):
    from litellm.proxy._experimental.mcp_server.operations import _prefetch_oauth_creds_for_user

    user_id = "caller\nFORGED-USER-LINE"
    fetch = AsyncMock(side_effect=RuntimeError("database\nFORGED-ERROR-LINE"))
    database = object()
    with (
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=database),
        patch("litellm.proxy._experimental.mcp_server.db.list_user_oauth_credentials", fetch),
        caplog.at_level("WARNING", logger="LiteLLM"),
    ):
        result = await _prefetch_oauth_creds_for_user(UserAPIKeyAuth(user_id=user_id))
    assert result == {}
    fetch.assert_awaited_once_with(database, user_id)
    warnings = [record.getMessage() for record in caplog.records if "prefetch" in record.getMessage()]
    assert len(warnings) == 1
    assert "failed" in warnings[0]
    assert "\n" not in warnings[0]
    assert "FORGED" not in warnings[0]


@pytest.mark.asyncio
async def test_dispatch_uses_explicit_context_when_ambient_caller_differs():
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from litellm.proxy._experimental.mcp_server.server import set_auth_context

    context = prepare_context(
        UserAPIKeyAuth(user_id="alpha"),
        raw_headers={"x-caller": "alpha"},
        mcp_servers=["alpha-server"],
        client_ip="192.0.2.1",
    )
    token = auth_context_var.set(None)
    handler = AsyncMock(return_value=GetPromptResult(messages=[]))
    try:
        set_auth_context(UserAPIKeyAuth(user_id="bravo"), raw_headers={"x-caller": "bravo"})
        with patch("litellm.proxy._experimental.mcp_server.operations.mcp_get_prompt", handler):
            result = await GatewayOperations().execute(
                GetPromptRequest(params=GetPromptRequestParams(name="alpha-prompt")), context
            )
        assert result.messages == []
        assert handler.await_args.kwargs["name"] == "alpha-prompt"
        assert handler.await_args.kwargs["user_api_key_auth"].user_id == "alpha"
        assert handler.await_args.kwargs["raw_headers"] == {"x-caller": "alpha"}
        assert handler.await_args.kwargs["mcp_servers"] == ["alpha-server"]
        assert handler.await_args.kwargs["client_ip"] == "192.0.2.1"
    finally:
        auth_context_var.reset(token)


@pytest.mark.asyncio
async def test_legacy_adapter_cleans_context_after_cancelled_operation():
    from types import SimpleNamespace
    from litellm.proxy._experimental.mcp_server import server
    from litellm.proxy._experimental.mcp_server.mcp_context import active_mcp_request_ctx_var

    previous_session = server.active_mcp_session_var.get()
    previous_request = active_mcp_request_ctx_var.get()
    request = SimpleNamespace(session=object())
    auth = (None, None, None, None, None, None, None)

    async def cancelled_operation():
        async with server._legacy_operation_context(request, trace=False):
            assert server.active_mcp_session_var.get() is request.session
            assert active_mcp_request_ctx_var.get() is request
            raise asyncio.CancelledError

    with patch(
        "litellm.proxy._experimental.mcp_server.server.get_or_extract_auth_context", AsyncMock(return_value=auth)
    ):
        with pytest.raises(asyncio.CancelledError):
            await cancelled_operation()
    assert server.active_mcp_session_var.get() is previous_session
    assert active_mcp_request_ctx_var.get() is previous_request


@pytest.mark.asyncio
async def test_legacy_adapter_cleans_context_when_trace_setup_fails():
    from types import SimpleNamespace
    from litellm.proxy._experimental.mcp_server import server
    from litellm.proxy._experimental.mcp_server.mcp_context import active_mcp_request_ctx_var

    previous_session = server.active_mcp_session_var.get()
    previous_request = active_mcp_request_ctx_var.get()
    request = SimpleNamespace(session=object())

    async def enter_operation():
        async with server._legacy_operation_context(request, trace=True):
            pytest.fail("Trace setup failure must prevent dispatch")

    with patch.object(server, "_otel_set_mcp_transport_span", side_effect=RuntimeError("trace failure")):
        with pytest.raises(RuntimeError, match="trace failure"):
            await enter_operation()
    assert server.active_mcp_session_var.get() is previous_session
    assert active_mcp_request_ctx_var.get() is previous_request


@pytest.mark.asyncio
async def test_prompt_sampling_receives_explicit_operation_caller_headers_and_ip():
    from unittest.mock import MagicMock
    from litellm.proxy._experimental.mcp_server import operations
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    upstream = MCPServer(
        server_id="explicit-prompt",
        name="explicit_prompt",
        url="https://example.invalid/mcp",
        transport=MCPTransport.http,
        allow_sampling=True,
    )
    context = prepare_context(
        UserAPIKeyAuth(user_id="prompt-caller"),
        raw_headers={"x-caller": "prompt-caller"},
        client_ip="192.0.2.41",
    )
    client = MagicMock()
    client.get_prompt = AsyncMock(return_value=GetPromptResult(messages=[]))
    sampling = AsyncMock()
    with (
        patch.object(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[upstream])),
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient", return_value=client) as factory,
        patch("litellm.proxy._experimental.mcp_server.sampling_handler.handle_sampling_create_message", sampling),
    ):
        result = await GatewayOperations().execute(
            GetPromptRequest(params=GetPromptRequestParams(name="explicit_prompt-prompt")), context
        )
        assert result.messages == []
        await factory.call_args.kwargs["sampling_callback"](None, None)
    captured = sampling.await_args.kwargs
    assert captured["user_api_key_auth"] is not None
    assert captured["user_api_key_auth"].user_id == "prompt-caller"
    assert captured["raw_headers"] == {"x-caller": "prompt-caller"}
    assert captured["client_ip"] == "192.0.2.41"


def _catalog_case(method):
    from mcp import types

    cases = {
        "prompts/list": (
            types.ListPromptsRequest(),
            "list_prompts",
            "get_prompts_from_server",
            [types.Prompt(name="catalog-prompt")],
            "prompts",
        ),
        "prompts/get": (
            types.GetPromptRequest(
                params=types.GetPromptRequestParams(name="catalog-prompt", arguments={"topic": "test"})
            ),
            "get_prompt",
            "get_prompt_from_server",
            types.GetPromptResult(messages=[]),
            None,
        ),
        "resources/list": (
            types.ListResourcesRequest(),
            "list_resources",
            "get_resources_from_server",
            [types.Resource(name="document", uri="https://example.com/document")],
            "resources",
        ),
        "resources/templates/list": (
            types.ListResourceTemplatesRequest(),
            "list_resource_templates",
            "get_resource_templates_from_server",
            [types.ResourceTemplate(name="document", uri_template="https://example.com/{name}")],
            "resource_templates",
        ),
        "resources/read": (
            types.ReadResourceRequest(params=types.ReadResourceRequestParams(uri="https://example.com/document")),
            "read_resource",
            "read_resource_from_server",
            types.ReadResourceResult(
                contents=[types.TextResourceContents(uri="https://example.com/document", text="document body")]
            ),
            None,
        ),
    }
    return cases[method]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["prompts/list", "prompts/get", "resources/list", "resources/templates/list", "resources/read"]
)
@pytest.mark.parametrize("state", ["success", "denied", "upstream_failure", "scope_failure"])
async def test_native_catalog_operations_preserve_context_results_and_failure_policy(method, state):
    from types import SimpleNamespace
    from fastapi import HTTPException
    from mcp.server.context import ServerRequestContext
    from mcp.types import PaginatedRequestParams
    from litellm.proxy._experimental.mcp_server import operations, server
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    operation, handler_name, manager_method, payload, collection = _catalog_case(method)
    caller = UserAPIKeyAuth(user_id="catalog-caller")
    headers = {"x-caller": "catalog-caller"}
    upstream_server = MCPServer(server_id="catalog", name="catalog", transport=MCPTransport.http)
    allowed = AsyncMock(
        return_value=[] if state == "denied" else [upstream_server],
        side_effect=HTTPException(status_code=403, detail="scope denied") if state == "scope_failure" else None,
    )
    upstream = AsyncMock(
        return_value=payload, side_effect=RuntimeError("upstream unavailable") if state == "upstream_failure" else None
    )
    ctx = ServerRequestContext(
        session=SimpleNamespace(), lifespan_context={}, protocol_version="2025-06-18", method=method
    )
    auth = (caller, None, ["catalog"], None, None, headers, "192.0.2.41")
    with (
        patch.object(server, "get_or_extract_auth_context", AsyncMock(return_value=auth)),
        patch.object(operations, "_get_allowed_mcp_servers", allowed),
        patch.object(operations.global_mcp_server_manager, manager_method, upstream),
    ):
        if collection is None and state != "success":
            expected_error = RuntimeError if state == "upstream_failure" else HTTPException
            with pytest.raises(expected_error):
                await getattr(server, handler_name)(ctx, operation.params)
        else:
            result = await getattr(server, handler_name)(ctx, operation.params or PaginatedRequestParams())
            if collection:
                assert getattr(result, collection) == (payload if state == "success" else [])
            else:
                assert result == payload
    assert allowed.await_args.kwargs == {
        "user_api_key_auth": caller,
        "mcp_servers": ["catalog"],
        "client_ip": "192.0.2.41",
    }
    if state in ("denied", "scope_failure"):
        upstream.assert_not_awaited()
    else:
        upstream.assert_awaited_once()
        forwarded = upstream.await_args.kwargs
        assert forwarded["user_api_key_auth"] == caller
        assert forwarded["raw_headers"] == headers
        assert forwarded["client_ip"] == "192.0.2.41"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["prompts/list", "prompts/get", "resources/list", "resources/templates/list", "resources/read"]
)
async def test_explicit_proxy_context_rejects_catalog_operations_before_upstream_access(method):
    from mcp.shared.exceptions import MCPError
    from mcp.types import METHOD_NOT_FOUND
    from litellm.proxy._experimental.mcp_server import operations

    operation, _, manager_method, _, _ = _catalog_case(method)
    upstream = AsyncMock()
    with patch.object(operations.global_mcp_server_manager, manager_method, upstream):
        with pytest.raises(MCPError) as rejected:
            await GatewayOperations().execute(operation, prepare_context(mcp_proxy_mode=True))
    assert rejected.value.error.code == METHOD_NOT_FOUND
    assert rejected.value.error.message == "Operation unavailable on /mcp/proxy"
    upstream.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_env", "pii", "guardrail", "unexpected"])
async def test_tool_operation_preserves_failure_messages_and_request_trace(failure):
    from mcp.types import CallToolRequest, CallToolRequestParams
    from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
    from litellm.proxy._experimental.mcp_server import operations
    from litellm.proxy._experimental.mcp_server.utils import MCPMissingUserEnvVarsError

    failures = {
        "missing_env": (
            MCPMissingUserEnvVarsError(
                server_id="server", server_name="server", missing=["TOKEN"], setup_url="https://example.com/setup"
            ),
            "https://example.com/setup",
        ),
        "pii": (
            BlockedPiiEntityError(entity_type="EMAIL_ADDRESS", guardrail_name="test"),
            "Blocked PII entity detected",
        ),
        "guardrail": (GuardrailRaisedException(message="request denied"), "Guardrail violation"),
        "unexpected": (RuntimeError("upstream unavailable"), "Error: upstream unavailable"),
    }
    error, expected = failures[failure]
    dispatch = AsyncMock(side_effect=error)
    context = prepare_context(
        raw_headers={"x-litellm-trace-id": "operation-trace", "authorization": "private-test-header"}
    )
    with patch.object(operations, "call_mcp_tool", dispatch):
        result = await GatewayOperations().execute(
            CallToolRequest(params=CallToolRequestParams(name="catalog-tool", arguments={})), context
        )
    assert result.is_error is True
    assert expected in result.content[0].text
    assert "private-test-header" not in result.content[0].text
    dispatch.assert_awaited_once()
    assert dispatch.await_args.kwargs["litellm_trace_id"] == "operation-trace"
    assert dispatch.await_args.kwargs["litellm_session_id"] == "operation-trace"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,helper",
    [
        ("prompts/list", "_list_mcp_prompts"),
        ("resources/list", "_list_mcp_resources"),
        ("resources/templates/list", "_list_mcp_resource_templates"),
    ],
)
async def test_catalog_operation_preserves_empty_result_for_malformed_upstream_items(method, helper):
    from litellm.proxy._experimental.mcp_server import operations

    operation, _, _, _, collection = _catalog_case(method)
    with patch.object(operations, helper, AsyncMock(return_value=[{"unexpected": "item"}])):
        result = await GatewayOperations().execute(operation, prepare_context())
    assert getattr(result, collection) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("catalog_unavailable", [False, True])
async def test_tool_listing_returns_empty_result_without_dispatch_for_unavailable_catalog(catalog_unavailable):
    from mcp.types import ListToolsRequest
    from litellm.proxy._experimental.mcp_server import operations

    allowed = AsyncMock(
        return_value=[], side_effect=RuntimeError("catalog unavailable") if catalog_unavailable else None
    )
    upstream = AsyncMock()
    with (
        patch.object(operations, "_get_allowed_mcp_servers", allowed),
        patch.object(operations.global_mcp_server_manager, "_get_tools_from_server", upstream),
    ):
        result = await GatewayOperations().execute(ListToolsRequest(), prepare_context())
    assert result.tools == []
    allowed.assert_awaited_once()
    upstream.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_proxy_context_lists_builtin_tools_and_blocks_direct_tool_dispatch():
    from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest
    from litellm.proxy._experimental.mcp_server import operations

    context = prepare_context(mcp_proxy_mode=True)
    allowed = AsyncMock()
    with patch.object(operations, "_get_allowed_mcp_servers", allowed):
        listing = await GatewayOperations().execute(ListToolsRequest(), context)
        denied = await GatewayOperations().execute(
            CallToolRequest(params=CallToolRequestParams(name="catalog-tool", arguments={})), context
        )
    assert {tool.name for tool in listing.tools} == {"search_tools", "get_tool_schema", "call_tool"}
    assert denied.is_error is True
    assert "unavailable on /mcp/proxy" in denied.content[0].text
    allowed.assert_not_awaited()


def _server(server_id: str, auth_type: MCPAuth) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=f"{server_id}-server",
        url="https://up.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=auth_type,
        token_exchange_endpoint="https://idp.example.com/token",
        client_id="cid",
        client_secret="csec",
    )


class TestChallengeMissingTokenExchangeSubject:
    """The REST cold-catalog path must answer a missing OBO subject with the RFC 9728 401 challenge
    before the best-effort listing swallows the upstream 401 and tool resolution turns it into a 500."""

    @staticmethod
    def _challenge(
        server: MCPServer | None,
        allowed: list[MCPServer],
        *,
        user: UserAPIKeyAuth | None = None,
        oauth2_headers: dict[str, str] | None = None,
        raw_headers: dict[str, str] | None = None,
        requested_server: MCPServer | None = None,
    ) -> None:
        from litellm.proxy._experimental.mcp_server.operations import _challenge_missing_token_exchange_subject

        return _challenge_missing_token_exchange_subject(
            server=server,
            requested_server=requested_server,
            allowed_mcp_servers=allowed,
            user_api_key_auth=user,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
        )

    def test_missing_subject_raises_401_challenge(self):
        from fastapi import HTTPException

        server = _server("te-cold", MCPAuth.oauth2_token_exchange)
        with pytest.raises(HTTPException) as exc_info:
            self._challenge(
                server,
                [server],
                user=UserAPIKeyAuth(api_key="sk-admission"),
                raw_headers={"x-litellm-api-key": "sk-admission"},
            )
        assert exc_info.value.status_code == 401
        challenge = (exc_info.value.headers or {}).get("WWW-Authenticate", "")
        assert challenge.startswith("Bearer ") and 'error="invalid_token"' in challenge, challenge
        assert "resource_metadata" in challenge, challenge

    @pytest.mark.parametrize(
        "authorization",
        ["Bearer sk-admission", "Bearer sk-some-other-virtual-key"],
        ids=["repeated-admission-key", "another-virtual-key"],
    )
    def test_litellm_key_in_authorization_is_not_a_subject(self, authorization: str):
        from fastapi import HTTPException

        server = _server("te-vk", MCPAuth.oauth2_token_exchange)
        with pytest.raises(HTTPException) as exc_info:
            self._challenge(
                server,
                [server],
                user=UserAPIKeyAuth(api_key="sk-admission"),
                oauth2_headers={"Authorization": authorization},
                raw_headers={"x-litellm-api-key": "sk-admission", "authorization": authorization},
            )
        assert exc_info.value.status_code == 401

    def test_subject_present_does_not_challenge(self):

        server = _server("te-ok", MCPAuth.oauth2_token_exchange)
        assert (
            self._challenge(
                server,
                [server],
                user=UserAPIKeyAuth(api_key="sk-admission"),
                oauth2_headers={"Authorization": "Bearer idp-subject"},
                raw_headers={"x-litellm-api-key": "sk-admission", "authorization": "Bearer idp-subject"},
            )
            is None
        )

    def test_server_outside_allowlist_is_not_challenged(self):

        server = _server("te-hidden", MCPAuth.oauth2_token_exchange)
        other = _server("te-visible", MCPAuth.oauth2_token_exchange)
        assert self._challenge(server, [other], user=UserAPIKeyAuth(api_key="sk-admission")) is None
        assert self._challenge(None, [other], user=UserAPIKeyAuth(api_key="sk-admission")) is None

    def test_prefix_owner_differing_from_server_id_is_not_challenged(self):
        """An explicit server_id that disagrees with the tool prefix keeps the existing mismatch answer."""
        from fastapi import HTTPException

        prefix_owner = _server("te-prefix", MCPAuth.oauth2_token_exchange)
        requested = _server("te-requested", MCPAuth.oauth2_token_exchange)
        user = UserAPIKeyAuth(api_key="sk-admission")
        allowed = [prefix_owner, requested]
        assert self._challenge(prefix_owner, allowed, user=user, requested_server=requested) is None
        with pytest.raises(HTTPException):
            self._challenge(prefix_owner, allowed, user=user, requested_server=prefix_owner)

    @pytest.mark.parametrize(
        "auth_type",
        ["oauth2", "oauth_delegate", "oauth2_id_jag", "bearer_token", "api_key", "none"],
    )
    def test_other_auth_types_are_untouched(self, auth_type: str):

        server = _server("na", MCPAuth(auth_type))
        assert self._challenge(server, [server], user=UserAPIKeyAuth(api_key="sk-admission")) is None


@pytest.mark.asyncio
async def test_execute_mcp_tool_challenges_missing_subject_before_cold_listing():
    """On a cold catalog the challenge fires before any listing or tool resolution is attempted."""
    from fastapi import HTTPException
    from datetime import datetime, timezone
    from litellm.proxy._experimental.mcp_server import operations

    server = _server("te-exec", MCPAuth.oauth2_token_exchange)
    listing = AsyncMock()
    with (
        patch.object(operations.global_mcp_server_manager, "get_mcp_server_by_id", return_value=server),
        patch.object(operations.global_mcp_server_manager, "server_exposes_tool", return_value=False),
        patch.object(operations, "_get_tools_from_mcp_servers", listing),
        pytest.raises(HTTPException) as exc_info,
    ):
        await operations.execute_mcp_tool(
            name="add",
            arguments={"a": 2, "b": 3},
            allowed_mcp_servers=[server],
            start_time=datetime.now(timezone.utc),
            user_api_key_auth=UserAPIKeyAuth(api_key="sk-admission"),
            raw_headers={"x-litellm-api-key": "sk-admission"},
            requested_server_id=server.server_id,
        )
    assert exc_info.value.status_code == 401
    listing.assert_not_awaited()

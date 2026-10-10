import asyncio
from collections.abc import Sequence
from typing import Final, Literal
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import GetPromptRequest, GetPromptRequestParams, GetPromptResult
from mcp.types import Tool as MCPTool

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._experimental.mcp_server import operations
from litellm.proxy._experimental.mcp_server import rest_endpoints
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    HTTPException as MCPServerManagerHTTPException,
    ListedToolsCaller,
)
from litellm.proxy._experimental.mcp_server.operations import GatewayOperations, prepare_context
from litellm.proxy._experimental.mcp_server.tool_registry import global_mcp_tool_registry
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class _CatalogHookCapture(CustomLogger):
    data: dict[str, object] | None = None

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> None:
        if call_type == "call_mcp_tool":
            self.data = data.copy()


async def _served_catalog_tool() -> str:
    return "ok"


@pytest.mark.parametrize(
    ("settings", "mcp_servers", "toolset_id", "auth_state", "raises"),
    [
        ({"mcp_require_explicit_server_scope": True}, None, None, None, True),
        ({"mcp_require_explicit_server_scope": True}, None, None, "unscoped", True),
        ({"mcp_require_explicit_server_scope": True}, ["alpha"], None, None, False),
        ({"mcp_require_explicit_server_scope": True}, None, "toolset-id", None, False),
        ({"mcp_require_explicit_server_scope": True}, None, None, "sealed", False),
        ({"mcp_require_explicit_server_scope": False}, None, None, None, False),
        ({}, None, None, None, False),
    ],
)
def test_explicit_server_scope_setting_only_rejects_unscoped_aggregate_requests(
    settings: dict[str, bool],
    mcp_servers: list[str] | None,
    toolset_id: str | None,
    auth_state: Literal["unscoped", "sealed"] | None,
    raises: bool,
) -> None:
    from fastapi import HTTPException

    auth: Final[UserAPIKeyAuth | None] = (
        None if auth_state is None else UserAPIKeyAuth(api_key=None, user_id="scope-test")
    )
    if auth_state == "sealed" and auth is not None:
        auth.mcp_session_resource_server_ids = ("alpha-id",)

    with patch("litellm.proxy.proxy_server.general_settings", settings):
        if raises:
            with pytest.raises(HTTPException) as exc_info:
                operations.raise_if_unscoped_aggregate_request(mcp_servers, auth, toolset_id)
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail == {
                "error": "mcp_server_scope_required",
                "message": (
                    "This gateway requires an explicit MCP server scope: connect to /mcp/<server_name> "
                    "or send the x-mcp-servers header."
                ),
            }
        else:
            operations.raise_if_unscoped_aggregate_request(mcp_servers, auth, toolset_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["mcp", "rest"])
@pytest.mark.parametrize("restriction", ["key", "server"])
async def test_listing_records_only_tools_the_caller_received(
    monkeypatch: pytest.MonkeyPatch, surface: str, restriction: str
) -> None:
    manager: Final = operations.global_mcp_server_manager
    server: Final = MCPServer(
        server_id="served-catalog", name="served-catalog", transport=MCPTransport.http,
        spec_path="/catalog.yaml", allow_all_keys=True,
        allowed_tools=["echo"] if restriction == "server" else None,
    )
    auth: Final = UserAPIKeyAuth(
        api_key="sk-served-catalog", user_id="lister",
        object_permission={
            "object_permission_id": "served-permission",
            "mcp_servers": [server.server_id],
            "mcp_tool_permissions": {server.server_id: ["echo"]} if restriction == "key" else None,
        },
    )
    monkeypatch.setitem(manager.registry, server.server_id, server)
    monkeypatch.setitem(manager.tool_name_to_mcp_server_name_mapping, "status", server.server_id)
    monkeypatch.setitem(manager.tool_name_to_mcp_server_name_mapping, "served-catalog-status", server.server_id)
    capture: Final = _CatalogHookCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    for name in ("echo", "status"):
        global_mcp_tool_registry.register_tool(
            name=f"served-catalog-{name}", description=f"{name} description",
            input_schema={"type": "object"}, handler=_served_catalog_tool,
        )
    try:
        if surface == "mcp":
            listing: Final = await operations._list_mcp_tools(
                user_api_key_auth=auth, mcp_servers=[server.server_id], record_listing=True,
            )
            assert [tool.name for tool in listing.tools] == ["served-catalog-echo"]
        else:
            rest_listing: Final = await rest_endpoints._get_tools_for_single_server(
                server, None, user_api_key_auth=auth,
            )
            assert [tool.name for tool in rest_listing] == ["echo"]
        granted: Final = auth.model_copy(update={"object_permission": None})
        caller: Final = ListedToolsCaller(user_api_key_auth=granted)
        assert manager.get_listed_tool(server, "status", caller) is None
        served: Final = manager.get_listed_tool(server, "echo", caller)
        assert served is not None
        assert (served.description, served.input_schema) == ("echo description", {"type": "object"})
        server.allowed_tools = None
        result: Final = await manager.call_tool(
            server_name=server.server_id, name="status", arguments={}, user_api_key_auth=granted,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=UserApiKeyCache()),
        )
        assert result.is_error is False
        assert capture.data is not None
        assert capture.data["messages"] == [{"role": "user", "content": "Tool: status\nArguments: {}"}]
        assert (capture.data.get("mcp_tool_description"), capture.data.get("mcp_input_schema")) == (None, None)
    finally:
        manager._drop_listed_tools(server.server_id)
        global_mcp_tool_registry.unregister_tools_with_prefix("served-catalog-")


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
    request = SimpleNamespace(session=object(), protocol_version="2025-06-18")
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
        patch("litellm.proxy._experimental.mcp_server.upstream.MCPClient", return_value=client) as factory,
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


def _mcp_rate_limited_proxy_logging() -> ProxyLogging:
    proxy_logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    proxy_logging.proxy_hook_mapping["parallel_request_limiter"] = PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    return proxy_logging


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["tools/list", "prompts/list", "resources/list", "resources/templates/list", "prompts/get", "resources/read"],
)
async def test_mcp_server_rpm_limits_every_catalog_operation(operation: str) -> None:
    from unittest.mock import MagicMock

    from mcp import types
    from mcp.shared.exceptions import MCPError
    from mcp.types import INVALID_REQUEST

    from litellm.proxy._experimental.mcp_server import catalog
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import ServerListOk

    server: Final = MCPServer(
        server_id="catalog-rpm",
        name="catalog",
        server_name="catalog",
        transport=MCPTransport.http,
        rpm=1,
    )
    caller: Final = UserAPIKeyAuth(api_key=hash_token("sk-catalog-rpm"))
    operation_to_manager_method: Final = {
        "tools/list": "get_tools_from_server",
        "prompts/list": "get_prompts_from_server",
        "resources/list": "get_resources_from_server",
        "resources/templates/list": "get_resource_templates_from_server",
        "prompts/get": "get_prompt_from_server",
        "resources/read": "read_resource_from_server",
    }
    upstream_results: Final = {
        "tools/list": (
            types.ListToolsResult(tools=[types.Tool(name="echo", inputSchema={"type": "object"})]),
            ServerListOk(tool_count=1),
        ),
        "prompts/list": types.ListPromptsResult(prompts=[types.Prompt(name="catalog-prompt")]),
        "resources/list": types.ListResourcesResult(
            resources=[types.Resource(name="document", uri="https://example.com/document")]
        ),
        "resources/templates/list": types.ListResourceTemplatesResult(
            resource_templates=[
                types.ResourceTemplate(name="document", uri_template="https://example.com/{name}")
            ]
        ),
        "prompts/get": GetPromptResult(messages=[]),
        "resources/read": types.ReadResourceResult(contents=[]),
    }
    upstream: Final = AsyncMock(return_value=upstream_results[operation])
    manager_method: Final = operation_to_manager_method[operation]
    manager: Final = operations.global_mcp_server_manager
    rate_limit_error: Final = ProxyRateLimitError(detail="server RPM exceeded")
    enforce_rate_limit: Final = AsyncMock(side_effect=[None, rate_limit_error, rate_limit_error])
    proxy_logging: Final = MagicMock(enforce_mcp_server_rate_limits=enforce_rate_limit)
    is_protocol_listing: Final = operation.endswith("/list")
    context: Final = prepare_context(caller, mcp_servers=[server.server_id])

    async def invoke() -> object:
        if operation == "tools/list":
            return await GatewayOperations().execute(types.ListToolsRequest(), context)
        if operation == "prompts/list":
            return await GatewayOperations().execute(types.ListPromptsRequest(), context)
        if operation == "resources/list":
            return await GatewayOperations().execute(types.ListResourcesRequest(), context)
        if operation == "resources/templates/list":
            return await GatewayOperations().execute(types.ListResourceTemplatesRequest(), context)
        if operation == "prompts/get":
            return await operations.mcp_get_prompt(
                name=f"{server.name}-catalog-prompt",
                user_api_key_auth=caller,
                mcp_servers=[server.server_id],
            )
        return await operations.mcp_read_resource(
            url="https://example.com/document",
            user_api_key_auth=caller,
            mcp_servers=[server.server_id],
        )

    with (
        patch.object(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server])),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch.dict(manager.registry, {server.server_id: server}),
        patch.object(manager, manager_method, upstream),
        patch.object(catalog, "get_filtered_server_tools", upstream),
        patch.object(catalog, "fetch_optional_catalog_page", upstream),
    ):
        await invoke()
        if is_protocol_listing:
            with pytest.raises(MCPError) as rejected:
                await invoke()
            assert rejected.value.error.code == INVALID_REQUEST
            assert rejected.value.error.message == "server RPM exceeded"
            if operation == "tools/list":
                with pytest.raises(ProxyRateLimitError) as rejected:
                    await operations._get_tools_from_mcp_servers(
                        user_api_key_auth=caller,
                        mcp_auth_header=None,
                        mcp_servers=[server.server_id],
                        params=None,
                    )
                assert rejected.value is rate_limit_error
                assert upstream.await_count == 1
                assert enforce_rate_limit.await_count == 3
        else:
            with pytest.raises(ProxyRateLimitError):
                await invoke()

    assert upstream.await_count == 1
    if operation != "tools/list":
        assert enforce_rate_limit.await_count == 2


@pytest.mark.asyncio
async def test_tools_call_warmup_does_not_consume_mcp_server_rpm() -> None:
    from mcp import types

    server: Final = MCPServer(
        server_id="catalog-warmup",
        name="catalog-warmup",
        server_name="catalog-warmup",
        transport=MCPTransport.http,
        rpm=1,
    )
    caller: Final = UserAPIKeyAuth(api_key=hash_token("sk-catalog-warmup"))
    proxy_logging: Final = _mcp_rate_limited_proxy_logging()
    upstream: Final = AsyncMock(
        return_value=[types.Tool(name="echo", inputSchema={"type": "object"})]
    )
    with (
        patch.object(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server])),
        patch.object(operations.global_mcp_server_manager, "server_exposes_tool", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging),
        patch.object(operations.global_mcp_server_manager, "get_tools_from_server", upstream),
    ):
        await operations._list_tools_before_first_call(
            server=server,
            tool_name="echo",
            allowed_mcp_servers=[server],
            user_api_key_auth=caller,
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
        )
        listing: Final = await operations._get_tools_from_mcp_servers(
            user_api_key_auth=caller,
            mcp_auth_header=None,
            mcp_servers=[server.server_id],
        )

    assert [tool.name for tool in listing.tools] == ["echo"]


@pytest.mark.asyncio
async def test_tools_call_pre_call_check_enforces_mcp_server_rpm() -> None:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    server: Final = MCPServer(
        server_id="catalog-call",
        name="catalog-call",
        server_name="catalog-call",
        transport=MCPTransport.http,
        rpm=1,
    )
    proxy_logging: Final = _mcp_rate_limited_proxy_logging()
    manager: Final = MCPServerManager()

    await manager.pre_call_tool_check(
        name="echo",
        arguments={},
        server_name=server.name,
        user_api_key_auth=None,
        proxy_logging_obj=proxy_logging,
        server=server,
    )
    with pytest.raises(ProxyRateLimitError):
        await manager.pre_call_tool_check(
            name="echo",
            arguments={},
            server_name=server.name,
            user_api_key_auth=None,
            proxy_logging_obj=proxy_logging,
            server=server,
        )


@pytest.mark.asyncio
async def test_tools_call_pre_call_hook_rejection_does_not_enforce_mcp_server_rpm() -> None:
    from unittest.mock import MagicMock

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    server: Final = MCPServer(
        server_id="catalog-call-pre-hook-rejected",
        name="catalog-call-pre-hook-rejected",
        server_name="catalog-call-pre-hook-rejected",
        transport=MCPTransport.http,
        rpm=1,
    )
    rate_limit_error: Final = ProxyRateLimitError(detail="ordinary key rate limit")
    proxy_logging: Final = MagicMock()
    proxy_logging.create_mcp_request_object_from_kwargs.return_value = {}
    proxy_logging.convert_mcp_to_llm_format.return_value = {}
    proxy_logging.pre_call_hook = AsyncMock(side_effect=rate_limit_error)
    proxy_logging.enforce_mcp_server_rate_limits = AsyncMock()

    with pytest.raises(ProxyRateLimitError) as rejected:
        await MCPServerManager().pre_call_tool_check(
            name="echo",
            arguments={},
            server_name=server.name,
            user_api_key_auth=None,
            proxy_logging_obj=proxy_logging,
            server=server,
        )

    assert rejected.value is rate_limit_error
    proxy_logging.enforce_mcp_server_rate_limits.assert_not_awaited()


@pytest.mark.asyncio
async def test_disallowed_tool_does_not_consume_mcp_server_rpm() -> None:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    server: Final = MCPServer(
        server_id="catalog-call-authorization",
        name="catalog-call-authorization",
        server_name="catalog-call-authorization",
        transport=MCPTransport.http,
        allowed_tools=["allowed"],
        rpm=1,
    )
    proxy_logging: Final = _mcp_rate_limited_proxy_logging()
    manager: Final = MCPServerManager()

    with pytest.raises(MCPServerManagerHTTPException) as denied_call:
        await manager.pre_call_tool_check(
            name="disallowed",
            arguments={},
            server_name=server.name,
            user_api_key_auth=None,
            proxy_logging_obj=proxy_logging,
            server=server,
        )

    assert denied_call.value.status_code == 403
    await manager.pre_call_tool_check(
        name="allowed",
        arguments={},
        server_name=server.name,
        user_api_key_auth=None,
        proxy_logging_obj=proxy_logging,
        server=server,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["prompts/list", "prompts/get", "resources/list", "resources/templates/list", "resources/read"]
)
@pytest.mark.parametrize("state", ["success", "denied", "upstream_failure", "scope_failure"])
async def test_catalog_helpers_preserve_context_results_and_failure_policy(method, state):
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
            if collection:
                helper = getattr(operations, "_list_mcp_" + collection)
                result = await helper(
                    user_api_key_auth=caller, mcp_auth_header=None, mcp_servers=["catalog"],
                    mcp_server_auth_headers=None, oauth2_headers=None, raw_headers=headers, client_ip="192.0.2.41",
                )
                assert result == (payload if state == "success" else [])
            else:
                result = await getattr(server, handler_name)(ctx, operation.params or PaginatedRequestParams())
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
        patch.object(operations.global_mcp_server_manager, "get_tools_from_server", upstream),
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
    from datetime import datetime, timezone

    from fastapi import HTTPException

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


@pytest.mark.asyncio
@pytest.mark.parametrize("compat", ["legacy", "modern"])
async def test_local_tool_json_array_is_converted_once_for_the_caller_revision(compat: str) -> None:
    """The local-registry arm used to convert at MODERN and let the legacy downgrade append a second
    text block; converting at the caller's revision keeps the upstream body exactly once."""
    from unittest.mock import MagicMock

    from litellm.proxy._experimental.mcp_server import operations
    from litellm.proxy._experimental.mcp_server.tool_outcome import WireCompat, parse_http_body
    from litellm.proxy._experimental.mcp_server.tool_registry import global_mcp_tool_registry

    body = '["a","b"]'
    tool = MagicMock()
    tool.server_id = None
    tool.handler = AsyncMock(return_value=parse_http_body(body))
    with patch.object(global_mcp_tool_registry, "get_tool", return_value=tool):
        result = await operations._handle_local_mcp_tool("reports-list_tags", {}, WireCompat(compat))

    assert [block.text for block in result.content] == [body]
    assert result.structured_content == (["a", "b"] if compat == "modern" else None)


@pytest.mark.asyncio
async def test_discovery_preserves_caller_scope_and_proxy_restrictions():
    from mcp.types import DiscoverRequest, ListToolsResult, Tool

    listed = AsyncMock(return_value=ListToolsResult(tools=[Tool(name="allowed", input_schema={"type": "object"})]))
    context = prepare_context(UserAPIKeyAuth(user_id="scoped"), mcp_servers=["only-this"], mcp_proxy_mode=True, protocol_version="2025-06-18")
    with patch("litellm.proxy._experimental.mcp_server.operations._execute_handle_list_tools", listed):
        result = await GatewayOperations().execute(DiscoverRequest(), context)
    assert result.capabilities.tools is not None
    assert result.capabilities.resources is None
    assert result.capabilities.prompts is None
    assert listed.await_args.args[0] is context
    assert listed.await_args.args[0].user_api_key_auth.user_id == "scoped"
    assert listed.await_args.args[0].mcp_servers == ("only-this",)


@pytest.mark.asyncio
async def test_discovery_denial_cannot_advertise_tools():
    from mcp.types import DiscoverRequest
    from fastapi import HTTPException

    denied = AsyncMock(side_effect=HTTPException(status_code=403, detail="Forbidden"))
    with patch("litellm.proxy._experimental.mcp_server.operations._execute_handle_list_tools", denied):
        with pytest.raises(HTTPException) as error:
            await GatewayOperations().execute(DiscoverRequest(), prepare_context(UserAPIKeyAuth(user_id="denied")))
    assert error.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("available", ["none", "resources", "templates", "prompts"])
async def test_discovery_lists_each_capability_with_the_same_caller(available):
    from mcp.types import (
        DiscoverRequest, ListToolsResult, ListPromptsResult, ListResourcesResult,
        ListResourceTemplatesResult, Prompt, Resource, ResourceTemplate,
    )
    from litellm.proxy._experimental.mcp_server import operations

    context = prepare_context(UserAPIKeyAuth(user_id="scoped"), mcp_servers=["authorized"])
    tools = AsyncMock(return_value=ListToolsResult(tools=[]))
    prompts = AsyncMock(return_value=ListPromptsResult(prompts=[Prompt(name="allowed")] if available == "prompts" else []))
    resources = AsyncMock(return_value=ListResourcesResult(resources=[Resource(name="allowed", uri="test://allowed")] if available == "resources" else []))
    templates = AsyncMock(return_value=ListResourceTemplatesResult(resource_templates=[ResourceTemplate(name="allowed", uri_template="test://{id}")] if available == "templates" else []))
    with (
        patch.object(operations, "_execute_handle_list_tools", tools),
        patch.object(operations, "_execute_list_prompts", prompts),
        patch.object(operations, "_execute_list_resources", resources),
        patch.object(operations, "_execute_list_resource_templates", templates),
    ):
        result = await GatewayOperations().execute(DiscoverRequest(), context)
    assert result.capabilities.tools is None
    assert (result.capabilities.prompts is not None) == (available == "prompts")
    assert (result.capabilities.resources is not None) == (available in {"resources", "templates"})
    for listing in (tools, prompts, resources, templates):
        assert listing.await_args.args[0] is context


@pytest.mark.asyncio
async def test_discovery_shares_one_server_admission_across_catalog_listings() -> None:
    from mcp import types

    from litellm.proxy._experimental.mcp_server import catalog
    from litellm.proxy._experimental.mcp_server.contracts import OperationContext
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import ServerListOk

    admitted: Final = MCPServer(
        server_id="discover-admitted",
        name="discover-admitted",
        server_name="discover-admitted",
        transport=MCPTransport.http,
    )
    rejected: Final = MCPServer(
        server_id="discover-rejected",
        name="discover-rejected",
        server_name="discover-rejected",
        transport=MCPTransport.http,
    )
    caller: Final = UserAPIKeyAuth(api_key="sk-discovery-admission")
    proxy_logging: Final = _mcp_rate_limited_proxy_logging()

    async def enforce_server_rpm(_user_api_key_auth: UserAPIKeyAuth | None, server: MCPServer) -> None:
        if server.server_id == rejected.server_id:
            raise ProxyRateLimitError(detail="server RPM exceeded")

    async def fetch_tools(server: MCPServer, **_: object) -> tuple[types.ListToolsResult, ServerListOk]:
        tools: Final = [types.Tool(name=f"{server.server_id}-tool", inputSchema={"type": "object"})]
        return types.ListToolsResult(tools=tools), ServerListOk(tool_count=len(tools))

    async def fetch_prompts(*, server: MCPServer, **_: object) -> types.ListPromptsResult:
        return types.ListPromptsResult(prompts=[types.Prompt(name=f"{server.server_id}-prompt")])

    async def fetch_resources(*, server: MCPServer, **_: object) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[types.Resource(name=f"{server.server_id}-resource", uri=f"test://{server.server_id}")]
        )

    async def fetch_resource_templates(*, server: MCPServer, **_: object) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(
            resource_templates=[
                types.ResourceTemplate(
                    name=f"{server.server_id}-template",
                    uri_template=f"test://{server.server_id}/{{name}}",
                )
            ]
        )

    enforcement: Final = AsyncMock(side_effect=enforce_server_rpm)
    upstream_calls: Final = (
        AsyncMock(side_effect=fetch_tools),
        AsyncMock(side_effect=fetch_prompts),
        AsyncMock(side_effect=fetch_resources),
        AsyncMock(side_effect=fetch_resource_templates),
    )
    async def fetch_optional_page(
        _context: OperationContext,
        request: types.ListPromptsRequest | types.ListResourcesRequest | types.ListResourceTemplatesRequest,
        server: MCPServer,
        _allowed: Sequence[MCPServer],
        _cursor: str | None,
    ) -> types.ListPromptsResult | types.ListResourcesResult | types.ListResourceTemplatesResult:
        if isinstance(request, types.ListPromptsRequest):
            return await upstream_calls[1](server=server)
        if isinstance(request, types.ListResourcesRequest):
            return await upstream_calls[2](server=server)
        return await upstream_calls[3](server=server)

    optional_fetch: Final = AsyncMock(side_effect=fetch_optional_page)
    with (
        patch.object(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[admitted, rejected])),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging),
        patch.object(proxy_logging, "enforce_mcp_server_rate_limits", enforcement),
        patch.object(catalog, "get_filtered_server_tools", upstream_calls[0]),
        patch.object(catalog, "fetch_optional_catalog_page", optional_fetch),
    ):
        result: Final = await GatewayOperations().execute(
            types.DiscoverRequest(),
            prepare_context(caller, mcp_servers=[admitted.server_id, rejected.server_id]),
        )

    assert enforcement.await_count == 2
    assert {call.args[1].server_id for call in enforcement.await_args_list} == {
        admitted.server_id,
        rejected.server_id,
    }
    assert tuple(call.args[0].server_id for call in upstream_calls[0].await_args_list) == (admitted.server_id,)
    assert optional_fetch.await_count == 3
    assert all(call.args[2].server_id == admitted.server_id for call in optional_fetch.await_args_list)
    assert all(upstream.await_count == 1 for upstream in upstream_calls)
    assert result.capabilities.tools is not None
    assert result.capabilities.prompts is not None
    assert result.capabilities.resources is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_discovery_concurrent_listings_drain_on_failure_and_cancellation(outcome):
    from mcp.types import DiscoverRequest, ListToolsResult, ListPromptsResult, ListResourcesResult, ListResourceTemplatesResult
    from litellm.proxy._experimental.mcp_server import operations

    ready = [asyncio.Event() for _ in range(4)]
    closed = [asyncio.Event() for _ in range(4)]
    release = asyncio.Event()
    responses = (ListToolsResult(tools=[]), ListPromptsResult(prompts=[]), ListResourcesResult(resources=[]), ListResourceTemplatesResult(resource_templates=[]))

    def listing(index):
        async def run(*args, **kwargs):
            ready[index].set()
            try:
                await release.wait()
                if index == 0 and outcome == "failure":
                    raise ValueError("discovery failed")
                if outcome != "success":
                    await asyncio.Event().wait()
                return responses[index]
            finally:
                closed[index].set()
        return run

    with (
        patch.object(operations, "_execute_handle_list_tools", side_effect=listing(0)) as tools,
        patch.object(operations, "_execute_list_prompts", side_effect=listing(1)),
        patch.object(operations, "_execute_list_resources", side_effect=listing(2)),
        patch.object(operations, "_execute_list_resource_templates", side_effect=listing(3)),
    ):
        task = asyncio.create_task(GatewayOperations().execute(DiscoverRequest(), prepare_context(UserAPIKeyAuth(user_id="scoped"))))
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 1)
            if outcome == "cancel":
                task.cancel()
            else:
                release.set()
            if outcome == "success":
                result = await asyncio.wait_for(task, 1)
                assert result.capabilities.model_dump(exclude_none=True) == {}
            else:
                with pytest.raises(asyncio.CancelledError if outcome == "cancel" else ValueError):
                    await asyncio.wait_for(task, 1)
            assert all(event.is_set() for event in closed)
            assert tools.call_args.kwargs["log_list_tools_to_spendlogs"] is False
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("log_enabled", [False, True])
async def test_tools_listing_preserves_explicit_spend_log_policy(log_enabled):
    from mcp.types import PaginatedRequestParams
    from litellm.proxy._experimental.mcp_server import operations

    listing = AsyncMock(return_value=operations.AggregateToolListing(tools=[], outcomes={}))
    with patch.object(operations, "_list_mcp_tools", listing):
        result = await operations._execute_handle_list_tools(
            prepare_context(UserAPIKeyAuth(user_id="caller")), PaginatedRequestParams(),
            log_list_tools_to_spendlogs=log_enabled,
        )
    assert result.tools == []
    assert listing.await_args.kwargs["log_list_tools_to_spendlogs"] is log_enabled
    assert listing.await_args.kwargs["record_listing"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(("listing_kwargs", "recorded"), [({}, False), ({"record_listing": True}, True)])
async def test_list_mcp_tools_records_the_catalog_only_when_asked(
    listing_kwargs: dict[str, bool], recorded: bool
) -> None:
    """The aggregate listing fills the caller's listed-tools slot only when asked: a listing an internal
    caller never serves must not hand a later tools/call a description the caller never saw."""
    manager = operations.global_mcp_server_manager
    server = MCPServer(server_id="listing-slot", name="listing-slot", transport=MCPTransport.http, url="http://slot")
    user = UserAPIKeyAuth(api_key="sk-listing-slot", user_id="lister")
    upstream = [MCPTool(name="echo", description="Echo text back", inputSchema={"type": "object"})]
    with (
        patch.object(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server])),
        patch.object(manager, "create_mcp_client", AsyncMock(return_value=object())),
        patch.object(manager, "_fetch_tools_with_timeout", AsyncMock(return_value=upstream)),
        patch.dict(manager.tool_name_to_mcp_server_name_mapping),
    ):
        try:
            listing = await operations._list_mcp_tools(user_api_key_auth=user, **listing_kwargs)
            listed = manager.get_listed_tool(server, "echo", ListedToolsCaller(user_api_key_auth=user))
        finally:
            manager._drop_listed_tools(server.server_id)
    assert [tool.name for tool in listing.tools] == ["listing-slot-echo"]
    assert (listed is not None) is recorded


@pytest.mark.asyncio
async def test_discovery_keeps_one_catalog_revision_across_concurrent_listings(monkeypatch):
    from mcp.types import (
        DiscoverRequest, ListToolsResult, ListPromptsResult, ListResourcesResult,
        ListResourceTemplatesResult,
    )
    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server import operations
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    manager = MCPServerManager()
    original = MCPServer(server_id="catalog-server", name="before", transport=MCPTransport.http)
    updated = original.model_copy(update={"name": "after"})
    manager.registry = {original.server_id: original}
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(operations, "global_mcp_server_manager", manager)
    observed = []
    responses = (ListToolsResult(tools=[]), ListPromptsResult(prompts=[]),
                 ListResourcesResult(resources=[]), ListResourceTemplatesResult(resource_templates=[]))

    def listing(index):
        async def run(*args, **kwargs):
            async with manager.catalog.operation():
                observed.append(manager.get_mcp_server_by_id(original.server_id).name)
                manager.registry = {updated.server_id: updated}
                return responses[index]
        return run

    with (
        patch.object(operations, "_execute_handle_list_tools", side_effect=listing(0)),
        patch.object(operations, "_execute_list_prompts", side_effect=listing(1)),
        patch.object(operations, "_execute_list_resources", side_effect=listing(2)),
        patch.object(operations, "_execute_list_resource_templates", side_effect=listing(3)),
    ):
        result = await GatewayOperations().execute(DiscoverRequest(), prepare_context(UserAPIKeyAuth(user_id="scoped")))
    assert result.capabilities.model_dump(exclude_none=True) == {}
    assert observed == ["before"] * 4
    async with manager.catalog.operation():
        assert manager.get_mcp_server_by_id(original.server_id).name == "after"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_server_id", ["private", "allowed"])
async def test_local_handler_freshness_tracks_registered_owner_with_overlapping_alias(monkeypatch, changed_server_id):
    from datetime import datetime, timedelta, timezone

    from fastapi import HTTPException

    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    manager = MCPServerManager()
    now = datetime.now(timezone.utc)
    private = MCPServer(server_id="private", name="billing", alias="billing", transport=MCPTransport.http, updated_at=now)
    allowed = MCPServer(server_id="allowed", name="billing_admin", alias="billing-admin", transport=MCPTransport.http, updated_at=now)
    manager.config_mcp_servers = {server.server_id: server for server in (private, allowed)}
    monkeypatch.setattr(operations, "global_mcp_server_manager", manager)
    monkeypatch.setattr(global_mcp_tool_registry, "published_tools", {})
    handler = AsyncMock(return_value="private result")
    global_mcp_tool_registry.register_tool("billing-admin-export", "export", {}, handler, server_id=private.server_id)

    async with manager.catalog.operation():
        manager.config_mcp_servers[changed_server_id] = manager.config_mcp_servers[changed_server_id].model_copy(update={"updated_at": now + timedelta(seconds=1)})
        if changed_server_id == "private":
            with pytest.raises(HTTPException) as denied:
                await operations._handle_local_mcp_tool("billing-admin-export", {})
            assert denied.value.status_code == 503
            handler.assert_not_awaited()
        else:
            result = await operations._handle_local_mcp_tool("billing-admin-export", {})
            assert result.is_error is False
            handler.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_present", [False, True])
@pytest.mark.parametrize("requested_server", [False, True])
async def test_local_call_cannot_borrow_another_servers_authority(
    monkeypatch: pytest.MonkeyPatch, owner_present: bool, requested_server: bool
) -> None:
    from datetime import datetime

    from fastapi import HTTPException

    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    manager: Final = MCPServerManager()
    allowed: Final = MCPServer(server_id="allowed", name="allowed", transport=MCPTransport.http)
    owner: Final = MCPServer(server_id="private", name="private", transport=MCPTransport.http)
    manager.config_mcp_servers = {
        server.server_id: server for server in ((allowed, owner) if owner_present else (allowed,))
    }
    handler: Final = AsyncMock(return_value="private result")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(operations, "global_mcp_server_manager", manager)
    monkeypatch.setattr(global_mcp_tool_registry, "published_tools", {})
    tool_name: Final = "export" if requested_server else "private-export"
    global_mcp_tool_registry.register_tool(tool_name, "export", {}, handler, server_id=owner.server_id)

    async with manager.catalog.operation():
        with pytest.raises(HTTPException) as denied:
            await operations._execute_mcp_tool(
                name=tool_name if requested_server else "allowed-private-export",
                arguments={},
                allowed_mcp_servers=[allowed],
                start_time=datetime.now(),
                requested_server_id=allowed.server_id if requested_server else None,
            )
    assert denied.value.status_code == 403
    handler.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("server_owned", [False, True])
@pytest.mark.parametrize("requested_server", [False, True])
async def test_local_call_preserves_matching_and_legacy_handlers(
    monkeypatch: pytest.MonkeyPatch, server_owned: bool, requested_server: bool
) -> None:
    from datetime import datetime

    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    manager: Final = MCPServerManager()
    server: Final = MCPServer(server_id="allowed", name="allowed", transport=MCPTransport.http)
    manager.config_mcp_servers = {server.server_id: server}
    handler: Final = AsyncMock(return_value="allowed result")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(operations, "global_mcp_server_manager", manager)
    monkeypatch.setattr(global_mcp_tool_registry, "published_tools", {})
    global_mcp_tool_registry.register_tool(
        "export", "export", {}, handler, server_id=server.server_id if server_owned else None
    )

    async with manager.catalog.operation():
        result: Final = await operations._execute_mcp_tool(
            name="export" if requested_server else "allowed-export",
            arguments={},
            allowed_mcp_servers=[server],
            start_time=datetime.now(),
            requested_server_id=server.server_id if requested_server else None,
        )
    assert result.is_error is False
    handler.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_local_handler_rejects_an_owner_absent_from_the_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    manager: Final = MCPServerManager()
    handler: Final = AsyncMock(return_value="private result")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(operations, "global_mcp_server_manager", manager)
    monkeypatch.setattr(global_mcp_tool_registry, "published_tools", {})
    global_mcp_tool_registry.register_tool("private-export", "export", {}, handler, server_id="private")

    async with manager.catalog.operation():
        with pytest.raises(HTTPException) as denied:
            await operations._handle_local_mcp_tool("private-export", {})
    assert denied.value.status_code == 503
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_listing_rejects_unrecognized_continuation() -> None:
    from mcp.shared.exceptions import MCPError
    from mcp.types import ListToolsRequest, PaginatedRequestParams

    with pytest.raises(MCPError, match=r"cursor|pagination"):
        await GatewayOperations().execute(
            ListToolsRequest(params=PaginatedRequestParams(cursor="forged-pagination-state")),
            prepare_context(mcp_proxy_mode=True),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["prompts/list", "resources/list", "resources/templates/list"])
async def test_continuation_preserves_current_authority_unavailable_error(monkeypatch, method):
    from mcp import MCPError
    from mcp.types import ListPromptsRequest, ListResourcesRequest, ListResourceTemplatesRequest, PaginatedRequestParams

    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    caller = UserAPIKeyAuth(api_key="sk-owned-key-without-database")
    caller.via_virtual_key = True
    request = {"prompts/list": ListPromptsRequest, "resources/list": ListResourcesRequest, "resources/templates/list": ListResourceTemplatesRequest}[method]
    with pytest.raises(MCPError, match="Server misconfigured: no database connection"):
        await GatewayOperations().execute(request(params=PaginatedRequestParams(cursor="existing-state")), prepare_context(caller))


@pytest.mark.asyncio
async def test_virtual_tool_catalog_rejects_a_cursor_and_preserves_its_complete_listing():
    from mcp import MCPError
    from mcp.types import ListToolsRequest, PaginatedRequestParams

    from litellm.proxy._types import LiteLLM_ObjectPermissionTable

    caller = UserAPIKeyAuth(object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="search", mcp_tool_search_enabled=True))
    context = prepare_context(caller)
    result = await GatewayOperations().execute(ListToolsRequest(), context)
    assert result.tools
    assert result.next_cursor is None
    with pytest.raises(MCPError, match="fresh listing"):
        await GatewayOperations().execute(ListToolsRequest(params=PaginatedRequestParams(cursor="existing-state")), context)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_name", ["ListPromptsRequest", "ListResourcesRequest", "ListResourceTemplatesRequest"])
@pytest.mark.parametrize("cursor", [None, "existing-state"])
async def test_optional_catalog_preserves_revoked_user_error(monkeypatch, request_name, cursor):
    from types import SimpleNamespace

    from mcp import MCPError, types
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": []})
    table = SimpleNamespace(find_unique=AsyncMock(return_value=None))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(writer_db=SimpleNamespace(litellm_usertable=table)))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
    caller = UserAPIKeyAuth(user_id="revoked-catalog-user")
    caller.mcp_admitted_user_subject = True
    request = getattr(types, request_name)(params=types.PaginatedRequestParams(cursor=cursor))
    with pytest.raises(MCPError, match="Invalid or expired credential"):
        await GatewayOperations().execute(request, prepare_context(caller))


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", ["continuation-state", ""])
async def test_tool_continuation_failure_requires_a_fresh_listing(monkeypatch: pytest.MonkeyPatch, cursor: str) -> None:
    from mcp import MCPError
    from mcp.types import INVALID_PARAMS, ListToolsRequest, PaginatedRequestParams

    failure: Final = RuntimeError("catalog temporarily unavailable")
    fetch: Final = AsyncMock(side_effect=failure)
    monkeypatch.setattr(operations, "_get_tools_from_mcp_servers", fetch)
    with pytest.raises(MCPError, match="start a fresh listing") as raised:
        await GatewayOperations().execute(
            ListToolsRequest(params=PaginatedRequestParams(cursor=cursor)), prepare_context()
        )
    assert raised.value.error.code == INVALID_PARAMS
    assert raised.value.__cause__ is failure
    assert fetch.await_count == 1
    assert fetch.await_args.kwargs["params"].cursor == cursor


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway", [False, True])
async def test_initial_tool_listing_preserves_legacy_error_fallback(monkeypatch: pytest.MonkeyPatch, gateway: bool) -> None:
    from mcp.types import ListToolsRequest

    fetch: Final = AsyncMock(side_effect=RuntimeError("catalog temporarily unavailable"))
    monkeypatch.setattr(operations, "_get_tools_from_mcp_servers", fetch)
    if gateway:
        result: Final = await GatewayOperations().execute(ListToolsRequest(), prepare_context())
        assert result.tools == []
        assert result.next_cursor is None
    else:
        listing: Final = await operations._list_mcp_tools()
        assert listing.tools == []
        assert listing.next_cursor is None
    fetch.assert_awaited_once()


@pytest.mark.parametrize("header_name", ("x-litellm-api-key", "X-LiteLLM-API-Key"))
def test_discovery_extra_headers_exclude_gateway_admission_key(header_name: str) -> None:
    caller_key: Final = "Bearer sk-admission-only"
    raw_headers: Final = {"x-litellm-api-key": caller_key, "x-tenant": "tenant-control"}
    server: Final = MCPServer(
        server_id="header-boundary", name="header-boundary", transport=MCPTransport.http,
        url="https://example.invalid/mcp", auth_type=MCPAuth.none,
        extra_headers=[header_name, "X-Tenant"],
    )
    auth_header, extra_headers = operations._prepare_mcp_server_headers(
        server, None, None, None, raw_headers, UserAPIKeyAuth(api_key="sk-admission-only"),
    )
    assert auth_header is None
    assert extra_headers == {"X-Tenant": "tenant-control"}
    assert raw_headers == {"x-litellm-api-key": caller_key, "x-tenant": "tenant-control"}

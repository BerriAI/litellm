from litellm.proxy._experimental.mcp_server import operations as mcp_operations
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._experimental.mcp_server import server
from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode
from litellm.proxy._experimental.mcp_server.mcp_server_manager import ListedToolsCaller
from litellm.proxy._experimental.mcp_server.tool_search import (
    handle_mcp_proxy_tool,
    mcp_proxy_tool_id,
    with_mcp_proxy_identity,
)
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

AUTH = UserAPIKeyAuth(api_key="key")


@pytest.fixture
def proxy_mode():
    token = _mcp_proxy_mode.set(True)
    try:
        yield
    finally:
        _mcp_proxy_mode.reset(token)


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_mode")
async def test_proxy_call_rejects_non_proxy_tool_names() -> None:
    result = await mcp_operations._dispatch_virtual_mcp_tool(
        name="math_stdio-add", arguments={"a": 1, "b": 2}, user_api_key_auth=AUTH, client_ip=None, mcp_proxy_mode=True
    )

    assert result is not None
    assert result.is_error is True
    assert "unavailable on /mcp/proxy" in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_mode")
async def test_proxy_rejects_non_tool_protocol_operations() -> None:
    options = server.server.create_initialization_options()
    assert options.capabilities.prompts is None
    assert options.capabilities.resources is None
    assert options.capabilities.tools is not None

    from types import SimpleNamespace

    from mcp.server.context import ServerRequestContext
    from mcp.types import GetPromptRequestParams, PaginatedRequestParams, ReadResourceRequestParams

    ctx = ServerRequestContext(
        session=SimpleNamespace(),
        lifespan_context={},
        protocol_version="2025-06-18",
        method="",
    )

    with pytest.raises(MCPError):
        await server.list_prompts(ctx, PaginatedRequestParams())
    with pytest.raises(MCPError):
        await server.get_prompt(ctx, GetPromptRequestParams(name="prompt", arguments={}))
    with pytest.raises(MCPError):
        await server.list_resources(ctx, PaginatedRequestParams())
    with pytest.raises(MCPError):
        await server.list_resource_templates(ctx, PaginatedRequestParams())
    with pytest.raises(MCPError):
        await server.read_resource(ctx, ReadResourceRequestParams(uri="https://example.com/resource"))


class FailureRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str]] = []

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.events.append(("failure", json.dumps(kwargs.get("standard_logging_object"), default=str)))

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.events.append(("success", json.dumps(kwargs.get("standard_logging_object"), default=str)))

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        self.events.append(("post_failure", json.dumps(request_data, default=str)))


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_mode")
async def test_proxy_scope_exception_emits_failure_log(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = FailureRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    auth = UserAPIKeyAuth(
        api_key="scope-denial-key-hash",
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="denied", mcp_servers=["no-mcp-servers"]),
    )
    arguments = {"tool_id": "denied-scope", "arguments": {}}

    with pytest.raises(HTTPException) as denied:
        await mcp_operations._dispatch_virtual_mcp_tool(
            name="call_tool",
            arguments=arguments,
            user_api_key_auth=auth,
            client_ip=None,
            mcp_servers=["ungranted"],
            mcp_proxy_mode=True,
            raw_headers={"authorization": "Bearer raw-scope-secret", "x-litellm-call-id": "scope-denial"},
        )

    assert denied.value.status_code == 403
    assert denied.value.detail == {"error": "The key is not allowed to access the requested MCP servers: ungranted"}
    assert [kind for kind, _ in recorder.events] == ["failure", "post_failure"]
    payload = json.loads(recorder.events[0][1])
    assert payload["id"] == "scope-denial"
    assert payload["call_type"] == "call_mcp_tool"
    assert payload["status"] == "failure"
    assert payload["response_cost"] == 0
    assert "ungranted" in payload["error_str"]
    hook_payload = json.loads(recorder.events[1][1])
    assert hook_payload["standard_logging_object"] == payload
    assert hook_payload["arguments"] == arguments
    assert "raw_headers" not in hook_payload
    assert "raw-scope-secret" not in recorder.events[1][1]


@pytest.mark.asyncio
async def test_proxy_call_tool_on_a_never_listed_tool_hands_the_pre_hook_no_listed_tool() -> None:
    """/mcp/proxy tools/list serves only the meta-tools, so the catalog call_tool reads to resolve its
    tool_id was never served: it must not fill the caller's listed-tools slot, and the pre-call hook
    must see no listed tool for the call."""
    manager = mcp_operations.global_mcp_server_manager
    server = MCPServer(server_id="proxy-meta", name="proxy-meta", transport=MCPTransport.http, url="http://meta")
    auth = UserAPIKeyAuth(api_key="sk-proxy-meta", user_id="proxy-caller")
    upstream = [MCPTool(name="echo", description="Echo text back", inputSchema={"type": "object"})]
    served_as = with_mcp_proxy_identity(MCPTool(name="proxy-meta-echo", inputSchema={}), server.server_id)
    pre_call_tool_check = AsyncMock(return_value={})

    async def call_regular_mcp_tool(*, tasks: list[asyncio.Task[object]], **_: object) -> CallToolResult:
        await asyncio.gather(*tasks)
        return CallToolResult(content=[TextContent(type="text", text="echoed")])

    with (
        patch.dict(manager.registry, {server.server_id: server}),
        patch.dict(manager.tool_name_to_mcp_server_name_mapping),
        patch.object(mcp_operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server])),
        patch.object(manager, "create_mcp_client", AsyncMock(return_value=object())),
        patch.object(manager, "_fetch_tools_with_timeout", AsyncMock(return_value=upstream)),
        patch.object(manager, "pre_call_tool_check", pre_call_tool_check),
        patch.object(manager, "_call_regular_mcp_tool", call_regular_mcp_tool),
    ):
        try:
            result = await handle_mcp_proxy_tool(
                name="call_tool",
                arguments={"tool_id": mcp_proxy_tool_id(served_as), "arguments": {}},
                user_api_key_dict=auth,
            )
            listed = manager.get_listed_tool(server, "echo", ListedToolsCaller(user_api_key_auth=auth))
        finally:
            manager._drop_listed_tools(server.server_id)

    assert result.is_error is False
    assert result.content[0].text == "echoed"
    pre_call_tool_check.assert_awaited_once()
    assert pre_call_tool_check.await_args.kwargs["name"] == "echo"
    assert pre_call_tool_check.await_args.kwargs["tool"] is None
    assert listed is None

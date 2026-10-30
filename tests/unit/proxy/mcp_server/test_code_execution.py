import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import TextContent, Tool, ToolExecution
from pydantic import ValidationError

import litellm
from litellm.proxy.mcp_server.code_execution import (
    CODE_MODE_META,
    CODE_TOOL_NAME,
    CodeInput,
    compatibility,
    run_code_tool,
    settings,
    with_code_tool,
    with_code_trace,
)
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.types.mcp import MCPCodeModeSettings


def test_disabled_preserves_existing_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "mcp_code_mode", False)
    tool = Tool.model_validate({"name": "orders_list", "inputSchema": {"type": "object"}, "_meta": {"vendor": "kept"}})
    listed = with_code_tool([tool])
    assert listed == [tool]
    assert listed[0] is tool


def test_enabled_annotates_without_losing_metadata_and_adds_only_on_first_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "mcp_code_mode", True)
    tool = Tool.model_validate({"name": "orders_list", "inputSchema": {"type": "object"}, "_meta": {"vendor": "kept"}})
    first = with_code_tool([tool])
    later = with_code_tool([tool], first_page=False)
    assert [entry.name for entry in first] == ["orders_list", CODE_TOOL_NAME]
    assert [entry.name for entry in later] == ["orders_list"]
    assert first[0].meta is not None
    assert first[0].meta["vendor"] == "kept"
    assert first[0].meta[CODE_MODE_META]["status"] == "available"
    assert tool.meta == {"vendor": "kept"}


@pytest.mark.parametrize("name", [CODE_TOOL_NAME, "call_tool", "search_tools", "mcp_tool_call"])
def test_orchestration_is_marked_unsupported(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr(litellm, "mcp_code_mode", True)
    assert compatibility(Tool(name=name, input_schema={"type": "object"}))["status"] == "unsupported"


def test_explicit_exclusion_keeps_direct_tool_and_explains_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm, "mcp_code_mode", {"enabled": True, "excluded_tools": {"orders_approve": "Requires human confirmation"}}
    )
    tool = Tool(name="orders_approve", input_schema={"type": "object"})
    result = with_code_tool([tool])
    assert result[0].name == tool.name
    assert result[0].meta is not None
    assert result[0].meta[CODE_MODE_META] == {"status": "unsupported", "reason": "Requires human confirmation"}


def test_required_task_is_marked_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "mcp_code_mode", True)
    tool = Tool(name="long_job", input_schema={"type": "object"}, execution=ToolExecution(task_support="required"))
    assert compatibility(tool)["status"] == "unsupported"


def test_reserved_name_collision_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "mcp_code_mode", True)
    with pytest.raises(MCPError, match="reserved"):
        with_code_tool([Tool(name=CODE_TOOL_NAME, input_schema={"type": "object"})])


@pytest.mark.asyncio
async def test_cached_code_tool_rejected_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "mcp_code_mode", False)
    result = await run_code_tool({"code": "text(1)"}, OperationContext(_caller=None))
    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "disabled" in result.content[0].text


@pytest.mark.asyncio
async def test_unauthenticated_code_tool_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "mcp_code_mode", True)
    result = await run_code_tool({"code": "text(1)"}, OperationContext(_caller=None))
    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "authenticated" in result.content[0].text


def test_config_and_script_reject_unrecognized_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        MCPCodeModeSettings.model_validate({"enabled": True, "allow_network": True})
    with pytest.raises(ValidationError):
        CodeInput.model_validate({"code": "text(1)", "credentials": "example"})
    monkeypatch.setattr(litellm, "mcp_code_mode", True)
    assert settings().enabled


def test_code_trace_preserves_existing_session_and_isolates_new_runs() -> None:
    existing = OperationContext(_caller=None, raw_headers={"X-LiteLLM-Session-ID": "original-session"})
    assert with_code_trace(existing) is existing
    original = OperationContext(_caller=None)
    first = with_code_trace(original)
    second = with_code_trace(original)
    assert original.raw_headers is None
    assert first.raw_headers is not None
    assert second.raw_headers is not None
    assert first.raw_headers["x-litellm-trace-id"] != second.raw_headers["x-litellm-trace-id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_mode", [True, False])
async def test_enabled_code_tool_appears_on_existing_discovery_surfaces(
    monkeypatch: pytest.MonkeyPatch, proxy_mode: bool
) -> None:
    from mcp.types import ListToolsRequest, ListToolsResult

    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
    from litellm.proxy.mcp_server.operations import GatewayOperations

    auth = UserAPIKeyAuth(
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="code-mode-test",
            mcp_tool_search_enabled=True,
        )
    )
    context = OperationContext(_caller=auth, mcp_proxy_mode=proxy_mode)
    monkeypatch.setattr(litellm, "mcp_code_mode", False)
    before = await GatewayOperations().execute(ListToolsRequest(), context)
    monkeypatch.setattr(litellm, "mcp_code_mode", True)
    after = await GatewayOperations().execute(ListToolsRequest(), context)
    assert isinstance(before, ListToolsResult)
    assert isinstance(after, ListToolsResult)
    assert [tool.name for tool in after.tools] == [*[tool.name for tool in before.tools], CODE_TOOL_NAME]


@pytest.mark.parametrize("canonical", ["echo", CODE_TOOL_NAME, "call_tool"])
def test_identity_covers_display_alias_exclusion_and_nested_orchestration(
    monkeypatch: pytest.MonkeyPatch, canonical: str
) -> None:
    from litellm.proxy.mcp_server.code_execution import tool_names
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager
    from litellm.proxy.mcp_server.tool_search import with_mcp_proxy_identity
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="code-mode-identity-test",
        name="gateway",
        alias="gateway",
        transport=MCPTransport.http,
        url="https://example.com/mcp",
        tool_name_to_display_name={canonical: "Friendly tool"},
    )
    registry = global_mcp_server_manager.registry
    previous = registry.get(server.server_id)
    registry[server.server_id] = server
    try:
        proxy_tool = with_mcp_proxy_identity(Tool(name=f"gateway-{canonical}", input_schema={}), server.server_id)
        displayed = proxy_tool.model_copy(update={"name": "Friendly tool"})
        assert {canonical, f"gateway-{canonical}", "Friendly tool"} <= tool_names(proxy_tool)
        monkeypatch.setattr(litellm, "mcp_code_mode", True)
        expected = "available" if canonical == "echo" else "unsupported"
        assert compatibility(proxy_tool)["status"] == expected
        assert compatibility(displayed)["status"] == expected
        monkeypatch.setattr(
            litellm, "mcp_code_mode", {"enabled": True, "excluded_tools": {"Friendly tool": "Needs confirmation"}}
        )
        assert compatibility(proxy_tool)["status"] == "unsupported"
        assert compatibility(displayed)["status"] == "unsupported"
    finally:
        if previous is None:
            del registry[server.server_id]
        else:
            registry[server.server_id] = previous


def test_worker_serialization_preserves_resource_uris_and_full_mcp_envelope() -> None:
    from mcp.types import CallToolResult, EmbeddedResource, ResourceLink, TextResourceContents
    from pydantic import AnyUrl

    from litellm.proxy.mcp_server.code_execution import serialize_tool_result

    result = CallToolResult(
        content=[
            ResourceLink(type="resource_link", name="report", uri="https://example.com/report"),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(uri="file:///report.txt", text="Report contents"),
            ),
        ],
        structured_content={"total": 42},
        is_error=False,
    )
    with_metadata = result.model_copy(update={"meta": {"source": AnyUrl("https://example.com/source")}})
    serialized = serialize_tool_result(with_metadata)
    assert serialized == {
        "content": [
            {"type": "resource_link", "name": "report", "uri": "https://example.com/report"},
            {"type": "resource", "resource": {"uri": "file:///report.txt", "text": "Report contents"}},
        ],
        "structuredContent": {"total": 42},
        "_meta": {"source": "https://example.com/source"},
        "resultType": "complete",
        "isError": False,
    }


@pytest.mark.asyncio
async def test_disabled_code_name_falls_through_to_existing_tool_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.mcp_server.operations import _dispatch_virtual_mcp_tool

    monkeypatch.setattr(litellm, "mcp_code_mode", False)
    result = await _dispatch_virtual_mcp_tool(
        name=CODE_TOOL_NAME,
        arguments={"message": "ordinary upstream arguments"},
        user_api_key_auth=None,
        client_ip=None,
    )
    assert result is None


def test_rest_keeps_nested_code_tool_visible_without_name_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.mcp_server.rest_endpoints import _code_mode_rest_tool
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    monkeypatch.setattr(litellm, "mcp_code_mode", True)
    server = MCPServer(
        server_id="nested-code-test",
        name="gateway",
        alias="gateway",
        transport=MCPTransport.http,
        url="https://example.com/mcp",
    )
    nested = _code_mode_rest_tool(Tool(name=CODE_TOOL_NAME, input_schema={}), server)
    listed = with_code_tool([nested])
    assert [tool.name for tool in listed] == ["gateway-litellm_execute_code", CODE_TOOL_NAME]
    assert listed[0].meta is not None
    assert listed[0].meta[CODE_MODE_META]["status"] == "unsupported"

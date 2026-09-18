"""Tests for litellm/responses/main.py MCP auto-execution."""

import importlib
import json
import types
from unittest.mock import AsyncMock

import pytest

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing
from litellm.proxy._types import UserAPIKeyAuth
from litellm.responses.main import aresponses_api_with_mcp
from litellm.responses.mcp.mcp_streaming_iterator import MAX_MCP_TOOL_CALL_ROUNDS
from litellm.types.llms.openai import ResponsesAPIResponse

MCP_TOOL_CONFIG = {
    "type": "mcp",
    "server_label": "x",
    "server_url": "litellm_proxy/mcp/deepwiki",
    "require_approval": "never",
}


def _auth(tool_search_enabled: bool) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="k",
        user_id="u",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="test",
            mcp_tool_search_enabled=tool_search_enabled,
        ),
    )


def _function_call_response(response_id: str, call_id: str, name: str, arguments: dict) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id=response_id,
        created_at=0,
        model="gpt-5.5",
        object="response",
        status="completed",
        output=[
            {
                "type": "function_call",
                "id": call_id,
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments),
            }
        ],
    )


def _text_response(response_id: str, text: str) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id=response_id,
        created_at=0,
        model="gpt-5.5",
        object="response",
        status="completed",
        output=[
            {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


def _text_result(text: str):
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


def _patch_aresponses(monkeypatch, responses):
    """Patch both references to aresponses (main.py's own and the handler's follow-up import)."""
    mock = AsyncMock(side_effect=list(responses))
    main_module = importlib.import_module("litellm.responses.main")
    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(main_module, "aresponses", mock)
    monkeypatch.setattr(handler_module, "aresponses", mock)
    return mock


def _patch_proxy_globals(monkeypatch):
    import sys

    from unittest.mock import MagicMock

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock()
    proxy_logging_obj.post_mcp_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["response"])
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        types.SimpleNamespace(proxy_logging_obj=proxy_logging_obj),
    )
    fake_manager = types.SimpleNamespace(
        get_registry=MagicMock(return_value={}),
        call_tool=AsyncMock(),
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
        get_mcp_server_by_name=MagicMock(return_value=None),
        get_allowed_mcp_servers=AsyncMock(return_value=[]),
        get_mcp_servers_from_ids=MagicMock(return_value=[]),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )


@pytest.mark.asyncio
async def test_search_then_call_then_answer_completes_in_one_request(monkeypatch):
    """The two-hop virtual tool flow (search -> call -> text) must finish inside a
    single client request; a single-round implementation would return the
    unexecuted mcp_tool_call instead."""
    _patch_proxy_globals(monkeypatch)

    search_mock = AsyncMock(
        return_value=_text_result(
            json.dumps([{"name": "deepwiki-read_wiki_structure", "description": "d", "inputSchema": {}}])
        )
    )
    call_mock = AsyncMock(return_value=_text_result("wiki structure"))
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_search", search_mock
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_call", call_mock
    )

    aresponses_mock = _patch_aresponses(
        monkeypatch,
        [
            _function_call_response("resp-1", "call-1", "mcp_tool_search", {"query": "wiki"}),
            _function_call_response(
                "resp-2",
                "call-2",
                "mcp_tool_call",
                {"tool_name": "deepwiki-read_wiki_structure", "arguments": {"repo": "litellm"}},
            ),
            _text_response("resp-3", "The wiki has 3 sections."),
        ],
    )

    final = await aresponses_api_with_mcp(
        input="describe the wiki",
        model="gpt-5.5",
        tools=[MCP_TOOL_CONFIG],
        litellm_metadata={"user_api_key_auth": _auth(True)},
        user_api_key_auth=_auth(True),
    )

    assert aresponses_mock.await_count == 3
    assert search_mock.await_count == 1
    assert call_mock.await_count == 1
    assert call_mock.await_args.kwargs["tool_name"] == "deepwiki-read_wiki_structure"

    # Only the two virtual tools ever reach the model.
    first_call_tools = aresponses_mock.await_args_list[0].kwargs["tools"]
    assert [tool["name"] for tool in first_call_tools] == ["mcp_tool_search", "mcp_tool_call"]

    assert isinstance(final, ResponsesAPIResponse)
    output_types = [item.get("type") if isinstance(item, dict) else item.type for item in final.output]
    assert "tool_execution_results" in output_types
    tool_results_item = next(
        item for item in final.output if isinstance(item, dict) and item.get("type") == "tool_execution_results"
    )
    logged_results = json.loads(tool_results_item["content"][0]["text"])
    assert [entry["name"] for entry in logged_results] == ["mcp_tool_search", "mcp_tool_call"]
    assert [entry["display_name"] for entry in logged_results] == [
        "mcp_tool_search",
        "deepwiki-read_wiki_structure",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_search_enabled", [True, False], ids=["virtual_tools", "plain_catalog"])
async def test_auto_execute_round_cap(monkeypatch, tool_search_enabled):
    """A model that keeps calling tools is capped and the final follow-up drops
    tools so it has to answer in text. The virtual tools spend two rounds per
    real tool use, so the flagged path gets twice the budget."""
    from mcp.types import Tool

    _patch_proxy_globals(monkeypatch)
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_tools_from_mcp_servers",
        AsyncMock(
            return_value=AggregateToolListing(
                tools=[Tool(name="deepwiki-read_wiki_contents", description="d", inputSchema={})], outcomes={}
            )
        ),
    )
    executed = AsyncMock(return_value=_text_result("[]"))
    if tool_search_enabled:
        called_tool = "mcp_tool_search"
        expected_cap = MAX_MCP_TOOL_CALL_ROUNDS * 2
        monkeypatch.setattr(
            "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_search", executed
        )
    else:
        called_tool = "deepwiki-read_wiki_contents"
        expected_cap = MAX_MCP_TOOL_CALL_ROUNDS
        monkeypatch.setattr(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.call_tool",
            executed,
        )

    aresponses_mock = _patch_aresponses(
        monkeypatch,
        [
            _function_call_response(f"resp-{i}", f"call-{i}", called_tool, {"query": "wiki"})
            for i in range(MAX_MCP_TOOL_CALL_ROUNDS * 2 + 5)
        ],
    )

    await aresponses_api_with_mcp(
        input="describe the wiki",
        model="gpt-5.5",
        tools=[MCP_TOOL_CONFIG],
        litellm_metadata={"user_api_key_auth": _auth(tool_search_enabled)},
        user_api_key_auth=_auth(tool_search_enabled),
    )

    assert executed.await_count == expected_cap
    assert aresponses_mock.await_count == expected_cap + 1
    assert aresponses_mock.await_args_list[-1].kwargs["tools"] is None

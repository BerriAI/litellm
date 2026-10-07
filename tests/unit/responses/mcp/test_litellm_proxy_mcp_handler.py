import asyncio
import importlib
import subprocess
import sys
import textwrap
import types
from contextlib import nullcontext
from typing import Any, Final, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from openai.types.responses.tool_param import Mcp

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._experimental.mcp_server import operations as mcp_operations
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing
from litellm.proxy._experimental.mcp_server.mcp_server_manager import ListedToolsCaller, MCPServerManager
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.responses import main as responses_main
from litellm.responses.mcp import litellm_proxy_mcp_handler as mcp_handler_module
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.responses.main import OutputFunctionToolCall
from litellm.types.utils import GenericGuardrailAPIInputs, ModelResponse


class _DummyMCPResult:
    def __init__(self):
        self.content = []


def _setup_mcp_call_environment(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch MCP globals so _execute_tool_calls can run in tests."""
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object(), prisma_client=None)
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    fake_manager = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        call_tool=AsyncMock(return_value=_DummyMCPResult()),
        # Newer logging path calls this to enrich spend logs metadata
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )
    return fake_manager.call_tool


def _setup_proxy_logging(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch proxy_logging_obj so failure hook can be asserted."""
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock()
    proxy_module = types.SimpleNamespace(proxy_logging_obj=proxy_logging_obj, prisma_client=None)
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)
    return proxy_logging_obj.post_call_failure_hook


def test_deduplicate_mcp_tools_single_allowed_server():
    tools = [{"name": "search"}, {"name": "search"}]  # duplicate on purpose

    deduped, server_map = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(
        tools,
        ["everything"],
    )

    assert len(deduped) == 1
    assert server_map == {"search": "everything"}


@pytest.mark.parametrize(
    "tool_name,expected_server",
    [
        ("alpha-tool", "alpha"),
        ("beta-another_tool", "beta"),
    ],
)
def test_deduplicate_mcp_tools_prefixed_names(tool_name, expected_server):
    tools = [{"name": tool_name}]

    _, server_map = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(
        tools,
        ["alpha", "beta"],
    )

    assert server_map[tool_name] == expected_server


def test_extract_tool_calls_from_chat_response_handles_tool_calls():
    response = ModelResponse(
        id="resp-1",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-123",
                            "type": "function",
                            "function": {"name": "foo", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        model="gpt",
        created=0,
        object="chat.completion",
    )

    tool_calls = LiteLLM_Proxy_MCP_Handler.extract_tool_calls_from_chat_response(response)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "foo"


def test_create_follow_up_messages_for_chat_appends_tool_results():
    original_messages = [{"role": "user", "content": "hi"}]
    response = ModelResponse(
        id="resp-2",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-abc",
                            "type": "function",
                            "function": {"name": "foo", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
        model="gpt",
        created=0,
        object="chat.completion",
    )
    tool_results = [
        {
            "tool_call_id": "call-abc",
            "name": "foo",
            "result": "done",
        }
    ]

    follow_up = LiteLLM_Proxy_MCP_Handler.create_follow_up_messages_for_chat(
        original_messages,
        response,
        tool_results,
    )

    assert follow_up[0]["role"] == "user"
    assert follow_up[-1]["role"] == "tool"
    assert follow_up[-1]["name"] == "foo"
    assert follow_up[-1]["content"] == "done"


def test_transform_mcp_tools_to_openai_uses_chat_format(monkeypatch):
    captured = {}

    def fake_transform_chat(tool):
        captured.setdefault("chat", []).append(tool)
        return {"chat": True}

    def fake_transform_responses(tool):
        captured.setdefault("responses", []).append(tool)
        return {"responses": True}

    monkeypatch.setattr(
        "litellm.experimental_mcp_client.tools.transform_mcp_tool_to_openai_tool",
        fake_transform_chat,
    )
    monkeypatch.setattr(
        "litellm.experimental_mcp_client.tools.transform_mcp_tool_to_openai_responses_api_tool",
        fake_transform_responses,
    )

    chat_tools = LiteLLM_Proxy_MCP_Handler.transform_mcp_tools_to_openai(["tool"], target_format="chat")
    resp_tools = LiteLLM_Proxy_MCP_Handler.transform_mcp_tools_to_openai(["tool"])

    assert chat_tools == [{"chat": True}]
    assert resp_tools == [{"responses": True}]
    assert captured["chat"] == ["tool"]
    assert captured["responses"] == ["tool"]


def test_create_follow_up_input_handles_response_function_tool_call():
    response = types.SimpleNamespace(
        output=[
            OutputFunctionToolCall(
                id="id",
                type="function_call",
                call_id="call-1",
                name="foo",
                arguments="{}",
                status="completed",
            )
        ]
    )

    follow_up = LiteLLM_Proxy_MCP_Handler.create_follow_up_input(
        response=cast(Any, response),
        tool_results=[],
        original_input=None,
    )

    assert follow_up == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "foo",
            "arguments": "{}",
        }
    ]


@pytest.mark.asyncio
async def test_execute_tool_calls_strips_server_prefix(monkeypatch):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    tool_name = "deepwiki-read_wiki_structure"
    tool_calls = [
        {
            "id": "call-1",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == "read_wiki_structure"


@pytest.mark.asyncio
async def test_execute_tool_calls_keeps_tool_name_without_prefix(monkeypatch):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    tool_name = "read_wiki_structure"
    tool_calls = [
        {
            "id": "call-2",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == tool_name


@pytest.mark.asyncio
async def test_execute_tool_calls_keeps_tool_name_when_equal_to_server(monkeypatch):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    tool_name = "echo"
    tool_calls = [
        {
            "id": "call-3",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "echo"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == tool_name


@pytest.mark.asyncio
async def test_execute_tool_calls_strips_prefix_when_alias_differs_from_server_name(
    monkeypatch,
):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    fake_server = types.SimpleNamespace(
        alias="my_deepwiki",
        server_name="deepwiki_test",
        server_id="test-server-id",
        short_prefix=None,
        mcp_info=None,
        tool_name_to_display_name=None,
    )
    from litellm.proxy._experimental.mcp_server import mcp_server_manager as _msm

    _msm.global_mcp_server_manager._get_mcp_server_from_tool_name = MagicMock(return_value=fake_server)

    tool_name = "my_deepwiki-read_wiki_structure"
    tool_calls = [
        {
            "id": "call-4",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki_test"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == "read_wiki_structure"


@pytest.mark.asyncio
async def test_execute_tool_calls_reverse_maps_display_name(monkeypatch):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    colliding_server = types.SimpleNamespace(
        alias=None,
        server_name="other_mcp",
        server_id="other-server-id",
        short_prefix=None,
        mcp_info=None,
        tool_name_to_display_name={"search": "search_docs"},
    )
    fake_server = types.SimpleNamespace(
        alias=None,
        server_name="deepwiki_mcp",
        server_id="test-server-id",
        short_prefix=None,
        mcp_info=None,
        tool_name_to_display_name={"read_wiki_structure": "browse_repo_docs"},
    )
    from litellm.proxy._experimental.mcp_server import mcp_server_manager as _msm

    _msm.global_mcp_server_manager._get_mcp_server_from_tool_name = MagicMock(return_value=colliding_server)
    _msm.global_mcp_server_manager.get_mcp_server_by_name = MagicMock(return_value=fake_server)

    tool_name = "browse_repo_docs"
    tool_calls = [
        {
            "id": "call-5",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki_mcp"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == "read_wiki_structure"


@pytest.mark.asyncio
async def test_execute_tool_calls_logs_failure_via_post_call_failure_hook(monkeypatch):
    """
    Regression test for ae4d92ad...:
    Ensure responses-side MCP tool execution logs failures via proxy_logging_obj.post_call_failure_hook.
    """
    post_call_failure_hook = _setup_proxy_logging(monkeypatch)

    fake_manager = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        call_tool=AsyncMock(side_effect=HTTPException(status_code=500, detail="boom")),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    tool_name = "deepwiki-read_wiki_structure"
    tool_calls = [{"id": "call-err", "function": {"name": tool_name, "arguments": "{}"}}]

    user_auth = types.SimpleNamespace(api_key="test_key", user_id="test_user")

    results = await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=user_auth,
        litellm_call_id="cid",
        litellm_trace_id="tid",
    )

    assert len(results) == 1
    assert results[0]["tool_call_id"] == "call-err"
    assert results[0]["name"] == tool_name

    post_call_failure_hook.assert_awaited_once()
    assert post_call_failure_hook.await_args is not None
    assert post_call_failure_hook.await_args.kwargs.get("route") == "/responses/mcp/call_tool"


@pytest.mark.asyncio
async def test_execute_tool_calls_passes_litellm_call_id_and_trace_id_to_function_setup(
    monkeypatch,
):
    """
    Regression test for ae4d92ad...:
    Ensure litellm_call_id / litellm_trace_id are forwarded into function_setup kwargs.
    """
    _setup_proxy_logging(monkeypatch)
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)

    captured = {}

    def fake_function_setup(*_args, **kwargs):
        captured.update(kwargs)
        return None, None

    # NOTE: Don't patch via dotted string path here because `litellm.responses`
    # is a function attribute on the `litellm` package (shadowing the submodule),
    # which breaks monkeypatch's importpath resolution.
    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(handler_module, "function_setup", fake_function_setup)

    tool_name = "deepwiki-read_wiki_structure"
    tool_calls = [{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}]

    await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
        litellm_call_id="cid",
        litellm_trace_id="tid",
    )

    # Ensure the tool call was attempted (sanity)
    assert call_tool_mock.await_count == 1

    assert captured.get("litellm_call_id") == "cid"
    assert captured.get("litellm_trace_id") == "tid"


@pytest.mark.asyncio
async def test_execute_tool_calls_threads_logging_obj_into_call_tool(monkeypatch):
    """The Responses-API MCP path must hand the request's litellm_logging_obj to
    global_mcp_server_manager.call_tool, otherwise pre_call_tool_check /
    _create_during_hook_task get None and no guardrail evaluation is bridged onto
    the request logger, so MCP tool calls made through the Responses API report zero
    guardrail evaluations in the monitor. Drop the litellm_logging_obj kwarg on the
    call_tool invocation and this fails."""
    _setup_proxy_logging(monkeypatch)
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)

    sentinel_logging_obj = MagicMock()
    sentinel_logging_obj.async_post_mcp_tool_call_hook = AsyncMock()
    sentinel_logging_obj.async_success_handler = AsyncMock()

    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(
        handler_module,
        "function_setup",
        lambda *_args, **_kwargs: (sentinel_logging_obj, None),
    )

    tool_name = "deepwiki-read_wiki_structure"
    tool_calls = [{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}]

    await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["litellm_logging_obj"] is sentinel_logging_obj


@pytest.mark.asyncio
async def test_execute_tool_calls_applies_post_call_hook_content(monkeypatch):
    proxy_module = types.SimpleNamespace(proxy_logging_obj=None)
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    result = CallToolResult(
        content=[TextContent(type="text", text="SECRET-1234")],
        structuredContent={"result": "SECRET-1234"},
        isError=False,
    )
    fake_manager = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        call_tool=AsyncMock(return_value=result),
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.async_post_mcp_tool_call_hook = AsyncMock(
        return_value=CallToolResult(content=[TextContent(type="text", text="[REDACTED]")], is_error=True)
    )
    logging_obj.async_success_handler = AsyncMock()
    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(handler_module, "function_setup", lambda *_args, **_kwargs: (logging_obj, None))

    tool_name = "deepwiki-read_wiki_structure"
    results = await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=[{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}],
        user_api_key_auth=None,
    )

    assert results == [{"tool_call_id": "call-1", "result": "[REDACTED]", "name": tool_name}]
    assert logging_obj.async_success_handler.await_args.kwargs["result"].content[0].text == "[REDACTED]"
    assert logging_obj.async_success_handler.await_args.kwargs["result"].structured_content is None


@pytest.mark.asyncio
async def test_execute_tool_calls_returns_proxy_result_without_logging(monkeypatch):
    result = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.post_mcp_call_hook = AsyncMock(side_effect=lambda response, **_: response)
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.proxy_server", types.SimpleNamespace(proxy_logging_obj=proxy_logging_obj)
    )

    fake_manager = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        call_tool=AsyncMock(return_value=result),
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )
    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(handler_module, "function_setup", lambda *_args, **_kwargs: (None, None))

    tool_name = "deepwiki-read_wiki_structure"
    results = await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=[{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}],
        user_api_key_auth=None,
    )

    assert results == [{"tool_call_id": "call-1", "result": "ok", "name": tool_name}]
    proxy_logging_obj.post_mcp_call_hook.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_tool_calls_passes_logging_details_to_proxy_hook(monkeypatch):
    result = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.post_mcp_call_hook = AsyncMock(side_effect=lambda response, **_: response)
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.proxy_server", types.SimpleNamespace(proxy_logging_obj=proxy_logging_obj)
    )

    fake_manager = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        call_tool=AsyncMock(return_value=result),
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"request_id": "request-1"}
    logging_obj.async_post_mcp_tool_call_hook = AsyncMock(return_value=result)
    logging_obj.async_success_handler = AsyncMock()
    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(handler_module, "function_setup", lambda *_args, **_kwargs: (logging_obj, None))

    tool_name = "deepwiki-read_wiki_structure"
    results = await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=[{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}],
        user_api_key_auth=None,
    )

    assert results == [{"tool_call_id": "call-1", "result": "ok", "name": tool_name}]
    assert proxy_logging_obj.post_mcp_call_hook.await_args.kwargs["request_data"] == logging_obj.model_call_details


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["post_call_hook", "success_handler"])
async def test_execute_tool_calls_continues_when_post_call_logging_fails(monkeypatch, failure_stage: str):
    proxy_module = types.SimpleNamespace(proxy_logging_obj=None)
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    result = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
    fake_manager = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        call_tool=AsyncMock(return_value=result),
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.post_call = MagicMock()
    logging_obj.async_post_mcp_tool_call_hook = AsyncMock(
        side_effect=RuntimeError("hook failed") if failure_stage == "post_call_hook" else None,
        return_value=result,
    )
    logging_obj.async_success_handler = AsyncMock(
        side_effect=RuntimeError("success logging failed") if failure_stage == "success_handler" else None
    )
    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(handler_module, "function_setup", lambda *_args, **_kwargs: (logging_obj, None))

    tool_name = "deepwiki-read_wiki_structure"
    results = await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=[{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}],
        user_api_key_auth=None,
    )

    assert results == [{"tool_call_id": "call-1", "result": "ok", "name": tool_name}]


@pytest.mark.asyncio
async def test_get_mcp_tools_from_manager_enables_list_tools_logging(monkeypatch):
    """
    Regression test for 872e5b98...:
    Ensure responses-side tool discovery enables list-tools SpendLogs logging flags.
    """
    served_tools: Final = [
        MCPTool(name="safe", description="Safe lookup", inputSchema={"type": "object"}),
        MCPTool(
            name="masked",
            description="Contact [MASKED]",
            inputSchema={"type": "object", "properties": {"query": {"type": "string", "description": "For [MASKED]"}}},
        ),
    ]
    mock_get_tools = AsyncMock(return_value=AggregateToolListing(tools=served_tools, outcomes={}))
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_tools_from_mcp_servers",
        mock_get_tools,
    )

    # Patch manager methods used by _get_mcp_tools_from_manager to avoid needing full UserAPIKeyAuth fields.
    fake_manager = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        get_allowed_mcp_servers=AsyncMock(return_value=[]),
        get_mcp_servers_from_ids=MagicMock(return_value=[]),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    user_auth = types.SimpleNamespace(api_key="test_key", user_id="test_user")
    tools, _server_names = await LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager(
        user_api_key_auth=user_auth,
        mcp_tools_with_litellm_proxy=[{"type": "mcp", "server_url": "litellm_proxy/mcp/deepwiki"}],
    )

    forwarded: Final = LiteLLM_Proxy_MCP_Handler.transform_mcp_tools_to_openai(tools)
    assert [tool["name"] for tool in forwarded] == ["safe", "masked"]
    assert forwarded[0]["description"] == "Safe lookup"
    assert forwarded[1]["description"] == "Contact [MASKED]"
    assert forwarded[1]["parameters"] == {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "For [MASKED]"}},
        "additionalProperties": False,
    }
    assert mock_get_tools.await_count == 1
    assert mock_get_tools.await_args is not None
    assert mock_get_tools.await_args.kwargs["log_list_tools_to_spendlogs"] is True
    assert mock_get_tools.await_args.kwargs["list_tools_log_source"] == "responses"


def test_get_parent_request_tags_from_metadata():
    tags = LiteLLM_Proxy_MCP_Handler.get_parent_request_tags({"metadata": {"tags": ["team-a", "prod"]}})
    assert tags == ["team-a", "prod"]


def test_get_parent_request_tags_from_nested_litellm_params():
    tags = LiteLLM_Proxy_MCP_Handler.get_parent_request_tags(
        {
            "metadata": {"tags": ["top-level"]},
            "litellm_params": {
                "metadata": {"tags": ["nested"]},
                "proxy_server_request": {"headers": {"user-agent": "client/1.0"}},
            },
        }
    )
    assert tags == ["nested", "User-Agent: client", "User-Agent: client/1.0"]


@pytest.mark.asyncio
async def test_get_mcp_tools_from_manager_forwards_request_tags(monkeypatch):
    mock_get_tools = AsyncMock(return_value=AggregateToolListing(tools=[], outcomes={}))
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_tools_from_mcp_servers",
        mock_get_tools,
    )
    fake_manager = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        get_allowed_mcp_servers=AsyncMock(return_value=[]),
        get_mcp_servers_from_ids=MagicMock(return_value=[]),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    await LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager(
        user_api_key_auth=types.SimpleNamespace(api_key="k", user_id="u"),
        mcp_tools_with_litellm_proxy=[{"type": "mcp", "server_url": "litellm_proxy/mcp/deepwiki"}],
        request_tags=["team-a"],
    )

    assert mock_get_tools.await_args.kwargs["request_tags"] == ["team-a"]


@pytest.mark.asyncio
async def test_execute_tool_calls_exposes_sanitized_client_headers_to_logging(monkeypatch):
    """The Responses API MCP bridge used to log an empty header dict, hiding the caller's
    headers from logging callbacks and hooks."""
    _setup_proxy_logging(monkeypatch)
    _setup_mcp_call_environment(monkeypatch)

    captured = {}

    def fake_function_setup(*_args, **kwargs):
        captured.update(kwargs)
        return None, None

    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(handler_module, "function_setup", fake_function_setup)

    tool_name = "deepwiki-read_wiki_structure"
    await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=[{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}],
        user_api_key_auth=None,
        raw_headers={"x-nuid": "nuid-1", "x-litellm-api-key": "sk-proxy", "cookie": "s=1"},
    )

    expected = {"x-nuid": "nuid-1", "cookie": "***REDACTED***"}
    assert captured["metadata"]["headers"] == expected
    assert captured["proxy_server_request"]["headers"] == expected


@pytest.mark.asyncio
async def test_execute_tool_calls_propagates_request_tags_to_function_setup(monkeypatch):
    _setup_proxy_logging(monkeypatch)
    _setup_mcp_call_environment(monkeypatch)
    captured = {}

    def fake_function_setup(*_args, **kwargs):
        captured.update(kwargs)
        return None, None

    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(handler_module, "function_setup", fake_function_setup)

    tool_name = "deepwiki-read_wiki_structure"
    await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=[{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}],
        user_api_key_auth=None,
        request_tags=["team-a", "prod"],
    )

    assert captured["metadata"]["tags"] == ["team-a", "prod"]


def test_completion_with_function_tools_works_without_fastapi_installed():
    script = textwrap.dedent(
        """
        import sys

        class _FastapiBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "fastapi" or fullname.startswith("fastapi."):
                    raise ModuleNotFoundError("No module named 'fastapi'")
                return None

        sys.meta_path.insert(0, _FastapiBlocker())

        import litellm

        response = litellm.completion(
            model="openai/gpt-5.5",
            messages=[{"role": "user", "content": "What is the weather in SF?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            mock_response="sunny",
        )
        assert response.choices[0].message.content == "sunny"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_extract_tool_call_details_reads_anthropic_tool_use_input():
    """
    Regression test (LIT-4517): an Anthropic tool_use block carries its arguments
    under `input`, not `arguments`.

    Given: A tool_use content block as /v1/messages returns it
    When:  The shared extractor reads it
    Then:  The arguments come back, so the MCP tool is called with them

    Reading only `arguments` fails silently rather than loudly: _parse_tool_arguments
    turns the resulting None into {}, so the tool still executes, just with every
    argument dropped.
    """
    tool_use_block = {
        "type": "tool_use",
        "id": "toolu_01ABC",
        "name": "read_wiki_structure",
        "input": {"repoName": "BerriAI/litellm"},
    }

    name, arguments, call_id = LiteLLM_Proxy_MCP_Handler.extract_tool_call_details(tool_use_block)

    assert name == "read_wiki_structure"
    assert call_id == "toolu_01ABC"
    assert arguments == {"repoName": "BerriAI/litellm"}
    assert LiteLLM_Proxy_MCP_Handler._parse_tool_arguments(arguments) == {"repoName": "BerriAI/litellm"}


def test_extract_tool_call_details_still_prefers_openai_arguments():
    """The OpenAI chat shape must keep winning; `input` is only the fallback."""
    openai_tool_call = {
        "id": "call_123",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }

    name, arguments, call_id = LiteLLM_Proxy_MCP_Handler.extract_tool_call_details(openai_tool_call)

    assert name == "get_weather"
    assert call_id == "call_123"
    assert arguments == '{"city": "Paris"}'


def _registered(
    server_id: str,
    name: str,
    alias: str | None = None,
    server_name: str | None = None,
    access_groups: list[str] | None = None,
):
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    return MCPServer(
        server_id=server_id,
        name=name,
        alias=alias,
        server_name=server_name,
        transport=MCPTransport.http,
        access_groups=access_groups,
    )


async def _no_toolset(_: str) -> bool:
    return False


ZAPIER_TOOL: Mcp = {
    "type": "mcp",
    "server_label": "zapier",
    "server_url": "https://mcp.zapier.com/api/mcp/mcp",
    "require_approval": "never",
}
EXPLICIT_GATEWAY_TOOL = {"type": "mcp", "server_label": "github", "server_url": "litellm_proxy/mcp/github"}
FUNCTION_TOOL = {"type": "function", "name": "get_weather", "parameters": {}}


@pytest.mark.asyncio
async def test_gateway_served_names_matches_alias_server_name_name_access_group_and_toolset():
    from litellm.responses.mcp.litellm_proxy_mcp_handler import _gateway_served_names

    servers = (
        _registered("id-1", "github-name", alias="github", server_name="github-server", access_groups=["prod-group"]),
        _registered("id-2", "deepwiki"),
    )

    async def toolset_exists(name: str) -> bool:
        return name == "my-toolset"

    served = await _gateway_served_names(
        {"github", "github-server", "github-name", "deepwiki", "prod-group", "my-toolset", "mcp", "nope"},
        servers=lambda: servers,
        toolset_exists=toolset_exists,
    )

    assert served == {"github", "github-server", "github-name", "deepwiki", "prod-group", "my-toolset"}


@pytest.mark.asyncio
async def test_gateway_served_names_matches_server_id_short_prefix_and_alias_case_like_the_gateway():
    from litellm.proxy._experimental.mcp_server.utils import compute_short_server_prefix
    from litellm.responses.mcp.litellm_proxy_mcp_handler import _gateway_served_names

    server_id = "0b9ae4ca-1bd2-4faa-b183-7dd812597e3b"
    short_prefix = compute_short_server_prefix(server_id)
    servers = (_registered(server_id, "github-name", alias="github", access_groups=["prod-group"]),)

    served = await _gateway_served_names(
        {server_id, short_prefix, "GitHub", "PROD-GROUP", "nope"}, servers=lambda: servers, toolset_exists=_no_toolset
    )

    assert served == {server_id, short_prefix, "GitHub"}


@pytest.mark.asyncio
async def test_routes_through_gateway_flags_explicit_and_served_tools_only():
    served_tool = {"type": "mcp", "server_label": "github", "server_url": "http://localhost:4000/mcp/github"}

    async def served_names(names):
        assert names == {"github", "mcp"}
        return frozenset({"github"})

    flags = await LiteLLM_Proxy_MCP_Handler.routes_through_gateway(
        [ZAPIER_TOOL, EXPLICIT_GATEWAY_TOOL, served_tool, FUNCTION_TOOL], served_names=served_names
    )

    assert flags == (False, True, True, False)


@pytest.mark.asyncio
async def test_split_mcp_tools_leaves_external_mcp_path_urls_for_the_provider():

    async def served_names(names):
        assert names == {"mcp"}
        return frozenset()

    gateway_tools, other_tools = await LiteLLM_Proxy_MCP_Handler.split_mcp_tools(
        [ZAPIER_TOOL, EXPLICIT_GATEWAY_TOOL, FUNCTION_TOOL], served_names=served_names
    )

    assert gateway_tools == [EXPLICIT_GATEWAY_TOOL]
    assert other_tools == [ZAPIER_TOOL, FUNCTION_TOOL]


@pytest.mark.asyncio
async def test_split_mcp_tools_repoints_served_proxy_urls_at_the_gateway():
    served_tool = {
        "type": "mcp",
        "server_label": "toolset",
        "server_url": "http://localhost:4000/mcp/my-toolset",
        "require_approval": "never",
        "allowed_tools": ["get_me"],
    }
    unserved_tool = {"type": "mcp", "server_label": "typo", "server_url": "http://localhost:4000/mcp/githb"}

    async def served_names(names):
        return frozenset({"my-toolset"})

    gateway_tools, other_tools = await LiteLLM_Proxy_MCP_Handler.split_mcp_tools(
        [served_tool, unserved_tool], served_names=served_names
    )

    assert gateway_tools == [{**served_tool, "server_url": "litellm_proxy/mcp/my-toolset"}]
    assert other_tools == [unserved_tool]


@pytest.mark.asyncio
async def test_split_mcp_tools_skips_resolution_when_nothing_points_at_the_proxy():
    async def served_names(names):
        raise AssertionError("no lookup expected")

    gateway_tools, other_tools = await LiteLLM_Proxy_MCP_Handler.split_mcp_tools(
        [EXPLICIT_GATEWAY_TOOL, FUNCTION_TOOL], served_names=served_names
    )

    assert gateway_tools == [EXPLICIT_GATEWAY_TOOL]
    assert other_tools == [FUNCTION_TOOL]


def test_should_use_gateway_still_triggers_on_http_mcp_path():
    assert LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway([ZAPIER_TOOL]) is True
    assert LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway([EXPLICIT_GATEWAY_TOOL]) is True
    assert LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway([FUNCTION_TOOL]) is False
    assert LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(None) is False


@pytest.mark.asyncio
async def test_aresponses_api_with_mcp_forwards_unserved_external_mcp_tool_to_the_provider(monkeypatch):
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager
    from litellm.responses import main as responses_main
    from litellm.types.llms.openai import ResponsesAPIResponse

    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", types.SimpleNamespace(prisma_client=None))
    monkeypatch.setattr(global_mcp_server_manager, "get_registry", lambda: {})
    provider_tools: list[object] = []

    def fake_provider(**kwargs: object) -> object:
        request_params = cast(dict[str, object], kwargs["response_api_optional_request_params"])
        provider_tools.append(request_params.get("tools"))

        async def respond() -> ResponsesAPIResponse:
            return ResponsesAPIResponse(id="resp_zapier", created_at=0, output=[])

        return respond()

    monkeypatch.setattr(responses_main.base_llm_http_handler, "response_api_handler", fake_provider)

    response = await responses_main.aresponses_api_with_mcp(
        input="Reply with the single word ok.", model="openai/gpt-4.1", tools=[ZAPIER_TOOL]
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert provider_tools == [[ZAPIER_TOOL]]


def _response_with_reasoning_and_tool_call() -> Any:
    """A first-turn response as a reasoning model returns it: reasoning item, then a function call."""
    return ResponsesAPIResponse(
        id="resp_first",
        created_at=1234567890,
        model="gpt-5",
        object="response",
        status="completed",
        output=[
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "gAAAAA-opaque-blob",
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call-1",
                "name": "foo",
                "arguments": "{}",
                "status": "completed",
            },
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def test_create_follow_up_input_preserves_reasoning_when_stateless():
    """
    Regression test (LIT-5427): a store=false follow-up has to replay the reasoning
    item, including reasoning.encrypted_content, since the provider kept no state.
    """
    follow_up = LiteLLM_Proxy_MCP_Handler.create_follow_up_input(
        response=_response_with_reasoning_and_tool_call(),
        tool_results=[{"tool_call_id": "call-1", "name": "foo", "result": "done"}],
        original_input="hi",
        preserve_reasoning=True,
    )

    assert follow_up[1] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "gAAAAA-opaque-blob",
    }
    assert follow_up[2] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "foo",
        "arguments": "{}",
    }
    assert follow_up[3] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "done",
    }


def _response_with_interleaved_reasoning_and_tool_calls() -> Any:
    """A first-turn response that reasons before each of two function calls."""
    return ResponsesAPIResponse(
        id="resp_first",
        created_at=1234567890,
        model="gpt-5",
        object="response",
        status="completed",
        output=[
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "blob-1"},
            {"type": "function_call", "id": "fc_1", "call_id": "call-1", "name": "foo", "arguments": "{}"},
            {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "blob-2"},
            {"type": "function_call", "id": "fc_2", "call_id": "call-2", "name": "bar", "arguments": "{}"},
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def test_create_follow_up_input_keeps_each_reasoning_item_before_its_function_call():
    """
    Regression test (LIT-5427): the provider pairs a replayed reasoning item with the
    item that follows it, so the replay has to keep the response's output order instead
    of grouping every reasoning item ahead of every function call.
    """
    follow_up = LiteLLM_Proxy_MCP_Handler.create_follow_up_input(
        response=_response_with_interleaved_reasoning_and_tool_calls(),
        tool_results=[
            {"tool_call_id": "call-1", "name": "foo", "result": "one"},
            {"tool_call_id": "call-2", "name": "bar", "result": "two"},
        ],
        original_input="hi",
        preserve_reasoning=True,
    )

    assert [cast(dict[str, Any], item)["type"] for item in follow_up] == [
        "message",
        "reasoning",
        "function_call",
        "reasoning",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert [
        cast(dict[str, Any], item).get("id") or cast(dict[str, Any], item).get("call_id") for item in follow_up[1:5]
    ] == [
        "rs_1",
        "call-1",
        "rs_2",
        "call-2",
    ]


def test_create_follow_up_input_omits_reasoning_when_stateful():
    """With store=true the provider still holds the reasoning item, so don't resend it."""
    follow_up = LiteLLM_Proxy_MCP_Handler.create_follow_up_input(
        response=_response_with_reasoning_and_tool_call(),
        tool_results=[{"tool_call_id": "call-1", "name": "foo", "result": "done"}],
        original_input="hi",
    )

    assert not [item for item in follow_up if isinstance(item, dict) and item.get("type") == "reasoning"]


@pytest.mark.parametrize(
    "call_params, expected",
    [
        ({"store": False}, True),
        ({"store": True}, False),
        ({"store": None}, False),
        ({}, False),
    ],
)
def test_is_persistence_disabled(call_params: dict[str, Any], expected: bool):
    assert LiteLLM_Proxy_MCP_Handler.is_persistence_disabled(call_params) is expected


@pytest.mark.parametrize(
    "store, caller_previous_response_id, expected_previous_response_id",
    [
        (False, None, None),
        (False, "resp_caller", "resp_caller"),
        (True, None, "resp_first"),
        (True, "resp_caller", "resp_first"),
    ],
)
@pytest.mark.asyncio
async def test_mcp_follow_up_call_is_stateless_when_store_is_false(
    monkeypatch: pytest.MonkeyPatch,
    store: bool,
    caller_previous_response_id: str | None,
    expected_previous_response_id: str | None,
):
    """
    Regression test (LIT-5427): linking the MCP follow-up call to the first response's id
    fails for zero data retention callers, because store=false means it was never persisted.
    The caller's own previous_response_id was valid for the first call, so it stays.
    """
    captured_calls: list[dict[str, Any]] = []
    first_response = _response_with_reasoning_and_tool_call()

    async def fake_aresponses(**kwargs: Any) -> ResponsesAPIResponse:
        captured_calls.append(kwargs)
        return (
            first_response
            if len(captured_calls) == 1
            else ResponsesAPIResponse(
                id="resp_follow_up",
                created_at=1234567891,
                model="gpt-5",
                object="response",
                status="completed",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            )
        )

    async def fake_process(**kwargs: Any) -> tuple[list[Any], dict[str, str]]:
        assert kwargs["raw_headers"] == {"x-app-id": "follow-up-caller"}
        return ([], {"foo": "litellm_proxy"})

    async def fake_execute(**kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs["guardrail_context"]["metadata"]["guardrails"] == ("block-all",)
        assert kwargs["guardrail_context"]["model"] == "gpt-5"
        return [{"tool_call_id": "call-1", "name": "foo", "result": "done"}]

    monkeypatch.setattr(responses_main, "aresponses", fake_aresponses)
    monkeypatch.setattr(mcp_handler_module, "aresponses", fake_aresponses)
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler, "process_mcp_tools_without_openai_transform", staticmethod(fake_process)
    )
    monkeypatch.setattr(LiteLLM_Proxy_MCP_Handler, "execute_tool_calls", staticmethod(fake_execute))

    await responses_main.aresponses_api_with_mcp(
        input="hi",
        model="gpt-5",
        tools=[{"type": "mcp", "server_url": "litellm_proxy", "require_approval": "never"}],
        litellm_metadata={"guardrails": ["block-all"]},
        secret_fields={"raw_headers": {"x-app-id": "follow-up-caller"}},
        store=store,
        previous_response_id=caller_previous_response_id,
    )

    assert len(captured_calls) == 2
    follow_up_call = captured_calls[1]
    assert follow_up_call["previous_response_id"] == expected_previous_response_id

    reasoning_items = [
        item for item in follow_up_call["input"] if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assert bool(reasoning_items) is (store is False)


@pytest.mark.asyncio
async def test_responses_discovery_logs_sanitized_caller_headers(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy._experimental.mcp_server import mcp_server_manager, operations

    headers: Final = {
        "x-app-id": "app-a",
        "x-nuid": "user-a",
        "x-user-id": "identity-a",
        "x-mcp-deepwiki-authorization": "upstream-sentinel",
        "authorization": "proxy-sentinel",
    }
    manager: Final = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        get_allowed_mcp_servers=AsyncMock(return_value=[]),
        get_mcp_servers_from_ids=MagicMock(return_value=[]),
    )
    logger: Final = MagicMock(model_call_details={})
    logger.async_success_handler = AsyncMock()
    setup: Final = MagicMock(return_value=(logger, None))
    monkeypatch.setattr(mcp_server_manager, "global_mcp_server_manager", manager)
    monkeypatch.setattr(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[]))
    monkeypatch.setattr(operations, "function_setup", setup)
    response: Final = ResponsesAPIResponse(
        id="resp_test",
        created_at=1234567891,
        model="test-model",
        object="response",
        status="completed",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    monkeypatch.setattr(responses_main, "aresponses", AsyncMock(return_value=response))
    result: Final = await responses_main.aresponses_api_with_mcp(
        input="hi",
        model="test-model",
        tools=[{"type": "mcp", "server_url": "litellm_proxy"}],
        secret_fields={"raw_headers": headers},
    )
    assert result is response
    logger.async_success_handler.assert_awaited_once()
    logged: Final = setup.call_args.kwargs["metadata"]["headers"]
    assert logged == {"x-app-id": "app-a", "x-nuid": "user-a", "x-user-id": "identity-a"}
    assert headers["x-mcp-deepwiki-authorization"] == "upstream-sentinel"


@pytest.mark.asyncio
@pytest.mark.parametrize("real_listing", [False, True])
@pytest.mark.parametrize(
    ("allowed_tools", "expected_names"),
    [
        ([], ["responses_slot-echo", "responses_slot-status"]),
        (["echo"], ["responses_slot-echo"]),
        (["responses_slot-echo"], ["responses_slot-echo"]),
        (["absent"], []),
    ],
)
async def test_bridge_listing_leaves_the_callers_catalog_unchanged(
    monkeypatch: pytest.MonkeyPatch, allowed_tools: list[str], expected_names: list[str], real_listing: bool
) -> None:
    manager: Final = mcp_operations.global_mcp_server_manager
    server: Final = MCPServer(
        server_id="responses-slot", name="responses_slot", alias="responses_slot", transport=MCPTransport.http
    )
    user: Final = UserAPIKeyAuth(api_key="sk-responses-slot", user_id="responder")
    upstream: Final = [
        MCPTool(name="echo", description="Echo text back", inputSchema={"type": "object"}),
        MCPTool(name="status", description="Report status", inputSchema={"type": "object"}),
        MCPTool(name="echo", description="Duplicate echo", inputSchema={"type": "object", "properties": {}}),
    ]
    fake_manager: Final = types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        get_allowed_mcp_servers=AsyncMock(return_value=[]),
        get_mcp_servers_from_ids=MagicMock(return_value=[]),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )
    with (
        patch.dict(manager.tool_name_to_mcp_server_name_mapping),
        patch.object(mcp_operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server])),
        patch.object(manager, "_create_mcp_client", AsyncMock(return_value=object())),
        patch.object(manager, "_fetch_tools_with_timeout", AsyncMock(return_value=upstream)),
    ):
        try:
            if real_listing:
                await manager._get_tools_from_server(server, user_api_key_auth=user, record_listing=True)
            caller: Final = ListedToolsCaller(user_api_key_auth=user)
            before: Final = {
                tool.name: (listed.description, listed.input_schema)
                for tool in upstream
                if (listed := manager.get_listed_tool(server, tool.name, caller)) is not None
            }
            assert bool(before) is real_listing
            tools, _server_names = await LiteLLM_Proxy_MCP_Handler.process_mcp_tools_without_openai_transform(
                user_api_key_auth=user,
                mcp_tools_with_litellm_proxy=[
                    {
                        "type": "mcp",
                        "server_url": "litellm_proxy/mcp/responses-slot",
                        "allowed_tools": allowed_tools,
                    }
                ],
            )
            recorded: Final = {
                tool.name: (listed.description, listed.input_schema)
                for tool in upstream
                if (listed := manager.get_listed_tool(server, tool.name, caller)) is not None
            }
            assert recorded == before
            assert (
                manager.get_listed_tool(
                    server, "echo", ListedToolsCaller(user_api_key_auth=UserAPIKeyAuth(api_key="sk-other-caller"))
                )
                is None
            )
        finally:
            manager._drop_listed_tools(server.server_id)

    assert [tool.name for tool in tools] == expected_names


class _BridgeMetadataGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="bridge-metadata", event_hook=GuardrailEventHooks.pre_mcp_call, default_on=True)
        self.calls: tuple[tuple[object, object], ...] = ()

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Logging | None = None,
    ) -> GenericGuardrailAPIInputs:
        if request_data.get("mcp_arguments") == {"probe": "bridge"}:
            self.calls += ((request_data.get("mcp_tool_description"), request_data.get("mcp_input_schema")),)
        return inputs


@pytest.mark.asyncio
async def test_concurrent_bridge_calls_use_their_own_served_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    manager: Final = MCPServerManager()
    server: Final = MCPServer(server_id="bridge", name="bridge", transport=MCPTransport.http, url="http://upstream")
    manager.registry = {server.server_id: server}
    user: Final = UserAPIKeyAuth(api_key="sk-bridge", user_id="bridge-user")
    upstream: Final = [
        MCPTool(
            name="echo",
            description="Echo text",
            inputSchema={"type": "object", "properties": {"text": {"type": "string"}}},
        ),
        MCPTool(name="status", description="Read status", inputSchema={"type": "object"}),
    ]
    client: Final = AsyncMock()
    client.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="ok")])
    manager._create_mcp_client = AsyncMock(return_value=client)
    manager._fetch_tools_with_timeout = AsyncMock(return_value=upstream)
    guardrail: Final = _BridgeMetadataGuardrail()
    logger: Final = ProxyLogging(user_api_key_cache=DualCache())
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", logger)
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    monkeypatch.setattr(mcp_operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server]))
    monkeypatch.setattr("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager", manager)
    first_listed: Final = asyncio.Event()
    second_listed: Final = asyncio.Event()

    async def bridge(name: str, first: bool) -> None:
        if not first:
            await first_listed.wait()
        tools, server_map = await LiteLLM_Proxy_MCP_Handler.process_mcp_tools_without_openai_transform(
            user_api_key_auth=user,
            mcp_tools_with_litellm_proxy=[
                {"type": "mcp", "server_url": "litellm_proxy/mcp/bridge", "allowed_tools": [name]}
            ],
        )
        (first_listed if first else second_listed).set()
        await second_listed.wait()
        result: Final = await LiteLLM_Proxy_MCP_Handler.execute_tool_calls(
            tool_server_map=server_map,
            tool_calls=[
                {"type": "function_call", "name": f"bridge-{name}", "arguments": '{"probe":"bridge"}', "call_id": name}
            ],
            user_api_key_auth=user,
            served_tools=tools,
        )
        assert [entry["result"] for entry in result] == ["ok"]

    try:
        await asyncio.gather(bridge("echo", True), bridge("status", False))
        assert sorted(guardrail.calls, key=str) == sorted(
            ((tool.description, tool.input_schema) for tool in upstream), key=str
        )
        await manager.call_tool("bridge", "echo", {"probe": "bridge"}, user_api_key_auth=user, proxy_logging_obj=logger)
        assert guardrail.calls[-1] == (None, None)
        await manager._get_tools_from_server(
            server, user_api_key_auth=user, proxy_logging_obj=logger, record_listing=True
        )
        await manager.call_tool("bridge", "echo", {"probe": "bridge"}, user_api_key_auth=user, proxy_logging_obj=logger)
        assert guardrail.calls[-1] == (upstream[0].description, upstream[0].input_schema)
    finally:
        manager._drop_listed_tools(server.server_id)
        ProxyLogging._callback_capabilities_cache.clear()


def _toolset_gateway_manager(toolset_id: str, server_id: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        catalog=types.SimpleNamespace(operation=nullcontext),
        get_registry=MagicMock(return_value={}),
        get_allowed_mcp_servers=AsyncMock(return_value=[]),
        get_mcp_servers_from_ids=MagicMock(return_value=[]),
        get_mcp_server_by_name=MagicMock(return_value=None),
        get_toolset_by_name_cached=AsyncMock(return_value=types.SimpleNamespace(toolset_id=toolset_id)),
        resolve_toolset_tool_permissions=AsyncMock(return_value={server_id: ["add"]}),
    )


async def _tools_listing_kwargs_for_toolset_url(monkeypatch, team_toolset_id: str) -> dict[str, object]:
    from litellm.proxy._experimental.mcp_server.ui_session_utils import granted_toolset_ids
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LitellmUserRoles, UserAPIKeyAuth

    mock_get_tools = AsyncMock(return_value=AggregateToolListing(tools=[], outcomes={}))
    monkeypatch.setattr("litellm.proxy._experimental.mcp_server.server._get_tools_from_mcp_servers", mock_get_tools)
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        _toolset_gateway_manager("ts-granted", "srv-1"),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MagicMock())

    async def team_permission(context: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable:
        return LiteLLM_ObjectPermissionTable(object_permission_id="op-team", mcp_toolsets=[team_toolset_id])

    async def granted_through_team(context: UserAPIKeyAuth) -> frozenset[str]:
        return await granted_toolset_ids(context, team_object_permission=team_permission, require_key_access=False)

    team_key: Final = UserAPIKeyAuth(api_key="sk-team", team_id="team-1", user_role=LitellmUserRoles.INTERNAL_USER)
    await LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager(
        user_api_key_auth=team_key,
        mcp_tools_with_litellm_proxy=[{"type": "mcp", "server_url": "litellm_proxy/mcp/team-toolset"}],
        granted_toolsets=granted_through_team,
    )
    assert mock_get_tools.await_args is not None
    return mock_get_tools.await_args.kwargs


@pytest.mark.asyncio
async def test_toolset_gateway_url_scopes_a_team_granted_toolset_for_a_key_without_its_own_grant(monkeypatch):
    kwargs: Final = await _tools_listing_kwargs_for_toolset_url(monkeypatch, team_toolset_id="ts-granted")
    scoped = kwargs["user_api_key_auth"].object_permission
    assert scoped is not None
    assert scoped.mcp_servers == ["srv-1"]
    assert scoped.mcp_tool_permissions == {"srv-1": ["add"]}
    assert kwargs["mcp_servers"] is None


@pytest.mark.asyncio
async def test_toolset_gateway_url_skips_a_toolset_the_team_does_not_grant(monkeypatch):
    kwargs: Final = await _tools_listing_kwargs_for_toolset_url(monkeypatch, team_toolset_id="ts-other")
    assert kwargs["user_api_key_auth"].object_permission is None
    assert kwargs["mcp_servers"] is None


@pytest.mark.asyncio
async def test_apply_toolset_permissions_pins_the_auth_to_explicit_grants_only(monkeypatch: pytest.MonkeyPatch):
    """A toolset gateway URL must not widen to operator-open (allow_all_keys) servers."""
    from litellm.proxy._types import UserAPIKeyAuth

    fake_manager = types.SimpleNamespace(
        resolve_toolset_tool_permissions=AsyncMock(return_value={"srv-1": ["add"]}),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    scoped = await LiteLLM_Proxy_MCP_Handler._apply_toolset_permissions(
        resolved_toolset_ids=["ts-1"],
        resolved_mcp_servers=[],
        user_api_key_auth=UserAPIKeyAuth(api_key="sk-test", user_id="u1"),
    )

    assert scoped.mcp_explicit_grants_only is True
    assert scoped.object_permission is not None
    assert scoped.object_permission.mcp_servers == ["srv-1"]
    assert scoped.object_permission.mcp_tool_permissions == {"srv-1": ["add"]}

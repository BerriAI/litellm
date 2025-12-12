import sys
import types
from unittest.mock import AsyncMock

import pytest

from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.types.utils import ModelResponse
from litellm.types.responses.main import OutputFunctionToolCall


class _DummyMCPResult:
    def __init__(self):
        self.content = []


def _setup_mcp_call_environment(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch MCP globals so _execute_tool_calls can run in tests."""
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object())
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    fake_manager = types.SimpleNamespace(
        call_tool=AsyncMock(return_value=_DummyMCPResult())
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )
    return fake_manager.call_tool


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

    tool_calls = LiteLLM_Proxy_MCP_Handler._extract_tool_calls_from_chat_response(
        response
    )

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

    follow_up = LiteLLM_Proxy_MCP_Handler._create_follow_up_messages_for_chat(
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

    chat_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(
        ["tool"], target_format="chat"
    )
    resp_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(["tool"])

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

    follow_up = LiteLLM_Proxy_MCP_Handler._create_follow_up_input(
        response=response,
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

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

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

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

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

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "echo"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_args.kwargs["name"] == tool_name

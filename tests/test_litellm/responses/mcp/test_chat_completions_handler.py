import pytest
from unittest.mock import AsyncMock, patch

from litellm.types.utils import ModelResponse

from litellm.responses.mcp.chat_completions_handler import (
    acompletion_with_mcp,
)
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils


@pytest.mark.asyncio
async def test_acompletion_with_mcp_returns_normal_completion_without_tools(monkeypatch):
    mock_acompletion = AsyncMock(return_value="normal_response")

    with patch("litellm.acompletion", mock_acompletion):
        result = await acompletion_with_mcp(
            model="test-model",
            messages=[],
            tools=None,
        )

    assert result == "normal_response"
    mock_acompletion.assert_awaited_once()


@pytest.mark.asyncio
async def test_acompletion_with_mcp_without_auto_execution_calls_model(monkeypatch):
    tools = [{"type": "function", "function": {"name": "tool"}}]
    mock_acompletion = AsyncMock(return_value="ok")

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda tools: (tools, [])),
    )
    async def mock_process(**_):
        return ([], {})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: ["openai-tool"]),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
        staticmethod(lambda **_: False),
    )
    captured_secret_fields = {}

    def mock_extract(**kwargs):
        captured_secret_fields["value"] = kwargs.get("secret_fields")
        return (None, None, None, None)

    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(mock_extract),
    )

    with patch("litellm.acompletion", mock_acompletion):
        result = await acompletion_with_mcp(
            model="test-model",
            messages=[],
            tools=tools,
            secret_fields={"api_key": "value"},
        )

    assert result == "ok"
    mock_acompletion.assert_awaited_once()
    assert mock_acompletion.await_args is not None
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs.get("_skip_mcp_handler") is True
    assert kwargs.get("tools") == ["openai-tool"]
    assert captured_secret_fields["value"] == {"api_key": "value"}


@pytest.mark.asyncio
async def test_acompletion_with_mcp_auto_exec_performs_follow_up(monkeypatch):
    tools = [{"type": "function", "function": {"name": "tool"}}]
    initial_response = ModelResponse(
        id="1",
        model="test",
        choices=[],
        created=0,
        object="chat.completion",
    )
    follow_up_response = ModelResponse(
        id="2",
        model="test",
        choices=[],
        created=0,
        object="chat.completion",
    )
    mock_acompletion = AsyncMock(
        side_effect=[initial_response, follow_up_response]
    )

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda tools: (tools, [])),
    )
    async def mock_process(**_):
        return (tools, {"tool": "server"})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        mock_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *_, **__: tools),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
        staticmethod(lambda **_: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_extract_tool_calls_from_chat_response",
        staticmethod(lambda **_: ["call"]),
    )
    async def mock_execute(**_):
        return ["result"]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        mock_execute,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_create_follow_up_messages_for_chat",
        staticmethod(lambda **_: ["follow-up"]),
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_: (None, None, None, None)),
    )

    with patch("litellm.acompletion", mock_acompletion):
        result = await acompletion_with_mcp(
            model="test-model",
            messages=["msg"],
            tools=tools,
            stream=True,
        )

    assert result is follow_up_response
    assert mock_acompletion.await_count == 2
    first_call = mock_acompletion.await_args_list[0].kwargs
    second_call = mock_acompletion.await_args_list[1].kwargs
    assert first_call["stream"] is False
    assert second_call["messages"] == ["follow-up"]
    assert second_call["stream"] is True

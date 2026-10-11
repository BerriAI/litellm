"""The SDK entry points stay off the MCP gateway handler unless a tool can name the gateway.

The handler module sits on the proxy layer and needs the optional ``mcp`` package. A base SDK install has
neither, so ordinary tools must never make ``completion()`` or ``responses()`` import it.
"""

import sys
from typing import Final

import httpx
import pytest
import respx

import litellm

GATEWAY_HANDLER_MODULES: Final = (
    "litellm.responses.mcp.litellm_proxy_mcp_handler",
    "litellm.responses.mcp.chat_completions_handler",
    "litellm.llms.anthropic.pass_through.messages.mcp_handler",
)

FUNCTION_TOOL: Final = {
    "type": "function",
    "function": {"name": "noop", "description": "No operation", "parameters": {"type": "object", "properties": {}}},
}
RESPONSES_FUNCTION_TOOL: Final = {
    "type": "function",
    "name": "noop",
    "description": "No operation",
    "parameters": {"type": "object", "properties": {}},
}
ANTHROPIC_FUNCTION_TOOL: Final = {
    "name": "noop",
    "description": "No operation",
    "input_schema": {"type": "object", "properties": {}},
}
ANTHROPIC_MESSAGE_PAYLOAD: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
GATEWAY_MCP_TOOL: Final = {"type": "mcp", "server_label": "litellm", "server_url": "litellm_proxy"}

RESPONSES_PAYLOAD: Final = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1734366691,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "output_tokens_details": {"reasoning_tokens": 0},
    },
    "tools": [],
}


@pytest.fixture
def gateway_handler_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing any gateway handler module raises, as it does on an install without proxy extras."""
    for module_name in GATEWAY_HANDLER_MODULES:
        monkeypatch.setitem(sys.modules, module_name, None)


@pytest.mark.usefixtures("gateway_handler_unavailable")
def test_completion_with_function_tool_does_not_import_the_gateway_handler():
    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[FUNCTION_TOOL],
        mock_response="ok",
    )

    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
@pytest.mark.usefixtures("gateway_handler_unavailable")
async def test_acompletion_with_function_tool_does_not_import_the_gateway_handler():
    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[FUNCTION_TOOL],
        mock_response="ok",
    )

    assert response.choices[0].message.content == "ok"


@pytest.mark.usefixtures("gateway_handler_unavailable")
def test_completion_with_gateway_mcp_tool_still_reaches_the_handler():
    with pytest.raises(ModuleNotFoundError, match=r"litellm_proxy_mcp_handler|chat_completions_handler"):
        litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=[GATEWAY_MCP_TOOL],
            mock_response="ok",
        )


@pytest.mark.parametrize("tools", [None, [RESPONSES_FUNCTION_TOOL]], ids=["no-tools", "function-tool"])
@pytest.mark.usefixtures("gateway_handler_unavailable")
def test_responses_without_a_gateway_tool_does_not_import_the_gateway_handler(tools, respx_mock: respx.MockRouter):
    upstream: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=RESPONSES_PAYLOAD)
    )

    response = litellm.responses(model="openai/gpt-4o-mini", input="hi", tools=tools, api_key="sk-test")

    assert upstream.called
    assert response.output[0].content[0].text == "Done."


@pytest.mark.asyncio
@pytest.mark.usefixtures("gateway_handler_unavailable")
async def test_anthropic_messages_with_function_tool_does_not_import_the_gateway_handler(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    upstream: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_MESSAGE_PAYLOAD)
    )

    response = await litellm.anthropic_interface.messages.acreate(
        model="anthropic/claude-sonnet-5-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16,
        tools=[ANTHROPIC_FUNCTION_TOOL],
        api_key="sk-test",
    )

    assert upstream.called
    assert response["content"][0]["text"] == "ok"


@pytest.mark.parametrize(
    "tools, expected",
    [
        (None, False),
        ([], False),
        ([FUNCTION_TOOL], False),
        ([RESPONSES_FUNCTION_TOOL], False),
        ([{"type": "web_search"}], False),
        (["not-a-mapping"], False),
        ([FUNCTION_TOOL, GATEWAY_MCP_TOOL], True),
        ([{"type": "mcp", "server_label": "x", "server_url": "https://gw.example.com/mcp/github"}], True),
        ([{"type": "mcp", "server_label": "x", "server_url": "https://elsewhere.example.com/sse"}], True),
    ],
    ids=[
        "none",
        "empty",
        "chat-function",
        "responses-function",
        "web-search",
        "non-mapping",
        "gateway-among-functions",
        "gateway-path-url",
        "remote-mcp",
    ],
)
def test_may_name_litellm_mcp_gateway_keeps_only_mcp_tools(tools, expected):
    from litellm.responses.mcp.tool_precheck import may_name_litellm_mcp_gateway

    assert may_name_litellm_mcp_gateway(tools) is expected


def test_precheck_leaves_a_one_shot_iterable_unread():
    tools: Final = (tool for tool in [FUNCTION_TOOL, GATEWAY_MCP_TOOL, GATEWAY_MCP_TOOL])
    from litellm.responses.mcp.tool_precheck import may_name_litellm_mcp_gateway

    assert may_name_litellm_mcp_gateway(tools) is True
    assert list(tools) == [FUNCTION_TOOL, GATEWAY_MCP_TOOL, GATEWAY_MCP_TOOL]


@pytest.mark.parametrize(
    "tool",
    [
        FUNCTION_TOOL,
        RESPONSES_FUNCTION_TOOL,
        {"type": "web_search"},
        GATEWAY_MCP_TOOL,
        {"type": "mcp", "server_label": "x", "server_url": "litellm_proxy/mcp/github"},
        {"type": "mcp", "server_label": "x", "server_url": "https://gw.example.com/mcp/github"},
        {"type": "mcp", "server_label": "x", "server_url": "https://elsewhere.example.com/sse"},
        {"type": "mcp", "server_label": "x"},
        {"type": "mcp", "server_label": "x", "server_url": None},
        {"server_url": "litellm_proxy"},
    ],
)
def test_precheck_never_rejects_a_tool_the_gateway_handler_accepts(tool):
    pytest.importorskip("mcp")
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    from litellm.responses.mcp.tool_precheck import may_name_litellm_mcp_gateway

    if LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(tools=[tool]):
        assert may_name_litellm_mcp_gateway([tool])

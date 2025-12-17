import pytest

import litellm
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_acompletion_mcp_auto_exec(monkeypatch):
    from types import SimpleNamespace

    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )
    from litellm.responses.utils import ResponsesAPIRequestUtils

    dummy_tool = SimpleNamespace(
        name="local_search",
        description="search",
        inputSchema={"type": "object", "properties": {}},
    )

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy):
        return [dummy_tool], {"local_search": "local"}

    async def fake_execute(**kwargs):
        fake_execute.called = True  # type: ignore[attr-defined]
        tool_calls = kwargs.get("tool_calls") or []
        assert tool_calls, "tool calls should be present during auto execution"
        call_entry = tool_calls[0]
        call_id = call_entry.get("id") or call_entry.get("call_id") or "call"
        return [
            {
                "tool_call_id": call_id,
                "result": "executed",
                "name": call_entry.get("name", "local_search"),
            }
        ]

    fake_execute.called = False  # type: ignore[attr-defined]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        fake_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda secret_fields, tools: (None, None, None, None)),
    )

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "mcp",
                "server_url": "litellm_proxy/mcp/local",
                "server_label": "local",
                "require_approval": "never",
            }
        ],
        mock_response="Final answer",
        mock_tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "local_search", "arguments": "{}"},
            }
        ],
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Final answer"
    assert fake_execute.called is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_acompletion_mcp_respects_manual_approval(monkeypatch):
    from types import SimpleNamespace

    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )
    from litellm.responses.utils import ResponsesAPIRequestUtils

    dummy_tool = SimpleNamespace(
        name="local_search",
        description="search",
        inputSchema={"type": "object", "properties": {}},
    )

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy):
        return [dummy_tool], {"local_search": "local"}

    async def fake_execute(**kwargs):
        pytest.fail("auto execution should not run when approval is required")

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        fake_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda secret_fields, tools: (None, None, None, None)),
    )

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "mcp",
                "server_url": "litellm_proxy/mcp/local",
                "server_label": "local",
                "require_approval": "manual",
            }
        ],
        mock_response="Pending tool",
        mock_tool_calls=[
            {
                "id": "call-2",
                "type": "function",
                "function": {"name": "local_search", "arguments": "{}"},
            }
        ],
    )

    assert isinstance(response, ModelResponse)
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1

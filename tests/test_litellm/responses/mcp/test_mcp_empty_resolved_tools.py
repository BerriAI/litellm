"""
Guard tests: a request that explicitly asks for MCP tools via the litellm_proxy
gateway but resolves zero of them (and has no other tools to fall back on) must
fail loudly with a 400 instead of silently calling the model with no tools —
which makes it hallucinate, with the only trace being a "success"
list_mcp_tools spend log with an empty response.
"""

import sys
from unittest.mock import AsyncMock

import pytest

import litellm
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.types.llms.openai import ResponsesAPIResponse

# See test_mcp_streaming_iterator.py: look the real submodule up in sys.modules
# to sidestep litellm.responses being shadowed by the re-exported function.
responses_main_module = sys.modules["litellm.responses.main"]

MCP_TOOL = {
    "type": "mcp",
    "server_url": "litellm_proxy/mcp/nonexistent_server",
    "require_approval": "never",
    "allowed_tools": ["get_links"],
}


def _patch_resolved_tools(monkeypatch: pytest.MonkeyPatch, resolved_tools: list) -> None:
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        AsyncMock(return_value=(resolved_tools, {})),
    )


def _model_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(id="resp-1", created_at=0, output=[])


@pytest.mark.asyncio
async def test_zero_resolved_mcp_tools_raises_before_model_call(monkeypatch):
    # The guard runs before the stream/non-stream branch in
    # aresponses_api_with_mcp, so one case covers both.
    _patch_resolved_tools(monkeypatch, [])
    aresponses_mock = AsyncMock()
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    with pytest.raises(litellm.BadRequestError) as excinfo:
        await responses_main_module.aresponses_api_with_mcp(
            input="how many links do i have?",
            model="gpt-4",
            stream=False,
            tools=[MCP_TOOL],
        )

    message = str(excinfo.value)
    assert "resolved 0 tools" in message
    assert "litellm_proxy/mcp/nonexistent_server" in message
    assert "allow_all_keys" in message
    # The model was never called without its tools.
    aresponses_mock.assert_not_called()


@pytest.mark.asyncio
async def test_zero_resolved_mcp_tools_with_function_tools_falls_back(monkeypatch):
    """Mixed requests keep working: with other (function) tools present, the
    request proceeds using those tools instead of hard-failing."""
    _patch_resolved_tools(monkeypatch, [])
    response = _model_response()
    aresponses_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    function_tool = {"type": "function", "name": "my_fn", "parameters": {}}
    result = await responses_main_module.aresponses_api_with_mcp(
        input="hello",
        model="gpt-4",
        stream=False,
        tools=[MCP_TOOL, function_tool],
    )

    assert result is response
    aresponses_mock.assert_called_once()
    assert aresponses_mock.call_args.kwargs["tools"] == [function_tool]


@pytest.mark.asyncio
async def test_zero_resolved_mcp_tools_flag_off_restores_old_behaviour(monkeypatch):
    _patch_resolved_tools(monkeypatch, [])
    response = _model_response()
    aresponses_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)
    monkeypatch.setattr(litellm, "reject_empty_mcp_resolved_tools", False)

    result = await responses_main_module.aresponses_api_with_mcp(
        input="how many links do i have?",
        model="gpt-4",
        stream=False,
        tools=[MCP_TOOL],
    )

    assert result is response
    aresponses_mock.assert_called_once()


@pytest.mark.asyncio
async def test_resolved_mcp_tools_proceed_to_model_call(monkeypatch):
    from mcp.types import Tool as MCPTool

    resolved = [MCPTool(name="get_links", description="List links", inputSchema={"type": "object"})]
    _patch_resolved_tools(monkeypatch, resolved)

    response = _model_response()
    aresponses_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    result = await responses_main_module.aresponses_api_with_mcp(
        input="how many links do i have?",
        model="gpt-4",
        stream=False,
        tools=[MCP_TOOL],
    )

    assert result is response
    aresponses_mock.assert_called_once()
    assert aresponses_mock.call_args.kwargs["tools"], "model call must carry the resolved tools"


@pytest.mark.asyncio
async def test_request_without_mcp_tools_is_unaffected(monkeypatch):
    """Plain function-tool requests never hit the guard."""
    response = _model_response()
    aresponses_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    result = await responses_main_module.aresponses_api_with_mcp(
        input="hello",
        model="gpt-4",
        stream=False,
        tools=[{"type": "function", "name": "my_fn", "parameters": {}}],
    )

    assert result is response
    aresponses_mock.assert_called_once()

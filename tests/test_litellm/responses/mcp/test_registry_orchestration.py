"""
Registry-backed orchestration tests for /v1/chat/completions.

Validates the feature where:
  - server_url: "litellm_proxy/mcp"     → expands to ALL registered MCP servers
  - server_url: "litellm_proxy/agents"  → expands to ALL registered A2A agents
  - Both MCP tool calls and A2A agent calls share the same trace
  - semantic_filter: true  pre-filters MCP tools by query relevance
  - Streaming (stream=True) works identically to non-streaming
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.agents import AgentResponse
from litellm.types.utils import ModelResponse


def _mcp_tool_to_openai(tool):
    """Convert a SimpleNamespace MCP tool to OpenAI function tool format without importing mcp."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MATH_MCP_TOOL = SimpleNamespace(
    name="add",
    description="Add two numbers",
    inputSchema={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First operand"},
            "b": {"type": "integer", "description": "Second operand"},
        },
        "required": ["a", "b"],
    },
)

CURRENCY_AGENT = AgentResponse(
    agent_id="agent-fx-001",
    agent_name="FX_Converter",
    agent_card_params={
        "url": "http://mock-agent.internal/a2a",
        "name": "FX_Converter",
        "description": "Converts amounts between currencies using live rates.",
        "skills": [
            {
                "id": "fx-convert",
                "name": "Currency Conversion",
                "description": "Convert a numeric amount from one currency to another",
                "tags": ["finance", "fx"],
            }
        ],
    },
)


def _make_fake_process(mcp_tools=None, tool_server_map=None):
    """Return a fake _process_mcp_tools_without_openai_transform."""
    _tools = mcp_tools or [MATH_MCP_TOOL]
    _map = tool_server_map or {MATH_MCP_TOOL.name: "math_server"}

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy, **kwargs):
        return _tools, _map

    return fake_process


def _no_mcp_headers(secret_fields, tools):
    return (None, None, None, None)


# ---------------------------------------------------------------------------
# Test 1 – Non-streaming: MCP + A2A tool calls in the same trace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_orchestration_nonstreaming(monkeypatch):
    """
    One LLM turn triggers both an MCP tool call (add) and an A2A agent call
    (FX_Converter). Both are executed in a single trace and the final answer
    is returned as a ModelResponse.
    """
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    # ── Setup registries ──────────────────────────────────────────────────
    original_agents = list(global_agent_registry.agent_list)
    global_agent_registry.agent_list = [CURRENCY_AGENT]

    executed: List[Dict[str, Any]] = []

    async def fake_execute(**kwargs):
        tool_calls: List[Any] = kwargs.get("tool_calls") or []
        agent_tool_map: Dict[str, Any] = kwargs.get("agent_tool_map") or {}

        results = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            call_id = tc.get("id") or "tc-unknown"

            if name == "add":
                executed.append({"type": "mcp", "tool": "add"})
                results.append({"tool_call_id": call_id, "result": "12", "name": "add"})
            elif name in agent_tool_map or name == "FX_Converter":
                executed.append({"type": "a2a", "tool": name})
                results.append(
                    {
                        "tool_call_id": call_id,
                        "result": "12 USD = 9.48 GBP",
                        "name": name,
                    }
                )
        return results

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        _make_fake_process(),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda tools, **kw: [_mcp_tool_to_openai(t) for t in tools]),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(_no_mcp_headers),
    )

    try:
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Add 5 and 7, then convert the result to GBP.",
                }
            ],
            tools=[
                # All registered MCP servers
                {
                    "type": "mcp",
                    "server_url": "litellm_proxy/mcp",
                    "require_approval": "never",
                },
                # All registered A2A agents
                {
                    "type": "a2a_agent",
                    "server_url": "litellm_proxy/agents",
                    "require_approval": "never",
                },
            ],
            # First LLM response: call both tools
            mock_tool_calls=[
                {
                    "id": "tc-mcp-1",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 5, "b": 7}',
                    },
                },
                {
                    "id": "tc-a2a-1",
                    "type": "function",
                    "function": {
                        "name": "FX_Converter",
                        "arguments": '{"message": "Convert 12 USD to GBP"}',
                    },
                },
            ],
            # Second LLM response after tool results are fed back
            mock_response="5 + 7 = 12. The FX Converter confirms: 12 USD = 9.48 GBP.",
        )
    finally:
        global_agent_registry.agent_list = original_agents

    # ── Assertions ────────────────────────────────────────────────────────
    assert isinstance(response, ModelResponse), "Expected a ModelResponse"
    assert "12 USD = 9.48 GBP" in response.choices[0].message.content

    mcp_calls = [e for e in executed if e["type"] == "mcp"]
    a2a_calls = [e for e in executed if e["type"] == "a2a"]
    assert mcp_calls, "MCP tool 'add' was never executed"
    assert a2a_calls, "A2A agent 'FX_Converter' was never executed"

    mcp_metadata = (
        response.choices[0].message.provider_specific_fields or {}
        if hasattr(response.choices[0].message, "provider_specific_fields")
        else {}
    )
    # Both MCP list and agent tools should appear in provider metadata
    assert (
        "mcp_list_tools" in mcp_metadata
    ), f"Expected mcp_list_tools in provider_specific_fields, got: {list(mcp_metadata.keys())}"


# ---------------------------------------------------------------------------
# Test 2 – litellm_proxy/mcp bare URL expands to ALL registered servers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_mcp_url_expands_to_all_servers(monkeypatch):
    """
    server_url: 'litellm_proxy/mcp' (no /server_name suffix) must call
    _process_mcp_tools_without_openai_transform with mcp_servers=None so that
    ALL registered MCP servers are queried, not just one.
    """
    captured: Dict[str, Any] = {}

    async def spy_process(user_api_key_auth, mcp_tools_with_litellm_proxy, **kwargs):
        # Record the tool config to check server_url below
        captured["tools"] = mcp_tools_with_litellm_proxy
        return [MATH_MCP_TOOL], {MATH_MCP_TOOL.name: "math_server"}

    async def fake_execute(**kwargs):
        return [{"tool_call_id": "tc-1", "result": "8", "name": "add"}]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        spy_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda tools, **kw: [_mcp_tool_to_openai(t) for t in tools]),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(_no_mcp_headers),
    )

    await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "add 3 and 5"}],
        tools=[
            {
                "type": "mcp",
                "server_url": "litellm_proxy/mcp",  # bare – no server suffix
                "require_approval": "never",
            }
        ],
        mock_tool_calls=[
            {
                "id": "tc-1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 3, "b": 5}'},
            }
        ],
        mock_response="3 + 5 = 8",
    )

    # The tool config passed into _process_mcp_tools must include the bare URL
    assert captured.get("tools"), "spy_process was never called"
    bare_url_tools = [
        t
        for t in captured["tools"]
        if isinstance(t, dict) and t.get("server_url") == "litellm_proxy/mcp"
    ]
    assert bare_url_tools, (
        "Expected a tool entry with server_url='litellm_proxy/mcp' "
        f"(all-servers sentinel). Got: {captured['tools']}"
    )


# ---------------------------------------------------------------------------
# Test 3 – Agent wrapping: agents exposed as OpenAI function tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agents_wrapped_as_function_tools(monkeypatch):
    """
    When agent_tool_configs are present, _wrap_agents_as_function_tools reads
    global_agent_registry and converts each agent to an OpenAI function tool
    with a sanitized name and enriched description.
    """
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    original_agents = list(global_agent_registry.agent_list)
    global_agent_registry.agent_list = [CURRENCY_AGENT]

    try:
        from litellm.proxy.agent_endpoints.registry_orchestrator import (
            RegistryOrchestrator,
        )

        function_tools, agent_tool_map = await RegistryOrchestrator.resolve_agent_tools(
            user_api_key_auth=None
        )
    finally:
        global_agent_registry.agent_list = original_agents

    assert len(function_tools) == 1
    ft = function_tools[0]
    assert ft["type"] == "function"
    fn = ft["function"]

    # Name must be a valid OpenAI function name (alphanumeric + _ + -)
    import re

    assert re.match(
        r"^[a-zA-Z0-9_-]{1,64}$", fn["name"]
    ), f"Function name '{fn['name']}' is not a valid OpenAI function name"

    # Description should be enriched with skill description
    assert (
        "Convert a numeric amount" in fn["description"]
    ), f"Skill description missing from function description: {fn['description']}"

    # Parameters schema must include 'message' field
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "message" in params["properties"]
    assert params["required"] == ["message"]

    # agent_tool_map maps the sanitized name to the agent URL
    assert fn["name"] in agent_tool_map
    assert agent_tool_map[fn["name"]]["url"] == "http://mock-agent.internal/a2a"


# ---------------------------------------------------------------------------
# Test 4 – A2A response parsing
# ---------------------------------------------------------------------------


def test_parse_a2a_response_artifacts():
    """Extracts text from A2A result.artifacts[].parts[]."""
    from litellm.proxy.agent_endpoints.registry_orchestrator import _parse_a2a_response

    data = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {
            "artifacts": [
                {
                    "parts": [
                        {"type": "text", "text": "12 USD = 9.48 GBP"},
                        {"type": "text", "text": "Rate: 0.79"},
                    ]
                }
            ]
        },
    }
    result = _parse_a2a_response(data)
    assert "12 USD = 9.48 GBP" in result
    assert "Rate: 0.79" in result


def test_parse_a2a_response_status_message():
    """Falls back to result.status.message.parts[] when no artifacts."""
    from litellm.proxy.agent_endpoints.registry_orchestrator import _parse_a2a_response

    data = {
        "jsonrpc": "2.0",
        "id": "req-2",
        "result": {
            "status": {
                "state": "completed",
                "message": {
                    "role": "agent",
                    "parts": [{"type": "text", "text": "Done: 9.48 GBP"}],
                },
            }
        },
    }
    assert _parse_a2a_response(data) == "Done: 9.48 GBP"


def test_parse_a2a_response_error():
    """Error responses surface the error message."""
    from litellm.proxy.agent_endpoints.registry_orchestrator import _parse_a2a_response

    data = {
        "jsonrpc": "2.0",
        "id": "req-3",
        "error": {"code": -32600, "message": "Invalid Request"},
    }
    result = _parse_a2a_response(data)
    assert "Invalid Request" in result


# ---------------------------------------------------------------------------
# Test 5 – Streaming mode: MCP + A2A in same trace, stream=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_orchestration_streaming(monkeypatch):
    """
    With stream=True the handler wraps streaming in MCPStreamingIterator.
    Collecting all chunks must yield a final text response that includes
    both the MCP tool result and the A2A agent result.
    """
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.utils import CustomStreamWrapper

    original_agents = list(global_agent_registry.agent_list)
    global_agent_registry.agent_list = [CURRENCY_AGENT]

    executed: List[Dict[str, Any]] = []

    async def fake_execute(**kwargs):
        tool_calls: List[Any] = kwargs.get("tool_calls") or []
        agent_tool_map: Dict[str, Any] = kwargs.get("agent_tool_map") or {}
        results = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            call_id = tc.get("id") or "tc-s"
            if name == "add":
                executed.append({"type": "mcp", "tool": "add"})
                results.append({"tool_call_id": call_id, "result": "12", "name": "add"})
            elif name in agent_tool_map or name == "FX_Converter":
                executed.append({"type": "a2a", "tool": name})
                results.append(
                    {
                        "tool_call_id": call_id,
                        "result": "12 USD = 9.48 GBP",
                        "name": name,
                    }
                )
        return results

    # Mock tool calls to be "found" after stream collection.
    # mock_tool_calls in streaming mode are not reliably embedded in chunk deltas,
    # so we inject them directly via _extract_tool_calls_from_chat_response.
    _stream_tool_calls = [
        {
            "id": "tc-s-mcp",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 5, "b": 7}'},
        },
        {
            "id": "tc-s-a2a",
            "type": "function",
            "function": {
                "name": "FX_Converter",
                "arguments": '{"message": "Convert 12 USD to GBP"}',
            },
        },
    ]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        _make_fake_process(),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda tools, **kw: [_mcp_tool_to_openai(t) for t in tools]),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_extract_tool_calls_from_chat_response",
        staticmethod(lambda response: _stream_tool_calls),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(_no_mcp_headers),
    )

    try:
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Add 5 and 7, then convert the result to GBP.",
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "server_url": "litellm_proxy/mcp",
                    "require_approval": "never",
                },
                {
                    "type": "a2a_agent",
                    "server_url": "litellm_proxy/agents",
                    "require_approval": "never",
                },
            ],
            stream=True,
            mock_tool_calls=[
                {
                    "id": "tc-s-mcp",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 5, "b": 7}',
                    },
                },
                {
                    "id": "tc-s-a2a",
                    "type": "function",
                    "function": {
                        "name": "FX_Converter",
                        "arguments": '{"message": "Convert 12 USD to GBP"}',
                    },
                },
            ],
            mock_response="5 + 7 = 12. The FX Converter confirms: 12 USD = 9.48 GBP.",
        )
    finally:
        global_agent_registry.agent_list = original_agents

    # Collect all chunks from the stream
    chunks = []
    final_text = ""
    if isinstance(response, CustomStreamWrapper):
        async for chunk in response:
            chunks.append(chunk)
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and getattr(delta, "content", None):
                final_text += delta.content
    elif isinstance(response, ModelResponse):
        # Non-streaming fallback (shouldn't happen but handle gracefully)
        final_text = response.choices[0].message.content or ""

    assert chunks, "No streaming chunks received"
    assert (
        "12 USD = 9.48 GBP" in final_text
    ), f"Expected final text to contain FX result. Got: {final_text!r}"

    # Both MCP and A2A calls must have fired during the stream loop
    assert any(e["type"] == "mcp" for e in executed), "MCP tool not executed in stream"
    assert any(e["type"] == "a2a" for e in executed), "A2A agent not executed in stream"


# ---------------------------------------------------------------------------
# Test 6 – semantic_filter flag: filter hook reduces tool count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_filter_reduces_tools(monkeypatch):
    """
    When semantic_filter: true is set on the MCP tool config, _apply_semantic_filter
    is invoked. This test verifies the hook integration: if the filter is applied,
    the tool list passed downstream is reduced.
    """
    from litellm.responses.mcp import chat_completions_handler

    # Two MCP tools available
    add_tool = MATH_MCP_TOOL
    multiply_tool = SimpleNamespace(
        name="multiply",
        description="Multiply two numbers",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )

    async def fake_process(user_api_key_auth, mcp_tools_with_litellm_proxy, **kwargs):
        return [add_tool, multiply_tool], {
            "add": "math_server",
            "multiply": "math_server",
        }

    async def fake_execute(**kwargs):
        return [{"tool_call_id": "tc-1", "result": "8", "name": "add"}]

    # Semantic filter: keep only the first tool (simulates "add" being most relevant)
    async def fake_semantic_filter(tools, messages):
        return tools[:1]  # keep only 'add', drop 'multiply'

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        fake_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda tools, **kw: [_mcp_tool_to_openai(t) for t in tools]),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(_no_mcp_headers),
    )
    # Patch RegistryOrchestrator.apply_semantic_filter (moved from module-level)
    from litellm.proxy.agent_endpoints.registry_orchestrator import RegistryOrchestrator

    monkeypatch.setattr(
        RegistryOrchestrator,
        "apply_semantic_filter",
        staticmethod(fake_semantic_filter),
    )

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "add 3 and 5"}],
        tools=[
            {
                "type": "mcp",
                "server_url": "litellm_proxy/mcp",
                "require_approval": "never",
                "semantic_filter": True,  # ← trigger filter
            }
        ],
        mock_tool_calls=[
            {
                "id": "tc-1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 3, "b": 5}'},
            }
        ],
        mock_response="3 + 5 = 8",
    )

    assert isinstance(response, ModelResponse)
    assert "8" in response.choices[0].message.content

    # Verify semantic filter was applied: only 'add' tool should appear in metadata
    mcp_metadata = (
        response.choices[0].message.provider_specific_fields or {}
        if hasattr(response.choices[0].message, "provider_specific_fields")
        else {}
    )
    listed = mcp_metadata.get("mcp_list_tools", [])
    tool_names = [t.get("function", {}).get("name") for t in listed]
    assert "add" in tool_names, f"Expected 'add' in mcp_list_tools, got: {tool_names}"
    assert (
        "multiply" not in tool_names
    ), f"Expected 'multiply' to be filtered out, got: {tool_names}"

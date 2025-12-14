from types import SimpleNamespace

import pytest

from litellm.responses.mcp import litellm_proxy_mcp_handler as handler_module


@pytest.mark.asyncio
async def test_apply_semantic_filter_limits_tools(monkeypatch):
    captured = {}

    class DummyRegistry:
        async def query(self, *, prompt, allowed_servers, top_k=None):
            captured["prompt"] = prompt
            captured["allowed_servers"] = list(allowed_servers)
            captured["top_k"] = top_k
            return ["serverA-tool1"]

    monkeypatch.setattr(
        handler_module,
        "semantic_mcp_filter_registry",
        DummyRegistry(),
    )

    tools = [
        SimpleNamespace(name="serverA-tool1"),
        SimpleNamespace(name="serverB-tool2"),
    ]

    context = {"query": "latest incidents", "server_labels": ["serverA"], "top_k": 1}
    result = await handler_module.LiteLLM_Proxy_MCP_Handler._apply_semantic_filter(
        tools=tools,
        semantic_filter_context=context,
        allowed_servers=["serverA", "serverB"],
    )

    assert [tool.name for tool in result] == ["serverA-tool1"]
    assert captured == {
        "prompt": "latest incidents",
        "allowed_servers": ["serverA"],
        "top_k": 1,
    }


@pytest.mark.asyncio
async def test_apply_semantic_filter_falls_back_when_no_hits(monkeypatch):
    class DummyRegistry:
        async def query(self, *, prompt, allowed_servers, top_k=None):
            return []

    monkeypatch.setattr(
        handler_module,
        "semantic_mcp_filter_registry",
        DummyRegistry(),
    )

    tools = [
        SimpleNamespace(name="serverA-tool1"),
        SimpleNamespace(name="serverB-tool2"),
    ]

    context = {"query": "something"}
    result = await handler_module.LiteLLM_Proxy_MCP_Handler._apply_semantic_filter(
        tools=tools,
        semantic_filter_context=context,
        allowed_servers=["serverA", "serverB"],
    )

    assert result == tools

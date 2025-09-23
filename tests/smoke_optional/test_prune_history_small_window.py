import pytest

@pytest.mark.smoke
def test_prune_history_small_window_preserves_last_pair(monkeypatch):
    import sys, types
    sys.modules.setdefault("fastuuid", types.SimpleNamespace(uuid4=lambda: "0"*32))
    sys.modules.setdefault("mcp", types.SimpleNamespace(ClientSession=object))
    sys.modules.setdefault("mcp.types", types.SimpleNamespace(
        CallToolRequestParams=type("CallToolRequestParams", (), {}),
        CallToolResult=type("CallToolResult", (), {}),
        Tool=type("Tool", (), {}),
    ))
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mod

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc_last", "type": "function", "function": {"name": "echo", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc_last", "content": "tool-resp"},
    ]
    pruned = mod._prune_history_preserve_pair(messages, max_non_system=3, hard_char_budget=1000)
    # Expect the last assistant tool_call + matching tool to be present even with small cap
    assert any(m.get("role") == "assistant" and mod._get_tool_calls(m) for m in pruned)
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "tc_last" for m in pruned)

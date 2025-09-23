import pytest


@pytest.mark.smoke
def test_prune_history_preserve_pair():
    # Stub required modules to satisfy litellm imports
    import sys, types
    sys.modules.setdefault("fastuuid", types.SimpleNamespace(uuid4=lambda: "0"*32))
    sys.modules.setdefault("mcp", types.SimpleNamespace(ClientSession=object))
    sys.modules.setdefault("mcp.types", types.SimpleNamespace(
        CallToolRequestParams=type("CallToolRequestParams", (), {}),
        CallToolResult=type("CallToolResult", (), {}),
        Tool=type("Tool", (), {}),
    ))
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mod

    # Build a convo with system, older turns, and a final assistant tool_call + tool reply
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old a"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "hi"},
    ]

    pruned = mod._prune_history_preserve_pair(messages, max_non_system=2, hard_char_budget=1000)
    # Ensure the last assistant tool_call and its tool reply are present
    assert any(m.get("role") == "assistant" and mod._get_tool_calls(m) for m in pruned)
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "tc1" for m in pruned)

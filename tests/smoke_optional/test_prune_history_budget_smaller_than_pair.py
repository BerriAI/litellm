import pytest, sys, types

@pytest.mark.smoke
def test_prune_history_budget_smaller_than_pair(monkeypatch):
    sys.modules.setdefault("fastuuid", types.SimpleNamespace(uuid4=lambda:"0"*32))
    sys.modules.setdefault("mcp", types.SimpleNamespace(ClientSession=object))
    sys.modules.setdefault("mcp.types", types.SimpleNamespace(CallToolRequestParams=object,CallToolResult=object,Tool=object))
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mod
    messages=[
        {"role":"system","content":"s"},
        {"role":"user","content":"u"*100},
        {"role":"assistant","content":None,"tool_calls":[{"id":"tc","type":"function","function":{"name":"echo","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"tc","content":"X"*200},
    ]
    pruned = mod._prune_history_preserve_pair(messages, max_non_system=1, hard_char_budget=10)
    assert any(m.get("role")=="assistant" and mod._get_tool_calls(m) for m in pruned)
    assert any(m.get("role")=="tool" and m.get("tool_call_id")=="tc" for m in pruned)

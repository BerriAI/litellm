---
sidebar_position: 4
---

# Use an MCP tool in chat

```python
tools = xct.mcp.list_tools()   # OpenAI-shape function definitions

reply = xct.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Find the latest issues on llm-routing."}],
    tools=tools,
    # Optional: narrow to specific MCPs by alias or access group
    extra_headers={"x-mcp-servers": "github,search"},
)
```

The proxy:
1. forwards tools to the model with `tools=...`
2. when the model emits `tool_calls`, invokes the MCP server
3. threads tool results back into the conversation transparently
4. records `entity_type="mcp"`, `entity_id="<namespaced_tool_name>"` in
   the spend log

URL-namespaced variant for a fixed set:
```python
# Only the "github" MCP is available on this call
reply = httpx.post(
    f"{base_url}/github/mcp/v1/chat/completions",
    json={...},
    headers={"Authorization": f"Bearer {token}"},
)
```

For comma-separated access groups:
```
POST /research-tools,coding-tools/mcp/v1/chat/completions
```

# Anthropic Memory Tool Schema (via LiteLLM Proxy)

LiteLLM automatically adds the required `anthropic-beta: context-management-2025-06-27` header when you pass Anthropic's memory tool through the proxy and the caller has not already set a beta header.

This example shows the tool schema to include in a `/chat/completions` or `/v1/messages` request routed to Claude via LiteLLM.

## Tool definition

```json
{
  "tools": [
    {
      "type": "memory_20250818",
      "name": "memory"
    }
  ]
}
```

## Python example (OpenAI-compatible client)

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "user", "content": "Remember that my project codename is Aurora."}
    ],
    tools=[{"type": "memory_20250818", "name": "memory"}],
)

print(response.choices[0].message)
```

## Proxy config snippet

Ensure the model is routed to Anthropic in your `config.yaml`:

```yaml
model_list:
  - model_name: claude-sonnet-4
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
```

## What LiteLLM does automatically

When the memory tool is present in `tools`, LiteLLM's Anthropic transformation layer:

1. Detects `type: memory_20250818`.
2. Injects `anthropic-beta: context-management-2025-06-27` if the caller did not supply a beta header.
3. Forwards the request to Anthropic with the correct protocol version.

You do **not** need to set the beta header manually unless you want to override LiteLLM's default.

## MCP integration note

If you expose Claude through the LiteLLM MCP gateway (`/mcp-rest/tools/call` or the Responses API MCP bridge), pass tool definitions in the upstream LLM request—not as MCP server tools. The memory tool is an **LLM provider tool**, not an MCP server tool.

## Related

- Test reference: `tests/test_litellm/llms/anthropic/chat/test_anthropic_chat_transformation.py::test_anthropic_memory_tool_auto_adds_beta_header`
- Anthropic memory docs: https://docs.anthropic.com/en/docs/build-with-claude/context-management
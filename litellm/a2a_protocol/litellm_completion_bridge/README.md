# A2A to LiteLLM Completion Bridge

Routes A2A protocol requests through `litellm.acompletion`, enabling any LiteLLM-supported provider to be invoked via A2A.

## Flow

```
A2A Request → Transform → litellm.acompletion → Transform → A2A Response
```

## SDK Usage

Use the existing `asend_message` and `asend_message_streaming` functions with `litellm_params`:

```python
from litellm.a2a_protocol import asend_message, asend_message_streaming
from a2a.types import SendMessageRequest, SendStreamingMessageRequest, MessageSendParams
from uuid import uuid4

# Non-streaming
request = SendMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={"role": "user", "parts": [{"kind": "text", "text": "Hello!"}], "messageId": uuid4().hex}
    )
)
response = await asend_message(
    request=request,
    api_base="http://localhost:2024",
    litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
)

# Streaming
stream_request = SendStreamingMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={"role": "user", "parts": [{"kind": "text", "text": "Hello!"}], "messageId": uuid4().hex}
    )
)
async for chunk in asend_message_streaming(
    request=stream_request,
    api_base="http://localhost:2024",
    litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
):
    print(chunk)
```

## Proxy Usage

Configure an agent with `custom_llm_provider` in `litellm_params`:

```yaml
agents:
  - agent_name: my-langgraph-agent
    agent_card_params:
      name: "LangGraph Agent"
      url: "http://localhost:2024"  # Used as api_base
    litellm_params:
      custom_llm_provider: langgraph
      model: agent
```

When an A2A request hits `/a2a/{agent_id}/message/send`, the bridge:

1. Detects `custom_llm_provider` in agent's `litellm_params`
2. Transforms A2A message → OpenAI messages
3. Calls `litellm.acompletion(model="langgraph/agent", api_base="http://localhost:2024")`
4. Transforms response → A2A format

## Classes

- `A2ACompletionBridgeTransformation` - Static methods for message format conversion
- `A2ACompletionBridgeHandler` - Static methods for handling requests (streaming/non-streaming)

## Agentic loop: MCP tools and agent-to-agent delegation

A plain bridge agent maps one A2A `message/send` to one `litellm.acompletion`, so
it can answer from the model alone but cannot use tools or other agents.
`agentic_loop.run_agentic_loop` turns that single call into a loop: the model is
offered the agent's tools, any tool or agent calls it makes are executed, their
results are fed back, and the loop repeats until the model stops asking for tools
and returns prose.

Two kinds of tool are offered to the model:

- MCP tools, when the agent has `enable_mcp_tools` set. The agent only ever sees
  the servers its key is authorised for; entitlement is enforced downstream by
  `user_api_key_auth`, not by the loop.
- One function tool per other agent the caller may reach, when the agent has
  `enable_agent_calls` set. The tool name carries the target agent id, and a call
  to it is dispatched as a real A2A `message/send` to that agent via
  `asend_message`, so a delegated agent behaves exactly as if invoked directly.
  Delegation is limited to one level so a sub-agent cannot recurse into further
  hops.

### The empty-answer bug this fixes

`LiteLLM_Proxy_MCP_Handler._execute_tool_calls` returns each result as
`{tool_call_id, result, name}`, which is not a valid chat message. Feeding those
dicts straight back into the conversation left the model unable to see the
result, so it re-issued the same tool call on every iteration until the loop hit
its cap and returned an empty final answer. `_mcp_results_to_tool_messages`
reshapes each result into the `{role: "tool", tool_call_id, content}` form the
chat API expects, so the model sees the result, stops re-calling the tool, and
answers. The regression is covered in
`tests/test_litellm/a2a_protocol/test_agentic_loop.py`.

### Configuring an agent

Set these on the agent's `litellm_params` (via `POST /v1/agents`, the agent edit
API, or the dashboard agent create / Settings form):

- `prompt_id`: a stored prompt whose persona the agent answers in. A prompt that
  carries a model overrides the agent's model; a prompt with no model leaves the
  agent's own model in place. Clear the model in the prompt editor (or omit the
  `model:` line in the dotprompt) to have the agent use its own model.
- `enable_mcp_tools`: allow the agent to call its entitled MCP tools.
- `enable_agent_calls`: allow the agent to delegate to other agents it can reach.

The agent's `model` may be provider-prefixed (`gemini/gemini-flash-latest`) or a
bare proxy model-list alias (`gemini-flash`); the endpoint resolves the alias
against the model list and derives the provider so the bridge can route without
an explicit URL.

### Invoking an agent

```bash
curl -sS -X POST "http://localhost:4000/a2a/$AGENT_ID" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
          "message": {
            "role": "user",
            "parts": [{ "kind": "text", "text": "your question here" }],
            "messageId": "m-1"
          }
        }
      }'
```

The answer comes back in `result.parts[].text`.


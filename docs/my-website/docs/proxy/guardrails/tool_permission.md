import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Tool Permission Guardrail

LiteLLM provides a Tool Permission Guardrail that lets you control which **tool calls** a model is allowed to invoke, using configurable allow/deny rules. This offers fine-grained, provider-agnostic control over tool execution (e.g., OpenAI Chat Completions `tool_calls`, Anthropic Messages `tool_use`, MCP tools).

## Quick Start
### 1. Define Guardrails on your LiteLLM config.yaml 

Define your guardrails under the `guardrails` section
```yaml
guardrails:
  - guardrail_name: "tool-permission-guardrail"
    litellm_params:
      guardrail: tool_permission
      mode: "post_call"
      rules:
        - id: "allow_bash"
          tool_name: "Bash"
          decision: "allow"
        - id: "allow_github_mcp"
          tool_name: "mcp__github_*"
          decision: "allow"
        - id: "allow_aws_documentation"
          tool_name: "mcp__aws-documentation_*_documentation"
          decision: "allow"
        - id: "deny_read_commands"
          tool_name: "Read"
          decision: "Deny"
      default_action: "deny"  # Fallback when no rule matches: "allow" or "deny"
      on_disallowed_action: "block"  # How to handle disallowed tools: "block" or "rewrite"
```

#### Rule Structure

```yaml
- id: "unique_rule_id"           # Unique identifier for the rule
  tool_name: "pattern"           # Tool name or pattern to match
  decision: "allow"              # "allow" or "deny"
```

#### Supported values for `mode`

- `pre_call` Run **before** LLM call, on **input**
- `post_call` Run **after** LLM call, on **input & output**

### `on_disallowed_action` behavior

| Value | What happens |
| --- | --- |
| `block` | The request is immediately rejected. Pre-call checks raise a `400` HTTP error. Post-call checks raise `GuardrailRaisedException`, so the proxy responds with an error instead of the model output. Use when invoking the forbidden tool must halt the workflow. |
| `rewrite` | LiteLLM silently strips disallowed tools from the payload before it reaches the model (pre-call) or rewrites the model response/tool calls after the fact. The guardrail inserts error text into `message.content`/`tool_result` entries so the client learns the tool was blocked while the rest of the completion continues. Use when you want graceful degradation instead of hard failures. |

### Custom denial message

Set `violation_message_template` when you want the guardrail to return a branded error (e.g., “this violates our org policy…”). LiteLLM replaces placeholders from the denied tool:

- `{tool_name}` – the tool/function name (e.g., `Read`)
- `{rule_id}` – the matching rule ID (or `None` when the default action kicks in)
- `{default_message}` – the original LiteLLM message if you need to append it

Example:

```yaml
guardrails:
  - guardrail_name: "tool-permission-guardrail"
    litellm_params:
      guardrail: tool_permission
      mode: "post_call"
      violation_message_template: "this violates our org policy, we don't support executing {tool_name} commands"
      rules:
        - id: "allow_bash"
          tool_name: "Bash"
          decision: "allow"
        - id: "deny_read"
          tool_name: "Read"
          decision: "deny"
      default_action: "deny"
      on_disallowed_action: "block"
```

If a request tries to invoke `Read`, the proxy now returns “this violates our org policy, we don't support executing Read commands” instead of the stock error text. Omit the field to keep the default messaging.

### 2. Start the Proxy

```shell
litellm --config config.yaml --port 4000
```

## Examples

<Tabs>
<TabItem value="block" label="Block Request">

**Block request (`on_disallowed_action: block`)**

```bash
# Test
curl -X POST "http://localhost:4000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-master-key-here" \
  -d '{
    "model": "gpt-5-mini",
    "messages": [{"role": "user","content": "What is the weather like in Tokyo today?"}],
    "tools": [
      {
        "type":"function",
        "function": {
          "name":"get_current_weather",
          "description": "Get the current weather in a given location"
        }
      }
    ]
  }'
```

**Expected response (Denied):**

```json
{
  "error":
    {
      "message": "Guardrail raised an exception, Guardrail: tool-permission-guardrail, Message: Tool 'get_current_weather' denied by default action",
      "type": "None",
      "param": "None",
      "code": "500"
    }
}
```

</TabItem>
<TabItem value="rewrite" label="Rewrite Request">

**Rewrite request (`on_disallowed_action: rewrite`)**

```bash
# Test
curl -X POST "http://localhost:4000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-master-key-here" \
  -d '{
    "model": "gpt-5-mini",
    "messages": [{"role": "user","content": "What is the weather like in Tokyo today?"}],
    "tools": [
      {
        "type":"function",
        "function": {
          "name":"get_current_weather",
          "description": "Get the current weather in a given location"
        }
      }
    ]
  }'
```

**Expected response (tool removed, completion continues):**

```json
{
	"id": "chatcmpl-xxxxxxxxxxxxxxx",
	"created": 1757716050,
	"model": "gpt-5-mini-2025-08-07",
	"object": "chat.completion",
	"choices": [
		{
			"finish_reason": "stop",
			"index": 0,
			"message": {
				"content": "I can’t fetch live weather — I don’t have real‑time internet access.",
				"role": "assistant",
				"annotations": []
			},
			"provider_specific_fields": {}
		}
	],
	"usage": {
		"prompt_tokens": 112,
		"total_tokens": 735,
		"completion_tokens_details": {
			"reasoning_tokens": 384,
		},
	},
	"service_tier": "default"
}
```

</TabItem>
</Tabs>

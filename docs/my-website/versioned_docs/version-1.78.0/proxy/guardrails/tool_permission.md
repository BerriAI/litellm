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

### 2. Start the Proxy

```shell
litellm --config config.yaml --port 4000
```

## Examples

<Tabs>
<TabItem value="block" label="Block Request">

**Block requset**

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

**Rewrite requset**

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

**Expected response:**

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

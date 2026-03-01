import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Context Management (Compaction)

Automatically manage conversation context across providers. Pass a single `context_management` parameter and LiteLLM transforms it to the correct provider-specific format.

## Provider Support Matrix

| Provider | API Path | Supported | Format |
|----------|----------|-----------|--------|
| **OpenAI** | Responses API | ✅ | Native (passthrough) |
| **Azure OpenAI** | Responses API | ❌ | Blocked by Azure |
| **Anthropic** | Chat / Messages | ✅ | Transformed to `edits` + beta headers |
| **Vertex AI** (Anthropic) | Chat / Messages | ✅ | Same as Anthropic |
| **Bedrock** (Invoke) | Chat | ✅ | Same as Anthropic |
| **Bedrock** (Messages API) | Messages | ✅ | Same as Anthropic |
| **Bedrock** (Converse) | Converse | ❌ | Not supported by AWS |
| **Volcengine** | Responses API | ✅ | Native (passthrough) |

## Unified Format (OpenAI-compatible)

LiteLLM accepts the OpenAI format for `context_management` across all supported providers:

```python
context_management=[{"type": "compaction", "compact_threshold": 200000}]
```

For Anthropic-based providers (Anthropic, Vertex AI, Bedrock), LiteLLM automatically transforms this to:

```python
# Transformed to Anthropic format
context_management={
    "edits": [
        {
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 200000},
        }
    ]
}
```

You can also pass the native Anthropic format directly — LiteLLM detects and passes it through unchanged.

## Usage by Provider

### OpenAI Responses API

Server-side compaction runs automatically when context crosses the threshold. See also [Responses API — Server-side compaction](/docs/response_api#server-side-compaction).

```python
import litellm

response = litellm.responses(
    model="openai/gpt-4o",
    input="Summarise the conversation so far",
    context_management=[{"type": "compaction", "compact_threshold": 200000}],
)
```

### Anthropic

Anthropic supports two types of context edits:
- **Compaction** (`compact_20260112`) — auto-summarise old messages
- **Clear tool uses** (`clear_tool_uses_20250919`) — remove old tool results

LiteLLM automatically attaches the required beta headers (`compact-2026-01-12` or `context-management-2025-06-27`).

See also [Anthropic — Context Management](/docs/providers/anthropic#context-management-beta).

<Tabs>
<TabItem value="unified" label="Unified (OpenAI) Format">

```python
from litellm import completion

response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello"}],
    context_management=[
        {"type": "compaction", "compact_threshold": 200000}
    ],
)
```

</TabItem>
<TabItem value="native" label="Native Anthropic Format">

```python
from litellm import completion

response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Summarise the latest tool results"}],
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
                "clear_at_least": {"type": "input_tokens", "value": 5000},
                "exclude_tools": ["web_search"],
            }
        ]
    },
)
```

</TabItem>
</Tabs>

### Vertex AI (Anthropic)

Works identically to direct Anthropic — LiteLLM applies the same transformation and beta headers.

```python
from litellm import completion

response = completion(
    model="vertex_ai/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello"}],
    context_management=[
        {"type": "compaction", "compact_threshold": 200000}
    ],
)
```

### Bedrock (Invoke & Messages API)

Both the Invoke and Messages API paths support context management. See also [Bedrock — Context Management](/docs/providers/bedrock#usage---context-management-beta).

<Tabs>
<TabItem value="invoke" label="Invoke">

```python
from litellm import completion

response = completion(
    model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
    messages=[{"role": "user", "content": "Hello"}],
    context_management=[
        {"type": "compaction", "compact_threshold": 200000}
    ],
)
```

</TabItem>
<TabItem value="messages" label="Messages API">

```python
from litellm import completion

response = completion(
    model="bedrock/messages/anthropic.claude-sonnet-4-20250514-v1:0",
    messages=[{"role": "user", "content": "Hello"}],
    context_management={
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 200000},
            }
        ]
    },
)
```

</TabItem>
</Tabs>

:::info
**Bedrock Converse** (`bedrock/converse/...`) does **not** support context management. The `compact-2026-01-12` beta is in Bedrock's unsupported list. If you need context management on Bedrock, use the Invoke or Messages API path instead.
:::

### Volcengine Responses API

Volcengine passes through `context_management` unchanged (same as OpenAI format).

```python
import litellm

response = litellm.responses(
    model="volcengine/doubao-seed-1.6",
    input="Summarise the conversation",
    context_management=[{"type": "compaction", "compact_threshold": 200000}],
)
```

### LiteLLM Proxy

<Tabs>
<TabItem value="yaml" label="Config">

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
  - model_name: claude-bedrock
    litellm_params:
      model: bedrock/anthropic.claude-sonnet-4-20250514-v1:0
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
```

</TabItem>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python
import openai

client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# Works with any supported provider behind the proxy
response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "context_management": [
            {"type": "compaction", "compact_threshold": 200000}
        ]
    },
)
```

</TabItem>
</Tabs>

## Format Reference

### OpenAI Format

```json
[{"type": "compaction", "compact_threshold": 200000}]
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Currently only `"compaction"` is supported |
| `compact_threshold` | `int` | Token threshold at which compaction triggers (minimum 1000) |

### Anthropic Format

```json
{
  "edits": [
    {
      "type": "compact_20260112",
      "trigger": {"type": "input_tokens", "value": 200000}
    }
  ]
}
```

| Edit Type | Beta Header | Description |
|-----------|-------------|-------------|
| `compact_20260112` | `compact-2026-01-12` | Auto-summarise old messages |
| `clear_tool_uses_20250919` | `context-management-2025-06-27` | Clear old tool results |
| `clear_thinking_20250919` | `context-management-2025-06-27` | Clear thinking blocks |

## Related

- [Responses API — Server-side compaction](/docs/response_api#server-side-compaction)
- [Responses API — Compact endpoint](/docs/response_api_compact)
- [Anthropic — Context Management](/docs/providers/anthropic#context-management-beta)
- [Bedrock — Context Management](/docs/providers/bedrock#usage---context-management-beta)

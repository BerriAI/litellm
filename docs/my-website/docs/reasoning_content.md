import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 'Thinking' / 'Reasoning Content'

Supported Providers:
- Deepseek (`deepseek/`)
- Anthropic API (`anthropic/`)
- Bedrock (Anthropic) (`bedrock/`)
- Vertex AI (Anthropic) (`vertexai/`)

```python
"message": {
    ...
    "reasoning_content": "The capital of France is Paris.",
    "thinking_blocks": [
        {
            "type": "thinking",
            "thinking": "The capital of France is Paris.",
            "signature_delta": "EqoBCkgIARABGAIiQL2UoU0b1OHYi+..."
        }
    ]
}
```

## Quick Start 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os 

os.environ["ANTHROPIC_API_KEY"] = ""

response = completion(
  model="anthropic/claude-3-7-sonnet-20250219",
  messages=[
    {"role": "user", "content": "What is the capital of France?"},
  ],
  thinking={"type": "enabled", "budget_tokens": 1024} # 👈 REQUIRED FOR ANTHROPIC models (on `anthropic/`, `bedrock/`, `vertexai/`)
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "anthropic/claude-3-7-sonnet-20250219",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "thinking": {"type": "enabled", "budget_tokens": 1024}
}'
```
</TabItem>
</Tabs>

**Expected Response**

```bash
{
    "id": "3b66124d79a708e10c603496b363574c",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": " won the FIFA World Cup in 2022.",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "created": 1723323084,
    "model": "deepseek/deepseek-chat",
    "object": "chat.completion",
    "system_fingerprint": "fp_7e0991cad4",
    "usage": {
        "completion_tokens": 12,
        "prompt_tokens": 16,
        "total_tokens": 28,
    },
    "service_tier": null
}
```

## Spec 

These fields can be accessed via `response.choices[0].message.reasoning_content` and `response.choices[0].message.thinking_blocks`.

- `reasoning_content` - str: The reasoning content from the model. Returned across all providers.
- `thinking_blocks` - Optional[List[Dict[str, str]]]: A list of thinking blocks from the model. Only returned for Anthropic models.
  - `type` - str: The type of thinking block.
  - `thinking` - str: The thinking from the model.
  - `signature_delta` - str: The signature delta from the model.


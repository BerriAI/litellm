import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# completion()

Call 100+ LLMs using OpenAI's chat completion format.

## Example

```python
from litellm import completion

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Signature

```python
def completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stop: Optional[Union[str, List[str]]] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    user: Optional[str] = None,
    response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    **kwargs
) -> ModelResponse
```

## Parameters

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model identifier (e.g., `"gpt-4o"`, `"anthropic/claude-3-opus"`) |
| `messages` | `List[Dict]` | List of message objects with `role` and `content` |

### Optional - OpenAI Compatible

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` | `None` | Sampling temperature (0-2) |
| `top_p` | `float` | `None` | Nucleus sampling parameter |
| `n` | `int` | `None` | Number of completions to generate |
| `stream` | `bool` | `None` | Stream response chunks |
| `stop` | `str \| List[str]` | `None` | Stop sequences |
| `max_tokens` | `int` | `None` | Maximum tokens to generate |
| `presence_penalty` | `float` | `None` | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | `float` | `None` | Frequency penalty (-2.0 to 2.0) |
| `logit_bias` | `Dict[str, int]` | `None` | Token bias dictionary |
| `user` | `str` | `None` | End-user identifier |
| `response_format` | `dict \| BaseModel` | `None` | JSON mode or structured output |
| `seed` | `int` | `None` | Random seed for determinism |
| `tools` | `List` | `None` | Function/tool definitions |
| `tool_choice` | `str \| dict` | `None` | Tool selection mode |
| `logprobs` | `bool` | `None` | Return log probabilities |
| `top_logprobs` | `int` | `None` | Number of top logprobs per token |

### Optional - LiteLLM Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float` | `None` | Request timeout in seconds |
| `api_key` | `str` | `None` | API key override |
| `api_base` | `str` | `None` | API base URL override |
| `api_version` | `str` | `None` | API version (Azure) |
| `num_retries` | `int` | `None` | Number of retries on failure |
| `fallbacks` | `List[str]` | `None` | Fallback models |
| `metadata` | `dict` | `None` | Custom metadata for logging |

## Returns

**`ModelResponse`** - OpenAI-compatible response object ([OpenAI Reference](https://platform.openai.com/docs/api-reference/chat/object))

```python
ModelResponse(
    id="chatcmpl-abc123",
    created=1703658209,
    model="gpt-4o",
    object="chat.completion",
    choices=[
        Choices(
            index=0,
            finish_reason="stop",
            message=Message(
                role="assistant",
                content="Hello! How can I help you?"
            )
        )
    ],
    usage=Usage(
        prompt_tokens=10,
        completion_tokens=15,
        total_tokens=25
    )
)
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique completion identifier |
| `created` | `int` | Unix timestamp |
| `model` | `str` | Model used |
| `object` | `str` | `"chat.completion"` |
| `choices` | `List[Choices]` | List of completion choices |
| `choices[].index` | `int` | Choice index |
| `choices[].finish_reason` | `str` | `"stop"`, `"length"`, `"tool_calls"` |
| `choices[].message.role` | `str` | `"assistant"` |
| `choices[].message.content` | `str` | Response text |
| `choices[].message.tool_calls` | `List` | Tool calls (if any) |
| `usage` | `Usage` | Token usage statistics |
| `usage.prompt_tokens` | `int` | Input tokens |
| `usage.completion_tokens` | `int` | Output tokens |
| `usage.total_tokens` | `int` | Total tokens |

### LiteLLM-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `_hidden_params` | `dict` | Internal metadata |
| `_hidden_params["model_id"]` | `str` | Router deployment ID |
| `_hidden_params["api_base"]` | `str` | API endpoint used |
| `_hidden_params["response_cost"]` | `float` | Calculated cost in USD |
| `_hidden_params["custom_llm_provider"]` | `str` | Provider used (e.g., `"openai"`, `"anthropic"`) |
| `_response_headers` | `dict` | Raw HTTP response headers |
| `response_ms` | `float` | Response latency in milliseconds |

**Accessing LiteLLM fields:**
```python
response = completion(model="gpt-4o", messages=[...])

# Latency
print(response.response_ms)  # 523.45

# Cost
print(response._hidden_params["response_cost"])  # 0.0023

# Provider
print(response._hidden_params["custom_llm_provider"])  # "openai"
```

## Async Variant

```python
response = await litellm.acompletion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## More Examples

<Tabs>
<TabItem value="streaming" label="Streaming">

```python
from litellm import completion

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

</TabItem>
<TabItem value="tools" label="Tools/Functions">

```python
from litellm import completion

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

print(response.choices[0].message.tool_calls)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```bash
curl -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

</TabItem>
</Tabs>

## Supported Providers

See provider-specific documentation for available models and configuration:

- [OpenAI](../providers/openai)
- [Anthropic](../providers/anthropic)
- [Azure OpenAI](../providers/azure/azure)
- [Google Vertex AI](../providers/vertex)
- [Google AI Studio](../providers/gemini)
- [AWS Bedrock](../providers/bedrock)
- [Mistral](../providers/mistral)
- [Cohere](../providers/cohere)

[**Browse all models →**](https://models.litellm.ai/)

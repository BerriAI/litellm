import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# responses()

Use OpenAI's Responses API format to call 100+ LLMs. Supports streaming, tool calling, and MCP integration.

## Example

```python
from litellm import responses

response = responses(
    input="What is the capital of France?",
    model="gpt-4o"
)

print(response.output[0].content[0].text)
```

## Signature

```python
def responses(
    input: Union[str, ResponseInputParam],
    model: str,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    tools: Optional[List[ToolParam]] = None,
    tool_choice: Optional[ToolChoice] = None,
    parallel_tool_calls: Optional[bool] = None,
    metadata: Optional[Dict[str, Any]] = None,
    store: Optional[bool] = None,
    reasoning: Optional[Reasoning] = None,
    truncation: Optional[Literal["auto", "disabled"]] = None,
    user: Optional[str] = None,
    timeout: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs
) -> ResponsesAPIResponse
```

## Parameters

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `str \| ResponseInputParam` | The input text or structured input |
| `model` | `str` | Model identifier (e.g., `"gpt-4o"`, `"anthropic/claude-3-opus"`) |

### Optional - OpenAI Compatible

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instructions` | `str` | `None` | System instructions for the model |
| `max_output_tokens` | `int` | `None` | Maximum tokens to generate |
| `temperature` | `float` | `None` | Sampling temperature (0-2) |
| `top_p` | `float` | `None` | Nucleus sampling parameter |
| `stream` | `bool` | `None` | Stream response chunks |
| `tools` | `List[ToolParam]` | `None` | Tool/function definitions |
| `tool_choice` | `ToolChoice` | `None` | Tool selection mode |
| `parallel_tool_calls` | `bool` | `None` | Allow parallel tool calls |
| `metadata` | `Dict[str, Any]` | `None` | Custom metadata |
| `store` | `bool` | `None` | Store the response |
| `reasoning` | `Reasoning` | `None` | Enable reasoning/thinking |
| `truncation` | `"auto" \| "disabled"` | `None` | Truncation behavior |
| `user` | `str` | `None` | End-user identifier |

### Optional - LiteLLM Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float` | `None` | Request timeout in seconds |
| `custom_llm_provider` | `str` | `None` | Override provider detection |
| `text_format` | `BaseModel \| dict` | `None` | Structured output format |

## Returns

**`ResponsesAPIResponse`** - OpenAI Responses API compatible object

```python
ResponsesAPIResponse(
    id="resp_abc123",
    created_at=1703658209,
    status="completed",
    model="gpt-4o",
    output=[
        ResponseOutputItem(
            type="message",
            role="assistant",
            content=[
                ContentPart(type="output_text", text="Hello! How can I help?")
            ]
        )
    ],
    usage=ResponseAPIUsage(
        input_tokens=10,
        output_tokens=15,
        total_tokens=25
    )
)
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique response identifier |
| `created_at` | `int` | Unix timestamp |
| `status` | `str` | `"completed"`, `"failed"`, `"in_progress"`, `"cancelled"` |
| `model` | `str` | Model used |
| `output` | `List[ResponseOutputItem]` | List of output items (messages, tool calls) |
| `usage` | `ResponseAPIUsage` | Token usage |
| `usage.input_tokens` | `int` | Input tokens |
| `usage.output_tokens` | `int` | Output tokens |
| `usage.total_tokens` | `int` | Total tokens |
| `error` | `dict` | Error details (if failed) |

### LiteLLM-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `_hidden_params` | `dict` | Contains `model_id`, `api_base` |

## Async Variant

```python
response = await litellm.aresponses(
    input="Hello!",
    model="gpt-4o"
)
```

## More Examples

<Tabs>
<TabItem value="streaming" label="Streaming">

```python
from litellm import responses

response = responses(
    input="Write a short poem",
    model="gpt-4o",
    stream=True
)

for event in response:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

</TabItem>
<TabItem value="tools" label="Tools">

```python
from litellm import responses

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
}]

response = responses(
    input="What's the weather in Paris?",
    model="gpt-4o",
    tools=tools
)

print(response.output)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```bash
curl -X POST 'http://0.0.0.0:4000/v1/responses' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-4o",
    "input": "Hello!"
  }'
```

</TabItem>
</Tabs>

## Additional Methods

### Get Response

```python
response = litellm.get_responses(response_id="resp_abc123")
```

### Delete Response

```python
result = litellm.delete_responses(response_id="resp_abc123")
```

### Cancel Response

```python
response = litellm.cancel_responses(response_id="resp_abc123")
```

## Supported Providers

See provider-specific documentation for available models:

- [OpenAI](../providers/openai)
- [Anthropic](../providers/anthropic)
- [Azure OpenAI](../providers/azure/azure)
- [Google Vertex AI](../providers/vertex)
- [AWS Bedrock](../providers/bedrock)

[**Browse all models →**](https://models.litellm.ai/)

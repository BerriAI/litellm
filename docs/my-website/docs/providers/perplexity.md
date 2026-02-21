import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Perplexity AI (pplx-api)
https://www.perplexity.ai

## API Key
```python
# env variable
os.environ['PERPLEXITYAI_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/sonar-pro", 
    messages=messages
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/sonar-pro", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Reasoning Effort

Requires v1.72.6+

:::info

See full guide on Reasoning with LiteLLM [here](../reasoning_content)

:::

You can set the reasoning effort by setting the `reasoning_effort` parameter.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/sonar-reasoning", 
    messages=messages,
    reasoning_effort="high"
)
print(response)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

1. Setup config.yaml

```yaml
model_list:
  - model_name: perplexity-sonar-reasoning-model
    litellm_params:
        model: perplexity/sonar-reasoning
        api_key: os.environ/PERPLEXITYAI_API_KEY
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

Replace `anything` with your LiteLLM Proxy Virtual Key, if [setup](../proxy/virtual_keys).

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer anything" \
  -d '{
    "model": "perplexity-sonar-reasoning-model",
    "messages": [{"role": "user", "content": "Who won the World Cup in 2022?"}],
    "reasoning_effort": "high"
  }'
```

</TabItem>
</Tabs>

## Supported Models
All models listed here https://docs.perplexity.ai/docs/model-cards are supported.  Just do `model=perplexity/<model-name>`.

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| sonar-deep-research | `completion(model="perplexity/sonar-deep-research", messages)` | 
| sonar-reasoning-pro | `completion(model="perplexity/sonar-reasoning-pro", messages)` | 
| sonar-reasoning | `completion(model="perplexity/sonar-reasoning", messages)` | 
| sonar-pro | `completion(model="perplexity/sonar-pro", messages)` | 
| sonar | `completion(model="perplexity/sonar", messages)` | 
| r1-1776 | `completion(model="perplexity/r1-1776", messages)` | 






## Agentic Research API (Responses API)

Requires v1.72.6+


### Using Presets

Presets provide optimized defaults for specific use cases. Start with a preset for quick setup:

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

# Using the pro-search preset
response = responses(
    model="perplexity/preset/pro-search",
    input="What are the latest developments in AI?",
    custom_llm_provider="perplexity",
)

print(response.output)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

1. Setup config.yaml

```yaml
model_list:
  - model_name: perplexity-pro-search
    litellm_params:
        model: perplexity/preset/pro-search
        api_key: os.environ/PERPLEXITY_API_KEY
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it!

```bash
curl http://0.0.0.0:4000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer anything" \
  -d '{
    "model": "perplexity-pro-search",
    "input": "What are the latest developments in AI?"
  }'
```

</TabItem>
</Tabs>

### Using Third-Party Models

Access models from OpenAI, Anthropic, Google, xAI, and other providers through Perplexity's unified API:

<Tabs>
<TabItem value="openai" label="OpenAI">

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

response = responses(
    model="perplexity/openai/gpt-4o",
    input="Explain quantum computing in simple terms",
    custom_llm_provider="perplexity",
    max_output_tokens=500,
)

print(response.output)
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

response = responses(
    model="perplexity/anthropic/claude-3-5-sonnet-20241022",
    input="Write a short story about a robot learning to paint",
    custom_llm_provider="perplexity",
    max_output_tokens=500,
)

print(response.output)
```

</TabItem>
<TabItem value="google" label="Google">

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

response = responses(
    model="perplexity/google/gemini-2.0-flash-exp",
    input="Explain the concept of neural networks",
    custom_llm_provider="perplexity",
    max_output_tokens=500,
)

print(response.output)
```

</TabItem>
<TabItem value="xai" label="xAI">

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

response = responses(
    model="perplexity/xai/grok-2-1212",
    input="What makes a good AI assistant?",
    custom_llm_provider="perplexity",
    max_output_tokens=500,
)

print(response.output)
```

</TabItem>
</Tabs>

### Web Search Tool

Enable web search capabilities to access real-time information:

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

response = responses(
    model="perplexity/openai/gpt-4o",
    input="What's the weather in San Francisco today?",
    custom_llm_provider="perplexity",
    tools=[{"type": "web_search"}],
    instructions="You have access to a web_search tool. Use it for questions about current events.",
)

print(response.output)
```


### Reasoning Effort (Responses API)

Control the reasoning effort level for reasoning-capable models:

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

response = responses(
    model="perplexity/openai/gpt-5.2",
    input="Solve this complex problem step by step",
    custom_llm_provider="perplexity",
    reasoning={"effort": "high"},  # Options: low, medium, high
    max_output_tokens=1000,
)

print(response.output)
```

### Multi-Turn Conversations

Use message arrays for multi-turn conversations with context:

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

response = responses(
    model="perplexity/anthropic/claude-3-5-sonnet-20241022",
    input=[
        {"type": "message", "role": "system", "content": "You are a helpful assistant."},
        {"type": "message", "role": "user", "content": "What are the latest AI developments?"},
    ],
    custom_llm_provider="perplexity",
    instructions="Provide detailed, well-researched answers.",
    max_output_tokens=800,
)

print(response.output)
```

### Streaming Responses

Stream responses for real-time output:

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

response = responses(
    model="perplexity/openai/gpt-4o",
    input="Tell me a story about space exploration",
    custom_llm_provider="perplexity",
    stream=True,
    max_output_tokens=500,
)

for chunk in response:
    if hasattr(chunk, 'type'):
        if chunk.type == "response.output_text.delta":
            print(chunk.delta, end="", flush=True)
```

### Supported Third-Party Models

| Provider | Model Name | Function Call |
|----------|------------|---------------|
| OpenAI | gpt-4o | `responses(model="perplexity/openai/gpt-4o", ...)` |
| OpenAI | gpt-4o-mini | `responses(model="perplexity/openai/gpt-4o-mini", ...)` |
| OpenAI | gpt-5.2 | `responses(model="perplexity/openai/gpt-5.2", ...)` |
| Anthropic | claude-3-5-sonnet-20241022 | `responses(model="perplexity/anthropic/claude-3-5-sonnet-20241022", ...)` |
| Anthropic | claude-3-5-haiku-20241022 | `responses(model="perplexity/anthropic/claude-3-5-haiku-20241022", ...)` |
| Google | gemini-2.0-flash-exp | `responses(model="perplexity/google/gemini-2.0-flash-exp", ...)` |
| Google | gemini-2.0-flash-thinking-exp | `responses(model="perplexity/google/gemini-2.0-flash-thinking-exp", ...)` |
| xAI | grok-2-1212 | `responses(model="perplexity/xai/grok-2-1212", ...)` |
| xAI | grok-2-vision-1212 | `responses(model="perplexity/xai/grok-2-vision-1212", ...)` |

### Available Presets

| Preset Name    | Function Call                                           |
|----------------|--------------------------------------------------------|
| fast-search    | `responses(model="perplexity/preset/fast-search", ...)`|
| pro-search     | `responses(model="perplexity/preset/pro-search", ...)` |
| deep-research  | `responses(model="perplexity/preset/deep-research", ...)`|

### Complete Example

```python
from litellm import responses
import os

os.environ['PERPLEXITY_API_KEY'] = ""

# Comprehensive example with multiple features
response = responses(
    model="perplexity/openai/gpt-4o",
    input="Research the latest developments in quantum computing and provide sources",
    custom_llm_provider="perplexity",
    tools=[
        {"type": "web_search"},
        {"type": "fetch_url"}
    ],
    instructions="Use web_search to find relevant information and fetch_url to retrieve detailed content from sources. Provide citations for all claims.",
    max_output_tokens=1000,
    temperature=0.7,
)

print(f"Response ID: {response.id}")
print(f"Model: {response.model}")
print(f"Status: {response.status}")
print(f"Output: {response.output}")
print(f"Usage: {response.usage}")
```

:::info

For more information about passing provider-specific parameters, [go here](../completion/provider_specific_params.md)
:::

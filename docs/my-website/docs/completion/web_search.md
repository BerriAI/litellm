import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Web Search

Use web search with litellm

| Feature | Details |
|---------|---------|
| Supported Endpoints | - `/chat/completions` <br/> - `/responses` |
| Supported Providers | `openai`, `xai`, `vertex_ai`, `anthropic`, `gemini`, `perplexity` |
| LiteLLM Cost Tracking | ✅ Supported |
| LiteLLM Version | `v1.71.0+` |

## Which Search Engine is Used?

Each provider uses their own search backend:

| Provider | Search Engine | Notes |
|----------|---------------|-------|
| **OpenAI** (`gpt-5-search-api`, `gpt-4o-search-preview`, `gpt-4o-mini-search-preview`) | OpenAI's internal search | Real-time web data |
| **xAI** (`grok-3`) | xAI's search + X/Twitter | Real-time social media data |
| **Google AI/Vertex** (`gemini-2.0-flash`) | **Google Search** | Uses actual Google search results |
| **Anthropic** (`claude-3-5-sonnet`) | Anthropic's web search | Real-time web data |
| **Perplexity** | Perplexity's search engine | AI-powered search and reasoning |

:::warning Important: Only Search Models Support `web_search_options`
For OpenAI, only dedicated search models support the `web_search_options` parameter:
- `gpt-4o-search-preview`
- `gpt-4o-mini-search-preview`
- `gpt-5-search-api`

**Regular models like `gpt-5`, `gpt-4.1`, `gpt-4o` do not support `web_search_options`**
:::

:::tip The `web_search_options` parameter is optional
Search models (like `gpt-4o-search-preview`) **automatically search the web** even without the `web_search_options` parameter.

Use `web_search_options` when you need to:
- Adjust `search_context_size` (`"low"`, `"medium"`, `"high"`)
- Specify `user_location` for localized results
:::

:::info
**Anthropic Web Search Models**: Claude models that support web search: `claude-3-5-sonnet-latest`, `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-latest`, `claude-3-5-haiku-20241022`, `claude-3-7-sonnet-20250219`
:::

## OpenAI Web Search: Two Approaches

OpenAI offers two distinct ways to use web search depending on the endpoint and model:

| Approach | Endpoint | Models | How to enable |
|----------|----------|--------|---------------|
| **Search Models** | `/chat/completions` | `gpt-5-search-api`, `gpt-4o-search-preview`, `gpt-4o-mini-search-preview` | Pass `web_search_options` parameter |
| **Web Search Tool** | `/responses` | `gpt-5`, `gpt-4.1`, `gpt-4o`, and other regular models | Pass `web_search_preview` tool |

:::tip Search models search automatically
Search models like `gpt-5-search-api` **automatically search the web** even without the `web_search_options` parameter. Use `web_search_options` to set `search_context_size` (`"low"`, `"medium"`, `"high"`) or specify `user_location` for localized results.
:::

## `/chat/completions` (litellm.completion)

### Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import completion

response = completion(
    model="openai/gpt-5-search-api",
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?",
        }
    ],
    web_search_options={
        "search_context_size": "medium"  # Options: "low", "medium", "high"
    }
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  # OpenAI search models
  - model_name: gpt-5-search-api
    litellm_params:
      model: openai/gpt-5-search-api
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4o-search-preview
    litellm_params:
      model: openai/gpt-4o-search-preview
      api_key: os.environ/OPENAI_API_KEY

  # xAI
  - model_name: grok-3
    litellm_params:
      model: xai/grok-3
      api_key: os.environ/XAI_API_KEY

  # Anthropic
  - model_name: claude-3-5-sonnet-latest
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_API_KEY

  # VertexAI
  - model_name: gemini-2-flash
    litellm_params:
      model: gemini-2.0-flash
      vertex_project: your-project-id
      vertex_location: us-central1

  # Google AI Studio
  - model_name: gemini-2-flash-studio
    litellm_params:
      model: gemini/gemini-2.0-flash
      api_key: os.environ/GOOGLE_API_KEY
```

2. Start the proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it!

```python showLineNumbers
from openai import OpenAI

# Point to your proxy server
client = OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-5-search-api",  # or any other web search enabled model
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?"
        }
    ],
    extra_body={
        "web_search_options": {
            "search_context_size": "medium"
        }
    }
)
```
</TabItem>
</Tabs>

### Search context size

<Tabs>
<TabItem value="sdk" label="SDK">

**OpenAI (using web_search_options)**
```python showLineNumbers
from litellm import completion

# Customize search context size
response = completion(
    model="openai/gpt-5-search-api",
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?",
        }
    ],
    web_search_options={
        "search_context_size": "low"  # Options: "low", "medium" (default), "high"
    }
)
```

**xAI (using web_search_options)**
```python showLineNumbers
from litellm import completion

# Customize search context size for xAI
response = completion(
    model="xai/grok-3",
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?",
        }
    ],
    web_search_options={
        "search_context_size": "high"  # Options: "low", "medium" (default), "high"
    }
)
```

**Anthropic (using web_search_options)**
```python showLineNumbers
from litellm import completion

# Customize search context size for Anthropic
response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?",
        }
    ],
    web_search_options={
        "search_context_size": "medium",  # Options: "low", "medium" (default), "high"
        "user_location": {
            "type": "approximate",
            "approximate": {
                "city": "San Francisco",
            },
        }
    }
)
```

**VertexAI/Gemini (using web_search_options)**
```python showLineNumbers
from litellm import completion

# Customize search context size for Gemini
response = completion(
    model="gemini-2.0-flash",
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?",
        }
    ],
    web_search_options={
        "search_context_size": "low"  # Options: "low", "medium" (default), "high"
    }
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```python showLineNumbers
from openai import OpenAI

# Point to your proxy server
client = OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

# Customize search context size
response = client.chat.completions.create(
    model="grok-3",  # works with any web search enabled model
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?"
        }
    ],
    web_search_options={
        "search_context_size": "low"  # Options: "low", "medium" (default), "high"
    }
)
```
</TabItem>
</Tabs>



## `/responses` (litellm.responses)

Use the `web_search_preview` tool with models like `gpt-5`, `gpt-4.1`, `gpt-4o`, etc.

:::info
Search-dedicated models like `gpt-5-search-api` and `gpt-4o-search-preview` do **not** support the `/responses` endpoint. Use them with `/chat/completions` + `web_search_options` instead (see above).
:::

### Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import responses

response = responses(
    model="openai/gpt-5",
    input="What is the capital of France?",
    tools=[{
        "type": "web_search_preview"  # enables web search with default medium context size
    }]
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-5
    litellm_params:
      model: openai/gpt-5
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4.1
    litellm_params:
      model: openai/gpt-4.1
      api_key: os.environ/OPENAI_API_KEY
```

2. Start the proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it!

```python showLineNumbers
from openai import OpenAI

# Point to your proxy server
client = OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.responses.create(
    model="gpt-5",
    tools=[{
        "type": "web_search_preview"
    }],
    input="What is the capital of France?",
)

print(response.output_text)
```
</TabItem>
</Tabs>

### Search context size

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import responses

# Customize search context size
response = responses(
    model="openai/gpt-5",
    input="What is the capital of France?",
    tools=[{
        "type": "web_search_preview",
        "search_context_size": "low"  # Options: "low", "medium" (default), "high"
    }]
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```python showLineNumbers
from openai import OpenAI

# Point to your proxy server
client = OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

# Customize search context size
response = client.responses.create(
    model="gpt-5",
    tools=[{
        "type": "web_search_preview",
        "search_context_size": "low"  # Options: "low", "medium" (default), "high"
    }],
    input="What is the capital of France?",
)

print(response.output_text)
```
</TabItem>
</Tabs>

## Configuring Web Search in config.yaml

You can set default web search options directly in your proxy config file:

<Tabs>
<TabItem value="default" label="Default Web Search">

```yaml
model_list:
  # Enable web search by default for all requests to this model
  - model_name: grok-3
    litellm_params:
      model: xai/grok-3
      api_key: os.environ/XAI_API_KEY
      web_search_options: {}  # Enables web search with default settings
```

### Advanced
You can configure LiteLLM's router to optionally drop models that do not support WebSearch, for example
```yaml
  - model_name: gpt-4.1
    litellm_params:
      model: openai/gpt-4.1
  - model_name: gpt-4.1
    litellm_params:
      model: azure/gpt-4.1
      api_base: "x.openai.azure.com/"
      api_version: 2025-03-01-preview
    model_info:
      supports_web_search: False <---- KEY CHANGE!
```
In this example, LiteLLM will still route LLM requests to both deployments, but for WebSearch, will solely route to OpenAI.

</TabItem>
<TabItem value="custom" label="Custom Search Context">

```yaml
model_list:
  # Set custom web search context size
  - model_name: grok-3
    litellm_params:
      model: xai/grok-3
      api_key: os.environ/XAI_API_KEY
      web_search_options:
        search_context_size: "high"  # Options: "low", "medium", "high"
  
  # OpenAI search model with custom context size
  - model_name: gpt-5-search-api
    litellm_params:
      model: openai/gpt-5-search-api
      api_key: os.environ/OPENAI_API_KEY
      web_search_options:
        search_context_size: "low"

  # Gemini with medium context (default)
  - model_name: gemini-2-flash
    litellm_params:
      model: gemini-2.0-flash
      vertex_project: your-project-id
      vertex_location: us-central1
      web_search_options:
        search_context_size: "medium"
```

</TabItem>
</Tabs>

**Note:** When `web_search_options` is set in the config, it applies to all requests to that model. Users can still override these settings by passing `web_search_options` in their API requests.

## Checking if a model supports web search

<Tabs>
<TabItem label="SDK" value="sdk">

Use `litellm.supports_web_search(model="model_name")` -> returns `True` if model can perform web searches

```python showLineNumbers
# Check OpenAI models
assert litellm.supports_web_search(model="openai/gpt-5-search-api") == True
assert litellm.supports_web_search(model="openai/gpt-4o-search-preview") == True

# Check xAI models
assert litellm.supports_web_search(model="xai/grok-3") == True

# Check Anthropic models
assert litellm.supports_web_search(model="anthropic/claude-3-5-sonnet-latest") == True

# Check VertexAI models
assert litellm.supports_web_search(model="gemini-2.0-flash") == True

# Check Google AI Studio models
assert litellm.supports_web_search(model="gemini/gemini-2.0-flash") == True
```
</TabItem>

<TabItem label="PROXY" value="proxy">

1. Define models in config.yaml

```yaml
model_list:
  # OpenAI
  - model_name: gpt-5-search-api
    litellm_params:
      model: openai/gpt-5-search-api
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      supports_web_search: True

  - model_name: gpt-4o-search-preview
    litellm_params:
      model: openai/gpt-4o-search-preview
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      supports_web_search: True

  # xAI
  - model_name: grok-3
    litellm_params:
      model: xai/grok-3
      api_key: os.environ/XAI_API_KEY
    model_info:
      supports_web_search: True
  
  # Anthropic
  - model_name: claude-3-5-sonnet-latest
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_API_KEY
    model_info:
      supports_web_search: True
  
  # VertexAI
  - model_name: gemini-2-flash
    litellm_params:
      model: gemini-2.0-flash
      vertex_project: your-project-id
      vertex_location: us-central1
    model_info:
      supports_web_search: True
  
  # Google AI Studio
  - model_name: gemini-2-flash-studio
    litellm_params:
      model: gemini/gemini-2.0-flash
      api_key: os.environ/GOOGLE_API_KEY
    model_info:
      supports_web_search: True
```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Call `/model_group/info` to check if a model supports web search

```shell
curl -X 'GET' \
  'http://localhost:4000/model_group/info' \
  -H 'accept: application/json' \
  -H 'x-api-key: sk-1234'
```

Expected Response 

```json showLineNumbers
{
  "data": [
    {
      "model_group": "gpt-5-search-api",
      "providers": ["openai"],
      "max_tokens": 128000,
      "supports_web_search": true
    },
    {
      "model_group": "gpt-4o-search-preview",
      "providers": ["openai"],
      "max_tokens": 128000,
      "supports_web_search": true
    },
    {
      "model_group": "grok-3",
      "providers": ["xai"],
      "max_tokens": 131072,
      "supports_web_search": true
    },
    {
      "model_group": "gemini-2-flash",
      "providers": ["vertex_ai"],
      "max_tokens": 8192,
      "supports_web_search": true
    }
  ]
}
```

</TabItem>
</Tabs>

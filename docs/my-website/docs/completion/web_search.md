import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using Web Search

Use web search with litellm

| Feature | Details |
|---------|---------|
| Supported Endpoints | - `/chat/completions` <br/> - `/responses` |
| Supported Providers | `openai` |
| LiteLLM Cost Tracking | ✅ Supported |
| LiteLLM Version | `v1.63.15-nightly` or higher |


## `/chat/completions` (litellm.completion)

### Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import completion

response = completion(
    model="openai/gpt-4o-search-preview",
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?",
        }
    ],
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-4o-search-preview
    litellm_params:
      model: openai/gpt-4o-search-preview
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

response = client.chat.completions.create(
    model="gpt-4o-search-preview",
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?"
        }
    ]
)
```
</TabItem>
</Tabs>

### Search context size

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import completion

# Customize search context size
response = completion(
    model="openai/gpt-4o-search-preview",
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
    model="gpt-4o-search-preview",
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

### Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import responses

response = responses(
    model="openai/gpt-4o",
    input=[
        {
            "role": "user",
            "content": "What was a positive news story from today?"
        }
    ],
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
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
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
    model="gpt-4o",
    tools=[{
        "type": "web_search_preview"
    }],
    input="What was a positive news story from today?",
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
    model="openai/gpt-4o",
    input=[
        {
            "role": "user",
            "content": "What was a positive news story from today?"
        }
    ],
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
    model="gpt-4o",
    tools=[{
        "type": "web_search_preview",
        "search_context_size": "low"  # Options: "low", "medium" (default), "high"
    }],
    input="What was a positive news story from today?",
)

print(response.output_text)
```
</TabItem>
</Tabs>






## Checking if a model supports web search

<Tabs>
<TabItem label="SDK" value="sdk">

Use `litellm.supports_web_search(model="openai/gpt-4o-search-preview")` -> returns `True` if model can perform web searches

```python showLineNumbers
assert litellm.supports_web_search(model="openai/gpt-4o-search-preview") == True
```
</TabItem>

<TabItem label="PROXY" value="proxy">

1. Define OpenAI models in config.yaml

```yaml
model_list:
  - model_name: gpt-4o-search-preview
    litellm_params:
      model: openai/gpt-4o-search-preview
      api_key: os.environ/OPENAI_API_KEY
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
      "model_group": "gpt-4o-search-preview",
      "providers": ["openai"],
      "max_tokens": 128000,
      "supports_web_search": true, # 👈 supports_web_search is true
    }
  ]
}
```

</TabItem>
</Tabs>

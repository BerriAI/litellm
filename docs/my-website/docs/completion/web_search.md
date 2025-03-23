import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using Web Search

How to use web search capabilities with the `/chat/completions` endpoint



Works for:
- OpenAI GPT-4 with web search

## Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="openai/gpt-4-turbo-preview",
    messages=[
        {
            "role": "user",
            "content": "What was a positive news story from today?",
        }
    ],
    tool_choice="auto",  # Enable web search capability
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-4-turbo
    litellm_params:
      model: openai/gpt-4-turbo-preview
      api_key: os.environ/OPENAI_API_KEY
```

2. Start the proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-4-turbo",
    "messages": [
        {
            "role": "user",
            "content": "What was a positive news story from today?"
        }
    ],
    "tool_choice": "auto"
}'
```
</TabItem>
</Tabs>

## Checking if a model supports web search

<Tabs>
<TabItem label="SDK" value="sdk">

Use `litellm.supports_web_search(model="openai/gpt-4-turbo-preview")` -> returns `True` if model can perform web searches

```python
assert litellm.supports_web_search(model="openai/gpt-4-turbo-preview") == True
```
</TabItem>

<TabItem label="PROXY" value="proxy">

1. Define OpenAI models in config.yaml

```yaml
model_list:
  - model_name: gpt-4-turbo
    litellm_params:
      model: openai/gpt-4-turbo-preview
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

```json
{
  "data": [
    {
      "model_group": "gpt-4-turbo",
      "providers": ["openai"],
      "max_tokens": 128000,
      "supports_web_search": true, # 👈 supports_web_search is true
    }
  ]
}
```

</TabItem>
</Tabs>

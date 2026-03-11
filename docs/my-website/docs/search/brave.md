# Brave Search

Get started by creating a free API key via https://brave.com/search/api/.

For documentation on other parameters supported by the Brave Search API, visit https://api-dashboard.search.brave.com/api-reference/web/search.

## LiteLLM Python SDK

```python showLineNumbers title="Brave Search"
import os
from litellm import search

os.environ["BRAVE_API_KEY"] = "BSATzx..."

response = search(
    query="Brave browser features",
    search_provider="brave",
    max_results=5
)
```

## LiteLLM AI Gateway

### 1. Setup config.yaml

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY

search_tools:
  - search_tool_name: brave-search
    litellm_params:
      search_provider: brave
      api_key: os.environ/BRAVE_API_KEY
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/brave-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{ "query": "Brave browser features", "max_results": 5 }'
```

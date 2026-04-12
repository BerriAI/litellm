# Serper Search

**Get API Key:** [https://serper.dev](https://serper.dev)

## LiteLLM Python SDK

```python showLineNumbers title="Serper Search"
import os
from litellm import search

os.environ["SERPER_API_KEY"] = "your-api-key"

response = search(
    query="latest AI developments",
    search_provider="serper",
    max_results=5
)
```

## LiteLLM AI Gateway

### 1. Setup config.yaml

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-5
    litellm_params:
      model: gpt-5
      api_key: os.environ/OPENAI_API_KEY

search_tools:
  - search_tool_name: serper-search
    litellm_params:
      search_provider: serper
      api_key: os.environ/SERPER_API_KEY
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/serper-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "max_results": 5
  }'
```

## Provider-specific Parameters

```python showLineNumbers title="Serper Search with Provider-specific Parameters"
import os
from litellm import search

os.environ["SERPER_API_KEY"] = "your-api-key"

response = search(
    query="latest tech news",
    search_provider="serper",
    max_results=10,
    # Serper-specific parameters
    gl="us",                    # Country/geolocation code
    hl="en",                    # Language code
    autocorrect=False,          # Disable autocorrect
    tbs="qdr:d",               # Time filter: past day ('qdr:h' hour, 'qdr:w' week, 'qdr:m' month)
    page=2                      # Page number
)
```

# Tavily Search

**Get API Key:** [https://tavily.com](https://tavily.com)

## LiteLLM Python SDK

```python showLineNumbers title="Tavily Search"
import os
from litellm import search

os.environ["TAVILY_API_KEY"] = "tvly-..."

response = search(
    query="latest AI developments",
    search_provider="tavily",
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
  - search_tool_name: tavily-search
    litellm_params:
      search_provider: tavily
      api_key: os.environ/TAVILY_API_KEY
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/tavily-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "max_results": 5
  }'
```

## Provider-specific Parameters

```python showLineNumbers title="Tavily Search with Provider-specific Parameters"
import os
from litellm import search

os.environ["TAVILY_API_KEY"] = "tvly-..."

response = search(
    query="latest tech news",
    search_provider="tavily",
    max_results=5,
    # Tavily-specific parameters
    topic="news",                    # 'general', 'news', 'finance'
    search_depth="advanced",         # 'basic', 'advanced'
    include_answer=True,             # Include AI-generated answer
    include_raw_content=True         # Include raw HTML content
)
```


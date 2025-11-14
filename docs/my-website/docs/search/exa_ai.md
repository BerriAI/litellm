# Exa AI Search

**Get API Key:** [https://exa.ai](https://exa.ai)

## LiteLLM Python SDK

```python showLineNumbers title="Exa AI Search"
import os
from litellm import search

os.environ["EXA_API_KEY"] = "exa-..."

response = search(
    query="latest AI developments",
    search_provider="exa_ai",
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
  - search_tool_name: exa-search
    litellm_params:
      search_provider: exa_ai
      api_key: os.environ/EXA_API_KEY
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/exa-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "max_results": 5
  }'
```

## Provider-specific Parameters

```python showLineNumbers title="Exa AI Search with Provider-specific Parameters"
import os
from litellm import search

os.environ["EXA_API_KEY"] = "exa-..."

response = search(
    query="AI research papers",
    search_provider="exa_ai",
    max_results=10,
    search_domain_filter=["arxiv.org"],
    # Exa-specific parameters
    type="neural",                   # 'neural', 'keyword', or 'auto'
    contents={"text": True},         # Request text content
    use_autoprompt=True              # Enable Exa's autoprompt
)
```


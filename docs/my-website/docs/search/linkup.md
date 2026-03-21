# Linkup Search

**Get API Key:** [https://linkup.so](https://linkup.so)

## LiteLLM Python SDK

```python showLineNumbers title="Linkup Search"
import os
from litellm import search

os.environ["LINKUP_API_KEY"] = "..."

response = search(
    query="latest AI developments",
    search_provider="linkup",
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
  - search_tool_name: linkup-search
    litellm_params:
      search_provider: linkup
      api_key: os.environ/LINKUP_API_KEY
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/linkup-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "max_results": 5
  }'
```

## Provider-specific Parameters

```python showLineNumbers title="Linkup Search with Provider-specific Parameters"
import os
from litellm import search

os.environ["LINKUP_API_KEY"] = "..."

response = search(
    query="machine learning research",
    search_provider="linkup",
    max_results=10,
    # Linkup-specific parameters
    depth="deep",                      # "standard" (faster) or "deep" (more comprehensive)
    outputType="searchResults",        # "searchResults", "sourcedAnswer", or "structured"
    includeSources=True,               # Include sources in response
    includeImages=True,                # Include images in results
    fromDate="2024-01-01",             # Start date filter (YYYY-MM-DD)
    toDate="2024-12-31",               # End date filter (YYYY-MM-DD)
    includeDomains=["arxiv.org", "nature.com"],  # Domains to search (max 100)
    excludeDomains=["wikipedia.com"],  # Domains to exclude
    includeInlineCitations=True,       # Include inline citations in sourcedAnswer
)
```

## Features

Linkup provides powerful web search with context retrieval capabilities:

### Search Depth
Control the precision and speed of your search:
- `standard` - Returns results faster
- `deep` - Takes longer but yields more comprehensive results

### Output Types
Choose how results are formatted:
- `searchResults` - Returns a list of search results with URLs and content
- `sourcedAnswer` - Returns an AI-generated answer with sources
- `structured` - Returns results in a custom JSON schema format

### Date Filtering
Filter results by date range:
```python
response = search(
    query="AI developments",
    search_provider="linkup",
    fromDate="2024-06-01",
    toDate="2024-12-31"
)
```

### Domain Filtering
Include or exclude specific domains:
```python
response = search(
    query="research papers",
    search_provider="linkup",
    includeDomains=["arxiv.org", "nature.com", "ieee.org"],
    excludeDomains=["wikipedia.com"]
)
```

### Structured Output
Get results in a custom JSON schema format:
```python
response = search(
    query="Microsoft 2024 revenue",
    search_provider="linkup",
    outputType="structured",
    structuredOutputSchema='{"type": "object", "properties": {"revenue": {"type": "string"}, "year": {"type": "string"}}}'
)
```

## Response Format

Linkup returns results in the following format:

```json
{
  "results": [
    {
      "type": "text",
      "name": "Microsoft 2024 Annual Report",
      "url": "https://www.microsoft.com/investor/reports/ar24/index.html",
      "content": "Highlights from fiscal year 2024..."
    }
  ]
}
```

LiteLLM transforms this to the standard `SearchResponse` format:
- `results[].name` → `SearchResult.title`
- `results[].url` → `SearchResult.url`
- `results[].content` → `SearchResult.snippet`


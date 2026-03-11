# SearchAPI.io (Google Search)

Get started by creating a free API key via https://www.searchapi.io/.

SearchAPI.io provides access to Google Search results with a simple API. It supports all Google Search parameters including location, language, time filters, and more.

For complete documentation on all supported parameters, visit https://www.searchapi.io/docs/google.

## LiteLLM Python SDK

```python showLineNumbers title="SearchAPI.io Search"
import os
from litellm import search

os.environ["SEARCHAPI_API_KEY"] = "your-api-key"

response = search(
    query="latest AI developments",
    search_provider="searchapi",
    max_results=10
)

# Access search results
for result in response.results:
    print(f"{result.title}: {result.url}")
    print(f"Snippet: {result.snippet}\n")
```

### Advanced Usage with SearchAPI.io Parameters

SearchAPI.io supports many Google Search-specific parameters:

```python showLineNumbers title="Advanced SearchAPI.io Parameters"
import os
from litellm import search

os.environ["SEARCHAPI_API_KEY"] = "your-api-key"

response = search(
    query="machine learning research",
    search_provider="searchapi",
    max_results=10,
    # Unified parameters
    country="US",
    search_domain_filter=["arxiv.org", "nature.com"],
    # SearchAPI.io specific parameters
    gl="us",              # Country code
    hl="en",              # Interface language
    time_period="last_month",  # Time filter
    safe="active",        # SafeSearch
    device="desktop",     # Device type
    location="New York"   # Geographic location
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
  - search_tool_name: google-search
    litellm_params:
      search_provider: searchapi
      api_key: os.environ/SEARCHAPI_API_KEY
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/google-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "max_results": 10,
    "country": "US"
  }'
```

## SearchAPI.io Specific Parameters

SearchAPI.io supports many Google Search parameters. Here are some commonly used ones:

| Parameter | Type | Description |
|-----------|------|-------------|
| `gl` | string | Country code (e.g., 'us', 'uk', 'de') |
| `hl` | string | Interface language (e.g., 'en', 'es', 'fr') |
| `location` | string | Geographic location (e.g., 'New York', 'London') |
| `device` | string | Device type: 'desktop', 'mobile', 'tablet' |
| `time_period` | string | Time filter: 'last_hour', 'last_day', 'last_week', 'last_month', 'last_year' |
| `time_period_min` | string | Start date (MM/DD/YYYY) |
| `time_period_max` | string | End date (MM/DD/YYYY) |
| `safe` | string | SafeSearch: 'active' or 'off' |
| `lr` | string | Language restriction (e.g., 'lang_en', 'lang_es') |
| `cr` | string | Country restriction |
| `page` | integer | Page number for pagination |

### Example with Time Filters

```python showLineNumbers title="Search with Time Filter"
response = search(
    query="AI breakthroughs",
    search_provider="searchapi",
    max_results=10,
    time_period="last_month"
)
```

### Example with Custom Date Range

```python showLineNumbers title="Search with Custom Date Range"
response = search(
    query="AI research papers",
    search_provider="searchapi",
    max_results=10,
    time_period_min="01/01/2024",
    time_period_max="03/01/2024"
)
```

### Example with Location

```python showLineNumbers title="Search with Location"
response = search(
    query="AI conferences",
    search_provider="searchapi",
    max_results=10,
    location="San Francisco",
    gl="us"
)
```

## Response Format

SearchAPI.io returns results in the standard LiteLLM search format:

```json
{
  "object": "search",
  "results": [
    {
      "title": "Latest AI Developments",
      "url": "https://example.com/ai-news",
      "snippet": "Recent breakthroughs in artificial intelligence...",
      "date": "2024-01-15"
    }
  ]
}
```

## Rate Limits

SearchAPI.io has different rate limits based on your plan:
- Free tier: 100 requests/month
- Paid plans: Higher limits available

Check your current usage at https://www.searchapi.io/dashboard.

## Error Handling

```python showLineNumbers title="Error Handling"
from litellm import search
import os

os.environ["SEARCHAPI_API_KEY"] = "your-api-key"

try:
    response = search(
        query="test query",
        search_provider="searchapi",
        max_results=10
    )
    print(f"Found {len(response.results)} results")
except Exception as e:
    print(f"Search failed: {str(e)}")
```

## Additional Resources

- SearchAPI.io Documentation: https://www.searchapi.io/docs
- API Dashboard: https://www.searchapi.io/dashboard
- Pricing: https://www.searchapi.io/pricing

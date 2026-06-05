# You.com Search

**Get API Key (optional, for higher rate limits):** [https://you.com/docs](https://you.com/docs)

You.com offers two tiers:

| Mode | Endpoint | Auth | Limits |
|---|---|---|---|
| **Keyless free tier** (default) | `https://api.you.com/v1/agents/search` | none | IP-throttled, ~100 queries/day |
| **Keyed tier** | `https://ydc-index.io/v1/search` | `X-API-Key` | higher rate limits |

If `YOUCOM_API_KEY` is not set, the adapter automatically uses the keyless endpoint — no signup required to start.

## LiteLLM Python SDK

### Keyless (zero config)

```python showLineNumbers title="You.com Search - keyless"
from litellm import search

response = search(
    query="latest AI developments",
    search_provider="you_com",
    max_results=5
)

for result in response.results:
    print(f"{result.title}: {result.url}")
    print(f"Snippet: {result.snippet}\n")
```

### With API key (higher limits)

```python showLineNumbers title="You.com Search - keyed"
import os
from litellm import search

os.environ["YOUCOM_API_KEY"] = "sk-..."

response = search(
    query="latest AI developments",
    search_provider="you_com",
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
  - search_tool_name: you-com-search
    litellm_params:
      search_provider: you_com
      # api_key optional - omit to use the keyless free tier
      api_key: os.environ/YOUCOM_API_KEY
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/you-com-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "max_results": 5
  }'
```

## Unified Parameters

You.com supports the standard Perplexity unified spec parameters:

```python showLineNumbers title="You.com Search with unified parameters"
from litellm import search

response = search(
    query="machine learning research",
    search_provider="you_com",
    max_results=10,                              # -> count
    search_domain_filter=["arxiv.org"],          # -> include_domains
    country="US"                                 # -> country (lowercased)
)
```

| Unified spec parameter | Mapped to You.com parameter |
|---|---|
| `max_results` | `count` |
| `search_domain_filter` | `include_domains` |
| `country` | `country` (lowercased) |
| `max_tokens_per_page` | _ignored (no equivalent)_ |

## Provider-specific Parameters

You can pass any You.com-specific parameter as a keyword argument; unrecognized parameters are forwarded to the upstream request body:

```python showLineNumbers title="You.com Search with provider-specific parameters"
from litellm import search

response = search(
    query="AI breakthroughs",
    search_provider="you_com",
    # You.com-specific parameters (passed through verbatim)
    freshness="week",                            # 'day', 'week', 'month', 'year', or date range
    exclude_domains=["example.com"],
    language="en"
)
```

## Response Notes

You.com's API returns results split into `web` and `news` arrays. The LiteLLM adapter flattens both into a single ordered `results` list (web first, then news) so the response matches the unified [`SearchResponse`](./index#response-format) shape.

For each result:
- `snippet` prefers the first entry of the upstream `snippets` array; falls back to `description`.
- `date` is populated from upstream `page_age` (ISO 8601 datetime).

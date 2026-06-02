# APISerpent Search

**Get API Key:** [https://apiserpent.com](https://apiserpent.com)

[APISerpent](https://apiserpent.com) is a multi-engine SERP API (Google, Bing, Yahoo, DuckDuckGo) with two endpoints, selected with the `deep` flag:

| Mode | Endpoint | `deep` | Results |
|---|---|---|---|
| **Quick search** (default) | `https://apiserpent.com/api/search/quick` | `False` | `num` 1–100 |
| **Deep search** | `https://apiserpent.com/api/search` | `True` | `num` 10–100 |

Both modes are billed at $0.60 per 1k searches.

## LiteLLM Python SDK

```python showLineNumbers title="APISerpent Search"
import os
from litellm import search

os.environ["APISERPENT_API_KEY"] = "your-api-key"

response = search(
    query="latest AI developments",
    search_provider="apiserpent",
    max_results=5
)

for result in response.results:
    print(f"{result.title}: {result.url}")
    print(f"Snippet: {result.snippet}\n")
```

### Deep search

```python showLineNumbers title="APISerpent Deep Search"
from litellm import search

response = search(
    query="open source vector databases comparison",
    search_provider="apiserpent",
    deep=True,            # routes to /api/search
    max_results=20
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
  - search_tool_name: apiserpent-search
    litellm_params:
      search_provider: apiserpent
      api_key: os.environ/APISERPENT_API_KEY
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/apiserpent-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "max_results": 5
  }'
```

## Unified Parameters

APISerpent supports the standard Perplexity unified spec parameters:

```python showLineNumbers title="APISerpent Search with unified parameters"
from litellm import search

response = search(
    query="machine learning research",
    search_provider="apiserpent",
    max_results=10,                              # -> num (clamped to the endpoint range)
    search_domain_filter=["arxiv.org"],          # -> site: filters appended to the query
    country="US"                                 # -> country (lowercased)
)
```

| Unified spec parameter | Mapped to APISerpent parameter |
|---|---|
| `max_results` | `num` (clamped: 1–100 quick, 10–100 deep) |
| `search_domain_filter` | `site:` clauses appended to `q` |
| `country` | `country` (lowercased) |
| `max_tokens_per_page` | _ignored (no equivalent)_ |

## Provider-specific Parameters

Pass any APISerpent-specific parameter as a keyword argument:

```python showLineNumbers title="APISerpent Search with Provider-specific Parameters"
import os
from litellm import search

os.environ["APISERPENT_API_KEY"] = "your-api-key"

response = search(
    query="elektroauto reichweite",
    search_provider="apiserpent",
    max_results=10,
    # APISerpent-specific parameters
    engine="bing",          # 'google' (default), 'bing', 'yahoo', or 'ddg'
    country="de",           # localized results
    language="de",          # 2-letter ISO language code
    freshness="d",          # time filter: 'h', 'd', '7d', 'w', 'm', 'y'
    safe="strict",          # SafeSearch: 'off', 'moderate', or 'strict'
    pages=2                 # pages to scrape (1-10)
)
```

| Parameter | Type | Description |
|---|---|---|
| `engine` | string | `google` (default), `bing`, `yahoo`, or `ddg` |
| `country` | string | Country code for localized results (default `us`) |
| `language` | string | 2-letter ISO language code (e.g. `en`, `es`, `de`) |
| `freshness` | string | Time filter: `h`, `d`, `7d`, `w`, `m`, `y` |
| `safe` | string | SafeSearch: `off`, `moderate`, or `strict` |
| `pages` | integer | Number of pages to scrape (1–10) |
| `format` | string | `full` (default) or `simple` |
| `pixel_position` | boolean | Include pixel coordinates (paid tiers only) |

`num` is clamped to its valid range (1–100 for quick search, 10–100 for deep) and `pages` must be 1–10; out-of-range values raise a `ValueError`.

## Response Notes

APISerpent's full-format response nests results under `results.organic`. The LiteLLM adapter maps each organic result's `title`, `url`, and `snippet` into the unified [`SearchResponse`](./index#response-format) shape.

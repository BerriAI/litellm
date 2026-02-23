# SearXNG Search

**Open Source:** [https://github.com/searxng/searxng](https://github.com/searxng/searxng)

**Public Instances:** [https://searx.space/](https://searx.space/)

## Overview

SearXNG is a free, open-source metasearch engine that aggregates results from multiple search engines while protecting user privacy. It can be self-hosted or used via public instances.

**Note:** SearXNG returns a fixed number of results per page (~20 by default) and does not support limiting results via the API. The `max_results` parameter is not directly supported by SearXNG.

## LiteLLM Python SDK

```python showLineNumbers title="SearXNG Search"
import os
from litellm import search

# Set your SearXNG instance URL (REQUIRED)
os.environ["SEARXNG_API_BASE"] = "https://serxng-deployment-production.up.railway.app"

response = search(
    query="latest AI developments",
    search_provider="searxng",
    max_results=10
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
  - search_tool_name: searxng-search
    litellm_params:
      search_provider: searxng
      api_base: https://serxng-deployment-production.up.railway.app
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/searxng-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "max_results": 10
  }'
```

## Provider-specific Parameters

```python showLineNumbers title="SearXNG Search with Provider-specific Parameters"
import os
from litellm import search

# REQUIRED: Set your SearXNG instance URL
os.environ["SEARXNG_API_BASE"] = "https://serxng-deployment-production.up.railway.app"

response = search(
    query="machine learning research",
    search_provider="searxng",
    max_results=10,
    # SearXNG-specific parameters
    categories="general,science",      # Comma-separated categories
    engines="google,duckduckgo,bing",  # Comma-separated engines
    language="en",                      # Language code
    pageno=1,                           # Page number
    time_range="month"                  # Time filter: day, month, year
)
```

## Features

SearXNG provides powerful metasearch capabilities:

### Multiple Search Engines
Aggregate results from multiple search engines simultaneously:
- Google, DuckDuckGo, Bing, Brave
- Wikipedia, Startpage
- And many more

### Categories
Search within specific categories:
- `general` - General web search
- `science` - Scientific articles and papers
- `images` - Image search
- `news` - News articles
- `videos` - Video content
- `music` - Music and audio
- `files` - File search
- `it` - IT and technology
- `map` - Maps and location

### Time-Based Filtering
Filter results by time range:
- `day` - Past day
- `month` - Past month
- `year` - Past year

### Privacy-Focused
- No user tracking
- No cookies required
- No profiling
- No ads

### Language Support
Support for 60+ languages with the `language` parameter.

## Self-Hosting

SearXNG can be self-hosted for complete control.

### Quick Deploy

Use our pre-configured deployment repository for easy setup:

**[Fork and Deploy: github.com/BerriAI/serxng-deployment](https://github.com/BerriAI/serxng-deployment)**

This repository includes:
- Docker and Docker Compose setup
- JSON API format pre-configured
- Ready to deploy

### Manual Installation

See the [official SearXNG installation instructions](https://docs.searxng.org/admin/installation.html) for detailed setup.

**Important:** When you install SearXNG, the only active output format by default is the HTML format. You need to activate the JSON format to use the API.

Add the following to your `settings.yml` file:

```yaml
search:
  formats:
    - html
    - json
```

Then restart SearXNG:

```bash
# Using Docker
docker run -d -p 8080:8080 \
  -v $(pwd)/settings.yml:/etc/searxng/settings.yml:ro \
  -e SEARXNG_BASE_URL=http://localhost:8080 \
  searxng/searxng

# Then configure LiteLLM to use your instance
export SEARXNG_API_BASE=http://localhost:8080
```

## Configuration

### Setting API Base URL (Required)

You **must** specify a SearXNG instance URL either via environment variable or in the search call:

```python
# Option 1: Environment variable (Recommended)
import os
os.environ["SEARXNG_API_BASE"] = "https://your-instance.com"

response = search(
    query="AI developments",
    search_provider="searxng"
)

# Option 2: Pass directly in search call
response = search(
    query="AI developments",
    search_provider="searxng",
    api_base="https://your-instance.com"
)
```

**Note:** There is no default instance URL. You must choose either a [public instance](https://searx.space/) or self-host your own.

### Optional Authentication

Some SearXNG instances may require authentication:

```python
import os

# Set API key if required
os.environ["SEARXNG_API_KEY"] = "your-api-key"

response = search(
    query="AI developments",
    search_provider="searxng"
)
```

## Cost

SearXNG is completely free:
- **Open source** - No licensing costs
- **Self-hosted** - Only infrastructure costs
- **Public instances** - Usually free, check instance policies

## Advanced Usage

### Custom Engine Selection

```python
response = search(
    query="Python tutorials",
    search_provider="searxng",
    engines="stackoverflow,github,reddit",  # Only search these engines
    categories="it"
)
```

### Multi-Category Search

```python
response = search(
    query="climate change",
    search_provider="searxng",
    categories="general,science,news",  # Search multiple categories
    time_range="month"
)
```

### Pagination

```python
# Get page 1
page1 = search(
    query="AI research",
    search_provider="searxng",
    pageno=1
)

# Get page 2
page2 = search(
    query="AI research",
    search_provider="searxng",
    pageno=2
)
```

## Response Format

SearXNG returns results in the standard LiteLLM search format:

```json
{
  "object": "search",
  "results": [
    {
      "title": "Example Result",
      "url": "https://example.com",
      "snippet": "This is the content snippet from the search result...",
      "date": "2024-01-15",
      "last_updated": null
    }
  ]
}
```

## Troubleshooting

### Test Your Instance First

If LiteLLM with searxng search provider is not working, test your SearXNG instance directly with curl:

```bash
# Test if JSON API is working
curl -s "https://your-searxng-instance.com/search?q=test&format=json" | head -50

# Example with specific instance
curl -s "https://serxng-deployment-production.up.railway.app/search?q=test&format=json" | head -50
```

**Expected response**: JSON with search results  
**If you get HTML**: JSON format is not enabled in the instance's `settings.yml`

### No Results

If you get no results:

1. **Try different engines**: Specify `engines` parameter
2. **Broaden categories**: Use multiple categories
3. **Adjust language**: Set appropriate `language` parameter

### JSON Format Not Enabled

If you get HTML instead of JSON:

1. **Test with curl**: Use the curl command above to verify JSON output
2. **Self-host your own instance**: Use [our deployment repo](https://github.com/BerriAI/serxng-deployment) with JSON pre-configured
3. **Check instance configuration**: Not all public instances have JSON enabled
4. **Enable JSON manually**: Add to `settings.yml`:
   ```yaml
   search:
     formats:
       - html
       - json
   ```


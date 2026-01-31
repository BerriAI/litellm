# DataForSEO Search

**Get API Access:** [DataForSEO](https://dataforseo.com/)

## Setup

1. Go to [DataForSEO](https://dataforseo.com/) and create an account
2. Navigate to your account dashboard
3. Generate API credentials:
   - You'll receive a **login** (username)
   - You'll receive a **password**
4. Set up your environment variables:
   - `DATAFORSEO_LOGIN` - Your DataForSEO login/username
   - `DATAFORSEO_PASSWORD` - Your DataForSEO password

## LiteLLM Python SDK

```python showLineNumbers title="DataForSEO Search"
import os
from litellm import search

os.environ["DATAFORSEO_LOGIN"] = "your-login"
os.environ["DATAFORSEO_PASSWORD"] = "your-password"

response = search(
    query="latest AI developments",
    search_provider="dataforseo",
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
  - search_tool_name: dataforseo-search
    litellm_params:
      search_provider: dataforseo
      api_key: "os.environ/DATAFORSEO_LOGIN:os.environ/DATAFORSEO_PASSWORD"
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test the search endpoint

```bash showLineNumbers title="Test Request"
curl http://0.0.0.0:4000/v1/search/dataforseo-search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI developments",
    "max_results": 10
  }'
```

## Provider-specific Parameters

```python showLineNumbers title="DataForSEO Search with Provider-specific Parameters"
import os
from litellm import search

os.environ["DATAFORSEO_LOGIN"] = "your-login"
os.environ["DATAFORSEO_PASSWORD"] = "your-password"

response = search(
    query="AI developments",
    search_provider="dataforseo",
    max_results=10,
    # DataForSEO-specific parameters
    country="United States",       # Country name for location_name
    language_code="en",            # Language code
    depth=20,                      # Number of results (max 700)
    device="desktop",              # Device type ('desktop', 'mobile', 'tablet')
    os="windows"                   # Operating system
)
```


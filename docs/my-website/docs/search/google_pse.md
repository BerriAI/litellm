# Google Programmable Search Engine (PSE)

**Get API Key:** [Google Cloud Console](https://console.cloud.google.com/apis/credentials)  
**Create Search Engine:** [Programmable Search Engine](https://programmablesearchengine.google.com/)

## Setup

1. Go to [Google Developers Programmable Search Engine](https://programmablesearchengine.google.com/) and log in or create an account
2. Click the **Add** button in the control panel
3. Enter a search engine name and configure properties:
   - Choose which sites to search (entire web or specific sites)
   - Set language and other preferences
   - Verify you're not a robot
4. Click **Create** button
5. Once created, you'll see:
   - **Search engine ID (cx)** - Copy this for `GOOGLE_PSE_ENGINE_ID`
   - Instructions to get your API key
6. Generate API key:
   - Go to [Google Cloud Console - Credentials](https://console.cloud.google.com/apis/credentials)
   - Create a new API key or use existing one
   - Enable **Custom Search API** for your project
   - Copy the API key for `GOOGLE_PSE_API_KEY`

## LiteLLM Python SDK

```python showLineNumbers title="Google PSE Search"
import os
from litellm import search

os.environ["GOOGLE_PSE_API_KEY"] = "AIza..."
os.environ["GOOGLE_PSE_ENGINE_ID"] = "your-search-engine-id"

response = search(
    query="latest AI developments",
    search_provider="google_pse",
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
  - search_tool_name: google-search
    litellm_params:
      search_provider: google_pse
      api_key: os.environ/GOOGLE_PSE_API_KEY
      search_engine_id: os.environ/GOOGLE_PSE_ENGINE_ID
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
    "max_results": 10
  }'
```

## Provider-specific Parameters

```python showLineNumbers title="Google PSE Search with Provider-specific Parameters"
import os
from litellm import search

os.environ["GOOGLE_PSE_API_KEY"] = "AIza..."
os.environ["GOOGLE_PSE_ENGINE_ID"] = "your-search-engine-id"

response = search(
    query="latest AI research papers",
    search_provider="google_pse",
    max_results=10,
    search_domain_filter=["arxiv.org"],
    # Google PSE-specific parameters (use actual Google PSE API parameter names)
    dateRestrict="m6",               # 'm6' = last 6 months, 'd7' = last 7 days
    lr="lang_en",                    # Language restriction (e.g., 'lang_en', 'lang_es')
    safe="active",                   # Search safety level ('active' or 'off')
    exactTerms="machine learning",   # Phrase that all documents must contain
    fileType="pdf"                   # File type to restrict results to
)
```

